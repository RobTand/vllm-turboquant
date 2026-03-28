# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Optimized fused TurboQuant decode kernel for SM121 (DGX Spark / GB10).

Three key optimizations over the baseline unfused kernel:

1. **Fused postprocess** — inverse transform runs in-register after the
   attention loop, eliminating intermediate HBM round-trip and a kernel launch.

2. **Split-K parallelism** — for long sequences, the KV scan is split across
   multiple programs to increase GPU occupancy (from ~32 programs to 512+).
   Partial softmax results are merged in a fused reduction+postprocess kernel.

3. **Optimized bit unpacking** — replaces the per-bit loop (3 loads for 3-bit
   indices) with 2-load straight-line extraction, saving ~33% of loads for the
   MSE index path.

4. **Autotuned BLOCK_N / num_warps** — Triton searches over tile sizes and
   warp counts at first invocation.
"""

from __future__ import annotations

from functools import cache

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.turboquant_kv_cache import (
    TURBOQUANT_QJL_SCALE,
    apply_turboquant_query_transforms,
    get_turboquant_layout,
    get_turboquant_mse_inverse_transform_matrix,
    get_turboquant_qjl_inverse_transform_matrix,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

@cache
def _norm_lut(device_type: str, device_index: int | None) -> torch.Tensor:
    device = torch.device(device_type, device_index)
    values = torch.arange(1 << 16, dtype=torch.int32, device=device)
    return values.to(torch.int16).view(torch.float16).to(torch.float32)


def get_turboquant_norm_lut(device: torch.device) -> torch.Tensor:
    return _norm_lut(device.type, device.index)


def _require_gb10_cuda(device: torch.device) -> None:
    if device.type != "cuda":
        raise ValueError("TurboQuant Triton decode requires CUDA tensors.")
    capability = torch.cuda.get_device_capability(device)
    if capability != (12, 1):
        raise ValueError("TurboQuant KV cache requires NVIDIA GB10 / SM121.")


@triton.jit
def _apply_softcap(logits, softcap):
    scaled = logits / softcap
    return softcap * (2 * tl.sigmoid(2 * scaled) - 1)


@triton.jit
def _load_half_from_bytes(
    cache_ptr, token_base, lut_ptr,
    byte_offset: tl.constexpr, stride_cache_d: tl.constexpr,
):
    lo = tl.load(cache_ptr + token_base + byte_offset * stride_cache_d,
                 mask=True, other=0).to(tl.int32)
    hi = tl.load(cache_ptr + token_base + (byte_offset + 1) * stride_cache_d,
                 mask=True, other=0).to(tl.int32)
    return tl.load(lut_ptr + lo + (hi << 8), mask=True, other=0.0)


@triton.jit
def _unpack_signs(
    cache_ptr, token_base, offs_d, stride_cache_d: tl.constexpr,
    byte_offset: tl.constexpr, mask_d,
):
    bit_position = offs_d[None, :]
    qjl_byte_offset = byte_offset + bit_position // 8
    qjl_bit_offset = bit_position % 8
    byte = tl.load(
        cache_ptr + token_base[:, None] + qjl_byte_offset * stride_cache_d,
        mask=mask_d, other=0,
    ).to(tl.int32)
    bits = (byte >> qjl_bit_offset) & 1
    return bits.to(tl.float32) * 2.0 - 1.0


@triton.jit
def _unpack_indices_fast(
    cache_ptr, token_base, offs_d, stride_cache_d: tl.constexpr,
    bits: tl.constexpr, BLOCK_N: tl.constexpr, PADDED: tl.constexpr,
    base_offset: tl.constexpr, mask_d,
):
    """Extract multi-bit indices with 2 byte loads instead of `bits` loads.

    For 3-bit MSE indices, the original does 3 loads in a loop.  This does 2
    loads + straight-line shift/mask, saving ~33% of memory transactions and
    eliminating loop overhead.
    """
    if bits == 0:
        return tl.zeros([BLOCK_N, PADDED], dtype=tl.int32)

    first_bit_pos = offs_d[None, :] * bits
    byte_idx = first_bit_pos // 8
    bit_off = first_bit_pos % 8

    # Load the byte containing the first bit of each index
    byte_0 = tl.load(
        cache_ptr + token_base[:, None] + (base_offset + byte_idx) * stride_cache_d,
        mask=mask_d, other=0,
    ).to(tl.int32)

    # Load the next byte (needed when index spans a byte boundary)
    byte_1 = tl.load(
        cache_ptr + token_base[:, None] + (base_offset + byte_idx + 1) * stride_cache_d,
        mask=mask_d, other=0,
    ).to(tl.int32)

    # Combine: shift first byte right to align, OR in bits from second byte
    raw = (byte_0 >> bit_off) | (byte_1 << (8 - bit_off))
    return raw & ((1 << bits) - 1)


# ---------------------------------------------------------------------------
# Non-split fused kernel (for short sequences / few splits)
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 16}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 16}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_N": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_N": 32}, num_warps=8, num_stages=2),
    ],
    key=["G0_DIM", "G1_DIM", "G0_MSE_BITS", "G1_MSE_BITS"],
)
@triton.jit
def _tq_fused_decode_kernel(
    # query (pre-rotated)
    q_rot_0_ptr, q_qjl_0_ptr, q_rot_1_ptr, q_qjl_1_ptr,
    # KV caches
    key_cache_ptr, value_cache_ptr,
    # sequence metadata
    block_table_ptr, token_seq_ids_ptr, token_kv_lens_ptr,
    token_query_positions_ptr,
    # output
    out_ptr,
    # inverse transform matrices
    value_mse_inv_0_ptr, value_qjl_inv_0_ptr,
    value_mse_inv_1_ptr, value_qjl_inv_1_ptr,
    # value group indices
    value_group0_idx_ptr, value_group1_idx_ptr,
    # norm lookup + centroids
    norm_lut_ptr, centroids_0_ptr, centroids_1_ptr,
    # optional
    lse_ptr, sink_ptr, mm_prefix_range_ptr,
    # scalars
    softmax_scale, softcap,
    # strides: query
    q0_stride_0, q0_stride_1, q1_stride_0, q1_stride_1,
    # strides: KV cache
    stride_k_cache_0, stride_k_cache_1, stride_k_cache_2, stride_k_cache_3,
    stride_v_cache_0, stride_v_cache_1, stride_v_cache_2, stride_v_cache_3,
    # strides: block table, output
    block_table_stride,
    out_stride_0, out_stride_1, out_stride_2,
    # strides: inverse matrices
    mse_inv_0_stride_0, mse_inv_0_stride_1,
    qjl_inv_0_stride_0, qjl_inv_0_stride_1,
    mse_inv_1_stride_0, mse_inv_1_stride_1,
    qjl_inv_1_stride_0, qjl_inv_1_stride_1,
    # strides: group indices, lse
    group0_idx_stride_0, group1_idx_stride_0,
    lse_stride_0, lse_stride_1,
    # constexprs
    kv_group_num: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    USE_SOFTCAP: tl.constexpr,
    USE_SINKS: tl.constexpr,
    USE_MM_PREFIX: tl.constexpr,
    MAX_MM_RANGES: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    RETURN_LSE: tl.constexpr,
    POSTPROCESS_BLOCK: tl.constexpr,
    G0_DIM: tl.constexpr, G0_PADDED: tl.constexpr,
    G0_MSE_BITS: tl.constexpr, G0_GROUP_OFFSET: tl.constexpr,
    G0_QJL_OFFSET: tl.constexpr,
    G0_VECTOR_NORM_OFFSET: tl.constexpr, G0_RESIDUAL_NORM_OFFSET: tl.constexpr,
    G0_QJL_SCALE: tl.constexpr,
    G1_DIM: tl.constexpr, G1_PADDED: tl.constexpr,
    G1_MSE_BITS: tl.constexpr, G1_GROUP_OFFSET: tl.constexpr,
    G1_QJL_OFFSET: tl.constexpr,
    G1_VECTOR_NORM_OFFSET: tl.constexpr, G1_RESIDUAL_NORM_OFFSET: tl.constexpr,
    G1_QJL_SCALE: tl.constexpr,
):
    cur_token = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_kv_head = cur_head // kv_group_num

    seq_idx = tl.load(token_seq_ids_ptr + cur_token)
    seq_len = tl.load(token_kv_lens_ptr + cur_token)
    query_pos = tl.load(token_query_positions_ptr + cur_token)

    offs_d0 = tl.arange(0, G0_PADDED)
    mask_d0 = offs_d0 < G0_DIM
    offs_d1 = tl.arange(0, G1_PADDED)
    mask_d1 = offs_d1 < G1_DIM

    q_rot_0 = tl.load(q_rot_0_ptr + cur_token * q0_stride_0 + cur_head * q0_stride_1 + offs_d0, mask=mask_d0, other=0.0)
    q_qjl_0 = tl.load(q_qjl_0_ptr + cur_token * q0_stride_0 + cur_head * q0_stride_1 + offs_d0, mask=mask_d0, other=0.0)
    q_rot_1 = tl.load(q_rot_1_ptr + cur_token * q1_stride_0 + cur_head * q1_stride_1 + offs_d1, mask=mask_d1, other=0.0)
    q_qjl_1 = tl.load(q_qjl_1_ptr + cur_token * q1_stride_0 + cur_head * q1_stride_1 + offs_d1, mask=mask_d1, other=0.0)

    if USE_SINKS:
        e_max = tl.load(sink_ptr + cur_head).to(tl.float32)
        e_sum = tl.where(e_max > float("-inf"), 1.0, 0.0)
    else:
        e_max = -float("inf")
        e_sum = 0.0
    acc_mse_0 = tl.zeros([G0_PADDED], dtype=tl.float32)
    acc_qjl_0 = tl.zeros([G0_PADDED], dtype=tl.float32)
    acc_mse_1 = tl.zeros([G1_PADDED], dtype=tl.float32)
    acc_qjl_1 = tl.zeros([G1_PADDED], dtype=tl.float32)

    # === ATTENTION LOOP ===
    for start_n in range(0, seq_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seq_len
        block_ids = tl.load(block_table_ptr + seq_idx * block_table_stride + offs_n // BLOCK_SIZE, mask=mask_n, other=0)
        block_offsets = offs_n % BLOCK_SIZE
        key_token_base = block_ids * stride_k_cache_0 + block_offsets * stride_k_cache_1 + cur_kv_head * stride_k_cache_2
        value_token_base = block_ids * stride_v_cache_0 + block_offsets * stride_v_cache_1 + cur_kv_head * stride_v_cache_2

        # --- Key group 0 ---
        key_qjl_signs_0 = _unpack_signs(key_cache_ptr, key_token_base, offs_d0, stride_k_cache_3, G0_QJL_OFFSET, mask_n[:, None] & mask_d0[None, :])
        key_vector_norm_0 = _load_half_from_bytes(key_cache_ptr, key_token_base, norm_lut_ptr, G0_VECTOR_NORM_OFFSET, stride_k_cache_3)
        key_residual_norm_0 = _load_half_from_bytes(key_cache_ptr, key_token_base, norm_lut_ptr, G0_RESIDUAL_NORM_OFFSET, stride_k_cache_3)
        key_logits = key_vector_norm_0 * key_residual_norm_0 * tl.sum(key_qjl_signs_0 * q_qjl_0[None, :], axis=1)
        if G0_MSE_BITS > 0:
            key_indices_0 = _unpack_indices_fast(key_cache_ptr, key_token_base, offs_d0, stride_k_cache_3, G0_MSE_BITS, BLOCK_N, G0_PADDED, G0_GROUP_OFFSET, mask_n[:, None] & mask_d0[None, :])
            key_centroids_0 = tl.load(centroids_0_ptr + key_indices_0, mask=mask_n[:, None] & mask_d0[None, :], other=0.0)
            key_logits += key_vector_norm_0 * tl.sum(key_centroids_0 * q_rot_0[None, :], axis=1)

        # --- Key group 1 ---
        key_qjl_signs_1 = _unpack_signs(key_cache_ptr, key_token_base, offs_d1, stride_k_cache_3, G1_QJL_OFFSET, mask_n[:, None] & mask_d1[None, :])
        key_vector_norm_1 = _load_half_from_bytes(key_cache_ptr, key_token_base, norm_lut_ptr, G1_VECTOR_NORM_OFFSET, stride_k_cache_3)
        key_residual_norm_1 = _load_half_from_bytes(key_cache_ptr, key_token_base, norm_lut_ptr, G1_RESIDUAL_NORM_OFFSET, stride_k_cache_3)
        key_logits += key_vector_norm_1 * key_residual_norm_1 * tl.sum(key_qjl_signs_1 * q_qjl_1[None, :], axis=1)
        if G1_MSE_BITS > 0:
            key_indices_1 = _unpack_indices_fast(key_cache_ptr, key_token_base, offs_d1, stride_k_cache_3, G1_MSE_BITS, BLOCK_N, G1_PADDED, G1_GROUP_OFFSET, mask_n[:, None] & mask_d1[None, :])
            key_centroids_1 = tl.load(centroids_1_ptr + key_indices_1, mask=mask_n[:, None] & mask_d1[None, :], other=0.0)
            key_logits += key_vector_norm_1 * tl.sum(key_centroids_1 * q_rot_1[None, :], axis=1)

        # --- Softmax + masking ---
        logits = key_logits * softmax_scale
        if USE_SOFTCAP:
            logits = _apply_softcap(logits, softcap)
        valid_mask = mask_n
        if CAUSAL:
            valid_mask = valid_mask & (offs_n <= query_pos)
            if SLIDING_WINDOW > 0:
                valid_mask = valid_mask & ((query_pos - offs_n) < SLIDING_WINDOW)
        if USE_MM_PREFIX:
            for i in range(MAX_MM_RANGES):
                range_start = tl.load(mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + i * 2)
                range_end = tl.load(mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + i * 2 + 1)
                is_valid = range_start < range_end
                q_in_range = (query_pos >= range_start) & (query_pos <= range_end) & is_valid
                k_in_range = (offs_n >= range_start) & (offs_n <= range_end) & is_valid
                valid_mask = valid_mask | (q_in_range & k_in_range)
        logits = tl.where(valid_mask, logits, float("-inf"))
        n_e_max = tl.maximum(tl.max(logits, axis=0), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(logits - n_e_max)

        # --- Value accumulation ---
        value_vector_norm_0 = _load_half_from_bytes(value_cache_ptr, value_token_base, norm_lut_ptr, G0_VECTOR_NORM_OFFSET, stride_v_cache_3)
        value_residual_norm_0 = _load_half_from_bytes(value_cache_ptr, value_token_base, norm_lut_ptr, G0_RESIDUAL_NORM_OFFSET, stride_v_cache_3)
        value_vector_norm_1 = _load_half_from_bytes(value_cache_ptr, value_token_base, norm_lut_ptr, G1_VECTOR_NORM_OFFSET, stride_v_cache_3)
        value_residual_norm_1 = _load_half_from_bytes(value_cache_ptr, value_token_base, norm_lut_ptr, G1_RESIDUAL_NORM_OFFSET, stride_v_cache_3)

        acc_mse_0 *= re_scale
        acc_qjl_0 *= re_scale
        acc_mse_1 *= re_scale
        acc_qjl_1 *= re_scale

        if G0_MSE_BITS > 0:
            value_indices_0 = _unpack_indices_fast(value_cache_ptr, value_token_base, offs_d0, stride_v_cache_3, G0_MSE_BITS, BLOCK_N, G0_PADDED, G0_GROUP_OFFSET, mask_n[:, None] & mask_d0[None, :])
            value_centroids_0 = tl.load(centroids_0_ptr + value_indices_0, mask=mask_n[:, None] & mask_d0[None, :], other=0.0)
            acc_mse_0 += tl.sum((p * value_vector_norm_0)[:, None] * value_centroids_0, axis=0)
        value_qjl_signs_0 = _unpack_signs(value_cache_ptr, value_token_base, offs_d0, stride_v_cache_3, G0_QJL_OFFSET, mask_n[:, None] & mask_d0[None, :])
        acc_qjl_0 += tl.sum((p * value_vector_norm_0 * value_residual_norm_0 * G0_QJL_SCALE)[:, None] * value_qjl_signs_0, axis=0)

        if G1_MSE_BITS > 0:
            value_indices_1 = _unpack_indices_fast(value_cache_ptr, value_token_base, offs_d1, stride_v_cache_3, G1_MSE_BITS, BLOCK_N, G1_PADDED, G1_GROUP_OFFSET, mask_n[:, None] & mask_d1[None, :])
            value_centroids_1 = tl.load(centroids_1_ptr + value_indices_1, mask=mask_n[:, None] & mask_d1[None, :], other=0.0)
            acc_mse_1 += tl.sum((p * value_vector_norm_1)[:, None] * value_centroids_1, axis=0)
        value_qjl_signs_1 = _unpack_signs(value_cache_ptr, value_token_base, offs_d1, stride_v_cache_3, G1_QJL_OFFSET, mask_n[:, None] & mask_d1[None, :])
        acc_qjl_1 += tl.sum((p * value_vector_norm_1 * value_residual_norm_1 * G1_QJL_SCALE)[:, None] * value_qjl_signs_1, axis=0)

        e_sum = e_sum * re_scale + tl.sum(p, axis=0)
        e_max = n_e_max

    # === FUSED POSTPROCESS ===
    inv_e_sum = 1.0 / e_sum
    acc_mse_0 *= inv_e_sum
    acc_qjl_0 *= inv_e_sum
    acc_mse_1 *= inv_e_sum
    acc_qjl_1 *= inv_e_sum
    out_base = cur_token * out_stride_0 + cur_head * out_stride_1

    for out_tile_start in tl.static_range(0, G0_PADDED, POSTPROCESS_BLOCK):
        offs_out = out_tile_start + tl.arange(0, POSTPROCESS_BLOCK)
        mask_out = offs_out < G0_DIM
        mse_inv = tl.load(value_mse_inv_0_ptr + offs_d0[:, None] * mse_inv_0_stride_0 + offs_out[None, :] * mse_inv_0_stride_1, mask=mask_d0[:, None] & mask_out[None, :], other=0.0)
        qjl_inv = tl.load(value_qjl_inv_0_ptr + offs_d0[:, None] * qjl_inv_0_stride_0 + offs_out[None, :] * qjl_inv_0_stride_1, mask=mask_d0[:, None] & mask_out[None, :], other=0.0)
        recon = tl.sum(acc_mse_0[:, None] * mse_inv, axis=0) + tl.sum(acc_qjl_0[:, None] * qjl_inv, axis=0)
        gidx = tl.load(value_group0_idx_ptr + cur_head * group0_idx_stride_0 + offs_out, mask=mask_out, other=0).to(tl.int64)
        tl.store(out_ptr + out_base + gidx * out_stride_2, recon, mask=mask_out)

    for out_tile_start in tl.static_range(0, G1_PADDED, POSTPROCESS_BLOCK):
        offs_out = out_tile_start + tl.arange(0, POSTPROCESS_BLOCK)
        mask_out = offs_out < G1_DIM
        mse_inv = tl.load(value_mse_inv_1_ptr + offs_d1[:, None] * mse_inv_1_stride_0 + offs_out[None, :] * mse_inv_1_stride_1, mask=mask_d1[:, None] & mask_out[None, :], other=0.0)
        qjl_inv = tl.load(value_qjl_inv_1_ptr + offs_d1[:, None] * qjl_inv_1_stride_0 + offs_out[None, :] * qjl_inv_1_stride_1, mask=mask_d1[:, None] & mask_out[None, :], other=0.0)
        recon = tl.sum(acc_mse_1[:, None] * mse_inv, axis=0) + tl.sum(acc_qjl_1[:, None] * qjl_inv, axis=0)
        gidx = tl.load(value_group1_idx_ptr + cur_head * group1_idx_stride_0 + offs_out, mask=mask_out, other=0).to(tl.int64)
        tl.store(out_ptr + out_base + gidx * out_stride_2, recon, mask=mask_out)

    if RETURN_LSE:
        tl.store(lse_ptr + cur_head * lse_stride_0 + cur_token * lse_stride_1, tl.log(e_sum) + e_max)


# ---------------------------------------------------------------------------
# Split-K attention kernel — scans a slice of the KV sequence
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 16}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_N": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_N": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_N": 64}, num_warps=8, num_stages=3),
    ],
    key=["G0_DIM", "G1_DIM", "G0_MSE_BITS", "G1_MSE_BITS"],
)
@triton.jit
def _tq_splitk_attention_kernel(
    # query (pre-rotated)
    q_rot_0_ptr, q_qjl_0_ptr, q_rot_1_ptr, q_qjl_1_ptr,
    # KV caches
    key_cache_ptr, value_cache_ptr,
    # sequence metadata
    block_table_ptr, token_seq_ids_ptr, token_kv_lens_ptr,
    token_query_positions_ptr,
    # partial outputs  [num_splits, num_tokens, num_heads, ...]
    partial_acc_mse_0_ptr, partial_acc_qjl_0_ptr,
    partial_acc_mse_1_ptr, partial_acc_qjl_1_ptr,
    partial_emax_ptr, partial_esum_ptr,
    # norm lookup + centroids
    norm_lut_ptr, centroids_0_ptr, centroids_1_ptr,
    # optional
    sink_ptr, mm_prefix_range_ptr,
    # scalars
    softmax_scale, softcap,
    num_tokens: tl.constexpr,
    num_splits: tl.constexpr,
    tokens_per_split: tl.constexpr,
    # strides: query
    q0_stride_0, q0_stride_1, q1_stride_0, q1_stride_1,
    # strides: KV cache
    stride_k_cache_0, stride_k_cache_1, stride_k_cache_2, stride_k_cache_3,
    stride_v_cache_0, stride_v_cache_1, stride_v_cache_2, stride_v_cache_3,
    # strides: block table
    block_table_stride,
    # strides: partial outputs [flat: split * num_tokens * num_heads, dim]
    partial_stride_0, partial_stride_1,
    # stride for scalar partials (emax/esum): [num_splits * num_tokens, num_heads]
    scalar_stride_0,
    # constexprs
    kv_group_num: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    USE_SOFTCAP: tl.constexpr,
    USE_SINKS: tl.constexpr,
    USE_MM_PREFIX: tl.constexpr,
    MAX_MM_RANGES: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    G0_DIM: tl.constexpr, G0_PADDED: tl.constexpr,
    G0_MSE_BITS: tl.constexpr, G0_GROUP_OFFSET: tl.constexpr,
    G0_QJL_OFFSET: tl.constexpr,
    G0_VECTOR_NORM_OFFSET: tl.constexpr, G0_RESIDUAL_NORM_OFFSET: tl.constexpr,
    G0_QJL_SCALE: tl.constexpr,
    G1_DIM: tl.constexpr, G1_PADDED: tl.constexpr,
    G1_MSE_BITS: tl.constexpr, G1_GROUP_OFFSET: tl.constexpr,
    G1_QJL_OFFSET: tl.constexpr,
    G1_VECTOR_NORM_OFFSET: tl.constexpr, G1_RESIDUAL_NORM_OFFSET: tl.constexpr,
    G1_QJL_SCALE: tl.constexpr,
):
    # Grid: (num_splits * num_tokens, num_heads)
    flat_id = tl.program_id(0)
    cur_head = tl.program_id(1)
    split_idx = flat_id // num_tokens
    cur_token = flat_id % num_tokens
    cur_kv_head = cur_head // kv_group_num

    seq_idx = tl.load(token_seq_ids_ptr + cur_token)
    seq_len = tl.load(token_kv_lens_ptr + cur_token)
    query_pos = tl.load(token_query_positions_ptr + cur_token)

    # This split's KV range
    kv_start = split_idx * tokens_per_split
    kv_end = tl.minimum(kv_start + tokens_per_split, seq_len)

    offs_d0 = tl.arange(0, G0_PADDED)
    mask_d0 = offs_d0 < G0_DIM
    offs_d1 = tl.arange(0, G1_PADDED)
    mask_d1 = offs_d1 < G1_DIM

    q_rot_0 = tl.load(q_rot_0_ptr + cur_token * q0_stride_0 + cur_head * q0_stride_1 + offs_d0, mask=mask_d0, other=0.0)
    q_qjl_0 = tl.load(q_qjl_0_ptr + cur_token * q0_stride_0 + cur_head * q0_stride_1 + offs_d0, mask=mask_d0, other=0.0)
    q_rot_1 = tl.load(q_rot_1_ptr + cur_token * q1_stride_0 + cur_head * q1_stride_1 + offs_d1, mask=mask_d1, other=0.0)
    q_qjl_1 = tl.load(q_qjl_1_ptr + cur_token * q1_stride_0 + cur_head * q1_stride_1 + offs_d1, mask=mask_d1, other=0.0)

    e_max = -float("inf")
    e_sum = 0.0
    # For the first split, incorporate sinks
    if USE_SINKS:
        if split_idx == 0:
            e_max = tl.load(sink_ptr + cur_head).to(tl.float32)
            e_sum = tl.where(e_max > float("-inf"), 1.0, 0.0)

    acc_mse_0 = tl.zeros([G0_PADDED], dtype=tl.float32)
    acc_qjl_0 = tl.zeros([G0_PADDED], dtype=tl.float32)
    acc_mse_1 = tl.zeros([G1_PADDED], dtype=tl.float32)
    acc_qjl_1 = tl.zeros([G1_PADDED], dtype=tl.float32)

    for start_n in range(kv_start, kv_end, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < kv_end
        block_ids = tl.load(block_table_ptr + seq_idx * block_table_stride + offs_n // BLOCK_SIZE, mask=mask_n, other=0)
        block_offsets = offs_n % BLOCK_SIZE
        key_token_base = block_ids * stride_k_cache_0 + block_offsets * stride_k_cache_1 + cur_kv_head * stride_k_cache_2
        value_token_base = block_ids * stride_v_cache_0 + block_offsets * stride_v_cache_1 + cur_kv_head * stride_v_cache_2

        # Key group 0
        key_qjl_signs_0 = _unpack_signs(key_cache_ptr, key_token_base, offs_d0, stride_k_cache_3, G0_QJL_OFFSET, mask_n[:, None] & mask_d0[None, :])
        key_vector_norm_0 = _load_half_from_bytes(key_cache_ptr, key_token_base, norm_lut_ptr, G0_VECTOR_NORM_OFFSET, stride_k_cache_3)
        key_residual_norm_0 = _load_half_from_bytes(key_cache_ptr, key_token_base, norm_lut_ptr, G0_RESIDUAL_NORM_OFFSET, stride_k_cache_3)
        key_logits = key_vector_norm_0 * key_residual_norm_0 * tl.sum(key_qjl_signs_0 * q_qjl_0[None, :], axis=1)
        if G0_MSE_BITS > 0:
            ki0 = _unpack_indices_fast(key_cache_ptr, key_token_base, offs_d0, stride_k_cache_3, G0_MSE_BITS, BLOCK_N, G0_PADDED, G0_GROUP_OFFSET, mask_n[:, None] & mask_d0[None, :])
            kc0 = tl.load(centroids_0_ptr + ki0, mask=mask_n[:, None] & mask_d0[None, :], other=0.0)
            key_logits += key_vector_norm_0 * tl.sum(kc0 * q_rot_0[None, :], axis=1)

        # Key group 1
        key_qjl_signs_1 = _unpack_signs(key_cache_ptr, key_token_base, offs_d1, stride_k_cache_3, G1_QJL_OFFSET, mask_n[:, None] & mask_d1[None, :])
        key_vector_norm_1 = _load_half_from_bytes(key_cache_ptr, key_token_base, norm_lut_ptr, G1_VECTOR_NORM_OFFSET, stride_k_cache_3)
        key_residual_norm_1 = _load_half_from_bytes(key_cache_ptr, key_token_base, norm_lut_ptr, G1_RESIDUAL_NORM_OFFSET, stride_k_cache_3)
        key_logits += key_vector_norm_1 * key_residual_norm_1 * tl.sum(key_qjl_signs_1 * q_qjl_1[None, :], axis=1)
        if G1_MSE_BITS > 0:
            ki1 = _unpack_indices_fast(key_cache_ptr, key_token_base, offs_d1, stride_k_cache_3, G1_MSE_BITS, BLOCK_N, G1_PADDED, G1_GROUP_OFFSET, mask_n[:, None] & mask_d1[None, :])
            kc1 = tl.load(centroids_1_ptr + ki1, mask=mask_n[:, None] & mask_d1[None, :], other=0.0)
            key_logits += key_vector_norm_1 * tl.sum(kc1 * q_rot_1[None, :], axis=1)

        # Softmax + masking
        logits = key_logits * softmax_scale
        if USE_SOFTCAP:
            logits = _apply_softcap(logits, softcap)
        valid_mask = mask_n
        if CAUSAL:
            valid_mask = valid_mask & (offs_n <= query_pos)
            if SLIDING_WINDOW > 0:
                valid_mask = valid_mask & ((query_pos - offs_n) < SLIDING_WINDOW)
        if USE_MM_PREFIX:
            for i in range(MAX_MM_RANGES):
                rs = tl.load(mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + i * 2)
                re = tl.load(mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + i * 2 + 1)
                iv = rs < re
                valid_mask = valid_mask | (((query_pos >= rs) & (query_pos <= re) & iv) & ((offs_n >= rs) & (offs_n <= re) & iv))
        logits = tl.where(valid_mask, logits, float("-inf"))
        n_e_max = tl.maximum(tl.max(logits, axis=0), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(logits - n_e_max)

        # Value accumulation
        vvn0 = _load_half_from_bytes(value_cache_ptr, value_token_base, norm_lut_ptr, G0_VECTOR_NORM_OFFSET, stride_v_cache_3)
        vrn0 = _load_half_from_bytes(value_cache_ptr, value_token_base, norm_lut_ptr, G0_RESIDUAL_NORM_OFFSET, stride_v_cache_3)
        vvn1 = _load_half_from_bytes(value_cache_ptr, value_token_base, norm_lut_ptr, G1_VECTOR_NORM_OFFSET, stride_v_cache_3)
        vrn1 = _load_half_from_bytes(value_cache_ptr, value_token_base, norm_lut_ptr, G1_RESIDUAL_NORM_OFFSET, stride_v_cache_3)

        acc_mse_0 *= re_scale
        acc_qjl_0 *= re_scale
        acc_mse_1 *= re_scale
        acc_qjl_1 *= re_scale

        if G0_MSE_BITS > 0:
            vi0 = _unpack_indices_fast(value_cache_ptr, value_token_base, offs_d0, stride_v_cache_3, G0_MSE_BITS, BLOCK_N, G0_PADDED, G0_GROUP_OFFSET, mask_n[:, None] & mask_d0[None, :])
            vc0 = tl.load(centroids_0_ptr + vi0, mask=mask_n[:, None] & mask_d0[None, :], other=0.0)
            acc_mse_0 += tl.sum((p * vvn0)[:, None] * vc0, axis=0)
        vs0 = _unpack_signs(value_cache_ptr, value_token_base, offs_d0, stride_v_cache_3, G0_QJL_OFFSET, mask_n[:, None] & mask_d0[None, :])
        acc_qjl_0 += tl.sum((p * vvn0 * vrn0 * G0_QJL_SCALE)[:, None] * vs0, axis=0)

        if G1_MSE_BITS > 0:
            vi1 = _unpack_indices_fast(value_cache_ptr, value_token_base, offs_d1, stride_v_cache_3, G1_MSE_BITS, BLOCK_N, G1_PADDED, G1_GROUP_OFFSET, mask_n[:, None] & mask_d1[None, :])
            vc1 = tl.load(centroids_1_ptr + vi1, mask=mask_n[:, None] & mask_d1[None, :], other=0.0)
            acc_mse_1 += tl.sum((p * vvn1)[:, None] * vc1, axis=0)
        vs1 = _unpack_signs(value_cache_ptr, value_token_base, offs_d1, stride_v_cache_3, G1_QJL_OFFSET, mask_n[:, None] & mask_d1[None, :])
        acc_qjl_1 += tl.sum((p * vvn1 * vrn1 * G1_QJL_SCALE)[:, None] * vs1, axis=0)

        e_sum = e_sum * re_scale + tl.sum(p, axis=0)
        e_max = n_e_max

    # Write partial results
    partial_base = flat_id * partial_stride_0 + cur_head * partial_stride_1
    tl.store(partial_acc_mse_0_ptr + partial_base + offs_d0, acc_mse_0, mask=mask_d0)
    tl.store(partial_acc_qjl_0_ptr + partial_base + offs_d0, acc_qjl_0, mask=mask_d0)
    tl.store(partial_acc_mse_1_ptr + partial_base + offs_d1, acc_mse_1, mask=mask_d1)
    tl.store(partial_acc_qjl_1_ptr + partial_base + offs_d1, acc_qjl_1, mask=mask_d1)
    # emax/esum are [num_splits * num_tokens, num_heads]
    scalar_base = flat_id * scalar_stride_0 + cur_head
    tl.store(partial_emax_ptr + scalar_base, e_max)
    tl.store(partial_esum_ptr + scalar_base, e_sum)


# ---------------------------------------------------------------------------
# Split-K reduction + fused postprocess
# ---------------------------------------------------------------------------

@triton.jit
def _tq_splitk_reduce_kernel(
    # partial inputs [num_splits * num_tokens, num_heads, ...]
    partial_acc_mse_0_ptr, partial_acc_qjl_0_ptr,
    partial_acc_mse_1_ptr, partial_acc_qjl_1_ptr,
    partial_emax_ptr, partial_esum_ptr,
    # output
    out_ptr,
    # inverse transform matrices
    value_mse_inv_0_ptr, value_qjl_inv_0_ptr,
    value_mse_inv_1_ptr, value_qjl_inv_1_ptr,
    # value group indices
    value_group0_idx_ptr, value_group1_idx_ptr,
    # optional
    lse_ptr,
    # strides
    partial_stride_0, partial_stride_1,
    scalar_stride_0,
    out_stride_0, out_stride_1, out_stride_2,
    mse_inv_0_stride_0, mse_inv_0_stride_1,
    qjl_inv_0_stride_0, qjl_inv_0_stride_1,
    mse_inv_1_stride_0, mse_inv_1_stride_1,
    qjl_inv_1_stride_0, qjl_inv_1_stride_1,
    group0_idx_stride_0, group1_idx_stride_0,
    lse_stride_0, lse_stride_1,
    # constexprs
    num_tokens: tl.constexpr,
    num_splits: tl.constexpr,
    RETURN_LSE: tl.constexpr,
    POSTPROCESS_BLOCK: tl.constexpr,
    G0_DIM: tl.constexpr, G0_PADDED: tl.constexpr,
    G1_DIM: tl.constexpr, G1_PADDED: tl.constexpr,
):
    # Grid: (num_tokens, num_heads)
    cur_token = tl.program_id(0)
    cur_head = tl.program_id(1)

    offs_d0 = tl.arange(0, G0_PADDED)
    mask_d0 = offs_d0 < G0_DIM
    offs_d1 = tl.arange(0, G1_PADDED)
    mask_d1 = offs_d1 < G1_DIM

    # Merge partial softmax results across splits
    merged_mse_0 = tl.zeros([G0_PADDED], dtype=tl.float32)
    merged_qjl_0 = tl.zeros([G0_PADDED], dtype=tl.float32)
    merged_mse_1 = tl.zeros([G1_PADDED], dtype=tl.float32)
    merged_qjl_1 = tl.zeros([G1_PADDED], dtype=tl.float32)
    merged_emax = -float("inf")
    merged_esum = 0.0

    for s in range(num_splits):
        flat_id = s * num_tokens + cur_token
        base = flat_id * partial_stride_0 + cur_head * partial_stride_1
        s_emax = tl.load(partial_emax_ptr + flat_id * scalar_stride_0 + cur_head)
        s_esum = tl.load(partial_esum_ptr + flat_id * scalar_stride_0 + cur_head)

        new_max = tl.maximum(merged_emax, s_emax)
        old_scale = tl.exp(merged_emax - new_max)
        new_scale = tl.exp(s_emax - new_max)

        merged_mse_0 = merged_mse_0 * old_scale + new_scale * tl.load(partial_acc_mse_0_ptr + base + offs_d0, mask=mask_d0, other=0.0)
        merged_qjl_0 = merged_qjl_0 * old_scale + new_scale * tl.load(partial_acc_qjl_0_ptr + base + offs_d0, mask=mask_d0, other=0.0)
        merged_mse_1 = merged_mse_1 * old_scale + new_scale * tl.load(partial_acc_mse_1_ptr + base + offs_d1, mask=mask_d1, other=0.0)
        merged_qjl_1 = merged_qjl_1 * old_scale + new_scale * tl.load(partial_acc_qjl_1_ptr + base + offs_d1, mask=mask_d1, other=0.0)
        merged_esum = merged_esum * old_scale + s_esum * new_scale
        merged_emax = new_max

    # Normalize
    inv_sum = 1.0 / merged_esum
    merged_mse_0 *= inv_sum
    merged_qjl_0 *= inv_sum
    merged_mse_1 *= inv_sum
    merged_qjl_1 *= inv_sum

    # Fused inverse transform + scatter
    out_base = cur_token * out_stride_0 + cur_head * out_stride_1

    for ts in tl.static_range(0, G0_PADDED, POSTPROCESS_BLOCK):
        offs_out = ts + tl.arange(0, POSTPROCESS_BLOCK)
        mask_out = offs_out < G0_DIM
        mi = tl.load(value_mse_inv_0_ptr + offs_d0[:, None] * mse_inv_0_stride_0 + offs_out[None, :] * mse_inv_0_stride_1, mask=mask_d0[:, None] & mask_out[None, :], other=0.0)
        qi = tl.load(value_qjl_inv_0_ptr + offs_d0[:, None] * qjl_inv_0_stride_0 + offs_out[None, :] * qjl_inv_0_stride_1, mask=mask_d0[:, None] & mask_out[None, :], other=0.0)
        recon = tl.sum(merged_mse_0[:, None] * mi, axis=0) + tl.sum(merged_qjl_0[:, None] * qi, axis=0)
        gidx = tl.load(value_group0_idx_ptr + cur_head * group0_idx_stride_0 + offs_out, mask=mask_out, other=0).to(tl.int64)
        tl.store(out_ptr + out_base + gidx * out_stride_2, recon, mask=mask_out)

    for ts in tl.static_range(0, G1_PADDED, POSTPROCESS_BLOCK):
        offs_out = ts + tl.arange(0, POSTPROCESS_BLOCK)
        mask_out = offs_out < G1_DIM
        mi = tl.load(value_mse_inv_1_ptr + offs_d1[:, None] * mse_inv_1_stride_0 + offs_out[None, :] * mse_inv_1_stride_1, mask=mask_d1[:, None] & mask_out[None, :], other=0.0)
        qi = tl.load(value_qjl_inv_1_ptr + offs_d1[:, None] * qjl_inv_1_stride_0 + offs_out[None, :] * qjl_inv_1_stride_1, mask=mask_d1[:, None] & mask_out[None, :], other=0.0)
        recon = tl.sum(merged_mse_1[:, None] * mi, axis=0) + tl.sum(merged_qjl_1[:, None] * qi, axis=0)
        gidx = tl.load(value_group1_idx_ptr + cur_head * group1_idx_stride_0 + offs_out, mask=mask_out, other=0).to(tl.int64)
        tl.store(out_ptr + out_base + gidx * out_stride_2, recon, mask=mask_out)

    if RETURN_LSE:
        tl.store(lse_ptr + cur_head * lse_stride_0 + cur_token * lse_stride_1, tl.log(merged_esum) + merged_emax)


# ---------------------------------------------------------------------------
# Python entry point
# ---------------------------------------------------------------------------

# GB10 has ~50 SMs. For decode with N heads, we want enough programs to
# saturate: target >= 2 programs per SM = ~100 programs.
# With 32 heads, need >= 4 splits.  Don't split-K unless it buys >= 4 splits.
_SPLITK_MIN_SPLITS = 4
_SPLITK_TARGET_SPLITS_MAX = 64


def turboquant_fused_decode_attention_fwd(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    key_group_indices: tuple[torch.Tensor, torch.Tensor],
    value_group_indices: tuple[torch.Tensor, torch.Tensor],
    key_rotations: tuple[torch.Tensor, torch.Tensor],
    key_qjl_matrices: tuple[torch.Tensor, torch.Tensor],
    value_rotations: tuple[torch.Tensor, torch.Tensor],
    value_qjl_matrices: tuple[torch.Tensor, torch.Tensor],
    centroids: dict[int, torch.Tensor],
    norm_lut: torch.Tensor,
    softmax_scale: float,
    kv_cache_dtype: str,
    token_seq_ids: torch.Tensor | None = None,
    token_kv_lens: torch.Tensor | None = None,
    token_query_positions: torch.Tensor | None = None,
    kv_head_for_query_head: torch.Tensor | None = None,
    key_query_group_indices: tuple[torch.Tensor, torch.Tensor] | None = None,
    value_query_group_indices: tuple[torch.Tensor, torch.Tensor] | None = None,
    value_mse_inverse_matrices: tuple[torch.Tensor, torch.Tensor] | None = None,
    value_qjl_inverse_matrices: tuple[torch.Tensor, torch.Tensor] | None = None,
    causal: bool = True,
    sliding_window: tuple[int, int] = (-1, -1),
    sinks: torch.Tensor | None = None,
    mm_prefix_range: torch.Tensor | None = None,
    logits_soft_cap: float = 0.0,
    output_lse: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Optimized TurboQuant decode with autotune + split-K + fused postprocess."""
    if query.ndim != 3:
        raise ValueError(f"Expected query shape [T, H, D], got {query.shape}")
    _require_gb10_cuda(query.device)

    layout = get_turboquant_layout(kv_cache_dtype, query.shape[-1])
    kv_group_num = query.shape[1] // key_cache.shape[2]
    if kv_head_for_query_head is None:
        kv_head_for_query_head = (
            torch.arange(query.shape[1], device=query.device, dtype=torch.int64)
            // kv_group_num
        )

    q_rot, q_qjl = apply_turboquant_query_transforms(
        query=query,
        group_indices=key_group_indices,
        rotations=key_rotations,
        qjl_matrices=key_qjl_matrices,
        kv_head_for_query_head=kv_head_for_query_head,
        per_query_group_indices=key_query_group_indices,
    )

    if token_seq_ids is None or token_kv_lens is None or token_query_positions is None:
        query_lens = query_start_loc[1:] - query_start_loc[:-1]
        token_seq_ids = torch.repeat_interleave(
            torch.arange(seq_lens.shape[0], device=query.device, dtype=torch.int32),
            query_lens,
        )
        token_offsets = torch.arange(
            query.shape[0], device=query.device, dtype=torch.int32
        ) - query_start_loc.index_select(0, token_seq_ids.to(torch.int64))
        token_kv_lens = seq_lens.index_select(0, token_seq_ids.to(torch.int64)).to(torch.int32)
        if causal:
            token_query_positions = (
                token_kv_lens
                - query_lens.index_select(0, token_seq_ids.to(torch.int64))
                + token_offsets
            ).to(torch.int32)
        else:
            token_query_positions = torch.zeros_like(token_offsets, dtype=torch.int32)

    if value_query_group_indices is None:
        value_query_group_indices = tuple(
            group.index_select(0, kv_head_for_query_head)
            for group in value_group_indices
        )

    group0 = layout.groups[0]
    group1 = layout.groups[1]
    if value_mse_inverse_matrices is None:
        value_mse_inverse_matrices = (
            get_turboquant_mse_inverse_transform_matrix(query.device, group0.dim, seed_offset=101),
            get_turboquant_mse_inverse_transform_matrix(query.device, group1.dim, seed_offset=211),
        )
    if value_qjl_inverse_matrices is None:
        value_qjl_inverse_matrices = (
            get_turboquant_qjl_inverse_transform_matrix(query.device, group0.dim, seed_offset=307),
            get_turboquant_qjl_inverse_transform_matrix(query.device, group1.dim, seed_offset=401),
        )

    g0_pad = triton.next_power_of_2(group0.dim)
    g1_pad = triton.next_power_of_2(group1.dim)
    postprocess_block = min(32, g0_pad, g1_pad)
    output = torch.empty_like(query) if out is None else out

    lse = (
        output_lse if output_lse is not None
        else torch.empty((query.shape[1], query.shape[0]), dtype=torch.float32, device=query.device)
    )
    sink_tensor = sinks if sinks is not None else torch.empty(1, dtype=torch.float32, device=query.device)
    mm_prefix_tensor = mm_prefix_range if mm_prefix_range is not None else torch.empty(1, dtype=torch.int32, device=query.device)
    max_mm_ranges = mm_prefix_range.shape[1] if mm_prefix_range is not None else 1

    max_seq_len = int(seq_lens.max().item()) if seq_lens.numel() > 0 else 0
    num_tokens = query.shape[0]
    num_heads = query.shape[1]

    # Decide: split-K or direct.
    # Only use split-K when it yields enough splits to improve occupancy,
    # and only during decode (few query tokens).
    # Dynamic tokens_per_split: aim for 4-16 splits depending on seq length.
    potential_splits = max_seq_len // 1024 if max_seq_len >= 4096 else 0
    use_splitk = potential_splits >= _SPLITK_MIN_SPLITS and num_tokens <= 4
    if use_splitk:
        num_splits = max(_SPLITK_MIN_SPLITS, min(_SPLITK_TARGET_SPLITS_MAX, potential_splits))
        tokens_per_split = (max_seq_len + num_splits - 1) // num_splits

        # Allocate partial result buffers
        partial_shape_0 = (num_splits * num_tokens, num_heads, max(g0_pad, g1_pad))
        partial_acc_mse_0 = torch.empty(partial_shape_0, dtype=torch.float32, device=query.device)
        partial_acc_qjl_0 = torch.empty(partial_shape_0, dtype=torch.float32, device=query.device)
        partial_acc_mse_1 = torch.empty(partial_shape_0, dtype=torch.float32, device=query.device)
        partial_acc_qjl_1 = torch.empty(partial_shape_0, dtype=torch.float32, device=query.device)
        partial_emax = torch.empty((num_splits * num_tokens, num_heads), dtype=torch.float32, device=query.device)
        partial_esum = torch.empty_like(partial_emax)

        _tq_splitk_attention_kernel[(num_splits * num_tokens, num_heads)](
            q_rot[0], q_qjl[0], q_rot[1], q_qjl[1],
            key_cache, value_cache,
            block_table, token_seq_ids, token_kv_lens, token_query_positions,
            partial_acc_mse_0, partial_acc_qjl_0,
            partial_acc_mse_1, partial_acc_qjl_1,
            partial_emax, partial_esum,
            norm_lut, centroids[group0.mse_bits], centroids[group1.mse_bits],
            sink_tensor, mm_prefix_tensor,
            softmax_scale, logits_soft_cap,
            num_tokens=num_tokens,
            num_splits=num_splits,
            tokens_per_split=tokens_per_split,
            q0_stride_0=q_rot[0].stride(0), q0_stride_1=q_rot[0].stride(1),
            q1_stride_0=q_rot[1].stride(0), q1_stride_1=q_rot[1].stride(1),
            stride_k_cache_0=key_cache.stride(0), stride_k_cache_1=key_cache.stride(1),
            stride_k_cache_2=key_cache.stride(2), stride_k_cache_3=key_cache.stride(3),
            stride_v_cache_0=value_cache.stride(0), stride_v_cache_1=value_cache.stride(1),
            stride_v_cache_2=value_cache.stride(2), stride_v_cache_3=value_cache.stride(3),
            block_table_stride=block_table.stride(0),
            partial_stride_0=partial_acc_mse_0.stride(0),
            partial_stride_1=partial_acc_mse_0.stride(1),
            scalar_stride_0=partial_emax.stride(0),
            kv_group_num=kv_group_num, BLOCK_SIZE=key_cache.shape[1],
            CAUSAL=causal, USE_SOFTCAP=logits_soft_cap > 0,
            USE_SINKS=sinks is not None,
            USE_MM_PREFIX=mm_prefix_range is not None, MAX_MM_RANGES=max_mm_ranges,
            SLIDING_WINDOW=(sliding_window[0] + 1 if sliding_window[0] >= 0 else 0),
            G0_DIM=group0.dim, G0_PADDED=g0_pad, G0_MSE_BITS=group0.mse_bits,
            G0_GROUP_OFFSET=0, G0_QJL_OFFSET=group0.qjl_offset,
            G0_VECTOR_NORM_OFFSET=group0.vector_norm_offset,
            G0_RESIDUAL_NORM_OFFSET=group0.residual_norm_offset,
            G0_QJL_SCALE=TURBOQUANT_QJL_SCALE / group0.dim,
            G1_DIM=group1.dim, G1_PADDED=g1_pad, G1_MSE_BITS=group1.mse_bits,
            G1_GROUP_OFFSET=group0.packed_bytes, G1_QJL_OFFSET=group1.qjl_offset,
            G1_VECTOR_NORM_OFFSET=group1.vector_norm_offset,
            G1_RESIDUAL_NORM_OFFSET=group1.residual_norm_offset,
            G1_QJL_SCALE=TURBOQUANT_QJL_SCALE / group1.dim,
        )

        _tq_splitk_reduce_kernel[(num_tokens, num_heads)](
            partial_acc_mse_0, partial_acc_qjl_0,
            partial_acc_mse_1, partial_acc_qjl_1,
            partial_emax, partial_esum,
            output,
            value_mse_inverse_matrices[0], value_qjl_inverse_matrices[0],
            value_mse_inverse_matrices[1], value_qjl_inverse_matrices[1],
            value_query_group_indices[0], value_query_group_indices[1],
            lse,
            partial_stride_0=partial_acc_mse_0.stride(0),
            partial_stride_1=partial_acc_mse_0.stride(1),
            scalar_stride_0=partial_emax.stride(0),
            out_stride_0=output.stride(0), out_stride_1=output.stride(1), out_stride_2=output.stride(2),
            mse_inv_0_stride_0=value_mse_inverse_matrices[0].stride(0),
            mse_inv_0_stride_1=value_mse_inverse_matrices[0].stride(1),
            qjl_inv_0_stride_0=value_qjl_inverse_matrices[0].stride(0),
            qjl_inv_0_stride_1=value_qjl_inverse_matrices[0].stride(1),
            mse_inv_1_stride_0=value_mse_inverse_matrices[1].stride(0),
            mse_inv_1_stride_1=value_mse_inverse_matrices[1].stride(1),
            qjl_inv_1_stride_0=value_qjl_inverse_matrices[1].stride(0),
            qjl_inv_1_stride_1=value_qjl_inverse_matrices[1].stride(1),
            group0_idx_stride_0=value_query_group_indices[0].stride(0),
            group1_idx_stride_0=value_query_group_indices[1].stride(0),
            lse_stride_0=lse.stride(0), lse_stride_1=lse.stride(1),
            num_tokens=num_tokens, num_splits=num_splits,
            RETURN_LSE=output_lse is not None,
            POSTPROCESS_BLOCK=postprocess_block,
            G0_DIM=group0.dim, G0_PADDED=g0_pad,
            G1_DIM=group1.dim, G1_PADDED=g1_pad,
            num_warps=4, num_stages=1,
        )
    else:
        # Direct fused path (short sequences / multi-token prefill)
        _tq_fused_decode_kernel[(num_tokens, num_heads)](
            q_rot[0], q_qjl[0], q_rot[1], q_qjl[1],
            key_cache, value_cache,
            block_table, token_seq_ids, token_kv_lens, token_query_positions,
            output,
            value_mse_inverse_matrices[0], value_qjl_inverse_matrices[0],
            value_mse_inverse_matrices[1], value_qjl_inverse_matrices[1],
            value_query_group_indices[0], value_query_group_indices[1],
            norm_lut, centroids[group0.mse_bits], centroids[group1.mse_bits],
            lse, sink_tensor, mm_prefix_tensor,
            softmax_scale, logits_soft_cap,
            q0_stride_0=q_rot[0].stride(0), q0_stride_1=q_rot[0].stride(1),
            q1_stride_0=q_rot[1].stride(0), q1_stride_1=q_rot[1].stride(1),
            stride_k_cache_0=key_cache.stride(0), stride_k_cache_1=key_cache.stride(1),
            stride_k_cache_2=key_cache.stride(2), stride_k_cache_3=key_cache.stride(3),
            stride_v_cache_0=value_cache.stride(0), stride_v_cache_1=value_cache.stride(1),
            stride_v_cache_2=value_cache.stride(2), stride_v_cache_3=value_cache.stride(3),
            block_table_stride=block_table.stride(0),
            out_stride_0=output.stride(0), out_stride_1=output.stride(1), out_stride_2=output.stride(2),
            mse_inv_0_stride_0=value_mse_inverse_matrices[0].stride(0),
            mse_inv_0_stride_1=value_mse_inverse_matrices[0].stride(1),
            qjl_inv_0_stride_0=value_qjl_inverse_matrices[0].stride(0),
            qjl_inv_0_stride_1=value_qjl_inverse_matrices[0].stride(1),
            mse_inv_1_stride_0=value_mse_inverse_matrices[1].stride(0),
            mse_inv_1_stride_1=value_mse_inverse_matrices[1].stride(1),
            qjl_inv_1_stride_0=value_qjl_inverse_matrices[1].stride(0),
            qjl_inv_1_stride_1=value_qjl_inverse_matrices[1].stride(1),
            group0_idx_stride_0=value_query_group_indices[0].stride(0),
            group1_idx_stride_0=value_query_group_indices[1].stride(0),
            lse_stride_0=lse.stride(0), lse_stride_1=lse.stride(1),
            kv_group_num=kv_group_num, BLOCK_SIZE=key_cache.shape[1],
            CAUSAL=causal, USE_SOFTCAP=logits_soft_cap > 0,
            USE_SINKS=sinks is not None,
            USE_MM_PREFIX=mm_prefix_range is not None, MAX_MM_RANGES=max_mm_ranges,
            SLIDING_WINDOW=(sliding_window[0] + 1 if sliding_window[0] >= 0 else 0),
            RETURN_LSE=output_lse is not None,
            POSTPROCESS_BLOCK=postprocess_block,
            G0_DIM=group0.dim, G0_PADDED=g0_pad, G0_MSE_BITS=group0.mse_bits,
            G0_GROUP_OFFSET=0, G0_QJL_OFFSET=group0.qjl_offset,
            G0_VECTOR_NORM_OFFSET=group0.vector_norm_offset,
            G0_RESIDUAL_NORM_OFFSET=group0.residual_norm_offset,
            G0_QJL_SCALE=TURBOQUANT_QJL_SCALE / group0.dim,
            G1_DIM=group1.dim, G1_PADDED=g1_pad, G1_MSE_BITS=group1.mse_bits,
            G1_GROUP_OFFSET=group0.packed_bytes, G1_QJL_OFFSET=group1.qjl_offset,
            G1_VECTOR_NORM_OFFSET=group1.vector_norm_offset,
            G1_RESIDUAL_NORM_OFFSET=group1.residual_norm_offset,
            G1_QJL_SCALE=TURBOQUANT_QJL_SCALE / group1.dim,
        )

    return output
