# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Triton kernels for decode context parallel (DCP).

Consolidated from the two merged DCP implementations:
  - create_triton_kv_indices_for_dcp_triton  (PR #25090, Triton/MHA path)
  - create_dcp_kv_indices / update_kv_lens_and_indices  (PR #14194, MLA path)
  - _correct_attn_cp_out_kernel / correct_attn_out / CPTritonContext  (PR #14194)
"""

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.utils import is_npu


# ---------------------------------------------------------------------------
# KV-index build (PR #25090, Triton/MHA): per-rank local KV indices.
# ---------------------------------------------------------------------------
@triton.jit
def create_triton_kv_indices_for_dcp_triton(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices_ptr,
    dcp_kernel_lens_ptr,
    kv_indptr,
    kv_start_idx,
    kv_indices_ptr,
    req_to_token_ptr_stride: tl.constexpr,
    dcp_size: tl.constexpr,
    dcp_rank: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(axis=0)
    req_pool_index = tl.load(req_pool_indices_ptr + pid)
    kv_indices_offset = tl.load(kv_indptr + pid)

    kv_start = 0
    if kv_start_idx:
        kv_start = tl.load(kv_start_idx + pid).to(tl.int32)

    # First absolute token position in this range owned by dcp_rank.
    # Triton follows C-style remainder for negative values, so avoid
    # computing the offset as a negative remainder when kv_start > dcp_rank.
    kv_start_mod = kv_start % dcp_size
    first = kv_start + ((dcp_rank + dcp_size - kv_start_mod) % dcp_size)
    local_len = tl.load(dcp_kernel_lens_ptr + pid).to(tl.int32)

    num_loop = tl.cdiv(local_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE).to(tl.int64) + i * BLOCK_SIZE
        mask = offset < local_len
        abs_pos = first + offset * dcp_size
        data = tl.load(
            req_to_token_ptr + req_pool_index * req_to_token_ptr_stride + abs_pos,
            mask=mask,
        )
        tl.store(
            kv_indices_ptr + kv_indices_offset + offset, data // dcp_size, mask=mask
        )


# ---------------------------------------------------------------------------
# KV-index build (PR #14194, MLA): global prefix+extend layout for the
# all-gathered dcp_kv_buffer, plus the per-rank shard/compact kernel.
# ---------------------------------------------------------------------------
@triton.jit
def create_dcp_kv_indices(
    kv_indptr,
    extend_lens_ptr,
    extend_cu_lens_ptr,
    extend_prefix_lens_ptr,
    extend_cu_prefix_lens_ptr,
    kv_indices_ptr,
    extend_prefix_lens_sum,
    dcp_world_size: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(axis=0)
    prefix_len = tl.load(extend_prefix_lens_ptr + pid)
    prefix_start = tl.load(extend_cu_prefix_lens_ptr + pid)
    kv_ind_start = tl.load(kv_indptr + pid)
    num_loop = tl.cdiv(prefix_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < prefix_len
        data = prefix_start + offset
        tl.store(kv_indices_ptr + kv_ind_start + offset, data, mask=mask)
    extend_len = tl.load(extend_lens_ptr + pid)
    extend_start = tl.load(extend_cu_lens_ptr + pid)
    num_loop = tl.cdiv(extend_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < extend_len
        data = extend_prefix_lens_sum + extend_start + offset
        tl.store(
            kv_indices_ptr + kv_ind_start + prefix_len + offset,
            data,
            mask=mask,
        )


@triton.jit
def update_kv_lens_and_indices(
    kv_lens: torch.Tensor,
    kv_lens_cumsum: torch.Tensor,
    kv_indices: torch.Tensor,
    local_kv_lens: torch.Tensor,
    local_kv_lens_cumsum: torch.Tensor,
    local_kv_indices: torch.Tensor,
    dcp_rank: tl.constexpr,
    dcp_world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    bs_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    local_kv_len = tl.load(local_kv_lens + bs_idx)
    local_kv_indices_start = tl.load(local_kv_lens_cumsum + bs_idx)
    kv_indices_start = tl.load(kv_lens_cumsum + bs_idx)

    block_start = block_idx * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < local_kv_len

    kv_indice_offsets = offsets * dcp_world_size + dcp_rank + kv_indices_start
    local_kv_indices_offsets = local_kv_indices_start + offsets

    kv_values = tl.load(kv_indices + kv_indice_offsets, mask=mask)
    tl.store(
        local_kv_indices + local_kv_indices_offsets,
        kv_values // dcp_world_size,
        mask=mask,
    )


# ---------------------------------------------------------------------------
# Partial-attention LSE correction (PR #14194, MLA path).
# ---------------------------------------------------------------------------
@triton.jit
def _correct_attn_cp_out_kernel(
    outputs_ptr,
    new_output_ptr,
    lses_ptr,
    vlse_ptr,
    outputs_stride_B,
    outputs_stride_H,
    outputs_stride_D,
    lses_stride_N,
    lses_stride_B,
    lses_stride_H,
    new_outputs_stride_H,
    new_outputs_stride_B,
    new_outputs_stride_D,
    lse_idx,
    HEAD_DIM: tl.constexpr,
    N_ROUNDED: tl.constexpr,
):
    """
    Apply the all-gathered lses to correct each local rank's attention
    output. we still need perform a cross-rank reduction to obtain the
    final attention output.

    Args:
        outputs_ptr (triton.PointerType):
            Pointer to input tensor of shape [ B, H, D ]
        lses_ptr (triton.PointerType):
            Pointer to input tensor of shape [ N, B, H ]
        new_output_ptr (triton.PointerType):
            Pointer to output tensor of shape [ H, B, D ]
        vlse_ptr (triton.PointerType):
            Pointer to output tensor of shape [ B, H ]
    """
    batch_idx = tl.program_id(axis=0).to(tl.int64)
    head_idx = tl.program_id(axis=1).to(tl.int64)

    # Use int32 for offsets where possible to reduce register pressure
    b_i32 = batch_idx.to(tl.int32)
    h_i32 = head_idx.to(tl.int32)

    # Vectorized load of LSE values: shape = [N]
    num_n_offsets = tl.arange(0, N_ROUNDED)
    lse_offsets = (
        num_n_offsets * lses_stride_N + b_i32 * lses_stride_B + h_i32 * lses_stride_H
    )

    # Compute final LSE using online softmax algorithm (more numerically stable)
    lse = tl.load(lses_ptr + lse_offsets)

    # Replace NaN and inf with -inf for numerical stability
    neg_inf = float("-inf")
    lse = tl.where((lse != lse) | (lse == float("inf")), neg_inf, lse)

    # Online softmax: find max, subtract, exp, sum, log
    lse_max = tl.max(lse, axis=0)
    lse_max = tl.where(lse_max == neg_inf, 0.0, lse_max)
    lse = lse - lse_max
    lse_exp = tl.exp2(lse)
    lse_acc = tl.sum(lse_exp, axis=0)
    final_lse = tl.log2(lse_acc) + lse_max

    # Compute correction factor
    lse_offset = lse_idx * lses_stride_N + b_i32 * lses_stride_B + h_i32 * lses_stride_H
    local_lse = tl.load(lses_ptr + lse_offset)
    lse_diff = local_lse - final_lse
    lse_diff = tl.where(
        (lse_diff != lse_diff) | (lse_diff == float("inf")),
        neg_inf,
        lse_diff,
    )
    factor = tl.exp2(lse_diff)

    # Store final LSE
    tl.store(vlse_ptr + b_i32 * lses_stride_B + h_i32 * lses_stride_H, final_lse)

    # Load output with vectorized access: shape = [D]
    d_offsets = tl.arange(0, HEAD_DIM)
    output_offsets = (
        batch_idx * outputs_stride_B
        + head_idx * outputs_stride_H
        + d_offsets * outputs_stride_D
    )

    new_output_offsets = (
        head_idx * new_outputs_stride_H
        + batch_idx * new_outputs_stride_B
        + d_offsets * new_outputs_stride_D
    )
    # Apply correction and store
    output = tl.load(outputs_ptr + output_offsets)
    output = output * factor
    tl.store(new_output_ptr + new_output_offsets, output)


class CPTritonContext:
    """The CPTritonContext is used to avoid recompilation of the Triton JIT."""

    def __init__(self):
        self.inner_kernel = None

    def call_kernel(self, kernel, grid, *regular_args, **const_args):
        if self.inner_kernel is None:
            self.inner_kernel = kernel[grid](*regular_args, **const_args)
        else:
            self.inner_kernel[grid](*regular_args)


def correct_attn_out(
    out: torch.Tensor,
    lses: torch.Tensor,
    cp_rank: int,
    ctx: Optional[CPTritonContext],
    new_output: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Correct the attention output using the all-gathered lses.

    Args:
        out: Tensor of shape [ B, H, D ]
        lses: Tensor of shape [ N, B, H ]
        cp_rank: Current rank in the context-parallel group
        ctx: Triton context to avoid recompilation

    Returns:
        Tuple of (out, lse) with corrected attention and final log-sum-exp.
    """
    if is_npu():
        return correct_attn_out_torch(out, lses, cp_rank, new_output)

    if ctx is None:
        ctx = CPTritonContext()

    # --- Normalize to 3D views ---
    if out.ndim == 4 and out.shape[1] == 1:
        out = out.squeeze(1)
    assert out.ndim == 3, f"expected out [B,H,D] or [B,1,H,D], got {tuple(out.shape)}"

    if lses.ndim == 4 and lses.shape[-1] == 1:
        lses = lses.squeeze(-1)
    if lses.ndim == 4 and lses.shape[1] == 1:
        lses = lses.squeeze(1)
    assert lses.ndim == 3, (
        f"expected lses [N,B,H] (optionally with a 1-sized extra dim), "
        f"got {tuple(lses.shape)}"
    )

    B, H, D = out.shape
    N = lses.shape[0]

    # Strides after we normalized shapes to 3-D views.  The kernel computes
    # offsets for `vlse_ptr` using lses_stride_B/H, so the output buffer must
    # have the same B/H stride layout as a slice of `lses`.
    o_sB, o_sH, o_sD = out.stride()
    l_sN, l_sB, l_sH = lses.stride()
    no_sH, no_sB, no_sD = new_output.stride()
    # Allocate LSE with the same B/H strides as `lses` so writes land correctly
    # even when `lses` is a non-contiguous view (e.g., 4-D to 3-D squeeze).
    lse = torch.empty_strided(
        (B, H), (l_sB, l_sH), device=lses.device, dtype=lses.dtype
    )

    # Kernel launch config
    grid = (B, H, 1)

    regular_args = (
        out,
        new_output,
        lses,
        lse,
        o_sB,
        o_sH,
        o_sD,
        l_sN,
        l_sB,
        l_sH,
        no_sH,
        no_sB,
        no_sD,
        cp_rank,
    )
    const_args = {"HEAD_DIM": D, "N_ROUNDED": N}

    ctx.call_kernel(_correct_attn_cp_out_kernel, grid, *regular_args, **const_args)
    return new_output, lse


# ---------------------------------------------------------------------------
# PyTorch fallback implementations for platforms without Triton (e.g. NPU).
# These replicate the Triton kernel logic using standard PyTorch operations.
# ---------------------------------------------------------------------------


def create_dcp_kv_indices_torch(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    dcp_kernel_lens: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_start_idx: Optional[torch.Tensor],
    kv_indices: Optional[torch.Tensor],
    req_to_token_stride: int,
    dcp_size: int,
    dcp_rank: int,
) -> torch.Tensor:
    """
    PyTorch fallback for ``create_triton_kv_indices_for_dcp_triton``.
    先计算每个req本rank分到的position id，根据position id计算每个token在虚拟的全局kvcache中的逻辑id
    逻辑id除以dcp_size，得到物理id，物理id即是本卡分到的kv在kvcache中的真实id

    Builds per-rank KV token indices by walking each request's owned
    positions (owner rule: ``pos % dcp_size == dcp_rank``), loading the
    global token index from ``req_to_token``, and converting to the local
    (sharded) format (``global_idx // dcp_size``).

    Args:
        req_to_token:    [max_batch, max_context_len] global token pool
        req_pool_indices:[num_seqs] int64 pool indices of active requests
        dcp_kernel_lens: [num_seqs] int32 per-rank visible KV length
        kv_indptr:       [num_seqs+1] int64 cumsum of dcp_kernel_lens
        kv_start_idx:    [num_seqs] int64 start offsets (or None → 0)
        kv_indices:      output buffer (int64) or None → allocate
        req_to_token_stride: stride of dim-0 in req_to_token
        dcp_size:        DCP world size
        dcp_rank:        current DCP rank

    Returns:
        kv_indices [sum(dcp_kernel_lens)] int64 tensor with local KV indices
    """
    num_seqs = req_pool_indices.shape[0]
    total_len = int(dcp_kernel_lens.sum().item())

    if kv_indices is None:
        kv_indices = torch.empty(
            total_len, dtype=torch.int64, device=req_to_token.device
        )

    for i in range(num_seqs):
        pool_idx = req_pool_indices[i].item()
        offset = kv_indptr[i].item()
        local_len = dcp_kernel_lens[i].item()

        if local_len == 0:
            continue

        kv_start = kv_start_idx[i].item() if kv_start_idx is not None else 0
        kv_start_mod = kv_start % dcp_size
        first = kv_start + ((dcp_rank + dcp_size - kv_start_mod) % dcp_size)

        # Vectorized: all absolute positions for this request
        abs_pos = first + torch.arange(
            local_len, dtype=torch.int64, device=req_to_token.device
        ) * dcp_size

        # Fetch global token indices and convert to local (sharded) indices
        global_indices = req_to_token[pool_idx, abs_pos]
        kv_indices[offset : offset + local_len] = global_indices // dcp_size

    return kv_indices


def create_global_dcp_kv_indices_torch(
    kv_indptr: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    extend_cu_lens: torch.Tensor,
    extend_prefix_lens: torch.Tensor,
    extend_cu_prefix_lens: torch.Tensor,
    kv_indices: torch.Tensor,
    extend_prefix_lens_sum: int,
    dcp_world_size: int,
) -> None:
    """
    PyTorch fallback for ``create_dcp_kv_indices``.

    Builds global KV indices that point into the ``dcp_kv_buffer``.
    Layout of dcp_kv_buffer:
      [rank0_r1.prefix, rank1_r1.prefix, ..., rankN_r1.prefix,
       rank0_r2.prefix, ..., rankN_r2.prefix,
       r1.extend, r2.extend, ..., rn.extend]
    dcp_kv_buffer的布局:[请求1的完整prefix, 请求2的完整prefix, ..., 请求n的完整prefix,
                         请求1的local extend, 请求2的local extend, ..., 请求n的local extend]
    
    dcp_kv_buffer可以视作kv cache
    kv_indices表示每个请求的各token在dcp_kv_buffer中的id
    kv_indices:[请求1的token1, 请求1的token2, ..., 请求1的tokenN,
                请求2的token1, 请求2的token2, ..., 
                ...,
                请求n的token1, 请求n的token2, ..., 请求n的tokenN]  


    The prefix section is all-gathered (all ranks interleaved), and
    the extend section is per-rank local KV appended at the end.

    Args:
        kv_indptr:      [num_seqs+1] int32 cumsum of total (prefix+extend) lens
        extend_seq_lens:[num_seqs] int32 extend token count per request
        extend_cu_lens: [num_seqs] int32 starting offset of extend per request
                        in the per-rank extend block
        extend_prefix_lens: [num_seqs] int32 prefix token count per request
        extend_cu_prefix_lens: [num_seqs] int32 cumsum offset of prefix per
                               request in the all-gathered prefix block
        kv_indices:     output [total_tokens] int32 tensor (pre-allocated)
        extend_prefix_lens_sum: total number of prefix tokens across all requests
        dcp_world_size: DCP world size (unused in index computation, kept for
                        signature compatibility with Triton version)
    """
    num_seqs = extend_seq_lens.shape[0]
    device = kv_indices.device

    for i in range(num_seqs):
        prefix_len = extend_prefix_lens[i].item()
        prefix_start = extend_cu_prefix_lens[i].item()
        extend_len = extend_seq_lens[i].item()
        extend_start = extend_cu_lens[i].item()
        kv_start = kv_indptr[i].item()

        # Prefix section: indices point into the all-gathered prefix block
        if prefix_len > 0:
            kv_indices[kv_start : kv_start + prefix_len] = (
                prefix_start + torch.arange(prefix_len, dtype=torch.int32, device=device)
            )

        # Extend section: indices point into the per-rank extend block
        # (starts after all prefix tokens)
        if extend_len > 0:
            kv_indices[
                kv_start + prefix_len : kv_start + prefix_len + extend_len
            ] = (
                extend_prefix_lens_sum
                + extend_start
                + torch.arange(extend_len, dtype=torch.int32, device=device)
            )


def update_kv_lens_and_indices_torch(
    kv_lens: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    local_kv_lens: torch.Tensor,
    local_kv_lens_cumsum: torch.Tensor,
    local_kv_indices: torch.Tensor,
    dcp_rank: int,
    dcp_world_size: int,
) -> None:
    """
    PyTorch fallback for ``update_kv_lens_and_indices``.

    Compacts the global KV indices down to per-rank local indices by
    applying the owner rule: for each request, take every ``dcp_world_size``-th
    element starting at offset ``dcp_rank`` from the global indices, and
    divide by ``dcp_world_size`` to produce the local (sharded) index.

    This is a decode-path kernel (called by ``plan_dcp_decode_metadata``).
    对每个请求，从全局 kv_indices 中每隔 dcp_world_size 取一个属于本 rank 的元素，除以 dcp_world_size 得到本地索引

    Args:
        kv_lens:       [num_seqs] int32 original KV lengths (unused by torch impl;
                       already consumed by ``update_local_kv_lens_for_dcp``)
        kv_indptr:     [num_seqs+1] int32 cumsum of original KV lengths
        kv_indices:    [total_orig_len] int32 global KV indices
        local_kv_lens: [num_seqs] int32 per-rank KV lengths (pre-computed)
        local_kv_lens_cumsum: [num_seqs+1] int32 cumsum of local_kv_lens
        local_kv_indices: [total_local_len] int32 output buffer (pre-allocated)
        dcp_rank:      current DCP rank
        dcp_world_size:DCP world size
    """
    num_seqs = local_kv_lens.shape[0]

    for i in range(num_seqs):
        local_len = local_kv_lens[i].item()
        if local_len == 0:
            continue

        global_start = kv_indptr[i].item() + dcp_rank
        global_end = global_start + local_len * dcp_world_size
        local_start = local_kv_lens_cumsum[i].item()

        # Slice with step: take every dcp_world_size-th element
        local_kv_indices[local_start : local_start + local_len] = (
            kv_indices[global_start:global_end:dcp_world_size] // dcp_world_size
        )


def correct_attn_out_torch(
    out: torch.Tensor,
    lses: torch.Tensor,
    cp_rank: int,
    new_output: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch fallback for ``correct_attn_out`` / ``_correct_attn_cp_out_kernel``.

    Merges partial attention outputs from different DCP ranks using the
    log-sum-exp (LSE) correction:
        global_lse  = logsumexp(lse_ranks, dim=0)
        scale       = exp(local_lse - global_lse)
        corrected   = attn_out * scale

    The caller is responsible for the cross-rank reduction (all-reduce or
    reduce-scatter) after this function returns.

    Args:
        out:         Attention output  [B, H, D]  (or [B, 1, H, D])
        lses:        All-gathered LSE  [N, B, H]  (or with extra dims)
        cp_rank:     Index of this rank's LSE in lses
        new_output:  Pre-allocated output buffer [H, B, D] (or None → allocate)

    Returns:
        (corrected_out [H, B, D], global_lse [B, H])
    """
    # --- Normalize to 3D views (same as Triton path) ---
    if out.ndim == 4 and out.shape[1] == 1:
        out = out.squeeze(1)
    assert out.ndim == 3, f"expected out [B,H,D], got {tuple(out.shape)}"

    if lses.ndim == 4 and lses.shape[-1] == 1:
        lses = lses.squeeze(-1)
    if lses.ndim == 4 and lses.shape[1] == 1:
        lses = lses.squeeze(1)
    assert lses.ndim == 3, (
        f"expected lses [N,B,H], got {tuple(lses.shape)}"
    )

    B, H, D = out.shape
    N = lses.shape[0]

    # Global LSE via logsumexp (natural log, equivalent to Triton's log2/exp2
    # w.r.t. the final scale factor since exp(ln(x))/exp(ln(y)) = exp2(log2(x))/exp2(log2(y)))
    global_lse = torch.logsumexp(lses, dim=0)  # [B, H]
    local_lse = lses[cp_rank]  # [B, H]

    scale = torch.nan_to_num(
        torch.exp(local_lse - global_lse),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )  # [B, H]

    # Apply correction: attn_out *= scale
    corrected = out * scale.unsqueeze(-1)  # [B, H, D]

    # Transpose to [H, B, D] layout expected by cp_lse_ag_out_rs_mla
    if new_output is None:
        new_output = corrected.transpose(0, 1).contiguous()  # [H, B, D]
    else:
        new_output.copy_(corrected.transpose(0, 1))

    return new_output, global_lse
