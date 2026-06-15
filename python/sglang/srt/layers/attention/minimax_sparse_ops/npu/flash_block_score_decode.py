from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


def _next_power_of_2(x: int) -> int:
    return 1 << (int(x) - 1).bit_length()


def _get_vectorcore_num_safe() -> int:
    try:

        props = triton.runtime.driver.active.utils.get_device_properties(
            torch.npu.current_device()
        )
        n = int(props.get("num_vectorcore", -1))
        return max(1, n)
    except Exception:
        return 32


def _choose_num_kv_chunks(
    batch_size: int,
    num_kv_heads: int,
    max_num_kv_chunks: int = 8,
) -> int:
    max_num_kv_chunks = max(1, int(max_num_kv_chunks))
    max_num_kv_chunks = 1 << (max_num_kv_chunks.bit_length() - 1)

    vectorcore_num = _get_vectorcore_num_safe()
    target_grid = max(1, vectorcore_num * 8)
    denom = max(1, int(batch_size) * int(num_kv_heads))

    target = max(1, min(max_num_kv_chunks, target_grid // denom))
    return 1 << (target.bit_length() - 1)


def _torch_topk_from_score(
    score: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    topk: int,
) -> torch.Tensor:
    num_q_heads, batch_size, max_seqblock = score.shape
    device = score.device

    if topk <= 0:
        return torch.empty(
            (num_q_heads, batch_size, 0),
            dtype=torch.int32,
            device=device,
        )

    num_blocks = torch.div(
        seq_lens.to(torch.int64) + block_size - 1,
        block_size,
        rounding_mode="floor",
    )

    block_ids = torch.arange(max_seqblock, device=device, dtype=torch.int64)
    valid_score_mask = block_ids[None, None, :] < num_blocks[None, :, None]
    score_masked = score.masked_fill(~valid_score_mask, -float("inf"))

    k_eff = min(topk, max_seqblock)
    _, idx_eff = torch.topk(score_masked, k=k_eff, dim=-1)

    topk_idx = torch.full(
        (num_q_heads, batch_size, topk),
        -1,
        dtype=torch.int32,
        device=device,
    )
    topk_idx[:, :, :k_eff] = idx_eff.to(torch.int32)

    rank = torch.arange(topk, device=device, dtype=torch.int64)
    valid_topk_mask = rank[None, None, :] < torch.minimum(
        num_blocks[None, :, None],
        torch.tensor(topk, device=device, dtype=torch.int64),
    )

    topk_idx = torch.where(
        valid_topk_mask,
        topk_idx,
        torch.full_like(topk_idx, -1),
    )
    return topk_idx


@triton.heuristics(
    {
        "BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["topk"]),
    }
)
@triton.jit
def _topk_index_streaming_bnsd_kernel(
    score_ptr,
    topk_idx_ptr,
    seq_lens,
    block_size: tl.constexpr,
    topk: tl.constexpr,
    max_seqblock: tl.constexpr,
    stride_s_h,
    stride_s_b,
    stride_s_k,
    stride_tif_h,
    stride_tif_b,
    stride_tif_t,
    BLOCK_SIZE_T: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    seq_len = tl.load(seq_lens + pid_b).to(tl.int32)
    num_blocks = tl.cdiv(seq_len, block_size)

    off_t = tl.arange(0, BLOCK_SIZE_T)
    valid_topk_lane = off_t < topk

    top_scores = tl.where(
        valid_topk_lane,
        tl.full((BLOCK_SIZE_T,), -1.0e30, dtype=tl.float32),
        tl.full((BLOCK_SIZE_T,), 1.0e30, dtype=tl.float32),
    )
    top_indices = tl.full((BLOCK_SIZE_T,), -1, dtype=tl.int32)

    for block_idx in tl.range(0, max_seqblock):
        valid_block = block_idx < num_blocks
        score = tl.load(
            score_ptr
            + pid_h * stride_s_h
            + pid_b * stride_s_b
            + block_idx * stride_s_k,
            mask=valid_block,
            other=-1.0e30,
        ).to(tl.float32)
        score = tl.where(score != score, -1.0e30, score)

        min_score = tl.min(top_scores, axis=0)

        candidate_pos = tl.where(
            (top_scores == min_score) & valid_topk_lane,
            off_t,
            tl.full((BLOCK_SIZE_T,), BLOCK_SIZE_T, dtype=tl.int32),
        )
        min_pos = tl.min(candidate_pos, axis=0)

        do_replace = valid_block & (score > min_score)
        replace_mask = off_t == min_pos

        top_scores = tl.where(replace_mask & do_replace, score, top_scores)
        top_indices = tl.where(
            replace_mask & do_replace,
            block_idx,
            top_indices,
        )

    tl.store(
        topk_idx_ptr
        + pid_h * stride_tif_h
        + pid_b * stride_tif_b
        + off_t * stride_tif_t,
        top_indices.to(topk_idx_ptr.dtype.element_ty),
        mask=off_t < topk,
    )


def _streaming_topk_from_score(
    score: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    topk: int,
) -> torch.Tensor:
    num_q_heads, batch_size, max_seqblock = score.shape
    device = score.device

    if topk <= 0:
        return torch.empty(
            (num_q_heads, batch_size, 0),
            dtype=torch.int32,
            device=device,
        )

    topk_idx = torch.empty(
        (num_q_heads, batch_size, topk),
        dtype=torch.int32,
        device=device,
    )

    grid = (batch_size, num_q_heads)
    _topk_index_streaming_bnsd_kernel[grid](
        score,
        topk_idx,
        seq_lens,
        block_size,
        topk,
        max_seqblock,
        score.stride(0),
        score.stride(1),
        score.stride(2),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        num_warps=1,
        num_stages=1,
    )
    return topk_idx


@triton.heuristics(
    {
        "BLOCK_SIZE_H": lambda args: max(
            16, triton.next_power_of_2(args["gqa_group_size"])
        ),
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
    }
)
@triton.jit
def _decode_bnsd_score_kernel(
    q_ptr,
    k_cache_ptr,
    block_table_ptr,
    score_ptr,
    seq_lens,
    batch_size,
    gqa_group_size,
    head_dim,
    max_seqblock: tl.constexpr,
    block_size: tl.constexpr,
    sm_scale,
    init_blocks,
    local_blocks,
    stride_q_b,
    stride_q_h,
    stride_q_d,
    stride_k_block,
    stride_k_offset,
    stride_k_h,
    stride_k_d,
    stride_bt_b,
    stride_bt_n,
    stride_s_h,
    stride_s_b,
    stride_s_n,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    SCORE_TYPE: tl.constexpr,
):
    tl.static_assert(SCORE_TYPE == "max" or SCORE_TYPE == "lse")
    tl.static_assert(BLOCK_SIZE_N >= block_size)

    pid_b = tl.program_id(0)
    pid_kh = tl.program_id(1)
    pid_blk = tl.program_id(2)

    seq_len = tl.load(seq_lens + pid_b).to(tl.int32)
    num_blocks = tl.cdiv(seq_len, block_size)

    if pid_blk >= num_blocks:
        return

    pid_h = pid_kh * gqa_group_size

    off_h = tl.arange(0, BLOCK_SIZE_H)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    off_n = tl.arange(0, BLOCK_SIZE_N)

    q_offsets = (
        pid_b * stride_q_b
        + (pid_h + off_h[:, None]) * stride_q_h
        + off_d[None, :] * stride_q_d
    )
    q = tl.load(
        q_ptr + q_offsets,
        mask=(off_h[:, None] < gqa_group_size) & (off_d[None, :] < head_dim),
        other=0.0,
    )

    physical_block = tl.load(
        block_table_ptr + pid_b * stride_bt_b + pid_blk * stride_bt_n
    ).to(tl.int64)

    pos = pid_blk * block_size + off_n
    pos_mask = pos < seq_len

    k_offsets = (
        physical_block * stride_k_block
        + off_n[None, :] * stride_k_offset
        + pid_kh * stride_k_h
        + off_d[:, None] * stride_k_d
    )
    k = tl.load(
        k_cache_ptr + k_offsets,
        mask=(off_d[:, None] < head_dim) & pos_mask[None, :],
        other=0.0,
    )

    sm_scale_log2e = sm_scale * 1.4426950409
    qk = tl.dot(q, k) * sm_scale_log2e
    qk = tl.where(pos_mask[None, :], qk, float("-inf"))

    sub_max = tl.max(qk, axis=1)

    if SCORE_TYPE == "max":
        score = sub_max
    else:
        score = sub_max + tl.log2(tl.sum(tl.exp2(qk - sub_max[:, None]), axis=1))
        score = tl.where(score != score, float("-inf"), score)

    local_start = tl.maximum(0, num_blocks - local_blocks)
    is_init = pid_blk < init_blocks
    is_local = (pid_blk >= local_start) & (pid_blk < num_blocks)

    score = tl.where(is_init, 1e30, score)
    score = tl.where(is_local, 1e29, score)

    s_offsets = (pid_h + off_h) * stride_s_h + pid_b * stride_s_b + pid_blk * stride_s_n

    tl.store(
        score_ptr + s_offsets,
        score.to(score_ptr.dtype.element_ty),
        mask=off_h < gqa_group_size,
    )


@triton.heuristics(
    {
        "BLOCK_SIZE_H": lambda args: max(
            16, triton.next_power_of_2(args["gqa_group_size"])
        ),
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
        "HAS_SINK": lambda args: args["sink_ptr"] is not None,
    }
)
@triton.jit
def _decode_bnsd_attn_chunk_kernel(
    q_ptr,
    sink_ptr,
    k_cache_ptr,
    v_cache_ptr,
    block_table_ptr,
    o_ptr,
    lse_ptr,
    seq_lens,
    batch_size,
    gqa_group_size,
    head_dim,
    block_size: tl.constexpr,
    sm_scale,
    stride_q_b,
    stride_q_h,
    stride_q_d,
    stride_sink_h,
    stride_sink_d,
    stride_k_block,
    stride_k_offset,
    stride_k_h,
    stride_k_d,
    stride_v_block,
    stride_v_offset,
    stride_v_h,
    stride_v_d,
    stride_bt_b,
    stride_bt_n,
    stride_o_c,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_b,
    stride_l_h,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    NUM_KV_CHUNKS: tl.constexpr,
    HAS_SINK: tl.constexpr,
):
    tl.static_assert(BLOCK_SIZE_N >= block_size)

    pid_bc = tl.program_id(0)
    pid_kh = tl.program_id(1)

    pid_b = pid_bc % batch_size
    pid_c = pid_bc // batch_size
    pid_h = pid_kh * gqa_group_size

    seq_len = tl.load(seq_lens + pid_b).to(tl.int32)
    num_blocks = tl.cdiv(seq_len, block_size)

    chunk_size_blocks = tl.maximum(1, tl.cdiv(num_blocks, NUM_KV_CHUNKS))
    chunk_start_block = pid_c * chunk_size_blocks
    chunk_end_block = tl.minimum(chunk_start_block + chunk_size_blocks, num_blocks)

    if chunk_start_block >= chunk_end_block:
        return

    off_h = tl.arange(0, BLOCK_SIZE_H)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    off_n = tl.arange(0, BLOCK_SIZE_N)

    q_offsets = (
        pid_b * stride_q_b
        + (pid_h + off_h[:, None]) * stride_q_h
        + off_d[None, :] * stride_q_d
    )
    q = tl.load(
        q_ptr + q_offsets,
        mask=(off_h[:, None] < gqa_group_size) & (off_d[None, :] < head_dim),
        other=0.0,
    )

    sm_scale_log2e = sm_scale * 1.4426950409

    if HAS_SINK:
        if pid_c == 0:
            sink_offsets = (pid_h + off_h[:, None]) * stride_sink_h + off_d[
                None, :
            ] * stride_sink_d
            sink = tl.load(
                sink_ptr + sink_offsets,
                mask=(off_h[:, None] < gqa_group_size) & (off_d[None, :] < head_dim),
                other=0.0,
            ).to(tl.float32)
            qsink = tl.sum(q.to(tl.float32) * sink, axis=1) * sm_scale_log2e
            m_i = qsink
            l_i = tl.full((BLOCK_SIZE_H,), 1.0, dtype=tl.float32)
        else:
            m_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
            l_i = tl.full((BLOCK_SIZE_H,), 0.0, dtype=tl.float32)
    else:
        m_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
        l_i = tl.full((BLOCK_SIZE_H,), 0.0, dtype=tl.float32)

    acc_o = tl.full((BLOCK_SIZE_H, BLOCK_SIZE_D), 0.0, dtype=tl.float32)

    num_steps = chunk_end_block - chunk_start_block
    for step in tl.range(num_steps):
        logical_block = chunk_start_block + step
        physical_block = tl.load(
            block_table_ptr + pid_b * stride_bt_b + logical_block * stride_bt_n
        ).to(tl.int64)

        pos = logical_block * block_size + off_n
        pos_mask = pos < seq_len

        k_offsets = (
            physical_block * stride_k_block
            + off_n[None, :] * stride_k_offset
            + pid_kh * stride_k_h
            + off_d[:, None] * stride_k_d
        )
        k = tl.load(
            k_cache_ptr + k_offsets,
            mask=(off_d[:, None] < head_dim) & pos_mask[None, :],
            other=0.0,
        )

        v_offsets = (
            physical_block * stride_v_block
            + off_n[:, None] * stride_v_offset
            + pid_kh * stride_v_h
            + off_d[None, :] * stride_v_d
        )
        v = tl.load(
            v_cache_ptr + v_offsets,
            mask=pos_mask[:, None] & (off_d[None, :] < head_dim),
            other=0.0,
        )

        qk = tl.dot(q, k) * sm_scale_log2e
        qk = tl.where(pos_mask[None, :], qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp2(qk - m_new[:, None])
        l_new = tl.sum(p, axis=1)

        acc_scale = tl.exp2(m_i - m_new)
        acc_o = acc_o * acc_scale[:, None]
        acc_o += tl.dot(p.to(v.dtype), v)

        l_i = l_i * acc_scale + l_new
        m_i = m_new

    acc_o = acc_o / l_i[:, None]
    lse_i = m_i + tl.log2(l_i)

    o_offsets = (
        pid_c * stride_o_c
        + pid_b * stride_o_b
        + (pid_h + off_h[:, None]) * stride_o_h
        + off_d[None, :] * stride_o_d
    )
    tl.store(
        o_ptr + o_offsets,
        acc_o.to(o_ptr.dtype.element_ty),
        mask=(off_h[:, None] < gqa_group_size) & (off_d[None, :] < head_dim),
    )

    l_offsets = pid_c * stride_l_c + pid_b * stride_l_b + (pid_h + off_h) * stride_l_h
    tl.store(
        lse_ptr + l_offsets,
        lse_i.to(lse_ptr.dtype.element_ty),
        mask=off_h < gqa_group_size,
    )


@triton.heuristics(
    {
        "BLOCK_SIZE_H": lambda args: max(
            16, triton.next_power_of_2(args["gqa_group_size"])
        ),
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
        "HAS_SINK": lambda args: args["sink_ptr"] is not None,
    }
)
@triton.jit
def _decode_bnsd_score_attn_chunk_kernel(
    q_ptr,
    sink_ptr,
    k_cache_ptr,
    v_cache_ptr,
    block_table_ptr,
    o_ptr,
    lse_ptr,
    score_ptr,
    seq_lens,
    batch_size,
    gqa_group_size,
    head_dim,
    block_size: tl.constexpr,
    sm_scale,
    init_blocks,
    local_blocks,
    stride_q_b,
    stride_q_h,
    stride_q_d,
    stride_sink_h,
    stride_sink_d,
    stride_k_block,
    stride_k_offset,
    stride_k_h,
    stride_k_d,
    stride_v_block,
    stride_v_offset,
    stride_v_h,
    stride_v_d,
    stride_bt_b,
    stride_bt_n,
    stride_o_c,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_b,
    stride_l_h,
    stride_s_h,
    stride_s_b,
    stride_s_n,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    NUM_KV_CHUNKS: tl.constexpr,
    HAS_SINK: tl.constexpr,
    SCORE_TYPE: tl.constexpr,
):
    tl.static_assert(SCORE_TYPE == "max" or SCORE_TYPE == "lse")
    tl.static_assert(BLOCK_SIZE_N >= block_size)

    pid_bc = tl.program_id(0)
    pid_kh = tl.program_id(1)

    pid_b = pid_bc % batch_size
    pid_c = pid_bc // batch_size
    pid_h = pid_kh * gqa_group_size

    seq_len = tl.load(seq_lens + pid_b).to(tl.int32)
    num_blocks = tl.cdiv(seq_len, block_size)

    chunk_size_blocks = tl.maximum(1, tl.cdiv(num_blocks, NUM_KV_CHUNKS))
    chunk_start_block = pid_c * chunk_size_blocks
    chunk_end_block = tl.minimum(chunk_start_block + chunk_size_blocks, num_blocks)

    if chunk_start_block >= chunk_end_block:
        return

    off_h = tl.arange(0, BLOCK_SIZE_H)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    off_n = tl.arange(0, BLOCK_SIZE_N)

    q_offsets = (
        pid_b * stride_q_b
        + (pid_h + off_h[:, None]) * stride_q_h
        + off_d[None, :] * stride_q_d
    )
    q = tl.load(
        q_ptr + q_offsets,
        mask=(off_h[:, None] < gqa_group_size) & (off_d[None, :] < head_dim),
        other=0.0,
    )

    sm_scale_log2e = sm_scale * 1.4426950409

    if HAS_SINK:
        if pid_c == 0:
            sink_offsets = (pid_h + off_h[:, None]) * stride_sink_h + off_d[
                None, :
            ] * stride_sink_d
            sink = tl.load(
                sink_ptr + sink_offsets,
                mask=(off_h[:, None] < gqa_group_size) & (off_d[None, :] < head_dim),
                other=0.0,
            ).to(tl.float32)
            qsink = tl.sum(q.to(tl.float32) * sink, axis=1) * sm_scale_log2e
            m_i = qsink
            l_i = tl.full((BLOCK_SIZE_H,), 1.0, dtype=tl.float32)
        else:
            m_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
            l_i = tl.full((BLOCK_SIZE_H,), 0.0, dtype=tl.float32)
    else:
        m_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
        l_i = tl.full((BLOCK_SIZE_H,), 0.0, dtype=tl.float32)

    acc_o = tl.full((BLOCK_SIZE_H, BLOCK_SIZE_D), 0.0, dtype=tl.float32)
    local_start = tl.maximum(0, num_blocks - local_blocks)

    num_steps = chunk_end_block - chunk_start_block
    for step in tl.range(num_steps):
        logical_block = chunk_start_block + step
        physical_block = tl.load(
            block_table_ptr + pid_b * stride_bt_b + logical_block * stride_bt_n
        ).to(tl.int64)

        pos = logical_block * block_size + off_n
        pos_mask = pos < seq_len

        k_offsets = (
            physical_block * stride_k_block
            + off_n[None, :] * stride_k_offset
            + pid_kh * stride_k_h
            + off_d[:, None] * stride_k_d
        )
        k = tl.load(
            k_cache_ptr + k_offsets,
            mask=(off_d[:, None] < head_dim) & pos_mask[None, :],
            other=0.0,
        )

        v_offsets = (
            physical_block * stride_v_block
            + off_n[:, None] * stride_v_offset
            + pid_kh * stride_v_h
            + off_d[None, :] * stride_v_d
        )
        v = tl.load(
            v_cache_ptr + v_offsets,
            mask=pos_mask[:, None] & (off_d[None, :] < head_dim),
            other=0.0,
        )

        qk = tl.dot(q, k) * sm_scale_log2e
        qk = tl.where(pos_mask[None, :], qk, float("-inf"))

        sub_max = tl.max(qk, axis=1)
        if SCORE_TYPE == "max":
            score = sub_max
        else:
            score = sub_max + tl.log2(tl.sum(tl.exp2(qk - sub_max[:, None]), axis=1))
            score = tl.where(score != score, float("-inf"), score)

        is_init = logical_block < init_blocks
        is_local = (logical_block >= local_start) & (logical_block < num_blocks)
        score = tl.where(is_init, 1e30, score)
        score = tl.where(is_local, 1e29, score)

        s_offsets = (
            (pid_h + off_h) * stride_s_h
            + pid_b * stride_s_b
            + logical_block * stride_s_n
        )
        tl.store(
            score_ptr + s_offsets,
            score.to(score_ptr.dtype.element_ty),
            mask=off_h < gqa_group_size,
        )

        m_new = tl.maximum(m_i, sub_max)
        p = tl.exp2(qk - m_new[:, None])
        l_new = tl.sum(p, axis=1)

        acc_scale = tl.exp2(m_i - m_new)
        acc_o = acc_o * acc_scale[:, None]
        acc_o += tl.dot(p.to(v.dtype), v)

        l_i = l_i * acc_scale + l_new
        m_i = m_new

    acc_o = acc_o / l_i[:, None]
    lse_i = m_i + tl.log2(l_i)

    o_offsets = (
        pid_c * stride_o_c
        + pid_b * stride_o_b
        + (pid_h + off_h[:, None]) * stride_o_h
        + off_d[None, :] * stride_o_d
    )
    tl.store(
        o_ptr + o_offsets,
        acc_o.to(o_ptr.dtype.element_ty),
        mask=(off_h[:, None] < gqa_group_size) & (off_d[None, :] < head_dim),
    )

    l_offsets = pid_c * stride_l_c + pid_b * stride_l_b + (pid_h + off_h) * stride_l_h
    tl.store(
        lse_ptr + l_offsets,
        lse_i.to(lse_ptr.dtype.element_ty),
        mask=off_h < gqa_group_size,
    )


@triton.heuristics(
    {
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
    }
)
@triton.jit
def _merge_bnsd_attn_out_kernel(
    o_ptr,
    lse_ptr,
    seq_lens,
    out_ptr,
    head_dim,
    block_size: tl.constexpr,
    stride_o_c,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_b,
    stride_l_h,
    stride_out_b,
    stride_out_h,
    stride_out_d,
    NUM_KV_CHUNKS: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    off_d = tl.arange(0, BLOCK_SIZE_D)

    seq_len = tl.load(seq_lens + pid_b).to(tl.int32)
    num_blocks = tl.cdiv(seq_len, block_size)

    chunk_size_blocks = tl.maximum(1, tl.cdiv(num_blocks, NUM_KV_CHUNKS))
    valid_chunks = tl.cdiv(num_blocks, chunk_size_blocks)

    m = tl.full((), float("-inf"), dtype=tl.float32)
    l = tl.full((), 0.0, dtype=tl.float32)
    acc = tl.full((BLOCK_SIZE_D,), 0.0, dtype=tl.float32)

    for c in tl.static_range(0, NUM_KV_CHUNKS):
        valid = c < valid_chunks

        lse_c = tl.load(
            lse_ptr + c * stride_l_c + pid_b * stride_l_b + pid_h * stride_l_h,
            mask=valid,
            other=float("-inf"),
        )

        o_c = tl.load(
            o_ptr
            + c * stride_o_c
            + pid_b * stride_o_b
            + pid_h * stride_o_h
            + off_d * stride_o_d,
            mask=valid & (off_d < head_dim),
            other=0.0,
        ).to(tl.float32)

        m_new = tl.maximum(m, lse_c)
        scale_old = tl.exp2(m - m_new)
        scale_new = tl.exp2(lse_c - m_new)

        acc = acc * scale_old + o_c * scale_new
        l = l * scale_old + scale_new
        m = m_new

    out = acc / l

    tl.store(
        out_ptr + pid_b * stride_out_b + pid_h * stride_out_h + off_d * stride_out_d,
        out.to(out_ptr.dtype.element_ty),
        mask=off_d < head_dim,
    )


@torch.no_grad()
def flash_decode_bnsd_with_topk_idx(
    q: torch.Tensor,
    sink: Optional[torch.Tensor],
    k_cache_bnsd: torch.Tensor,
    v_cache_bnsd: Optional[torch.Tensor],
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seqlen: int,
    block_size: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    sm_scale: Optional[float] = None,
    score_type: str = "max",
    disable_index_value: bool = False,
    num_kv_chunks: Optional[int] = None,
    max_num_kv_chunks: int = 8,
    use_triton_topk: bool = True,
    num_topk_chunks: Optional[int] = None,
    use_fused_score_attn: bool = True,
) -> tuple:
    assert score_type in ("max", "lse")
    assert q.dtype in (torch.float16, torch.bfloat16)
    assert k_cache_bnsd.dtype == q.dtype
    assert block_table.dtype in (torch.int32, torch.int64)
    assert seq_lens.dtype in (torch.int32, torch.int64)

    if not disable_index_value:
        assert v_cache_bnsd is not None
        assert v_cache_bnsd.dtype == q.dtype
        assert v_cache_bnsd.shape == k_cache_bnsd.shape

    batch_size, num_q_heads, head_dim = q.shape
    _, block_size_from_cache, num_kv_heads, cache_head_dim = k_cache_bnsd.shape

    assert block_size_from_cache == block_size
    assert cache_head_dim == head_dim
    assert num_q_heads % num_kv_heads == 0
    assert block_table.shape[0] == batch_size
    assert seq_lens.shape[0] == batch_size

    gqa_group_size = num_q_heads // num_kv_heads

    if sm_scale is None:
        sm_scale = head_dim**-0.5

    max_seqblock = (max_seqlen + block_size - 1) // block_size
    block_size_n = _next_power_of_2(block_size)

    score = torch.full(
        (num_q_heads, batch_size, max_seqblock),
        -float("inf"),
        dtype=torch.float32,
        device=q.device,
    )

    if disable_index_value:
        grid_score = (batch_size, num_kv_heads, max_seqblock)
        _decode_bnsd_score_kernel[grid_score](
            q,
            k_cache_bnsd,
            block_table,
            score,
            seq_lens,
            batch_size,
            gqa_group_size,
            head_dim,
            max_seqblock,
            block_size,
            sm_scale,
            init_blocks,
            local_blocks,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k_cache_bnsd.stride(0),
            k_cache_bnsd.stride(1),
            k_cache_bnsd.stride(2),
            k_cache_bnsd.stride(3),
            block_table.stride(0),
            block_table.stride(1),
            score.stride(0),
            score.stride(1),
            score.stride(2),
            BLOCK_SIZE_N=block_size_n,
            SCORE_TYPE=score_type,
            num_warps=4,
            num_stages=2,
        )

        if use_triton_topk:
            topk_idx = _streaming_topk_from_score(score, seq_lens, block_size, topk)
        else:
            topk_idx = _torch_topk_from_score(score, seq_lens, block_size, topk)
        return None, topk_idx

    if num_kv_chunks is None:
        num_kv_chunks = _choose_num_kv_chunks(
            batch_size,
            num_kv_heads,
            max_num_kv_chunks=max_num_kv_chunks,
        )
    else:
        num_kv_chunks = int(num_kv_chunks)

    assert num_kv_chunks >= 1
    assert (num_kv_chunks & (num_kv_chunks - 1)) == 0

    o_chunks = torch.empty(
        (num_kv_chunks, batch_size, num_q_heads, head_dim),
        dtype=q.dtype,
        device=q.device,
    )
    lse_chunks = torch.empty(
        (num_kv_chunks, batch_size, num_q_heads),
        dtype=torch.float32,
        device=q.device,
    )

    grid_attn = (batch_size * num_kv_chunks, num_kv_heads)

    if use_fused_score_attn:
        _decode_bnsd_score_attn_chunk_kernel[grid_attn](
            q,
            sink,
            k_cache_bnsd,
            v_cache_bnsd,
            block_table,
            o_chunks,
            lse_chunks,
            score,
            seq_lens,
            batch_size,
            gqa_group_size,
            head_dim,
            block_size,
            sm_scale,
            init_blocks,
            local_blocks,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            sink.stride(0) if sink is not None else 0,
            sink.stride(1) if sink is not None else 0,
            k_cache_bnsd.stride(0),
            k_cache_bnsd.stride(1),
            k_cache_bnsd.stride(2),
            k_cache_bnsd.stride(3),
            v_cache_bnsd.stride(0),
            v_cache_bnsd.stride(1),
            v_cache_bnsd.stride(2),
            v_cache_bnsd.stride(3),
            block_table.stride(0),
            block_table.stride(1),
            o_chunks.stride(0),
            o_chunks.stride(1),
            o_chunks.stride(2),
            o_chunks.stride(3),
            lse_chunks.stride(0),
            lse_chunks.stride(1),
            lse_chunks.stride(2),
            score.stride(0),
            score.stride(1),
            score.stride(2),
            BLOCK_SIZE_N=block_size_n,
            NUM_KV_CHUNKS=num_kv_chunks,
            SCORE_TYPE=score_type,
            num_warps=4,
            num_stages=2,
        )
    else:
        grid_score = (batch_size, num_kv_heads, max_seqblock)
        _decode_bnsd_score_kernel[grid_score](
            q,
            k_cache_bnsd,
            block_table,
            score,
            seq_lens,
            batch_size,
            gqa_group_size,
            head_dim,
            max_seqblock,
            block_size,
            sm_scale,
            init_blocks,
            local_blocks,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k_cache_bnsd.stride(0),
            k_cache_bnsd.stride(1),
            k_cache_bnsd.stride(2),
            k_cache_bnsd.stride(3),
            block_table.stride(0),
            block_table.stride(1),
            score.stride(0),
            score.stride(1),
            score.stride(2),
            BLOCK_SIZE_N=block_size_n,
            SCORE_TYPE=score_type,
            num_warps=4,
            num_stages=2,
        )
        _decode_bnsd_attn_chunk_kernel[grid_attn](
            q,
            sink,
            k_cache_bnsd,
            v_cache_bnsd,
            block_table,
            o_chunks,
            lse_chunks,
            seq_lens,
            batch_size,
            gqa_group_size,
            head_dim,
            block_size,
            sm_scale,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            sink.stride(0) if sink is not None else 0,
            sink.stride(1) if sink is not None else 0,
            k_cache_bnsd.stride(0),
            k_cache_bnsd.stride(1),
            k_cache_bnsd.stride(2),
            k_cache_bnsd.stride(3),
            v_cache_bnsd.stride(0),
            v_cache_bnsd.stride(1),
            v_cache_bnsd.stride(2),
            v_cache_bnsd.stride(3),
            block_table.stride(0),
            block_table.stride(1),
            o_chunks.stride(0),
            o_chunks.stride(1),
            o_chunks.stride(2),
            o_chunks.stride(3),
            lse_chunks.stride(0),
            lse_chunks.stride(1),
            lse_chunks.stride(2),
            BLOCK_SIZE_N=block_size_n,
            NUM_KV_CHUNKS=num_kv_chunks,
            num_warps=4,
            num_stages=2,
        )

    if use_triton_topk:
        topk_idx = _streaming_topk_from_score(score, seq_lens, block_size, topk)
    else:
        topk_idx = _torch_topk_from_score(score, seq_lens, block_size, topk)

    o = torch.empty_like(q)

    grid_merge = (batch_size, num_q_heads)
    _merge_bnsd_attn_out_kernel[grid_merge](
        o_chunks,
        lse_chunks,
        seq_lens,
        o,
        head_dim,
        block_size,
        o_chunks.stride(0),
        o_chunks.stride(1),
        o_chunks.stride(2),
        o_chunks.stride(3),
        lse_chunks.stride(0),
        lse_chunks.stride(1),
        lse_chunks.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        NUM_KV_CHUNKS=num_kv_chunks,
        num_warps=4,
        num_stages=2,
    )

    return o, topk_idx
