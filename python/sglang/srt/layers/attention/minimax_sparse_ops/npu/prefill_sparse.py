from __future__ import annotations

from typing import List, Optional, Tuple

import torch


def _select_sparse_blocks(
    index_query: torch.Tensor,
    index_key: torch.Tensor,
    query_positions: torch.Tensor,
    seq_len: int,
    block_size: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
) -> torch.Tensor:
    num_blocks = (seq_len + block_size - 1) // block_size
    topk_blocks = min(topk, num_blocks)
    scores = torch.einsum("qhd,kd->qhk", index_query, index_key)
    scores = scores.float()

    padded_tokens = num_blocks * block_size
    if padded_tokens != seq_len:
        pad_len = padded_tokens - seq_len
        scores = torch.nn.functional.pad(scores, (0, pad_len), value=-1.0e30)

    key_positions = torch.arange(
        padded_tokens,
        device=index_query.device,
        dtype=query_positions.dtype,
    )
    valid = (key_positions[None, :] < seq_len) & (
        key_positions[None, :] <= query_positions[:, None]
    )
    scores = scores.masked_fill(~valid[:, None, :], -1.0e30)

    num_blocks_actual = (seq_len + block_size - 1) // block_size
    local_start = max(0, num_blocks_actual - local_blocks)
    for b in range(num_blocks_actual):
        block_start = b * block_size
        block_end = min(block_start + block_size, seq_len)
        if b < init_blocks:
            scores[:, :, block_start:block_end] = 1e30
        elif b >= local_start:
            scores[:, :, block_start:block_end] = 1e29

    block_scores = scores.view(
        index_query.shape[0],
        index_query.shape[1],
        num_blocks,
        block_size,
    ).amax(dim=-1)
    blocks = torch.topk(block_scores, k=topk_blocks, dim=-1).indices.to(torch.int32)
    if topk_blocks < topk:
        blocks = torch.nn.functional.pad(
            blocks,
            (0, topk - topk_blocks),
            value=-1,
        )
    return blocks


def _expand_blocks_to_tokens(
    block_indices: torch.Tensor,
    block_size: int,
    seq_len: int,
) -> torch.Tensor:
    offsets = torch.arange(
        block_size,
        device=block_indices.device,
        dtype=block_indices.dtype,
    )
    token_indices = block_indices[..., None] * block_size + offsets
    valid_tokens = (block_indices[..., None] >= 0) & (token_indices < seq_len)
    token_indices = token_indices.flatten(start_dim=-2)
    return torch.where(
        valid_tokens.flatten(start_dim=-2),
        token_indices,
        torch.full_like(token_indices, -1),
    )


def _apply_sparse_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    token_indices: torch.Tensor,
    query_positions: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    q_len = query.shape[0]
    seq_len = key_cache.shape[0]
    valid = (token_indices >= 0) & (token_indices < seq_len)
    valid = valid & (token_indices <= query_positions[:, None])
    safe_idx = token_indices.clamp(0, max(seq_len - 1, 0)).long()
    flat_idx = safe_idx.reshape(-1)
    k_selected = key_cache.index_select(0, flat_idx).view(
        q_len, -1, key_cache.shape[-1]
    )
    v_selected = value_cache.index_select(0, flat_idx).view(
        q_len, -1, value_cache.shape[-1]
    )
    scores = torch.einsum("qhd,qkd->qhk", query, k_selected)
    scores = scores.float()
    scores = scores * sm_scale
    scores = scores.masked_fill(~valid[:, None, :], -1.0e30)
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("qhk,qkd->qhd", probs.to(v_selected.dtype), v_selected)


@torch.no_grad()
def minimax_npu_sparse_prefill(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    sink: Optional[torch.Tensor],
    idx_q: torch.Tensor,
    idx_k_cache: torch.Tensor,
    idx_v_cache: Optional[torch.Tensor],
    idx_sink: Optional[torch.Tensor],
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    block_size_q: int,
    block_size_k: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    sm_scale: Optional[float] = None,
    idx_sm_scale: Optional[float] = None,
    score_type: str = "max",
    disable_index_value: bool = False,
    use_msa: bool = False,
    cu_seqblocks_q: Optional[torch.Tensor] = None,
    max_seqblock_q: Optional[int] = None,
    all_seqblock_q: Optional[int] = None,
    seqlens_cpu: Optional[List[int]] = None,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    num_q_heads = q.shape[1]
    num_idx_heads = idx_q.shape[1]
    num_kv_heads = k_cache.shape[1]
    head_dim = q.shape[2]
    idx_head_dim = idx_q.shape[2]
    head_dim_v = v_cache.shape[2]

    if sm_scale is None:
        sm_scale = head_dim**-0.5
    if idx_sm_scale is None:
        idx_sm_scale = idx_head_dim**-0.5

    batch_size = len(seq_lens)
    seqlens_list = seqlens_cpu if seqlens_cpu is not None else seq_lens.cpu().tolist()
    prefix_lens_list = prefix_lens.cpu().tolist()

    idx_o_parts = []
    o_parts = []

    for i in range(batch_size):
        s_len = seqlens_list[i]
        p_len = prefix_lens_list[i]
        q_len_i = s_len - p_len

        q_start = cu_seqlens[i].item()
        q_end = cu_seqlens[i + 1].item()

        q_i = q[q_start:q_end]
        idx_q_i = idx_q[q_start:q_end]

        idx_k_i = idx_k_cache[:s_len].reshape(s_len, num_idx_heads, idx_head_dim)
        idx_v_i = (
            idx_v_cache[:s_len].reshape(s_len, num_idx_heads, idx_head_dim)
            if idx_v_cache is not None and not disable_index_value
            else None
        )

        query_positions = torch.arange(p_len, s_len, device=q.device, dtype=torch.int32)

        topk_idx_i = _select_sparse_blocks(
            idx_q_i,
            idx_k_i,
            query_positions,
            s_len,
            block_size_k,
            topk,
            init_blocks,
            local_blocks,
        )

        idx_group_size = num_idx_heads // num_kv_heads
        if idx_group_size > 1:
            from ..common.index import topk_index_reduce

            topk_idx_i = topk_index_reduce(
                topk_idx_i.view(num_kv_heads, idx_group_size, -1, topk), dim=1
            )

        token_indices = _expand_blocks_to_tokens(topk_idx_i, block_size_k, s_len)

        k_i = k_cache[:s_len].reshape(s_len, num_kv_heads, head_dim)
        v_i = v_cache[:s_len].reshape(s_len, num_kv_heads, head_dim_v)

        o_i = _apply_sparse_attention(
            q_i,
            k_i,
            v_i,
            token_indices,
            query_positions,
            sm_scale,
        )

        if not disable_index_value and idx_v_i is not None:
            idx_token_indices = _expand_blocks_to_tokens(
                topk_idx_i, block_size_k, s_len
            )
            idx_o_i = _apply_sparse_attention(
                idx_q_i,
                idx_k_i,
                idx_v_i,
                idx_token_indices,
                query_positions,
                idx_sm_scale,
            )
        else:
            idx_o_i = torch.zeros(
                q_len_i,
                num_idx_heads,
                idx_head_dim,
                dtype=q.dtype,
                device=q.device,
            )

        idx_o_parts.append(idx_o_i)
        o_parts.append(o_i)

    idx_o = torch.cat(idx_o_parts, dim=0)
    o = torch.cat(o_parts, dim=0)

    return idx_o, o
