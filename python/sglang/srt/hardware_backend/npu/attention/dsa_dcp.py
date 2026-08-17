"""NPU DSA adaptation for token-sharded decode context parallelism."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.dcp.layout import (
    get_dcp_chain_spec_lens,
    get_dcp_lens,
    remap_dcp_sparse_indices,
)
from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def forward_dcp_sparse_attention(
    *,
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    topk_indices: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    forward_metadata,
    forward_batch: ForwardBatch,
    speculative_num_draft_tokens: int | None,
    page_size: int,
    scaling: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute one rank's sparse partial attention and natural-log LSE."""
    parallel = get_parallel()
    topk_indices = remap_dcp_sparse_indices(
        topk_indices, parallel.attn_dcp_size, parallel.attn_dcp_rank
    )
    topk_indices = _expand_sparse_indices(topk_indices)

    is_speculative = (
        forward_batch.forward_mode.is_target_verify()
        or forward_batch.forward_mode.is_draft_extend_v2()
    )
    if is_speculative:
        assert speculative_num_draft_tokens is not None
        tokens_per_request = speculative_num_draft_tokens
        block_tables = forward_metadata.dcp_spec_block_tables[::tokens_per_request]
    else:
        block_tables = forward_metadata.dcp_block_tables

    assert (
        block_tables is not None
    ), "NPU DSA+DCP sparse attention requires rank-local paged-KV metadata"
    local_lens = _get_local_kv_lens(
        forward_metadata=forward_metadata,
        is_speculative=is_speculative,
        tokens_per_request=speculative_num_draft_tokens,
        device=q_nope.device,
    )
    actual_seq_lengths_query = actual_seq_lengths_query.to(
        device=q_nope.device, dtype=torch.int32
    )
    sparse_kwargs = dict(
        query=q_nope,
        key=k_nope,
        value=k_nope,
        query_rope=q_rope,
        key_rope=k_rope,
        sparse_indices=topk_indices,
        scale_value=scaling,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_kv=local_lens,
        block_table=block_tables,
        sparse_block_size=1,
        layout_query="TND",
        layout_kv="PA_BSND",
        # Global top-k already enforces causal visibility. Local indices no
        # longer carry enough information for the kernel's causal sparse mode.
        sparse_mode=0,
        attention_mode=2,
    )

    custom_sparse_attention = getattr(
        torch.ops._C_ascend, "npu_sparse_flash_attention", None
    )
    if custom_sparse_attention is not None:
        attn_out, softmax_max, softmax_sum = custom_sparse_attention(
            return_softmax_lse=True, **sparse_kwargs
        )
        lse = softmax_max.float() + torch.log(softmax_sum.float())
        return attn_out, lse.permute(1, 0, 2).reshape(lse.shape[1], -1)

    import torch_npu

    attn_out, _, _ = torch_npu.npu_sparse_flash_attention(
        return_softmax_lse=False, **sparse_kwargs
    )
    lse = _compute_sparse_lse(
        q_nope=q_nope,
        q_rope=q_rope,
        k_nope=k_nope,
        k_rope=k_rope,
        sparse_indices=topk_indices,
        block_tables=block_tables,
        local_kv_lens=local_lens,
        actual_seq_lengths_query=actual_seq_lengths_query,
        is_prefill=forward_batch.forward_mode.is_extend_without_speculative(),
        page_size=page_size,
        scaling=scaling,
    )
    return attn_out, lse


def _expand_sparse_indices(topk_indices: torch.Tensor) -> torch.Tensor:
    return topk_indices.unsqueeze(-2) if topk_indices.dim() == 2 else topk_indices


def _get_local_kv_lens(
    *,
    forward_metadata,
    is_speculative: bool,
    tokens_per_request: int | None,
    device: torch.device,
) -> torch.Tensor:
    """Return local KV lengths without host transfers during graph capture."""
    parallel = get_parallel()
    dcp_spec_seq_lens = getattr(forward_metadata, "dcp_spec_seq_lens", None)
    dcp_seq_lens = getattr(forward_metadata, "dcp_seq_lens", None)
    if is_speculative and dcp_spec_seq_lens is not None:
        assert tokens_per_request is not None
        return dcp_spec_seq_lens.view(-1, tokens_per_request)[:, -1]
    if not is_speculative and dcp_seq_lens is not None:
        return dcp_seq_lens
    if forward_metadata.seq_lens_cpu_int is None:
        if is_speculative:
            assert tokens_per_request is not None
            return (
                get_dcp_chain_spec_lens(
                    forward_metadata.seq_lens,
                    tokens_per_request,
                    parallel.attn_dcp_size,
                    parallel.attn_dcp_rank,
                )
                .view(-1, tokens_per_request)[:, -1]
                .to(dtype=torch.int32)
            )
        return get_dcp_lens(
            forward_metadata.seq_lens,
            parallel.attn_dcp_size,
            parallel.attn_dcp_rank,
        ).to(dtype=torch.int32)

    if is_speculative:
        assert tokens_per_request is not None
        local_lens = forward_metadata.dcp_spec_seq_lens_cpu_int.view(
            -1, tokens_per_request
        )[:, -1]
    else:
        local_lens = forward_metadata.dcp_seq_lens_cpu_int
    assert local_lens is not None
    return local_lens.to(device=device, dtype=torch.int32)


def _compute_sparse_lse(
    *,
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    sparse_indices: torch.Tensor,
    block_tables: torch.Tensor,
    local_kv_lens: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    is_prefill: bool,
    page_size: int,
    scaling: float,
) -> torch.Tensor:
    """Compatibility path for CANN versions without paged sparse LSE."""
    if is_prefill:
        query_lens = torch.diff(
            torch.nn.functional.pad(actual_seq_lengths_query, (1, 0))
        )
        request_rows = torch.repeat_interleave(
            torch.arange(query_lens.shape[0], dtype=torch.int64, device=q_nope.device),
            query_lens.to(torch.int64),
        )
        num_valid_queries = request_rows.shape[0]
        request_rows = torch.nn.functional.pad(
            request_rows, (0, q_nope.shape[0] - num_valid_queries)
        )
    else:
        num_valid_queries = q_nope.shape[0]
        request_rows = (
            torch.arange(q_nope.shape[0], dtype=torch.int64, device=q_nope.device)
            * block_tables.shape[0]
            // q_nope.shape[0]
        )

    logical_indices = sparse_indices.squeeze(-2)
    lse_chunks = []
    for start in range(0, q_nope.shape[0], 128):
        end = min(start + 128, q_nope.shape[0])
        lse_chunks.append(
            _compute_sparse_lse_chunk(
                q_nope=q_nope[start:end],
                q_rope=q_rope[start:end],
                k_nope=k_nope,
                k_rope=k_rope,
                logical_indices=logical_indices[start:end],
                request_rows=request_rows[start:end],
                block_tables=block_tables,
                local_kv_lens=local_kv_lens,
                page_size=page_size,
                scaling=scaling,
            )
        )
    lse = torch.cat(lse_chunks)
    if num_valid_queries < lse.shape[0]:
        lse[num_valid_queries:].zero_()
    return lse


def _compute_sparse_lse_chunk(
    *,
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    logical_indices: torch.Tensor,
    request_rows: torch.Tensor,
    block_tables: torch.Tensor,
    local_kv_lens: torch.Tensor,
    page_size: int,
    scaling: float,
) -> torch.Tensor:
    valid = (logical_indices >= 0) & (
        logical_indices < local_kv_lens[request_rows].unsqueeze(1)
    )
    safe_indices = torch.where(
        valid, logical_indices, torch.zeros_like(logical_indices)
    ).to(torch.int64)
    logical_pages = safe_indices // page_size
    valid &= logical_pages < block_tables.shape[1]
    logical_pages.clamp_(max=block_tables.shape[1] - 1)
    physical_indices = (
        block_tables[request_rows.unsqueeze(1), logical_pages].to(torch.int64)
        * page_size
        + safe_indices % page_size
    )

    flat_k_nope = k_nope.view(-1, k_nope.shape[-1])
    flat_k_rope = k_rope.view(-1, k_rope.shape[-1])
    valid &= (physical_indices >= 0) & (physical_indices < flat_k_nope.shape[0])
    physical_indices.clamp_(min=0, max=flat_k_nope.shape[0] - 1)
    selected_k_nope = flat_k_nope[physical_indices]
    selected_k_rope = flat_k_rope[physical_indices]

    scores = torch.bmm(q_nope, selected_k_nope.transpose(1, 2))
    scores.add_(torch.bmm(q_rope, selected_k_rope.transpose(1, 2)))
    scores.mul_(scaling)
    scores.masked_fill_(~valid.unsqueeze(1), float("-inf"))
    return torch.logsumexp(scores.float(), dim=-1)
