"""NPU DSA adaptation for token-sharded decode context parallelism."""

from __future__ import annotations

from typing import TYPE_CHECKING

import sgl_kernel_npu  # noqa: F401  Registers torch.ops.sgl_kernel_npu.
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
    scaling: float,
    replicated_kv: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute one rank's sparse partial attention and natural-log LSE."""
    parallel = get_parallel()
    topk_indices = _remap_dcp_sparse_indices(
        topk_indices,
        parallel.attn_dcp_size,
        parallel.attn_dcp_rank,
        replicated_kv=replicated_kv,
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
        replicated_kv=replicated_kv,
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

    attn_out, softmax_max, softmax_sum = (
        torch.ops.sgl_kernel_npu.npu_sparse_flash_attention(
            return_softmax_lse=True, **sparse_kwargs
        )
    )
    lse = softmax_max.float() + torch.log(softmax_sum.float())
    return attn_out, lse.permute(1, 0, 2).reshape(lse.shape[1], -1)


def _expand_sparse_indices(topk_indices: torch.Tensor) -> torch.Tensor:
    return topk_indices.unsqueeze(-2) if topk_indices.dim() == 2 else topk_indices


def _remap_dcp_sparse_indices(
    topk_indices: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
    *,
    replicated_kv: bool = False,
) -> torch.Tensor:
    """Filter global positions; compact only the rank-local target layout."""
    if not replicated_kv:
        return remap_dcp_sparse_indices(topk_indices, dcp_size, dcp_rank)
    local_mask = (topk_indices >= 0) & (topk_indices % dcp_size == dcp_rank)
    local_indices = torch.where(
        local_mask, topk_indices, torch.full_like(topk_indices, -1)
    )
    valid_dest = torch.cumsum(local_mask, dim=-1) - 1
    invalid_dest = (
        local_mask.sum(dim=-1, keepdim=True) + torch.cumsum(~local_mask, dim=-1) - 1
    )
    destination = torch.where(local_mask, valid_dest, invalid_dest).to(torch.int64)
    return torch.empty_like(local_indices).scatter(-1, destination, local_indices)


def _get_local_kv_lens(
    *,
    forward_metadata,
    is_speculative: bool,
    tokens_per_request: int | None,
    device: torch.device,
    replicated_kv: bool = False,
) -> torch.Tensor:
    """Return local KV lengths without host transfers during graph capture."""
    parallel = get_parallel()
    dcp_spec_seq_lens = getattr(forward_metadata, "dcp_spec_seq_lens", None)
    dcp_seq_lens = getattr(forward_metadata, "dcp_seq_lens", None)
    if replicated_kv:
        full_lens = getattr(forward_metadata, "seq_lens_cpu_int", None)
        if full_lens is None:
            full_lens = forward_metadata.seq_lens
        if is_speculative:
            assert tokens_per_request is not None
            steps = torch.arange(
                1,
                tokens_per_request + 1,
                dtype=full_lens.dtype,
                device=full_lens.device,
            )
            return (
                (full_lens.unsqueeze(1) - tokens_per_request + steps)
                .clamp(min=0)[:, -1]
                .to(device=device, dtype=torch.int32)
            )
        return full_lens.to(device=device, dtype=torch.int32)
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
