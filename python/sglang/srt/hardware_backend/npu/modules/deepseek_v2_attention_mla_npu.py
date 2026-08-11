import re
from typing import TYPE_CHECKING

import torch
import torch_npu
from sgl_kernel_npu.norm.fused_split_qk_norm import fused_split_qk_norm

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.attention.mla_preprocess import (
    NPUFusedMLAPreprocess,
    is_fia_nz,
    is_mla_preprocess_enabled,
)
from sglang.srt.layers.attention.dsa.dsa_npu_indexer import scattered_to_tp_attn_full
from sglang.srt.layers.attention.dsa.utils import (
    dsa_use_prefill_cp,
)
from sglang.srt.layers.communicator import ScatterMode, get_attn_tp_context
from sglang.srt.layers.dcp.comm import (
    all_gather_packed_kv_cache_for_dcp,
    all_gather_q_for_mla_decode,
    cp_lse_ag_out_rs_mla_npu,
)
from sglang.srt.layers.dcp.metadata import NPUMLAPrefixDCPMetadata
from sglang.srt.layers.dcp.planner import plan_npu_dcp_prefix_segments
from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
    get_token_to_kv_pool,
)
from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA
    from sglang.srt.utils import BumpAllocator


_use_ag_after_qlora = envs.SGLANG_USE_AG_AFTER_QLORA.get()


def _use_dcp_mla_partial_attention(forward_batch: "ForwardBatch") -> bool:
    mode = forward_batch.forward_mode
    return get_parallel().dcp_enabled and (
        mode.is_decode() or mode.is_target_verify() or mode.is_draft_extend_v2()
    )


def _should_use_mha_chunked_kv_npu(
    m: "DeepseekV2AttentionMLA", forward_batch: "ForwardBatch"
) -> bool:
    if forward_batch.num_prefix_chunks is not None:
        return True

    prefix_lens = forward_batch.extend_prefix_lens_cpu
    if not (
        forward_batch.forward_mode.is_extend_without_speculative()
        and prefix_lens is not None
        and any(prefix_lens)
        and not m.disable_chunked_prefix_cache
    ):
        return False

    page_size = get_attn_backend().page_size
    if get_parallel().dcp_enabled:
        page_size *= get_parallel().attn_dcp_size
    max_chunk_len = forward_batch.get_max_chunk_capacity() // forward_batch.batch_size
    max_chunk_len -= max_chunk_len % page_size
    return max(prefix_lens) > max_chunk_len


def _prepare_mha_prefix_segments_npu(
    m: "DeepseekV2AttentionMLA",
    q: torch.Tensor,
    forward_batch: "ForwardBatch",
) -> None:
    prefix_lens = forward_batch.extend_prefix_lens_cpu
    if not (
        forward_batch.forward_mode.is_extend_without_speculative()
        and prefix_lens is not None
        and any(prefix_lens)
    ):
        return

    backend = get_attn_backend()
    parallel = get_parallel()
    dcp_world_size = parallel.attn_dcp_size
    use_chunked_prefix = _should_use_mha_chunked_kv_npu(m, forward_batch)
    if use_chunked_prefix and forward_batch.num_prefix_chunks is None:
        forward_batch.prepare_chunked_prefix_cache_info(
            q.device,
            prepare_kv_indices=False,
            chunk_alignment=backend.page_size * dcp_world_size,
        )

    if not parallel.dcp_enabled or forward_batch.attn_dcp_metadata is not None:
        return

    if use_chunked_prefix:
        prefix_segment_starts = forward_batch.prefix_chunk_starts_cpu
        prefix_segment_lens = forward_batch.prefix_chunk_seq_lens_cpu
        assert prefix_segment_starts is not None
        assert prefix_segment_lens is not None
    else:
        prefix_segment_lens = torch.tensor(prefix_lens, dtype=torch.int32).unsqueeze(0)
        prefix_segment_starts = torch.zeros_like(prefix_segment_lens)

    forward_batch.attn_dcp_metadata = plan_npu_dcp_prefix_segments(
        prefix_segment_starts,
        prefix_segment_lens,
        dcp_world_size=dcp_world_size,
        page_size=backend.page_size,
        kv_cache_device=q.device,
    )


def _load_mha_prefix_segment_npu(
    m: "DeepseekV2AttentionMLA",
    forward_batch: "ForwardBatch",
    dcp_metadata: NPUMLAPrefixDCPMetadata | None,
    segment_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_to_kv_pool = get_token_to_kv_pool()
    k_buffer = token_to_kv_pool.get_key_buffer(m.attn_mqa.layer_id)
    v_buffer = token_to_kv_pool.get_value_buffer(m.attn_mqa.layer_id)
    if dcp_metadata is None:
        assert forward_batch.prefix_chunk_seq_lens_cpu is not None
        assert forward_batch.prefix_chunk_starts_cpu is not None
        assert forward_batch.prefix_chunk_num_tokens is not None
        segment_lens = forward_batch.prefix_chunk_seq_lens_cpu[segment_idx].to(
            device=k_buffer.device, dtype=torch.int32
        )
        segment_starts = forward_batch.prefix_chunk_starts_cpu[segment_idx].to(
            device=k_buffer.device, dtype=torch.int32
        )
        local_token_count = int(forward_batch.prefix_chunk_num_tokens[segment_idx])
    else:
        segment_lens = dcp_metadata.prefix_segment_local_lens[segment_idx]
        segment_starts = dcp_metadata.prefix_segment_local_starts[segment_idx]
        local_token_count = dcp_metadata.prefix_segment_local_token_counts[segment_idx]

    local_kv = k_buffer.new_empty(local_token_count, *k_buffer.shape[2:])
    local_k_pe = v_buffer.new_empty(local_token_count, *v_buffer.shape[2:])
    if local_token_count > 0:
        prefix_block_tables = get_attn_backend().forward_metadata.prefix_block_tables
        assert prefix_block_tables is not None
        torch_npu.npu_gather_pa_kv_cache(
            k_buffer,
            v_buffer,
            prefix_block_tables,
            segment_lens.contiguous(),
            seq_offset=segment_starts.contiguous(),
            key=local_kv,
            value=local_k_pe,
        )

    if dcp_metadata is None:
        return local_kv, local_k_pe

    restored = all_gather_packed_kv_cache_for_dcp(
        local_kv,
        dcp_metadata.prefix_segment_restore_indices[segment_idx],
        local_k_pe=local_k_pe,
    )
    return restored.split([local_kv.shape[-1], local_k_pe.shape[-1]], dim=-1)


def _forward_mha_prefix_segments_npu(
    m: "DeepseekV2AttentionMLA",
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    forward_batch: "ForwardBatch",
) -> torch.Tensor:
    num_token_padding = q.shape[0]
    num_tokens = forward_batch.num_token_non_padded_cpu
    q, k, v = [tensor[:num_tokens] for tensor in (q, k, v)]
    forward_batch.mha_return_lse = True
    forward_batch.set_attn_attend_prefix_cache(False)
    attn_output, attn_lse = m.attn_mha(q, k, v, forward_batch, save_kv_cache=False)
    output_dtype = attn_output.dtype
    out_list = [attn_output.reshape(-1, m.v_head_dim).float()]
    lse_list = [attn_lse.squeeze(-1).reshape(-1).float()]

    dcp_metadata = forward_batch.attn_dcp_metadata
    use_chunked_prefix = forward_batch.num_prefix_chunks is not None
    if get_parallel().dcp_enabled:
        assert dcp_metadata is not None
    num_prefix_segments = forward_batch.num_prefix_chunks if use_chunked_prefix else 1

    forward_batch.set_attn_attend_prefix_cache(True)
    for segment_idx in range(num_prefix_segments):
        if use_chunked_prefix:
            forward_batch.set_prefix_chunk_idx(segment_idx)
        kv_a, k_pe = _load_mha_prefix_segment_npu(
            m, forward_batch, dcp_metadata, segment_idx
        )
        k_nope, chunk_v = (
            m.kv_b_proj(kv_a.contiguous())[0]
            .view(-1, m.num_local_heads, m.qk_nope_head_dim + m.v_head_dim)
            .split([m.qk_nope_head_dim, m.v_head_dim], dim=-1)
        )
        chunk_k = m._concat_and_cast_mha_k(k_nope, k_pe, forward_batch)
        chunk_output, chunk_lse = m.attn_mha(
            q, chunk_k, chunk_v, forward_batch, save_kv_cache=False
        )
        out_list.append(chunk_output.reshape(-1, m.v_head_dim).float())
        lse_list.append(chunk_lse.squeeze(-1).reshape(-1).float())

    attn_output, _ = torch_npu.npu_attention_update(tuple(lse_list), tuple(out_list), 0)
    attn_output = attn_output.view(num_tokens, m.num_local_heads, m.v_head_dim).to(
        output_dtype
    )
    if num_token_padding != num_tokens:
        padding = attn_output.new_zeros(
            num_token_padding - num_tokens, *attn_output.shape[1:]
        )
        attn_output = torch.cat([attn_output, padding], dim=0)
    attn_output = attn_output.reshape(-1, m.num_local_heads * m.v_head_dim)
    output, _ = m.o_proj(attn_output)
    return output


# region MHA
def forward_mha_prepare_npu(
    m: "DeepseekV2AttentionMLA",
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    forward_batch: "ForwardBatch",
    zero_allocator: "BumpAllocator",
    layer_scatter_modes,
):
    if m.q_lora_rank is not None:
        q, latent_cache = (
            get_attn_tp_context()
            .fetch_qkv_latent()
            .split(
                [m.q_lora_rank, m.kv_lora_rank + m.qk_rope_head_dim],
                dim=-1,
            )
        )

        # DSA Indexer: cache quantized keys, auto-skip topk for sequences <= dsa_index_topk

        if m.use_dsa:
            q_lora = m.q_a_layernorm(q)
            q = m.q_b_proj(q_lora)[0].view(-1, m.num_local_heads, m.qk_head_dim)
            _ = m.indexer(
                x=hidden_states,
                q_lora=q_lora,
                positions=positions,
                forward_batch=forward_batch,
                layer_id=m.layer_id,
                return_indices=False,
            )

        else:
            q = m.q_a_layernorm(q)
            if (
                _use_ag_after_qlora
                and layer_scatter_modes.layer_input_mode == ScatterMode.SCATTERED
                and layer_scatter_modes.attn_mode == ScatterMode.TP_ATTN_FULL
            ):
                q = scattered_to_tp_attn_full(q, forward_batch)
                latent_cache = scattered_to_tp_attn_full(latent_cache, forward_batch)
            q = m.q_b_proj(q)[0].view(-1, m.num_local_heads, m.qk_head_dim)

    else:
        q = m.q_proj(hidden_states)[0].view(-1, m.num_local_heads, m.qk_head_dim)
        latent_cache = m.kv_a_proj_with_mqa(hidden_states)[0]

    _, q_pe = q.split([m.qk_nope_head_dim, m.qk_rope_head_dim], dim=-1)
    kv_a, _ = latent_cache.split([m.kv_lora_rank, m.qk_rope_head_dim], dim=-1)
    latent_cache = latent_cache.unsqueeze(1)

    if m.use_deepseek_yarn_rope:
        B, S = q.shape[0], 1
        cos, sin = m.rotary_emb.get_cos_sin_cache(
            positions, hidden_states.dtype, offsets=None
        )
        q_pe = torch_npu.npu_interleave_rope(
            q_pe.reshape(B, -1, S, m.qk_rope_head_dim),
            cos,
            sin,
        )
        q_pe = q_pe.reshape(B, -1, m.qk_rope_head_dim)

        ckv_cache, k_rope_cache = get_token_to_kv_pool().get_kv_buffer(m.layer_id)
        _, _, k_pe, kv_a = torch_npu.npu_kv_rmsnorm_rope_cache(
            latent_cache.view(-1, 1, 1, m.kv_lora_rank + m.qk_rope_head_dim),  # bnsd
            m.kv_a_layernorm.weight,
            cos,
            sin,
            forward_batch.out_cache_loc.to(torch.int64),
            k_rope_cache,
            ckv_cache,
            k_rope_scale=None,
            c_kv_scale=None,
            k_rope_offset=None,
            c_kv_offset=None,
            epsilon=m.kv_a_layernorm.variance_epsilon,
            cache_mode="PA_NZ" if is_fia_nz() else "PA_BNSD",
            is_output_kv=True,
        )  # adapter NZ

        k_pe = k_pe.reshape(B, -1, m.qk_rope_head_dim)
    else:
        kv_a = m.kv_a_layernorm(kv_a)
        k_pe = latent_cache[:, :, m.kv_lora_rank :]
        if m.rotary_emb is not None:
            q_pe, k_pe = m.rotary_emb(positions, q_pe, k_pe)
        # this is for model kimi-vl-a3B-instruct
        get_token_to_kv_pool().set_kv_buffer(
            m, forward_batch.out_cache_loc, kv_a.unsqueeze(1), k_pe
        )

    q[..., m.qk_nope_head_dim :] = q_pe

    _prepare_mha_prefix_segments_npu(m, q, forward_batch)

    kv = m.kv_b_proj(kv_a)[0]
    kv = kv.view(-1, m.num_local_heads, m.qk_nope_head_dim + m.v_head_dim)
    k_nope = kv[..., : m.qk_nope_head_dim]
    v = kv[..., m.qk_nope_head_dim :]

    k = m._concat_and_cast_mha_k(k_nope, k_pe, forward_batch)
    return q, k, v, forward_batch


def forward_mha_core_npu(
    m: "DeepseekV2AttentionMLA",
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    forward_batch: "ForwardBatch",
) -> torch.Tensor:
    use_dcp_prefix = (
        get_parallel().dcp_enabled and forward_batch.attn_dcp_metadata is not None
    )
    if _should_use_mha_chunked_kv_npu(m, forward_batch) or use_dcp_prefix:
        return _forward_mha_prefix_segments_npu(m, q, k, v, forward_batch)

    attn_output = m.attn_mha(q, k, v, forward_batch, save_kv_cache=False)
    attn_output = attn_output.reshape(-1, m.num_local_heads * m.v_head_dim)
    output, _ = m.o_proj(attn_output)
    return output


# endregion


# region MLA
def forward_mla_prepare_npu(
    m: "DeepseekV2AttentionMLA",
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    forward_batch: "ForwardBatch",
    zero_allocator: "BumpAllocator",
    layer_scatter_modes,
):
    if is_mla_preprocess_enabled():
        if not hasattr(m, "mla_preprocess"):
            m.mla_preprocess = NPUFusedMLAPreprocess(
                m.fused_qkv_a_proj_with_mqa,
                m.q_a_layernorm,
                m.kv_a_layernorm,
                m.q_b_proj,
                m.w_kc,
                m.rotary_emb,
                m.layer_id,
                m.num_local_heads,
                m.qk_nope_head_dim,
                m.qk_rope_head_dim,
                m.quant_config,
            )
        (
            q_pe,
            k_pe,
            q_nope_out,
            k_nope,
            forward_batch,
            zero_allocator,
            positions,
        ) = m.mla_preprocess.forward(
            positions, hidden_states, forward_batch, zero_allocator
        )
        topk_indices = None
    else:
        q_lora = None
        if m.q_lora_rank is not None:
            qkv_latent = get_attn_tp_context().fetch_qkv_latent()
            if (
                _use_ag_after_qlora
                and layer_scatter_modes.layer_input_mode == ScatterMode.SCATTERED
                and layer_scatter_modes.attn_mode == ScatterMode.TP_ATTN_FULL
            ):
                q, latent_cache = qkv_latent.split(
                    [m.q_lora_rank, m.kv_lora_rank + m.qk_rope_head_dim],
                    dim=-1,
                )
                k_nope = latent_cache[..., : m.kv_lora_rank]

                q = m.q_a_layernorm(q)
                q = scattered_to_tp_attn_full(q, forward_batch)
                latent_cache = scattered_to_tp_attn_full(latent_cache, forward_batch)

                k_nope = m.kv_a_layernorm(k_nope).unsqueeze(1)
                k_pe = latent_cache[..., m.kv_lora_rank :].unsqueeze(1)
            else:
                if qkv_latent.shape[0] < 65536 and not dsa_use_prefill_cp(
                    forward_batch
                ):
                    q, k_nope, k_pe = fused_split_qk_norm(
                        qkv_latent,
                        m.q_a_layernorm,
                        m.kv_a_layernorm,
                        m.q_lora_rank,
                        m.kv_lora_rank,
                        m.qk_rope_head_dim,
                        eps=m.q_a_layernorm.variance_epsilon,
                    )
                else:
                    q, latent_cache = qkv_latent.split(
                        [m.q_lora_rank, m.kv_lora_rank + m.qk_rope_head_dim],
                        dim=-1,
                    )
                    k_nope = latent_cache[..., : m.kv_lora_rank]

                    q = m.q_a_layernorm(q)

                    k_nope = m.kv_a_layernorm(k_nope).unsqueeze(1)
                    k_pe = latent_cache[..., m.kv_lora_rank :].unsqueeze(1)

            # q_lora needed by indexer
            if m.use_dsa:
                q_lora = q

            q = m.q_b_proj(q)[0].view(-1, m.num_local_heads, m.qk_head_dim)
        else:
            q = m.q_proj(hidden_states)[0].view(-1, m.num_local_heads, m.qk_head_dim)
            latent_cache = m.kv_a_proj_with_mqa(hidden_states)[0]
            k_nope = latent_cache[..., : m.kv_lora_rank]
            k_nope = m.kv_a_layernorm(k_nope).unsqueeze(1)
            k_pe = latent_cache[..., m.kv_lora_rank :].unsqueeze(1)

        q_nope, q_pe = q.split([m.qk_nope_head_dim, m.qk_rope_head_dim], dim=-1)

        q_nope_out = torch.bmm(q_nope.transpose(0, 1), m.w_kc)

        q_nope_out = q_nope_out.transpose(0, 1)

        q_pe, k_pe = m.rotary_emb(positions, q_pe, k_pe)

        if dsa_use_prefill_cp(forward_batch):
            # support allgather+rerrange
            k_nope, k_pe = m.rebuild_cp_kv_cache(
                latent_cache, forward_batch, k_nope, k_pe
            )
        topk_indices = None
        if q_lora is not None:
            topk_indices = m.indexer(
                x=hidden_states,
                q_lora=q_lora,
                positions=positions,
                forward_batch=forward_batch,
                layer_id=m.layer_id,
            )

    # DCP decode/speculative attention: all-gather Q across DCP ranks so each
    # local KV shard computes partial outputs for the complete TP head set.
    if _use_dcp_mla_partial_attention(forward_batch):
        q_nope_out, q_pe = all_gather_q_for_mla_decode(q_nope_out, q_pe)

    return (
        q_pe,
        k_pe,
        q_nope_out,
        k_nope,
        forward_batch,
        zero_allocator,
        positions,
        topk_indices,
    )


def forward_mla_core_npu(
    m: "DeepseekV2AttentionMLA",
    q_pe: torch.Tensor,
    k_pe: torch.Tensor,
    q_nope_out: torch.Tensor,
    k_nope: torch.Tensor,
    forward_batch: "ForwardBatch",
    zero_allocator: "BumpAllocator",
    positions: torch.Tensor,
    topk_indices: torch.Tensor,
) -> torch.Tensor:
    if _use_dcp_mla_partial_attention(forward_batch):
        attn_output, lse = m.attn_mqa_for_dcp_decode(
            q_nope_out,
            k_nope,
            k_nope,
            forward_batch,
            q_rope=q_pe,
            k_rope=k_pe,
        )
        # Merge partial attention outputs across DCP ranks
        attn_output = attn_output.view(
            -1,
            m.num_local_heads * get_parallel().attn_dcp_size,
            m.kv_lora_rank,
        )
        attn_output = cp_lse_ag_out_rs_mla_npu(
            attn_output, lse, get_parallel().dcp_group
        )
    else:
        attn_output = m.attn_mqa(
            q_nope_out,
            k_nope,
            k_nope,
            forward_batch,
            q_rope=q_pe,
            k_rope=k_pe,
            **(dict(topk_indices=topk_indices) if topk_indices is not None else {}),
        )

    attn_output = attn_output.view(-1, m.num_local_heads, m.kv_lora_rank)

    attn_bmm_output = torch.empty(
        (attn_output.shape[0], m.num_local_heads, m.v_head_dim),
        dtype=attn_output.dtype,
        device=attn_output.device,
    )

    attn_output = attn_output.contiguous()
    torch.ops.npu.batch_matmul_transpose(attn_output, m.w_vc, attn_bmm_output)

    attn_bmm_output = attn_bmm_output.reshape(-1, m.num_local_heads * m.v_head_dim)
    output, _ = m.o_proj(attn_bmm_output)

    return output


# endregion


# region DSA
def forward_dsa_prepare_npu(
    m: "DeepseekV2AttentionMLA",
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    forward_batch: "ForwardBatch",
    zero_allocator: "BumpAllocator",
    layer_scatter_modes,
    prev_topk_indices: torch.Tensor = None,
):
    dynamic_scale = None
    if is_mla_preprocess_enabled() and forward_batch.forward_mode.is_decode():
        (
            q_pe,
            k_pe,
            q_nope_out,
            k_nope,
            q_lora,
            forward_batch,
            zero_allocator,
            positions,
            dynamic_scale,
        ) = npu_mla_preprocess(
            m,
            hidden_states,
            positions,
            forward_batch,
            zero_allocator,
        )
    else:
        fused_qkv_a_proj_out = m.fused_qkv_a_proj_with_mqa(hidden_states)[0]
        if m.rotary_emb.is_neox_style:
            q, latent_cache = fused_qkv_a_proj_out.split(
                [m.q_lora_rank, m.kv_lora_rank + m.qk_rope_head_dim], dim=-1
            )
            # overlap qk norm
            q = m.q_a_layernorm(q)
            if (
                _use_ag_after_qlora
                and layer_scatter_modes.layer_input_mode == ScatterMode.SCATTERED
                and layer_scatter_modes.attn_mode == ScatterMode.TP_ATTN_FULL
            ):
                q = scattered_to_tp_attn_full(q, forward_batch)
                latent_cache = scattered_to_tp_attn_full(latent_cache, forward_batch)
            q_lora = q.clone()  # required for topk_indices

            q_event = None
            if m.alt_stream is not None:
                m.alt_stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(m.alt_stream):
                    q = m.q_b_proj(q_lora)[0].view(-1, m.num_local_heads, m.qk_head_dim)
                    # record q to ensure memory space will not be released
                    q.record_stream(m.alt_stream)
                    q_event = m.alt_stream.record_event()
            else:
                q = m.q_b_proj(q_lora)[0].view(-1, m.num_local_heads, m.qk_head_dim)

            k_nope, k_pe = latent_cache.unsqueeze(1).split(
                [m.kv_lora_rank, m.qk_rope_head_dim], dim=-1
            )
            k_nope = m.kv_a_layernorm(k_nope)
            # main stream waits for the completion of the event on the alt stream to ensure data dependency is complete
            if q_event is not None:
                torch.npu.current_stream().wait_event(q_event)
        else:
            if fused_qkv_a_proj_out.shape[0] < 65535 and not dsa_use_prefill_cp(
                forward_batch
            ):
                q_lora, k_nope, k_pe = fused_split_qk_norm(
                    fused_qkv_a_proj_out,
                    m.q_a_layernorm,
                    m.kv_a_layernorm,
                    m.q_lora_rank,
                    m.kv_lora_rank,
                    m.qk_rope_head_dim,
                    eps=m.q_a_layernorm.variance_epsilon,
                )
            else:
                q, latent_cache = fused_qkv_a_proj_out.split(
                    [m.q_lora_rank, m.kv_lora_rank + m.qk_rope_head_dim], dim=-1
                )
                # overlap qk norm
                q = m.q_a_layernorm(q)

                q_lora = q.clone()  # required for topk_indices
                k_nope, k_pe = latent_cache.unsqueeze(1).split(
                    [m.kv_lora_rank, m.qk_rope_head_dim], dim=-1
                )
                k_nope = m.kv_a_layernorm(k_nope)
            q = m.q_b_proj(q_lora)[0].view(-1, m.num_local_heads, m.qk_head_dim)

        q_nope, q_pe = q.split([m.qk_nope_head_dim, m.qk_rope_head_dim], dim=-1)

        q_nope_out = torch.bmm(q_nope.transpose(0, 1), m.w_kc)

        q_nope_out = q_nope_out.transpose(0, 1)

        if m.layer_id == 0:
            m.rotary_emb.sin_cos_cache = m.rotary_emb.cos_sin_cache.index_select(
                0, positions
            )

        q_pe, k_pe = m.rotary_emb(positions, q_pe, k_pe)

        if dsa_use_prefill_cp(forward_batch):
            # support allgather+rerrange
            k_nope, k_pe = m.rebuild_cp_kv_cache(
                latent_cache, forward_batch, k_nope, k_pe
            )

    if not m.skip_topk or (m.is_nextn and prev_topk_indices is None):
        topk_indices = m.indexer(
            hidden_states,
            q_lora,
            positions,
            forward_batch,
            m.layer_id,
            layer_scatter_modes,
            dynamic_scale,
        )
    else:
        topk_indices = prev_topk_indices

    return (
        q_pe,
        k_pe,
        q_nope_out,
        k_nope,
        topk_indices,
        forward_batch,
        zero_allocator,
        positions,
    )


def forward_dsa_core_npu(
    m: "DeepseekV2AttentionMLA",
    q_pe: torch.Tensor,
    k_pe: torch.Tensor,
    q_nope_out: torch.Tensor,
    k_nope: torch.Tensor,
    topk_indices: torch.Tensor,
    forward_batch: "ForwardBatch",
    zero_allocator: "BumpAllocator",
    positions: torch.Tensor,
) -> torch.Tensor:
    attn_output = m.attn_mqa(
        q_nope_out.contiguous(),
        k_nope.contiguous(),
        k_nope.contiguous(),
        forward_batch,
        save_kv_cache=True,  # False if forward_batch.forward_mode.is_extend() else True,
        q_rope=q_pe.contiguous(),
        k_rope=k_pe.contiguous(),
        topk_indices=topk_indices,
    )
    attn_output = attn_output.view(-1, m.num_local_heads, m.kv_lora_rank)

    attn_bmm_output = torch.empty(
        (attn_output.shape[0], m.num_local_heads, m.v_head_dim),
        dtype=attn_output.dtype,
        device=attn_output.device,
    )

    if (
        forward_batch.forward_mode.is_extend()
        and not forward_batch.forward_mode.is_draft_extend_v2()
        and not forward_batch.forward_mode.is_target_verify()
    ):
        attn_output = attn_output.transpose(0, 1)
        torch.bmm(
            attn_output,
            m.w_vc,
            out=attn_bmm_output.view(-1, m.num_local_heads, m.v_head_dim).transpose(
                0, 1
            ),
        )
    else:
        attn_output = attn_output.contiguous()
        torch.ops.npu.batch_matmul_transpose(attn_output, m.w_vc, attn_bmm_output)

    attn_bmm_output = attn_bmm_output.reshape(-1, m.num_local_heads * m.v_head_dim)

    output, _ = m.o_proj(attn_bmm_output)
    if not m.next_skip_topk:
        return output, None
    else:
        return output, topk_indices


def npu_mla_preprocess(
    m: "DeepseekV2AttentionMLA",
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: "ForwardBatch",
    zero_allocator: "BumpAllocator",
):
    dynamic_scale = None
    if not hasattr(m, "mla_preprocess"):
        m.mla_preprocess = NPUFusedMLAPreprocess(
            m.fused_qkv_a_proj_with_mqa,
            m.q_a_layernorm,
            m.kv_a_layernorm,
            m.q_b_proj,
            m.w_kc,
            m.rotary_emb,
            m.layer_id,
            m.num_local_heads,
            m.qk_nope_head_dim,
            m.qk_rope_head_dim,
            m.v_head_dim,
            m.quant_config,
        )
    # mlaprolog does not require additional calculation of q_lora
    _is_mlaprolog = hasattr(m.quant_config, "ignore") and any(
        re.fullmatch(r".*kv_b_proj", l) for l in m.quant_config.ignore
    )
    if _is_mlaprolog:
        (
            q_pe,
            k_pe,
            q_nope_out,
            k_nope,
            q_lora,
            forward_batch,
            positions,
            dynamic_scale,
        ) = m.mla_preprocess.forward(
            positions, hidden_states, forward_batch, zero_allocator
        )
    else:
        if m.alt_stream is not None:
            mla_event = torch.npu.Event()
            mla_event.record()
            with torch.npu.stream(m.alt_stream):
                # alt stream waits for the completion of the event on the main stream to ensure data dependency is complete
                torch.npu.current_stream().wait_event(mla_event)
                (
                    q_pe,
                    k_pe,
                    q_nope_out,
                    k_nope,
                    forward_batch,
                    zero_allocator,
                    positions,
                ) = m.mla_preprocess.forward(
                    positions, hidden_states, forward_batch, zero_allocator
                )

            fused_qkv_a_proj_out = m.fused_qkv_a_proj_with_mqa(hidden_states)[0]
            q, _ = fused_qkv_a_proj_out.split(
                [m.q_lora_rank, m.kv_lora_rank + m.qk_rope_head_dim], dim=-1
            )
            q_lora = m.q_a_layernorm(q)
            torch.npu.current_stream().wait_event(m.alt_stream)
        else:
            (
                q_pe,
                k_pe,
                q_nope_out,
                k_nope,
                forward_batch,
                zero_allocator,
                positions,
            ) = m.mla_preprocess.forward(
                positions, hidden_states, forward_batch, zero_allocator
            )
            fused_qkv_a_proj_out = m.fused_qkv_a_proj_with_mqa(hidden_states)[0]
            q, _ = fused_qkv_a_proj_out.split(
                [m.q_lora_rank, m.kv_lora_rank + m.qk_rope_head_dim], dim=-1
            )
            q_lora = m.q_a_layernorm(q)

    return (
        q_pe,
        k_pe,
        q_nope_out,
        k_nope,
        q_lora,
        forward_batch,
        zero_allocator,
        positions,
        dynamic_scale,
    )


# endregion
