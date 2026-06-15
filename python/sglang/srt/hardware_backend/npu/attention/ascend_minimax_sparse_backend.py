from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.configs.model_config import (
    get_minimax_sparse_attention_config,
    get_minimax_sparse_disable_value_layer_ids,
    get_minimax_sparse_layer_ids,
    get_minimax_sparse_score_type,
)
from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMiniMaxSparseKVPool
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.minimax_sparse_ops.minimax_sparse import (
    minimax_sparse_decode,
    minimax_sparse_prefill,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class AscendMiniMaxSparseAttnBackend(AttentionBackend):
    def __init__(self, runner: ModelRunner):
        assert isinstance(runner.token_to_kv_pool, NPUMiniMaxSparseKVPool)
        self.kv_pool = runner.token_to_kv_pool
        self.req_to_token = runner.req_to_token_pool.req_to_token
        self.max_context_len = int(runner.model_config.context_len)

        hf_config = runner.model_config.hf_config
        sparse_cfg = get_minimax_sparse_attention_config(hf_config)
        self.idx_head_dim = sparse_cfg["sparse_index_dim"]
        self.dense_layer_ids, self.sparse_layer_ids = get_minimax_sparse_layer_ids(
            sparse_cfg
        )
        self.disable_value_layer_ids: set[int] = set(
            get_minimax_sparse_disable_value_layer_ids(sparse_cfg)
        )
        self.score_type: str = get_minimax_sparse_score_type(sparse_cfg)

        self._max_seqlen_q: int = 1
        self._max_seqlen_k: int = 1

        self.block_size_q = 1
        self.block_size_k = sparse_cfg["sparse_block_size"]
        if "sparse_init_block" in sparse_cfg:
            self.init_blocks = sparse_cfg["sparse_init_block"]
        else:
            init_tokens = sparse_cfg["sparse_init_tokens"]
            self.init_blocks = (
                init_tokens + self.block_size_k - 1
            ) // self.block_size_k
        if "sparse_local_block" in sparse_cfg:
            self.local_blocks = sparse_cfg["sparse_local_block"]
        else:
            local_tokens = sparse_cfg["sparse_local_tokens"]
            self.local_blocks = (
                local_tokens + self.block_size_k - 1
            ) // self.block_size_k + 1
        self.topk_blocks = sparse_cfg["sparse_topk_blocks"]

        self.page_size = self.kv_pool.page_size

        self.dense_backend: Optional[AttentionBackend] = None

        logger.info(
            f"[AscendMiniMaxSparse] Backend initialized "
            f"(score_type={self.score_type!r}, "
            f"disable_value_layers={sorted(self.disable_value_layer_ids)})"
        )

    def init_forward_metadata_out_graph(
        self, forward_batch: ForwardBatch, in_capture: bool = False
    ):
        extend_lens = getattr(forward_batch, "extend_seq_lens_cpu", None)
        if extend_lens is not None:
            self._max_seqlen_q = int(max(extend_lens))
        else:
            self._max_seqlen_q = 1
        if in_capture and forward_batch.forward_mode.is_decode_or_idle():
            self._max_seqlen_k = self.max_context_len
        else:
            self._max_seqlen_k = int(forward_batch.seq_lens_cpu.max().item())

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch):
        pass

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        pass

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    @staticmethod
    def _is_sparse_kv_cached_by_fusion(
        forward_batch: ForwardBatch, layer_id: int
    ) -> bool:
        layer_ids = forward_batch.minimax_m3_precached_sparse_layers
        return layer_ids is not None and layer_id in layer_ids

    def forward(
        self,
        q,
        k,
        v,
        layer,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        if forward_batch.forward_mode.is_idle():
            idx_q = kwargs.get("idx_q")
            num_idx_heads = idx_q.shape[1]
            disable_value = layer.layer_id in self.disable_value_layer_ids
            idx_out: Optional[torch.Tensor] = (
                None
                if disable_value
                else q.new_zeros(q.shape[0], num_idx_heads * self.idx_head_dim)
            )
            out = q.new_zeros(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
            return idx_out, out
        else:
            return super().forward(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )

    def _get_npu_kv_cache(self, layer_id: int):
        k_cache = self.kv_pool.get_key_buffer(layer_id)
        v_cache = self.kv_pool.get_value_buffer(layer_id)
        page_size = self.kv_pool.page_size
        num_pages = k_cache.shape[0]
        head_num = k_cache.shape[-2]
        head_dim = k_cache.shape[-1]
        v_head_dim = v_cache.shape[-1]
        k_bnsd = k_cache.view(num_pages, page_size, head_num, head_dim)
        v_bnsd = v_cache.view(num_pages, page_size, head_num, v_head_dim)
        return k_bnsd, v_bnsd

    def _get_npu_index_kv_cache(self, layer_id: int, disable_value: bool):
        if disable_value:
            idx_k_cache = self.kv_pool.get_index_k_buffer(layer_id)
            page_size = self.kv_pool.page_size
            num_pages = idx_k_cache.shape[0]
            idx_head_dim = idx_k_cache.shape[-1]
            idx_k_bnsd = idx_k_cache.view(num_pages, page_size, 1, idx_head_dim)
            return idx_k_bnsd, None
        else:
            idx_k_cache, idx_v_cache = self.kv_pool.get_index_kv_buffer(layer_id)
            page_size = self.kv_pool.page_size
            num_pages = idx_k_cache.shape[0]
            idx_head_dim = idx_k_cache.shape[-1]
            idx_k_bnsd = idx_k_cache.view(num_pages, page_size, 1, idx_head_dim)
            idx_v_bnsd = idx_v_cache.view(num_pages, page_size, 1, idx_head_dim)
            return idx_k_bnsd, idx_v_bnsd

    def _build_block_table(self, forward_batch: ForwardBatch) -> torch.Tensor:
        slot_ids = forward_batch.req_pool_indices
        block_table = self.req_to_token[slot_ids]
        page_size = self.kv_pool.page_size
        block_size_k = self.block_size_k
        if page_size == block_size_k:
            return (block_table // page_size).to(torch.int32)
        num_blocks_per_page = block_size_k // page_size
        block_ids = block_table // page_size
        batch_size = block_ids.shape[0]
        max_pages = block_ids.shape[1]
        max_blocks = max_pages // num_blocks_per_page
        block_table_out = block_ids[:, : max_blocks * num_blocks_per_page].reshape(
            batch_size, max_blocks, num_blocks_per_page
        )[:, :, 0]
        return block_table_out.to(torch.int32)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        *,
        idx_q: torch.Tensor,
        idx_k: torch.Tensor,
        idx_v: Optional[torch.Tensor],
    ):
        disable_value = layer.layer_id in self.disable_value_layer_ids
        kv_cached_by_fusion = self._is_sparse_kv_cached_by_fusion(
            forward_batch, layer.layer_id
        )
        if not kv_cached_by_fusion:
            self.kv_pool.set_fused_kv_index_buffer(
                layer,
                forward_batch.out_cache_loc,
                k,
                v,
                idx_k,
                None if disable_value else idx_v,
            )

        k_cache, v_cache = self.kv_pool.get_kv_buffer(layer.layer_id)
        if disable_value:
            idx_k_cache = self.kv_pool.get_index_k_buffer(layer.layer_id)
            idx_v_cache = None
        else:
            idx_k_cache, idx_v_cache = self.kv_pool.get_index_kv_buffer(layer.layer_id)

        cu_seqlens = torch.cat(
            [
                torch.zeros(
                    1, dtype=torch.int32, device=forward_batch.extend_seq_lens.device
                ),
                forward_batch.extend_seq_lens.to(torch.int32).cumsum(0).to(torch.int32),
            ]
        )
        seq_lens = forward_batch.seq_lens.to(torch.int32)
        if forward_batch.extend_prefix_lens is not None:
            prefix_lens = forward_batch.extend_prefix_lens.to(torch.int32)
        else:
            prefix_lens = torch.zeros_like(seq_lens)

        if forward_batch.extend_seq_lens_cpu is not None:
            actual_num_tokens = int(sum(forward_batch.extend_seq_lens_cpu))
        else:
            actual_num_tokens = int(cu_seqlens[-1].item())
        original_num_tokens = q.shape[0]
        if actual_num_tokens < original_num_tokens:
            q = q[:actual_num_tokens]
            idx_q = idx_q[:actual_num_tokens]

        idx_o, o = minimax_sparse_prefill(
            q,
            k_cache,
            v_cache,
            None,
            idx_q,
            idx_k_cache,
            idx_v_cache,
            None,
            self.req_to_token,
            forward_batch.req_pool_indices,
            cu_seqlens,
            seq_lens,
            prefix_lens,
            self._max_seqlen_q,
            self._max_seqlen_k,
            self.block_size_q,
            self.block_size_k,
            self.topk_blocks,
            self.init_blocks,
            self.local_blocks,
            score_type=self.score_type,
            disable_index_value=disable_value,
            use_msa=False,
            seqlens_cpu=forward_batch.extend_seq_lens_cpu,
        )

        if actual_num_tokens < original_num_tokens:
            pad_len = original_num_tokens - actual_num_tokens
            o = torch.cat([o, o.new_zeros(pad_len, *o.shape[1:])], dim=0)
            if idx_o is not None:
                idx_o = torch.cat(
                    [idx_o, idx_o.new_zeros(pad_len, *idx_o.shape[1:])], dim=0
                )

        return (
            (
                None
                if idx_o is None
                else idx_o.reshape(original_num_tokens, -1).contiguous()
            ),
            o.reshape(original_num_tokens, -1).contiguous(),
        )

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        *,
        idx_q: torch.Tensor,
        idx_k: torch.Tensor,
        idx_v: Optional[torch.Tensor],
        **kwargs,
    ):
        assert len(kwargs) == 0
        disable_value = layer.layer_id in self.disable_value_layer_ids
        self.kv_pool.set_fused_kv_index_buffer(
            layer,
            forward_batch.out_cache_loc,
            k,
            v,
            idx_k,
            None if disable_value else idx_v,
        )

        k_cache, v_cache = self.kv_pool.get_kv_buffer(layer.layer_id)
        if disable_value:
            idx_k_cache = self.kv_pool.get_index_k_buffer(layer.layer_id)
            idx_v_cache = None
        else:
            idx_k_cache, idx_v_cache = self.kv_pool.get_index_kv_buffer(layer.layer_id)

        idx_o, o = minimax_sparse_decode(
            q,
            None,
            k_cache,
            v_cache,
            idx_q,
            None,
            idx_k_cache,
            idx_v_cache,
            self.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            self._max_seqlen_k,
            1,
            self.block_size_k,
            self.topk_blocks,
            self.init_blocks,
            self.local_blocks,
            score_type=self.score_type,
            disable_index_value=disable_value,
            use_msa=False,
        )
        return (
            None if idx_o is None else idx_o.reshape(q.shape[0], -1).contiguous(),
            o.reshape(q.shape[0], -1).contiguous(),
        )
