import concurrent.futures
import enum
import logging
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from sglang.srt.disaggregation.ascend.transfer_engine import AscendTransferEngine
from sglang.srt.disaggregation.ascend.utils import (
    build_dcp_replicated_kv_entry_mask,
    build_replicated_dcp_token_indices,
)
from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.common.utils import (
    build_dcp_token_transfer_plan,
    group_concurrent_contiguous,
)
from sglang.srt.disaggregation.mooncake.conn import (
    MooncakeKVBootstrapServer,
    MooncakeKVManager,
    MooncakeKVReceiver,
    MooncakeKVSender,
)
from sglang.srt.disaggregation.utils import resolve_dcp_dst_entry_indices
from sglang.srt.distributed import get_pp_group
from sglang.srt.utils.network import get_local_ip_auto

logger = logging.getLogger(__name__)


class AscendStateType(str, enum.Enum):
    """DSV4-on-NPU PD components without a cross-hardware equivalent."""

    DSV4_C128 = "dsv4_c128"
    # C4 compress-state rows (attention + indexer) addressed within each
    # req_pool_idx bank on A5 (CYCLE cache_mode).  Separate from StateType.SWA
    # because each peer maps logical positions into its own local ring.
    DSV4_C4_STATE = "dsv4_c4_state"


_DSV4_KVCACHE_STATE_TYPES = tuple(AscendStateType)


class AscendKVManager(MooncakeKVManager):
    def _dcp_replicated_kv_group(self) -> Optional[int]:
        # NPU MLA registers [K, V], while NPU DSA adds index-K as group 2.
        return 2 if getattr(self.kv_args, "kv_buf_groups", 1) == 3 else None

    def _requires_exact_state_index_match(self, st: StateType) -> bool:
        return (
            super()._requires_exact_state_index_match(st)
            or st in _DSV4_KVCACHE_STATE_TYPES
        )

    def init_engine(self):
        # TransferEngine initialized on ascend.
        local_ip = get_local_ip_auto()
        self.engine = AscendTransferEngine(
            hostname=local_ip,
            npu_id=self.kv_args.gpu_id,
            disaggregation_mode=self.disaggregation_mode,
        )

    def register_buffer_to_engine(self):
        # MemFabric aligns registered buffers to 2 MiB. Register everything in
        # one batch so overlapping aligned ranges from small tensors are merged
        # before they are published to the peer.
        ptrs = list(self.kv_args.kv_data_ptrs)
        lens = list(self.kv_args.kv_data_lens)
        ptrs.extend(self.kv_args.aux_data_ptrs)
        lens.extend(self.kv_args.aux_data_lens)
        for component_ptrs, component_lens in zip(
            self.kv_args.state_data_ptrs or [],
            self.kv_args.state_data_lens or [],
        ):
            ptrs.extend(component_ptrs)
            lens.extend(component_lens)
        if ptrs:
            self.engine.batch_register(ptrs, lens)

    def requires_dcp_relayout(self, dst_dcp_size: int, dst_dcp_rank: int) -> bool:
        if self._dcp_replicated_kv_group() is not None and self.dcp_size != 1:
            raise RuntimeError(
                "Ascend DSA PD currently requires prefill dcp_size=1, got "
                f"prefill dcp_size={self.dcp_size}"
            )
        return super().requires_dcp_relayout(dst_dcp_size, dst_dcp_rank)

    def _init_dcp_pack_buffers_once(self, dcp_size: int) -> None:
        # The common DCP packer is CUDA-only. Ascend submits the rank-owned
        # MLA rows as one batched transfer request instead.
        self._dcp_pack_buffers = []

    def prepare_dcp_token_item_lens(self, dst_page_item_lens: List[int]) -> List[int]:
        replicated_group = self._dcp_replicated_kv_group()
        if replicated_group is None:
            return super().prepare_dcp_token_item_lens(dst_page_item_lens)

        page_size = self.kv_args.page_size
        src_page_item_lens = self.kv_args.kv_item_lens
        if src_page_item_lens and (
            not dst_page_item_lens or dst_page_item_lens[0] != src_page_item_lens[0]
        ):
            raise RuntimeError(
                "PD DCP source/destination MLA geometry differs: "
                f"src_first={src_page_item_lens[0] if src_page_item_lens else None}, "
                f"dst_first={dst_page_item_lens[0] if dst_page_item_lens else None}"
            )
        if any(item_len % page_size != 0 for item_len in src_page_item_lens):
            raise RuntimeError(
                "Ascend PD DCP requires page-aligned KV item lengths, got "
                f"{src_page_item_lens} with page_size={page_size}"
            )
        return [item_len // page_size for item_len in src_page_item_lens]

    def get_mla_kv_ptrs_with_pp(
        self, src_kv_ptrs: List[int], dst_kv_ptrs: List[int], state_type=None
    ) -> Tuple[List[int], List[int], int]:
        mla_ratios = getattr(self.kv_args, "mla_compression_ratios", None)
        if mla_ratios:
            if len(src_kv_ptrs) == len(dst_kv_ptrs):
                return src_kv_ptrs, dst_kv_ptrs, len(src_kv_ptrs)

            start_layer = self.kv_args.prefill_start_layer
            end_layer = self.kv_args.prefill_end_layer
            c4_full = sum(ratio == 4 for ratio in mla_ratios)
            c4_start = sum(ratio == 4 for ratio in mla_ratios[:start_layer])
            c4_end = sum(ratio == 4 for ratio in mla_ratios[:end_layer])
            c128_start = sum(ratio == 128 for ratio in mla_ratios[:start_layer])
            c128_end = sum(ratio == 128 for ratio in mla_ratios[:end_layer])

            if state_type == AscendStateType.DSV4_C128:
                dst = dst_kv_ptrs[c128_start:c128_end]
                return src_kv_ptrs, dst, len(src_kv_ptrs)

            if state_type == AscendStateType.DSV4_C4_STATE:
                # Layout: [attn_state_0..attn_{c4_full-1},
                #          idx_state_0..idx_{c4_full-1}]
                # Two groups, each c4_full entries; slice both by PP stage.
                dst = []
                for offset in (0, c4_full):
                    dst.extend(dst_kv_ptrs[offset + c4_start : offset + c4_end])
                return src_kv_ptrs, dst, len(src_kv_ptrs)

            # NPU main KV layout: [C4 KV, index K, index scale].
            if state_type is None and len(dst_kv_ptrs) == 3 * c4_full:
                dst = []
                for offset in (0, c4_full, 2 * c4_full):
                    dst.extend(dst_kv_ptrs[offset + c4_start : offset + c4_end])
                return src_kv_ptrs, dst, len(src_kv_ptrs)

            # On A5 (CYCLE cache_mode), StateType.SWA only contains SWA KV
            # buffers (C4 compress state is registered separately as
            # DSV4_C4_STATE).  The common _mla_slice_ptrs_for_pp assumes
            # SWA + C4 state are bundled (swa_L + 2*c4_full), so intercept
            # here and slice SWA KV by layer index directly.
            if state_type == StateType.SWA and AscendStateType.DSV4_C4_STATE in (
                self.kv_args.state_types or []
            ):
                dst = list(dst_kv_ptrs[start_layer:end_layer])
                return src_kv_ptrs, dst, len(src_kv_ptrs)

            return super().get_mla_kv_ptrs_with_pp(src_kv_ptrs, dst_kv_ptrs, state_type)

        # src_kv_ptrs: k_data, v_data, index_k_data(optional)
        # dst_kv_ptrs: k_data, v_data, index_k_data(optional)
        # state_type is accepted for parity with the common disaggregation path;
        # the NPU kv_buf_groups slicing below is state-type agnostic.
        kv_buf_groups = getattr(self.kv_args, "kv_buf_groups", 1)
        hidden_kv_layers = getattr(self.kv_args, "hidden_kv_layers", 0)
        draft_kv_layers = getattr(self.kv_args, "draft_kv_layers", 0)
        src_layers = len(src_kv_ptrs) // kv_buf_groups
        dst_layers = len(dst_kv_ptrs) // kv_buf_groups
        if src_layers == dst_layers:
            sliced_dst_kv_ptrs = dst_kv_ptrs
        else:
            sliced_dst_kv_ptrs = []
            start_layer = self.kv_args.prefill_start_layer
            transfer_draft_kv = get_pp_group().is_last_rank and draft_kv_layers
            if transfer_draft_kv:
                end_layer = start_layer + src_layers - draft_kv_layers
            else:
                end_layer = start_layer + src_layers

            # target kv
            for i in range(kv_buf_groups):
                layer_offset = i * hidden_kv_layers
                sliced_dst_kv_ptrs.extend(
                    dst_kv_ptrs[layer_offset + start_layer : layer_offset + end_layer]
                )
            # draft kv
            if transfer_draft_kv:
                for i in range(kv_buf_groups):
                    layer_offset = (
                        i * draft_kv_layers + kv_buf_groups * hidden_kv_layers
                    )
                    sliced_dst_kv_ptrs.extend(
                        dst_kv_ptrs[layer_offset : layer_offset + draft_kv_layers]
                    )
        layers_current_pp_stage = len(src_kv_ptrs)
        return src_kv_ptrs, sliced_dst_kv_ptrs, layers_current_pp_stage

    def send_kvcache(
        self,
        mooncake_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int32],
        executor: concurrent.futures.ThreadPoolExecutor,
        dst_layer_ids: Optional[List[int]] = None,
        dst_device_kv_indices: Optional[npt.NDArray[np.int32]] = None,
        dst_kv_item_len: Optional[int] = None,
        dst_attn_tp_size: Optional[int] = None,
    ):
        if dst_device_kv_indices is not None:
            raise NotImplementedError(
                "Ascend PD transfer does not support HiSparse "
                "destination device KV indices"
            )
        self._validate_envelope_kv_layout(
            dst_kv_ptrs, dst_kv_item_len, dst_attn_tp_size
        )
        # Group by indices
        prefill_kv_blocks, dst_kv_blocks = group_concurrent_contiguous(
            prefill_kv_indices, dst_kv_indices
        )

        if self.pp_size > 1:
            if self.is_mla_backend:
                src_kv_ptrs, sliced_dst_kv_ptrs, layers_current_pp_stage = (
                    self.get_mla_kv_ptrs_with_pp(self.kv_args.kv_data_ptrs, dst_kv_ptrs)
                )
                layers_params = [
                    (
                        src_kv_ptrs[layer_id],
                        sliced_dst_kv_ptrs[layer_id],
                        self.kv_args.kv_item_lens[layer_id],
                    )
                    for layer_id in range(layers_current_pp_stage)
                ]
            else:
                (
                    src_k_ptrs,
                    src_v_ptrs,
                    dst_k_ptrs,
                    dst_v_ptrs,
                    layers_current_pp_stage,
                ) = self.get_mha_kv_ptrs_with_pp(self.kv_args.kv_data_ptrs, dst_kv_ptrs)

                layers_params = [
                    (
                        src_k_ptrs[layer_id],
                        dst_k_ptrs[layer_id],
                        self.kv_args.kv_item_lens[layer_id],
                    )
                    for layer_id in range(layers_current_pp_stage)
                ] + [
                    (
                        src_v_ptrs[layer_id],
                        dst_v_ptrs[layer_id],
                        self.kv_args.kv_item_lens[layers_current_pp_stage + layer_id],
                    )
                    for layer_id in range(layers_current_pp_stage)
                ]
        else:
            num_layers = len(self.kv_args.kv_data_ptrs)
            layers_params = [
                (
                    self.kv_args.kv_data_ptrs[layer_id],
                    dst_kv_ptrs[layer_id],
                    self.kv_args.kv_item_lens[layer_id],
                )
                for layer_id in range(num_layers)
            ]

        def set_transfer_blocks(
            src_ptr: int, dst_ptr: int, item_len: int
        ) -> List[Tuple[int, int, int]]:
            transfer_blocks = []
            for prefill_index, decode_index in zip(prefill_kv_blocks, dst_kv_blocks):
                src_addr = src_ptr + int(prefill_index[0]) * item_len
                dst_addr = dst_ptr + int(decode_index[0]) * item_len
                length = item_len * len(prefill_index)
                transfer_blocks.append((src_addr, dst_addr, length))
            return transfer_blocks

        # Worker function for processing a single layer
        def process_layer(src_ptr: int, dst_ptr: int, item_len: int) -> int:
            transfer_blocks = set_transfer_blocks(src_ptr, dst_ptr, item_len)
            return self._transfer_data(mooncake_session_id, transfer_blocks)

        # Worker function for processing all layers in a batch
        def process_layers(layers_params: List[Tuple[int, int, int]]) -> int:
            transfer_blocks = []
            for src_ptr, dst_ptr, item_len in layers_params:
                transfer_blocks.extend(set_transfer_blocks(src_ptr, dst_ptr, item_len))
            return self._transfer_data(mooncake_session_id, transfer_blocks)

        if self.enable_custom_mem_pool:
            futures = [
                executor.submit(
                    process_layer,
                    src_ptr,
                    dst_ptr,
                    item_len,
                )
                for (src_ptr, dst_ptr, item_len) in layers_params
            ]
            for future in concurrent.futures.as_completed(futures):
                status = future.result()
                if status != 0:
                    for f in futures:
                        f.cancel()
                    return status
        else:
            # Combining all layers' params in one batch transfer is more efficient
            # compared to using multiple threads
            return process_layers(layers_params)

        return 0

    def send_kvcache_dcp(
        self,
        mooncake_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int32],
        *,
        dcp_token_item_lens: List[int],
        dst_dcp_size: int,
        dst_dcp_rank: int,
        src_page_offset: int,
        decode_prefix_len: int,
        num_kv_tokens: int,
        executor: concurrent.futures.ThreadPoolExecutor,
        dst_layer_ids: List[int],
        pack_buffer=None,
    ) -> int:
        replicated_group = self._dcp_replicated_kv_group()
        if replicated_group is None:
            return super().send_kvcache_dcp(
                mooncake_session_id,
                prefill_kv_indices,
                dst_kv_ptrs,
                dst_kv_indices,
                dcp_token_item_lens=dcp_token_item_lens,
                dst_dcp_size=dst_dcp_size,
                dst_dcp_rank=dst_dcp_rank,
                src_page_offset=src_page_offset,
                decode_prefix_len=decode_prefix_len,
                num_kv_tokens=num_kv_tokens,
                executor=executor,
                dst_layer_ids=dst_layer_ids,
                pack_buffer=None,
            )

        if num_kv_tokens is None:
            raise ValueError("PD DCP transfer requires num_kv_tokens")
        if not self.kv_args.kv_data_ptrs:
            return 0
        plan = build_dcp_token_transfer_plan(
            prefill_kv_indices,
            dst_kv_indices,
            physical_page_size=self.kv_args.page_size,
            dcp_size=dst_dcp_size,
            dcp_rank=dst_dcp_rank,
            src_page_offset=src_page_offset,
            decode_prefix_len=decode_prefix_len,
            num_kv_tokens=num_kv_tokens,
        )
        full_src_token_indices, full_dst_token_indices = (
            build_replicated_dcp_token_indices(
                prefill_kv_indices,
                dst_kv_indices,
                physical_page_size=self.kv_args.page_size,
                dcp_size=dst_dcp_size,
                src_page_offset=src_page_offset,
                num_kv_tokens=num_kv_tokens,
            )
        )
        if plan.src_token_indices.size == 0 and full_src_token_indices.size == 0:
            return 0

        src_layer_ids = self.kv_args.kv_layer_ids
        if src_layer_ids or dst_layer_ids:
            dst_indices = resolve_dcp_dst_entry_indices(
                src_layer_ids,
                dst_layer_ids,
                len(self.kv_args.kv_data_ptrs),
                len(dst_kv_ptrs),
            )
            src_kv_ptrs = self.kv_args.kv_data_ptrs
            dst_kv_ptrs = [dst_kv_ptrs[j] for j in dst_indices]
        else:
            src_kv_ptrs, dst_kv_ptrs, _ = self.get_mla_kv_ptrs_with_pp(
                self.kv_args.kv_data_ptrs,
                dst_kv_ptrs,
            )

        group_count = getattr(self.kv_args, "kv_buf_groups", 1)
        if replicated_group != 2 or group_count != 3:
            raise RuntimeError(
                "Ascend DSA PD DCP requires [MLA-K, MLA-V, index-K] buffer "
                "groups, got "
                f"group={replicated_group}, group_count={group_count}"
            )
        if len(src_kv_ptrs) != len(dst_kv_ptrs) or len(src_kv_ptrs) % group_count:
            raise RuntimeError(
                "Ascend PD DCP grouped KV layout is inconsistent: "
                f"src={len(src_kv_ptrs)}, dst={len(dst_kv_ptrs)}, "
                f"groups={group_count}"
            )
        if len(dcp_token_item_lens) < len(src_kv_ptrs):
            raise RuntimeError(
                "Ascend PD DCP item-length list is shorter than the KV layout: "
                f"items={len(dcp_token_item_lens)}, kv={len(src_kv_ptrs)}"
            )

        try:
            replicated_entries = build_dcp_replicated_kv_entry_mask(
                len(src_kv_ptrs),
                replicated_group=replicated_group,
                group_count=group_count,
                num_draft_layers=getattr(self.kv_args, "draft_kv_layers", 0),
            )
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        if not replicated_entries:
            raise RuntimeError(
                "Ascend PD DCP has no KV layers after PP/layer-id filtering"
            )
        mla_src_groups, mla_dst_groups = group_concurrent_contiguous(
            plan.src_token_indices,
            plan.dst_token_indices,
        )
        full_src_groups, full_dst_groups = group_concurrent_contiguous(
            full_src_token_indices,
            full_dst_token_indices,
        )

        def set_transfer_blocks(
            src_ptr: int, dst_ptr: int, token_item_len: int, replicated: bool
        ) -> List[Tuple[int, int, int]]:
            if replicated:
                src_groups, dst_groups = full_src_groups, full_dst_groups
            else:
                src_groups, dst_groups = mla_src_groups, mla_dst_groups
            return [
                (
                    src_ptr + int(src_group[0]) * token_item_len,
                    dst_ptr + int(dst_group[0]) * token_item_len,
                    len(src_group) * token_item_len,
                )
                for src_group, dst_group in zip(src_groups, dst_groups)
            ]

        layers_params = [
            (
                src_ptr,
                dst_ptr,
                dcp_token_item_lens[layer_id],
                replicated_entries[layer_id],
            )
            for layer_id, (src_ptr, dst_ptr) in enumerate(zip(src_kv_ptrs, dst_kv_ptrs))
        ]

        def process_layer(
            src_ptr: int, dst_ptr: int, token_item_len: int, replicated: bool
        ) -> int:
            return self._transfer_data(
                mooncake_session_id,
                set_transfer_blocks(src_ptr, dst_ptr, token_item_len, replicated),
            )

        if self.enable_custom_mem_pool:
            futures = [
                executor.submit(process_layer, *params) for params in layers_params
            ]
            return self._await_transfer_futures(futures)

        transfer_blocks = []
        for params in layers_params:
            transfer_blocks.extend(set_transfer_blocks(*params))
        return self._transfer_data(mooncake_session_id, transfer_blocks)

    def _is_generic_kvcache_state_type(self, st) -> bool:
        # DSV4 per-pool components also use the page-indexed send path.
        return (
            super()._is_generic_kvcache_state_type(st)
            or st in _DSV4_KVCACHE_STATE_TYPES
        )


class AscendKVSender(MooncakeKVSender):
    pass


class AscendKVReceiver(MooncakeKVReceiver):
    pass


class AscendKVBootstrapServer(MooncakeKVBootstrapServer):
    pass
