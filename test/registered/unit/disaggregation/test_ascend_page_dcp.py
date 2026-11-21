import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from sglang.srt.disaggregation.ascend.conn import AscendKVManager
from sglang.srt.disaggregation.base.conn import KVArgs, StateType
from sglang.srt.disaggregation.utils import build_kv_layer_ids, setup_state_kv_args
from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool
from sglang.srt.mem_cache.static_kv_storage import StaticKVStorage
from sglang.srt.runtime_context import get_parallel, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestAscendPageDcpSend(CustomTestCase):
    def test_dsa_indexer_int8_pairs_all_buffers_across_pp(self):
        """BF16 MLA with an INT8 indexer has four buffer groups, not three."""
        publish(
            ServerArgs(model_path="dummy", disaggregation_transfer_backend="ascend"),
            role="tokenizer",
        )
        self.addCleanup(reset_context)
        self.enterContext(
            patch("sglang.srt.disaggregation.utils.is_npu", return_value=True)
        )
        parallel = get_parallel()
        self.enterContext(
            parallel.override(pp_group=SimpleNamespace(is_last_rank=True))
        )

        def register(start_layer, dcp_size):
            with parallel.override(attn_dcp_size=dcp_size, attn_dcp_rank=0):
                pool = NPUMLATokenToKVPool(
                    size=16,
                    page_size=2,
                    dtype=torch.bfloat16,
                    kv_lora_rank=512,
                    qk_rope_head_dim=64,
                    layer_num=2 - start_layer,
                    device="cpu",
                    enable_memory_saver=False,
                    index_head_dim=128,
                    indexer_dtype=torch.int8,
                    index_size=16 * dcp_size,
                    start_layer=start_layer,
                    end_layer=2,
                )
            storage = StaticKVStorage(
                None, SimpleNamespace(get_kvcache=lambda: pool, state=None)
            )
            args = KVArgs()
            args.kv_data_ptrs, _, args.kv_item_lens = pool.get_contiguous_buf_infos()
            args.kv_layer_ids = build_kv_layer_ids(
                token_to_kv_pool=pool,
                draft_token_to_kv_pool=None,
                num_draft_entries=0,
                num_hidden_layers=2,
                storage=storage,
            )
            args.page_size = pool.page_size
            args.prefill_start_layer = start_layer
            setup_state_kv_args(args, storage, total_kv_layers=2)
            return pool, args

        prefill_pool, src = register(start_layer=1, dcp_size=1)
        mgr = object.__new__(AscendKVManager)
        mgr.kv_args = src
        mgr.pp_size = 2
        mgr.is_mla_backend = True
        mgr.is_hybrid_mla_backend = False
        mgr.enable_custom_mem_pool = False
        mgr.max_transfer_batch_indices = 0
        mgr._dcp_remote_decode_layout = prefill_pool.get_dcp_remote_decode_layout()
        mgr._transfer_data = Mock(return_value=0)
        for dcp_size in (1, 2):
            with self.subTest(dcp_size=dcp_size):
                _decode_pool, dst = register(start_layer=0, dcp_size=dcp_size)
                kwargs = dict(
                    mooncake_session_id="decode",
                    dst_kv_ptrs=dst.kv_data_ptrs,
                    dst_kv_indices=np.array([2], dtype=np.int32),
                    dst_layer_ids=dst.kv_layer_ids,
                    executor=None,
                )
                if dcp_size == 1:
                    rc = mgr.send_kvcache(
                        **kwargs, prefill_kv_indices=np.array([1], dtype=np.int32)
                    )
                    pages = [(1, 2, 1)] * 4
                else:
                    rc = mgr.send_kvcache_dcp(
                        **kwargs,
                        prefill_kv_indices=np.array([1, 2], dtype=np.int32),
                        dcp_token_item_lens=mgr.prepare_dcp_token_item_lens(
                            dst.kv_item_lens, 2
                        ),
                        dst_dcp_size=2,
                        dst_dcp_rank=1,
                        src_page_offset=0,
                        decode_prefix_len=0,
                        num_kv_tokens=4,
                        dst_kv_item_lens=dst.kv_item_lens,
                    )
                    # Main K and RoPE take rank 1's page. Index K and scale
                    # take both pages in the global slot space.
                    pages = [(2, 2, 1)] * 2 + [(1, 4, 2)] * 2
                self.assertEqual(rc, 0)
                self.assertEqual(
                    mgr._transfer_data.call_args.args[1],
                    [
                        (s + sp * width, d + dp * width, n * width)
                        for s, d, width, (sp, dp, n) in zip(
                            src.kv_data_ptrs,
                            dst.kv_data_ptrs[1::2],
                            src.kv_item_lens,
                            pages,
                        )
                    ],
                )

    def test_draft_global_slots_preserve_target_layout(self):
        for target_layout in (
            [False, False],
            [False, False, True],
            [False, False, True, True],
        ):
            for draft_layout in ([], [False], [False, False], [True, True]):
                with self.subTest(target=target_layout, draft=draft_layout):
                    pool_layout = target_layout + draft_layout
                    mgr = object.__new__(AscendKVManager)
                    mgr.is_hybrid_mla_backend = False
                    mgr.kv_args = SimpleNamespace(
                        kv_data_ptrs=list(range(len(pool_layout))),
                        num_draft_entries=len(draft_layout),
                    )
                    mgr._dcp_remote_decode_layout = pool_layout
                    self.assertEqual(
                        mgr._get_dcp_remote_decode_layout(),
                        target_layout + [True] * len(draft_layout),
                    )
                    self.assertEqual(pool_layout, target_layout + draft_layout)

    def test_hybrid_dsa_preserves_index_buffer_layout(self):
        mgr = object.__new__(AscendKVManager)
        mgr.is_hybrid_mla_backend = True
        mgr.kv_args = SimpleNamespace(
            kv_data_ptrs=[1_000, 2_000, 3_000, 4_000, 5_000],
            kv_layer_ids=[3, 3, 3, 16, 16],
            kv_item_lens=[20, 8, 4, 20, 8],
            page_size=2,
            num_draft_entries=2,
            state_types=[StateType.MAMBA, StateType.DSA],
        )
        mgr._dcp_remote_decode_layout = [False, False, True, False, False]
        self.assertEqual(
            mgr._get_dcp_remote_decode_layout(), [False, False, True, True, True]
        )
        mgr._transfer_data = Mock(return_value=0)
        self.assertEqual(
            mgr.send_kvcache_dcp(
                "decode",
                np.array([5, 6, 7], dtype=np.int32),
                [10_000, 20_000, 30_000, 40_000, 50_000],
                np.array([10, 11], dtype=np.int32),
                dcp_token_item_lens=[10, 4, 2, 10, 4],
                dst_dcp_size=2,
                dst_dcp_rank=1,
                src_page_offset=0,
                decode_prefix_len=0,
                num_kv_tokens=6,
                executor=None,
                dst_layer_ids=mgr.kv_args.kv_layer_ids,
                dst_kv_item_lens=[20, 8, 8, 40, 16],
            ),
            0,
        )
        mgr._transfer_data.assert_called_once_with(
            "decode",
            [
                (1_120, 10_200, 20),
                (2_048, 20_080, 8),
                (3_020, 30_080, 12),
                (4_100, 40_400, 60),
                (5_040, 50_160, 24),
            ],
        )
        mgr._dcp_remote_decode_layout = []
        with self.assertRaisesRegex(RuntimeError, "layout does not match"):
            mgr._get_dcp_remote_decode_layout()

    def test_rejects_incorrect_destination_geometry(self):
        mgr = object.__new__(AscendKVManager)
        mgr.is_hybrid_mla_backend = False
        mgr.kv_args = SimpleNamespace(
            kv_buf_groups=3,
            kv_data_ptrs=[1_000, 2_000, 3_000],
            kv_layer_ids=[],
            kv_item_lens=[20, 8, 4],
            page_size=2,
            mla_compression_ratios=None,
            num_draft_entries=0,
        )
        mgr._dcp_remote_decode_layout = [False, True, True]
        mgr._transfer_data = Mock(return_value=0)

        with self.assertRaisesRegex(RuntimeError, "destination geometry"):
            mgr.send_kvcache_dcp(
                "decode",
                np.array([5], dtype=np.int32),
                [10_000, 20_000, 30_000],
                np.array([7], dtype=np.int32),
                dcp_token_item_lens=[10, 4, 2],
                dst_dcp_size=4,
                dst_dcp_rank=0,
                src_page_offset=0,
                decode_prefix_len=0,
                num_kv_tokens=2,
                executor=None,
                dst_layer_ids=[],
                dst_kv_item_lens=[20, 8, 16],
            )

    def _hybrid_case(self, stage_ids, with_draft):
        model_ids = [3, 7, 11, 15]
        widths = [20] * len(stage_ids) + [8] * len(stage_ids)
        src_ids = stage_ids * 2
        dst_ids = model_ids * 2
        dst_widths = [20] * len(model_ids) + [8] * len(model_ids)
        if with_draft:
            src_ids += [16, 16]
            dst_ids += [16, 16]
            widths += [20, 8]
            dst_widths += [40, 16]
        src_ptrs = [10_000 * (i + 1) for i in range(len(src_ids))]
        dst_ptrs = [100_000 * (i + 1) for i in range(len(dst_ids))]
        mgr = object.__new__(AscendKVManager)
        mgr.is_hybrid_mla_backend = True
        # Hybrid pairing must not need the legacy NPU group/layer counts.
        mgr.kv_args = SimpleNamespace(
            kv_data_ptrs=src_ptrs,
            kv_layer_ids=src_ids,
            kv_item_lens=widths,
            page_size=2,
            num_draft_entries=2 if with_draft else 0,
            state_types=[StateType.MAMBA],
        )
        # Hybrid MLA must work without a layout hook on the public pool wrapper.
        mgr._dcp_remote_decode_layout = []
        mgr._get_dcp_remote_decode_layout = Mock(
            side_effect=AssertionError("dense MLA must select target/draft directly")
        )
        mgr.get_mla_kv_ptrs_with_pp = Mock(side_effect=AssertionError("legacy pairing"))
        mgr._transfer_data = Mock(return_value=0)
        kwargs = dict(
            mooncake_session_id="decode",
            prefill_kv_indices=np.array([5, 6, 7], dtype=np.int32),
            dst_kv_ptrs=dst_ptrs,
            dst_kv_indices=np.array([10, 11], dtype=np.int32),
            dcp_token_item_lens=[width // 2 for width in widths],
            dst_dcp_size=2,
            dst_dcp_rank=1,
            src_page_offset=0,
            decode_prefix_len=0,
            num_kv_tokens=6,
            executor=None,
            dst_layer_ids=dst_ids,
            dst_kv_item_lens=dst_widths,
        )
        expected = []
        for group, width in enumerate((20, 8)):
            for i, layer_id in enumerate(stage_ids):
                src = src_ptrs[group * len(stage_ids) + i]
                dst = dst_ptrs[group * len(model_ids) + model_ids.index(layer_id)]
                # Rank 1 owns source page 6, stored at local decode page 10.
                expected.append((src + 6 * width, dst + 10 * width, width))
        if with_draft:
            for i, width in enumerate((20, 8)):
                # Replicated draft keeps all three pages at global slots 20..22.
                expected.append(
                    (
                        src_ptrs[-2 + i] + 5 * width,
                        dst_ptrs[-2 + i] + 20 * width,
                        3 * width,
                    )
                )
        return mgr, kwargs, expected

    def test_hybrid_pairs_pages_by_layer_without_group_counts(self):
        for stage_ids in ([3, 7, 11, 15], [3, 7], [11, 15], [7, 11, 15], []):
            for with_draft in (False, True):
                for with_widths in (False, True):
                    with self.subTest(
                        stage_ids=stage_ids, draft=with_draft, widths=with_widths
                    ):
                        mgr, kwargs, expected = self._hybrid_case(stage_ids, with_draft)
                        self.assertEqual(
                            mgr.prepare_dcp_token_item_lens(
                                kwargs["dst_kv_item_lens"], kwargs["dst_dcp_size"]
                            ),
                            kwargs["dcp_token_item_lens"],
                        )
                        if not with_widths:
                            kwargs.pop("dst_kv_item_lens")
                        self.assertEqual(mgr.send_kvcache_dcp(**kwargs), 0)
                        mgr._transfer_data.assert_called_once_with("decode", expected)
                        mgr.get_mla_kv_ptrs_with_pp.assert_not_called()

    def test_hybrid_rejects_incomplete_layer_metadata_before_transfer(self):
        for missing_metadata in (False, True):
            with self.subTest(missing_metadata=missing_metadata):
                mgr, kwargs, _ = self._hybrid_case([11, 15], True)
                if missing_metadata:
                    kwargs["dst_layer_ids"] = []
                else:
                    # Keep latent layer 11 but remove its RoPE entry.
                    for name in ("dst_layer_ids", "dst_kv_ptrs", "dst_kv_item_lens"):
                        kwargs[name].pop(6)
                with self.assertRaises(RuntimeError):
                    mgr.send_kvcache_dcp(**kwargs)
                mgr._transfer_data.assert_not_called()

    def test_hybrid_rejects_invalid_draft_entry_count(self):
        for num_draft in (-1, 3):
            with self.subTest(num_draft=num_draft):
                mgr, kwargs, _ = self._hybrid_case([7], False)
                mgr.kv_args.num_draft_entries = num_draft
                with self.assertRaisesRegex(RuntimeError, "metadata is inconsistent"):
                    mgr.send_kvcache_dcp(**kwargs)
                mgr._transfer_data.assert_not_called()


if __name__ == "__main__":
    unittest.main()
