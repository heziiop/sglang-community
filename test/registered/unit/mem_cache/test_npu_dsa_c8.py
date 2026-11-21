"""CPU checks for the shared NPU C8 cache layout and INT8 scale contracts."""

import importlib
import unittest
from contextlib import ExitStack, contextmanager, nullcontext
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.hardware_backend.npu import memory_pool_npu as pools
from sglang.srt.layers.attention.dsa import dsa_npu_indexer as indexer
from sglang.srt.layers.cp import base as cp_base
from sglang.srt.layers.cp import interleave as cp
from sglang.srt.mem_cache.kv_cache_dtype import configure_kv_cache_dtype
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestNpuDsaC8(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        norm = ModuleType("sgl_kernel_npu.operators.norm")
        norm.fused_split_qk_norm = Mock()
        matmul = ModuleType("sgl_kernel_npu.operators.matmul")
        matmul.batch_matmul_transpose = Mock()
        with patch.dict(
            "sys.modules",
            {
                "torch_npu": ModuleType("torch_npu"),
                norm.__name__: norm,
                matmul.__name__: matmul,
            },
        ):
            cls.attention = importlib.import_module(
                "sglang.srt.hardware_backend.npu.modules.deepseek_v2_attention_mla_npu"
            )

    def setUp(self):
        super().setUp()
        parallel = patch.object(
            pools,
            "get_parallel",
            return_value=SimpleNamespace(attn_dcp_size=1, attn_dcp_rank=0),
        )
        parallel.start()
        self.addCleanup(parallel.stop)
        data_move = ModuleType("sgl_kernel_npu.operators.data_move")
        self.hp_scatter = Mock(side_effect=self.scatter)
        data_move.scatter_nd_update_hp = self.hp_scatter
        self.enterContext(patch.dict("sys.modules", {data_move.__name__: data_move}))

    def make_pool(self, dtype, indexer_dtype=None):
        return pools.NPUMLATokenToKVPool(
            size=128,
            page_size=128,
            dtype=dtype,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            layer_num=2,
            device="cpu",
            enable_memory_saver=False,
            index_head_dim=128,
            indexer_layer_ids=(0,),
            kv_cache_dim=656 if dtype != torch.bfloat16 else None,
            indexer_dtype=indexer_dtype,
        )

    @staticmethod
    def quantize(x, dst_type, **kwargs):
        # CPU stand-in for independent per-row/per-128-element quantization.
        blocks = x.float().reshape(-1, 128)
        scales = blocks.abs().amax(-1).clamp_min(1e-6) / 127
        values = (blocks / scales[:, None]).round().to(dst_type).reshape(x.shape)
        return values, scales

    @staticmethod
    def scatter(cache, indices, values):
        cache[indices.flatten().long()] = values

    @contextmanager
    def cp_gather(self, batch, rank, full_values):
        """Run the real interleave reorder with a simulated CP16 all-gather."""
        strategy = cp.InterleaveCPStrategy(cp_size=16)
        batch.attn_cp_metadata = strategy.build_metadata(
            num_tokens=full_values[0].shape[0], seqs_len=None
        )
        pending = iter(full_values)

        def collective(output, local):
            full = next(pending)
            shards = [full[r::16] for r in range(16)]
            rows = max(x.shape[0] for x in shards)
            padded = [
                torch.cat([x, x.new_zeros((rows - x.shape[0], *x.shape[1:]))])
                for x in shards
            ]
            torch.testing.assert_close(local, padded[rank], rtol=0, atol=0)
            output.copy_(torch.cat(padded))

        with (
            patch.object(
                cp_base, "get_parallel", return_value=SimpleNamespace(attn_cp_rank=rank)
            ),
            patch.object(
                cp, "get_parallel", return_value=SimpleNamespace(attn_cp_group=None)
            ),
            patch.object(cp, "use_symmetric_memory", return_value=nullcontext()),
            patch.object(cp, "is_allocation_symmetric", return_value=False),
            patch.object(cp, "attn_cp_all_gather_into_tensor", side_effect=collective),
        ):
            yield strategy.gather_hidden_states
        self.assertIsNone(next(pending, None), "Expected collective was skipped")

    def test_cp_main_kv_matches_full_quantization_without_rewriting_cache(self):
        mod = self.attention
        vendor = SimpleNamespace(
            npu_dynamic_block_quant=Mock(side_effect=self.quantize),
            npu_scatter_nd_update_=Mock(side_effect=self.scatter),
        )
        for tokens, rank, use_cp, dtype in (
            (32, 0, True, torch.int8),
            (35, 15, True, torch.int8),
            (35, 2, True, torch.int8),
            (35, 0, False, torch.int8),
            (35, 15, True, torch.bfloat16),
        ):
            with (
                self.subTest(tokens=tokens, rank=rank, cp=use_cp),
                ExitStack() as stack,
            ):
                pool = self.make_pool(dtype)
                stack.enter_context(
                    patch.object(pools, "torch_npu", vendor, create=True)
                )
                full = torch.randn(tokens, 576, dtype=torch.bfloat16)
                quantized_cp = use_cp and pool.dsa_kv_cache_store_c8
                expected = (
                    pool._pack_dsa_c8_kv_cache(full[:, :512], full[:, 512:])
                    if pool.dsa_kv_cache_store_c8
                    else None
                )
                vendor.npu_dynamic_block_quant.reset_mock()
                vendor.npu_scatter_nd_update_.reset_mock()
                self.hp_scatter.reset_mock()
                local = full[rank::16] if use_cp else full
                if use_cp and rank == 15:
                    # Physical tail padding must not become a cached token.
                    local = torch.cat([local, torch.full_like(local[:1], 123)])
                positions = torch.arange(local.shape[0])
                batch = SimpleNamespace(
                    forward_mode=ForwardMode.EXTEND,
                    out_cache_loc=torch.arange(tokens) + 7,
                )
                rotary = Mock(side_effect=lambda positions, q, k: (q, k))
                rotary.is_neox_style = True
                rotary.cos_sin_cache = torch.zeros(tokens, 64)
                model = SimpleNamespace(
                    layer_id=0,
                    q_lora_rank=2,
                    kv_lora_rank=512,
                    qk_rope_head_dim=64,
                    qk_nope_head_dim=2,
                    qk_head_dim=66,
                    num_local_heads=1,
                    v_head_dim=2,
                    alt_stream=None,
                    q_a_layernorm=lambda x: x,
                    kv_a_layernorm=lambda x: x,
                    fused_qkv_a_proj_with_mqa=lambda x: (
                        torch.cat([x[:, :2], x], -1),
                        None,
                    ),
                    q_b_proj=lambda x: (x.new_zeros((x.shape[0], 66)), None),
                    rotary_emb=rotary,
                    w_kc=None,
                    w_vc=None,
                    skip_topk=True,
                    is_nextn=False,
                    next_skip_topk=False,
                    o_proj=lambda x: (x, None),
                    rebuild_cp_kv_cache=Mock(
                        return_value=(
                            full[:, :512].unsqueeze(1),
                            full[:, 512:].unsqueeze(1),
                        )
                    ),
                )
                for name, value in (
                    ("get_token_to_kv_pool", lambda: pool),
                    ("get_dsa_token_shard_plan", lambda batch: None),
                    ("is_mla_preprocess_enabled", lambda: False),
                    ("dsa_use_prefill_cp", lambda batch: use_cp),
                    ("maybe_capture_indexer_topk", lambda layer, topk: topk),
                    (
                        "get_mla_dcp_backend",
                        lambda backend: SimpleNamespace(
                            uses_dcp_partial_attention=lambda batch: False
                        ),
                    ),
                    ("batch_matmul_transpose", lambda x, w, **kwargs: x),
                ):
                    stack.enter_context(patch.object(mod, name, value))
                stack.enter_context(
                    patch.object(
                        mod.forward_context, "get_attn_backend", return_value=None
                    )
                )
                gather = (
                    stack.enter_context(
                        self.cp_gather(batch, rank, [expected.view(torch.uint8)])
                    )
                    if quantized_cp
                    else Mock(side_effect=AssertionError("Non-CP gather"))
                )
                stack.enter_context(
                    patch.object(mod, "cp_gather_full_sequence_states", gather)
                )
                state = mod.forward_dsa_prepare_npu(
                    model, positions, local, batch, None, False
                )
                self.assertEqual(state[-1], quantized_cp)
                if use_cp and not quantized_cp:
                    model.rebuild_cp_kv_cache.assert_called_once()
                else:
                    model.rebuild_cp_kv_cache.assert_not_called()
                with patch.object(
                    mod,
                    "forward_mla_attention_npu",
                    return_value=local.new_zeros((local.shape[0], 1, 2)),
                ) as attn:
                    mod.forward_dsa_core_npu(model, *state)
                self.assertEqual(
                    attn.call_args.kwargs["save_kv_cache"], not quantized_cp
                )
                if quantized_cp:
                    self.assertEqual(
                        vendor.npu_dynamic_block_quant.call_args.args[0].shape[0],
                        local.shape[0],
                    )
                    vendor.npu_dynamic_block_quant.assert_called_once()
                    self.hp_scatter.assert_called_once()
                    vendor.npu_scatter_nd_update_.assert_not_called()
                    torch.testing.assert_close(
                        pool.k_buffer[0].view(-1, 656)[batch.out_cache_loc],
                        expected,
                        rtol=0,
                        atol=0,
                    )
                else:
                    vendor.npu_dynamic_block_quant.assert_not_called()

    def test_cp_indexer_gathers_quantized_k_and_matching_scales(self):
        vendor = SimpleNamespace(
            npu_dynamic_quant=Mock(side_effect=self.quantize),
            npu_scatter_nd_update_=Mock(side_effect=self.scatter),
        )
        for tokens, rank, use_cp, dtype, indexer_dtype in (
            (32, 0, True, torch.int8, None),
            (35, 2, True, torch.int8, None),
            (35, 15, True, torch.int8, None),
            (35, 0, False, torch.int8, None),
            (35, 15, True, torch.bfloat16, None),
            (35, 2, True, torch.bfloat16, torch.int8),
            (35, 2, True, torch.int8, torch.bfloat16),
        ):
            with (
                self.subTest(
                    tokens=tokens, rank=rank, dtype=dtype, indexer_dtype=indexer_dtype
                ),
                ExitStack() as stack,
            ):
                self.hp_scatter.reset_mock()
                vendor.npu_scatter_nd_update_.reset_mock()
                pool = self.make_pool(dtype, indexer_dtype)
                self.assertEqual(pool.k_buffer.dtype, dtype)
                for module in (pools, indexer):
                    stack.enter_context(
                        patch.object(module, "torch_npu", vendor, create=True)
                    )
                full = torch.randn(tokens, 128, dtype=torch.bfloat16)
                expected_dtype = indexer_dtype or dtype
                quantized = expected_dtype == torch.int8
                self.assertEqual(pool.get_index_k_buffer(0).dtype, expected_dtype)
                expected_k, expected_scale = (
                    indexer._quantize_npu_indexer_activation(
                        full, pool.indexer_hadamard_128, expected_dtype
                    )
                    if quantized
                    else (full, None)
                )
                local = full[rank::16] if use_cp else full
                if use_cp and rank == 15:
                    local = torch.cat([local, torch.full_like(local[:1], 123)])
                rows = local.shape[0]
                batch = SimpleNamespace(
                    forward_mode=ForwardMode.EXTEND,
                    out_cache_loc=torch.arange(tokens) + 7,
                    attn_cp_metadata=None,
                    seq_lens=torch.tensor([tokens]),
                    extend_seq_lens=torch.tensor([tokens]),
                )
                full_values = (
                    [expected_k.view(torch.uint8), expected_scale.view(-1, 1)]
                    if quantized
                    else [expected_k]
                )
                gather = (
                    stack.enter_context(self.cp_gather(batch, rank, full_values))
                    if use_cp
                    else Mock(side_effect=AssertionError("Non-CP gather"))
                )
                rotary = Mock(side_effect=lambda positions, q, k: (q, k))
                rotary.is_neox_style = False
                rotary.cos_sin_cache = torch.zeros(tokens, 64)
                model = SimpleNamespace(
                    dsa_enable_prefill_cp=True,
                    rotary_emb=rotary,
                    n_heads=1,
                    head_dim=128,
                    rope_head_dim=64,
                    hidden_size=128,
                    wq_b=lambda x: (x, None),
                    wk=lambda x: (x, None),
                    k_norm=lambda x: x,
                    num_init_tokens=0,
                    num_local_tokens=0,
                    weights_proj=Mock(return_value=(torch.ones(rows, 1), None)),
                    index_topk=2,
                    do_npu_cp_balance_indexer=Mock(
                        return_value=torch.zeros(rows, 2, dtype=torch.int32)
                    ),
                )
                model.weights_proj.weight = torch.ones(1, dtype=torch.bfloat16)
                backend = SimpleNamespace(
                    forward_metadata=SimpleNamespace(
                        seq_lens_cpu_int=[tokens],
                        actual_seq_lengths_q=torch.tensor([rows]),
                        actual_seq_lengths_kv=torch.tensor([tokens]),
                        block_tables=torch.zeros(1, 1, dtype=torch.int32),
                    )
                )
                for name, value in (
                    ("get_token_to_kv_pool", lambda: pool),
                    ("get_parallel", lambda: SimpleNamespace(dcp_enabled=False)),
                    ("get_attn_backend", lambda: backend),
                    ("cp_gather_full_sequence_states", gather),
                    ("_get_indexer_query_shard", lambda batch, bs: None),
                ):
                    stack.enter_context(patch.object(indexer, name, value))
                stack.enter_context(
                    patch.object(
                        indexer.envs.SGLANG_NPU_USE_MULTI_STREAM,
                        "get",
                        return_value=False,
                    )
                )
                kernel = Mock(return_value=torch.zeros(rows, 1, 2, dtype=torch.int32))
                stack.enter_context(
                    patch.object(
                        torch.ops,
                        "npu",
                        SimpleNamespace(npu_quant_lightning_indexer=kernel),
                    )
                )
                indexer.DSANPUIndexerMixin.forward_npu(
                    model, local, local, torch.arange(rows), batch, 0
                )
                torch.testing.assert_close(
                    pool.get_index_k_buffer(0).view(-1, 128)[batch.out_cache_loc],
                    expected_k,
                    rtol=0,
                    atol=0,
                )
                if quantized:
                    self.assertEqual(self.hp_scatter.call_count, 2)
                    vendor.npu_scatter_nd_update_.assert_not_called()
                    torch.testing.assert_close(
                        pool.get_index_k_scale_buffer(0).view(-1)[batch.out_cache_loc],
                        expected_scale,
                        rtol=0,
                        atol=0,
                    )
                    self.assertEqual(kernel.call_args.kwargs["query"].shape[0], rows)
                    self.assertEqual(kernel.call_args.kwargs["query"].dtype, torch.int8)
                    self.assertEqual(
                        kernel.call_args.kwargs["weights"].dtype, torch.float16
                    )
                else:
                    self.assertIsNone(pool.index_k_scale_buffer)
                    self.hp_scatter.assert_not_called()
                    vendor.npu_scatter_nd_update_.assert_called_once()
                    model.do_npu_cp_balance_indexer.assert_called_once()

    def test_hp_writes_accept_strided_inputs_and_preserve_other_rows(self):
        pool = self.make_pool(torch.int8)
        loc = torch.tensor([3, 99, 7, 99, 1, 99], dtype=torch.int32)[::2]
        packed = torch.randint(-128, 128, (6, 656), dtype=torch.int8)[::2]
        index_k = torch.randint(-128, 128, (6, 128), dtype=torch.int8)[::2]
        scales = torch.randn(6, dtype=torch.float16)[::2]
        self.assertFalse(loc.is_contiguous())
        before = [
            (buffer, buffer.clone(), updates)
            for buffer, updates in (
                (pool.k_buffer[0].view(-1, 656), packed),
                (pool.get_index_k_buffer(0).view(-1, 128), index_k),
                (pool.get_index_k_scale_buffer(0).view(-1, 1), scales[:, None]),
            )
        ]
        pool.set_dsa_c8_kv_buffer(0, loc, packed)
        pool.set_index_k_buffer(0, loc, index_k)
        pool.set_index_k_scale_buffer(0, loc, scales)

        self.assertEqual(self.hp_scatter.call_count, 3)
        self.assertEqual(
            [call.args[0].dtype for call in self.hp_scatter.call_args_list],
            [torch.int8, torch.int8, torch.float16],
        )
        for call in self.hp_scatter.call_args_list:
            self.assertTrue(all(t.is_contiguous() for t in call.args))
        for buffer, expected, updates in before:
            expected[loc.long()] = updates
            torch.testing.assert_close(buffer, expected, rtol=0, atol=0)

    def test_fp8_and_int8_allocate_the_same_packed_main_cache(self):
        for dtype, scale_dtype in (
            (torch.int8, torch.float16),
            (torch.float8_e4m3fn, torch.float32),
        ):
            with self.subTest(dtype=dtype):
                pool = self.make_pool(dtype)
                self.assertTrue(pool.dsa_kv_cache_store_c8)
                self.assertEqual(pool.k_buffer.shape, (2, 2, 128, 1, 656))
                self.assertEqual(pool.k_buffer.dtype, dtype)
                self.assertEqual(pool.v_buffer.numel(), 0)
                self.assertEqual(pool.v_buffer.dtype, torch.bfloat16)
                self.assertEqual(pool.index_k_scale_buffer.shape, (1, 2, 128, 1))
                self.assertEqual(pool.index_k_scale_buffer.dtype, scale_dtype)
                # Only the active indexer layer owns K and scale buffers.
                self.assertEqual(len(pool._get_disagg_buffer_entries()), 4)
                self.assertEqual(pool.get_state_layer_ids(), [0, 0])
                expected_bytes = (
                    2 * 656 + 128 + torch.empty((), dtype=scale_dtype).element_size()
                ) * 256
                self.assertEqual(pool.get_kv_size_bytes(), expected_bytes)

    def test_int8_write_preserves_rope_and_scale_bytes_at_the_requested_slots(self):
        pool = self.make_pool(torch.int8)
        latent = torch.randn(2, 1, 512, dtype=torch.bfloat16)
        rope = torch.randn(2, 1, 64, dtype=torch.bfloat16)
        quantized = torch.arange(1024).to(torch.int8).view(2, 1, 512)
        scales = torch.tensor([[0.125, 0.25, 0.5, 1.0], [2.0, 4.0, 8.0, 16.0]])

        def scatter(cache, indices, values):
            cache[indices.flatten().long()] = values

        vendor = SimpleNamespace(
            npu_dynamic_block_quant=Mock(return_value=(quantized, scales)),
            npu_scatter_nd_update_=Mock(side_effect=scatter),
        )
        loc = torch.tensor([7, 129])
        with patch.object(pools, "torch_npu", vendor, create=True):
            pool.set_kv_buffer(SimpleNamespace(layer_id=0), loc, latent, rope)
        quant_args = vendor.npu_dynamic_block_quant.call_args
        self.assertEqual(quant_args.args[0].shape, (2, 1, 512))
        self.assertEqual(
            quant_args.kwargs,
            dict(dst_type=torch.int8, row_block_size=1, col_block_size=128),
        )
        rows = pool.k_buffer[0].view(-1, 656)[loc]
        torch.testing.assert_close(rows[:, :512], quantized.view(2, 512))
        torch.testing.assert_close(
            rows[:, 512:640].contiguous().view(torch.bfloat16), rope.view(2, 64)
        )
        torch.testing.assert_close(
            rows[:, 640:].contiguous().view(torch.float32), scales
        )
        self.assertFalse(pool.k_buffer[0].view(-1, 656)[0].any())
        # Retraction/offload receives the full packed record and the separate LI scale.
        buffers = pool._get_cpu_offload_layer_buffers(0)
        self.assertEqual(buffers[0].shape, (256, 1, 656))
        self.assertEqual(buffers[-1].dtype, torch.float16)

    def test_indexer_quantizes_after_hadamard_and_casts_scales_for_each_dtype(self):
        x = torch.randn(2, 3, 128, dtype=torch.bfloat16)
        hadamard = indexer.create_npu_hadamard_128(128, "cpu")
        rotated = (x @ hadamard).reshape(-1, 128)

        def dynamic_quant(input_, dst_type):
            return (
                torch.zeros(input_.shape, dtype=dst_type),
                torch.arange(1, input_.shape[0] + 1, dtype=torch.float32),
            )

        vendor = SimpleNamespace(npu_dynamic_quant=Mock(side_effect=dynamic_quant))
        with patch.object(indexer, "torch_npu", vendor, create=True):
            for dtype, scale_dtype in (
                (torch.int8, torch.float16),
                (torch.float8_e4m3fn, torch.float32),
            ):
                with self.subTest(dtype=dtype):
                    q, scale = indexer._quantize_npu_indexer_activation(
                        x, hadamard, dtype
                    )
                    empty_q, empty_scale = indexer._quantize_npu_indexer_activation(
                        x[:0], hadamard, dtype
                    )
                    self.assertEqual(q.shape, x.shape)
                    self.assertEqual(q.dtype, dtype)
                    self.assertEqual(scale.dtype, scale_dtype)
                    torch.testing.assert_close(
                        scale,
                        torch.arange(1, 7, dtype=scale_dtype).view(2, 3),
                    )
                    self.assertEqual(empty_q.shape, (0, 3, 128))
                    self.assertEqual(empty_scale.dtype, scale_dtype)

        self.assertEqual(vendor.npu_dynamic_quant.call_count, 2)
        for call in vendor.npu_dynamic_quant.call_args_list:
            torch.testing.assert_close(call.args[0], rotated)

    def test_int8_dtype_resolution_preserves_independent_draft_dtype(self):
        for is_draft, expected in (
            (False, ("int8", torch.int8)),
            (True, ("bf16", torch.bfloat16)),
        ):
            with patch(
                "sglang.srt.mem_cache.kv_cache_dtype.current_platform"
            ) as platform:
                platform.is_npu.return_value = True
                result = configure_kv_cache_dtype(
                    server_args_kv_cache_dtype="int8",
                    quant_config=None,
                    model_dtype=torch.bfloat16,
                    is_draft_worker=is_draft,
                    is_dflash=False,
                    speculative_draft_attention_backend="ascend",
                    speculative_draft_kv_cache_dtype="bf16",
                )
                self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
