from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from sglang.srt.layers.moe.token_dispatcher import deepep
from sglang.srt.layers.moe.utils import DeepEPMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_exact_modes_select_independent_process_buffers():
    normal_buffer = object()
    low_latency_buffer = object()
    legacy_auto_buffer = object()
    state = SimpleNamespace(
        normal_buffer=normal_buffer,
        low_latency_buffer=low_latency_buffer,
        buffer=legacy_auto_buffer,
    )

    with patch.object(deepep.DeepEPBuffer, "_state", return_value=state):
        kwargs = dict(
            group=Mock(),
            hidden_size=128,
            param_bytes=2,
            num_max_dispatch_tokens_per_rank=16,
            num_experts=16,
        )
        assert (
            deepep.DeepEPBuffer.get_deepep_buffer(
                deepep_mode=DeepEPMode.NORMAL, **kwargs
            )
            is normal_buffer
        )
        assert (
            deepep.DeepEPBuffer.get_deepep_buffer(
                deepep_mode=DeepEPMode.LOW_LATENCY, **kwargs
            )
            is low_latency_buffer
        )
        assert (
            deepep.DeepEPBuffer.get_deepep_buffer(
                deepep_mode=DeepEPMode.AUTO, **kwargs
            )
            is legacy_auto_buffer
        )


def test_npu_auto_uses_distinct_groups_and_exact_modes():
    normal_group = Mock()
    low_latency_group = Mock()
    low_latency_coordinator = Mock(device_group=low_latency_group)

    with (
        patch.object(deepep, "_is_npu", True),
        patch.object(deepep, "use_deepep", True),
        patch.object(
            deepep,
            "get_moe_ep_low_latency_group",
            return_value=low_latency_coordinator,
        ) as get_low_latency_group,
        patch.object(deepep, "_DeepEPDispatcherImplNormal") as normal_impl,
        patch.object(
            deepep, "_DeepEPDispatcherImplLowLatency"
        ) as low_latency_impl,
    ):
        deepep.DeepEPDispatcher(
            group=normal_group,
            router_topk=2,
            num_experts=16,
            num_local_experts=1,
            hidden_size=128,
            params_dtype=None,
            deepep_mode=DeepEPMode.AUTO,
        )

    get_low_latency_group.assert_called_once_with()
    normal_kwargs = normal_impl.call_args.kwargs
    low_latency_kwargs = low_latency_impl.call_args.kwargs
    assert normal_kwargs["group"] is normal_group
    assert normal_kwargs["deepep_mode"] == DeepEPMode.NORMAL
    assert low_latency_kwargs["group"] is low_latency_group
    assert low_latency_kwargs["deepep_mode"] == DeepEPMode.LOW_LATENCY


def test_npu_auto_rejects_shared_group():
    group = Mock()

    with (
        patch.object(deepep, "_is_npu", True),
        patch.object(deepep, "use_deepep", True),
        patch.object(deepep, "_DeepEPDispatcherImplNormal"),
        patch.object(deepep, "_DeepEPDispatcherImplLowLatency"),
        pytest.raises(RuntimeError, match="requires distinct"),
    ):
        deepep.DeepEPDispatcher(
            group=group,
            low_latency_group=group,
            router_topk=2,
            num_experts=16,
            num_local_experts=1,
            hidden_size=128,
            params_dtype=None,
            deepep_mode=DeepEPMode.AUTO,
        )
