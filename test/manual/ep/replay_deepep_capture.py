#!/usr/bin/env python3
"""Replay a captured DeepEP dispatch/combine pair.

Capture during sglang execution with::

    SGLANG_NPU_COMM_DEBUG=1 \
    SGLANG_NPU_DEEPEP_CAPTURE_DIR=/tmp/deepep-capture \
    python -m sglang.launch_server ...

Then launch this script with the same number of ranks and DeepEP buffer sizes::

    torchrun --nproc_per_node=16 test/manual/ep/replay_deepep_capture.py \
      --capture-dir /tmp/deepep-capture --mode low_latency

The script loads one dispatch and one combine capture per global rank, creates
a fresh Buffer, and replays both calls.  Process-local handles/events from the
original run are replaced by the handle/event produced by replay dispatch.
Buffer sizes are derived with DeepEP's size-hint APIs; explicit byte values
can be supplied to override them.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import os
from pathlib import Path
import sys
from typing import Any


def _prepend_env_path(name: str, path: Path) -> bool:
    value = str(path)
    entries = [item for item in os.environ.get(name, "").split(":") if item]
    if value in entries:
        return False
    os.environ[name] = ":".join([value, *entries])
    return True


def _bootstrap_deepep_custom_ops() -> None:
    """Expose DeepEP's bundled Ascend custom ops before loading torch/CANN."""
    spec = importlib.util.find_spec("deep_ep")
    if spec is None or spec.origin is None:
        raise ImportError("deep_ep is not installed in this Python environment")

    package_dir = Path(spec.origin).resolve().parent
    vendor_dir = package_dir / "vendors" / "hwcomputing"
    op_api_dir = vendor_dir / "op_api" / "lib"
    op_api_lib = op_api_dir / "libcust_opapi.so"
    if not op_api_lib.is_file():
        raise RuntimeError(
            f"DeepEP custom op library is missing: {op_api_lib}. "
            "Install a DeepEP-Ascend wheel built for this CANN/device version."
        )

    changed = _prepend_env_path("ASCEND_CUSTOM_OPP_PATH", vendor_dir)
    changed = _prepend_env_path("LD_LIBRARY_PATH", op_api_dir) or changed
    # glibc reads LD_LIBRARY_PATH when the process starts. Re-exec once so
    # deep_ep_cpp's dlopen("libcust_opapi.so") resolves this library rather
    # than falling back to the system libopapi.so.
    if changed and os.environ.get("SGLANG_DEEPEP_REPLAY_BOOTSTRAPPED") != "1":
        environment = os.environ.copy()
        environment["SGLANG_DEEPEP_REPLAY_BOOTSTRAPPED"] = "1"
        os.execvpe(
            sys.executable,
            [sys.executable, *sys.argv],
            environment,
        )


_bootstrap_deepep_custom_ops()

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch_npu  # noqa: E402, F401
from deep_ep import Buffer  # noqa: E402


def _restore(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, tuple):
        return tuple(_restore(item, device) for item in value)
    if isinstance(value, list):
        return [_restore(item, device) for item in value]
    if isinstance(value, dict):
        if set(value) == {"type"}:
            return None
        return {key: _restore(item, device) for key, item in value.items()}
    return value


def _latest_capture(directory: Path, op: str, rank: int) -> Path:
    pattern = str(directory / f"{op.replace('.', '_')}.rank{rank}.pid*.seq*.pt")
    paths = glob.glob(pattern)
    if not paths:
        raise FileNotFoundError(f"no capture found for {op}, global rank {rank}")
    return max(paths, key=lambda path: int(path.rsplit(".seq", 1)[1][:-3]))


def _load(path: Path, device: torch.device) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["args"] = _restore(payload["args"], device)
    payload["kwargs"] = _restore(payload["kwargs"], device)
    return payload


def _first_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return _first_tensor(item)
            except ValueError:
                pass
    if isinstance(value, dict):
        for item in value.values():
            try:
                return _first_tensor(item)
            except ValueError:
                pass
    raise ValueError("capture does not contain a tensor input")


def _infer_buffer_settings(
    buffer_type: Any,
    dispatch: dict[str, Any],
    *,
    low_latency: bool,
    world_size: int,
) -> tuple[int, int, int]:
    dispatch_args = dispatch["args"]
    hidden_size = int(_first_tensor(dispatch_args[0]).shape[-1])
    if low_latency:
        max_tokens_per_rank = int(dispatch_args[2])
        num_experts = int(dispatch_args[3])
        if num_experts % world_size != 0:
            raise ValueError(
                f"num_experts={num_experts} is not divisible by world_size={world_size}"
            )
        rdma_bytes = buffer_type.get_low_latency_rdma_size_hint(
            max_tokens_per_rank,
            hidden_size,
            world_size,
            num_experts,
        )
        return 0, int(rdma_bytes), num_experts // world_size

    # SGLang allocates normal-mode DeepEP buffers using BF16-width hidden
    # elements, including when the dispatch payload itself is quantized.
    hidden_bytes = hidden_size * 2
    configs = (
        buffer_type.get_dispatch_config(world_size),
        buffer_type.get_combine_config(world_size),
    )
    nvl_bytes = max(
        config.get_nvl_buffer_size_hint(hidden_bytes, world_size)
        for config in configs
    )
    rdma_bytes = max(
        config.get_rdma_buffer_size_hint(hidden_bytes, world_size)
        for config in configs
    )
    return int(nvl_bytes), int(rdma_bytes), int(buffer_type.num_sms)


def _get_process_group_options(backend: str) -> tuple[Any, int | None]:
    if backend != "hccl":
        return None, None

    hccl_buffer_size = int(
        os.environ.get("DEEPEP_HCCL_BUFFSIZE")
        or os.environ.get("HCCL_BUFFSIZE")
        or 200
    )
    options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
    options.hccl_config = {"hccl_buffer_size": hccl_buffer_size}
    return options, hccl_buffer_size


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("normal", "low_latency"), required=True)
    parser.add_argument("--nvl-bytes", type=int, default=None)
    parser.add_argument("--rdma-bytes", type=int, default=None)
    parser.add_argument("--backend", default="hccl")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    global_rank = int(os.environ.get("RANK", str(local_rank)))
    device = torch.device(args.device or f"npu:{local_rank}")
    if hasattr(torch, "npu"):
        torch.npu.set_device(device)
    pg_options, hccl_buffer_size = _get_process_group_options(args.backend)
    dist.init_process_group(args.backend, pg_options=pg_options)

    prefix = f"deepep.{args.mode}"
    dispatch = _load(
        _latest_capture(args.capture_dir, f"{prefix}.dispatch", global_rank),
        device,
    )
    combine = _load(
        _latest_capture(args.capture_dir, f"{prefix}.combine", global_rank),
        device,
    )
    low_latency = args.mode == "low_latency"
    inferred_nvl_bytes, inferred_rdma_bytes, num_qps_per_rank = (
        _infer_buffer_settings(
            Buffer,
            dispatch,
            low_latency=low_latency,
            world_size=dist.get_world_size(),
        )
    )
    nvl_bytes = inferred_nvl_bytes if args.nvl_bytes is None else args.nvl_bytes
    rdma_bytes = inferred_rdma_bytes if args.rdma_bytes is None else args.rdma_bytes
    if global_rank == 0:
        print(
            f"[DEEPEP-REPLAY] nvl_bytes={nvl_bytes} rdma_bytes={rdma_bytes} "
            f"num_qps_per_rank={num_qps_per_rank} "
            f"hccl_buffer_size_mb={hccl_buffer_size}",
            flush=True,
        )
    buffer = Buffer(
        dist.group.WORLD,
        nvl_bytes,
        rdma_bytes,
        low_latency_mode=low_latency,
        num_qps_per_rank=num_qps_per_rank,
    )

    dispatch_kwargs = dict(dispatch["kwargs"])
    dispatch_kwargs["async_finish"] = False
    dispatch_kwargs.pop("previous_event", None)
    if low_latency:
        dispatched = buffer.low_latency_dispatch(*dispatch["args"], **dispatch_kwargs)
        packed_hidden, _, handle, event, hook = dispatched
        event.current_stream_wait() if event is not None else None
        combine_kwargs = dict(combine["kwargs"])
        combine_kwargs["handle"] = handle
        combine_kwargs["async_finish"] = False
        combine_kwargs.pop("return_recv_hook", None)
        combined = buffer.low_latency_combine(**combine_kwargs)
    else:
        dispatched = buffer.dispatch(*dispatch["args"], **dispatch_kwargs)
        recv_x, _, _, _, handle, event = dispatched
        event.current_stream_wait() if event is not None else None
        combine_args = list(combine["args"])
        if len(combine_args) > 1:
            combine_args[1] = handle
        combine_kwargs = dict(combine["kwargs"])
        combine_kwargs["async_finish"] = False
        combine_kwargs.pop("previous_event", None)
        combined = buffer.combine(*combine_args, **combine_kwargs)

    if hasattr(torch, "npu"):
        torch.npu.synchronize()
    print(f"[DEEPEP-REPLAY] rank={global_rank} mode={args.mode} completed", flush=True)
    dist.barrier()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
