#!/usr/bin/env python3
"""Replay a captured DeepEP dispatch/combine pair.

Capture during sglang execution with::

    SGLANG_NPU_COMM_DEBUG=1 \
    SGLANG_NPU_DEEPEP_CAPTURE_DIR=/tmp/deepep-capture \
    python -m sglang.launch_server ...

Then launch this script with the same number of ranks and DeepEP buffer sizes::

    torchrun --nproc_per_node=16 test/manual/ep/replay_deepep_capture.py \
      --capture-dir /tmp/deepep-capture --mode low_latency \
      --nvl-bytes 0 --rdma-bytes 1073741824

The script loads one dispatch and one combine capture per global rank, creates
a fresh Buffer, and replays both calls.  Process-local handles/events from the
original run are replaced by the handle/event produced by replay dispatch.
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("normal", "low_latency"), required=True)
    parser.add_argument("--nvl-bytes", type=int, default=0)
    parser.add_argument("--rdma-bytes", type=int, required=True)
    parser.add_argument("--backend", default="hccl")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    global_rank = int(os.environ.get("RANK", str(local_rank)))
    device = torch.device(args.device or f"npu:{local_rank}")
    if hasattr(torch, "npu"):
        torch.npu.set_device(device)
    dist.init_process_group(args.backend)

    import deep_ep

    prefix = f"deepep.{args.mode}"
    dispatch = _load(_latest_capture(args.capture_dir, f"{prefix}.dispatch", global_rank), device)
    combine = _load(_latest_capture(args.capture_dir, f"{prefix}.combine", global_rank), device)
    low_latency = args.mode == "low_latency"
    buffer = deep_ep.Buffer(
        dist.group.WORLD,
        args.nvl_bytes,
        args.rdma_bytes,
        low_latency_mode=low_latency,
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

