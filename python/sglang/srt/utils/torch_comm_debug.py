"""Optional instrumentation for diagnosing NPU distributed deadlocks.

Set ``SGLANG_NPU_COMM_DEBUG=1`` before starting workers to instrument public
``torch.distributed`` communication APIs.  Every call is synchronized and
logged before and after invocation.  The default is intentionally disabled so
there is no impact on normal runs.
"""

from __future__ import annotations

import functools
import os
import sys
import threading
from typing import Any, Callable


# Public APIs used by torch.distributed across the torch versions supported by
# sglang.  Missing attributes are skipped at installation time.
_COMM_OPS = (
    "all_reduce",
    "all_gather",
    "all_gather_into_tensor",
    "all_gather_into_tensor_coalesced",
    "all_gather_coalesced",
    "all_gather_object",
    "all_reduce_coalesced",
    "all_to_all",
    "all_to_all_single",
    "broadcast",
    "broadcast_object_list",
    "reduce",
    "reduce_scatter",
    "reduce_scatter_tensor",
    "reduce_scatter_tensor_coalesced",
    "scatter",
    "gather",
    "send",
    "recv",
    "isend",
    "irecv",
    "batch_isend_irecv",
    "barrier",
    "monitored_barrier",
)

_PROCESS_GROUP_OPS = (
    "allreduce",
    "allgather",
    "allgather_into_tensor",
    "alltoall",
    "alltoall_base",
    "barrier",
    "broadcast",
    "recv",
    "reduce",
    "reduce_scatter",
    "send",
)

_installed = False
_sequence = 0
_sequence_lock = threading.Lock()
_reentrancy = threading.local()


def _enabled() -> bool:
    value = os.getenv("SGLANG_NPU_COMM_DEBUG", "")
    return value.lower() not in ("", "0", "false", "no", "off")


def _next_sequence() -> int:
    global _sequence
    with _sequence_lock:
        _sequence += 1
        return _sequence


def _sync_npu() -> None:
    """Synchronize the current NPU stream, if available."""
    import torch

    npu = getattr(torch, "npu", None)
    synchronize = getattr(npu, "synchronize", None)
    if synchronize is not None:
        synchronize()


def _rank_and_world(group: Any) -> tuple[int, int]:
    try:
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            return -1, -1
        return dist.get_rank(group=group), dist.get_world_size(group=group)
    except Exception:
        # Logging must not hide the original communication exception.
        return -1, -1


def _tensor_summary(value: Any) -> str | None:
    """Return a compact, side-effect-free summary for tensors in arguments."""
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return (
                f"shape={tuple(value.shape)},dtype={value.dtype},"
                f"device={value.device},numel={value.numel()}"
            )
    except Exception:
        pass
    return None


def _arguments_summary(
    args: tuple[Any, ...], kwargs: dict[str, Any], *, process_group_method: bool = False
) -> str:
    # Avoid repr(tensor), which may synchronize or dump huge values.  Include
    # only tensor metadata and scalar arguments useful for matching call sites.
    tensors: list[str] = []
    for value in (*args, *kwargs.values()):
        summary = _tensor_summary(value)
        if summary is not None:
            tensors.append(summary)
    # ProcessGroup methods receive the ProcessGroup instance as ``self``;
    # public torch.distributed functions receive it as the ``group`` kwarg.
    group = args[0] if process_group_method and args else kwargs.get("group")
    rank, world = _rank_and_world(group)
    return f"rank={rank},world={world},tensors=[{' ; '.join(tensors)}]"


def _log(op_name: str, phase: str, sequence: int, details: str) -> None:
    # A plain print is intentional: logging handlers differ between worker
    # processes, while flushed one-line records are easy to grep and merge.
    import os as _os

    print(
        f"[NPU-COMM] pid={_os.getpid()} seq={sequence} phase={phase} "
        f"op={op_name} {details}",
        flush=True,
    )


def _wrap(
    op_name: str,
    original: Callable[..., Any],
    *,
    process_group_method: bool = False,
) -> Callable[..., Any]:
    @functools.wraps(original)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        # Protect against accidental recursion if a backend implementation
        # reaches another instrumented public API while synchronizing.
        if getattr(_reentrancy, "active", False):
            return original(*args, **kwargs)

        _reentrancy.active = True
        sequence = _next_sequence()
        details = _arguments_summary(
            args, kwargs, process_group_method=process_group_method
        )
        try:
            _log(op_name, "before", sequence, details)
            # Log before synchronizing as well: if a previous NPU kernel is
            # already stuck, the record still tells us which call reached it.
            _sync_npu()
            result = original(*args, **kwargs)
            # This also makes async_op launches observable as completed from
            # the device stream perspective, while preserving the Work object.
            _sync_npu()
            _log(op_name, "after", sequence, details)
            return result
        except BaseException as exc:
            _log(
                op_name,
                "error",
                sequence,
                f"{details},exc={type(exc).__name__}:{exc}",
            )
            raise
        finally:
            _reentrancy.active = False

    wrapped.__sglang_npu_comm_debug__ = True  # type: ignore[attr-defined]
    return wrapped


def _rebind_loaded_sglang_aliases(
    original: Callable[..., Any], wrapped: Callable[..., Any]
) -> None:
    """Update direct communication aliases already imported by sglang."""
    for module_name, module in tuple(sys.modules.items()):
        if not module_name.startswith("sglang.") or module is None:
            continue
        try:
            namespace = vars(module)
            for name, value in tuple(namespace.items()):
                if value is original:
                    namespace[name] = wrapped
        except (RuntimeError, TypeError):
            # A module can be mutating while workers import concurrently.
            continue


def install_torch_comm_debug() -> bool:
    """Install NPU communication instrumentation when enabled.

    Returns ``True`` if wrappers were installed and ``False`` when disabled or
    already installed.  The function is safe to call repeatedly.
    """
    global _installed
    if _installed or not _enabled():
        return False

    import torch.distributed as dist

    # Patch both public and c10d module attributes.  The latter covers code
    # that imported an operator directly from ``distributed_c10d``.
    modules = [dist]
    try:
        from torch.distributed import distributed_c10d

        modules.append(distributed_c10d)
    except ImportError:
        pass

    patched_count = 0
    for module in modules:
        for op_name in _COMM_OPS:
            if not hasattr(module, op_name):
                continue
            original = getattr(module, op_name)
            if getattr(original, "__sglang_npu_comm_debug__", False):
                continue
            if not callable(original):
                continue
            wrapped = _wrap(op_name, original)
            setattr(module, op_name, wrapped)
            _rebind_loaded_sglang_aliases(original, wrapped)
            patched_count += 1

    # ProcessGroup is a C extension on most torch versions.  Some builds allow
    # replacing its methods, others reject it; public API instrumentation above
    # remains useful in either case, so method patch failures are ignored.
    try:
        process_group_type = dist.ProcessGroup
        for op_name in _PROCESS_GROUP_OPS:
            if not hasattr(process_group_type, op_name):
                continue
            original = getattr(process_group_type, op_name)
            if getattr(original, "__sglang_npu_comm_debug__", False):
                continue
            setattr(
                process_group_type,
                op_name,
                _wrap(op_name, original, process_group_method=True),
            )
            patched_count += 1
    except (AttributeError, RuntimeError, TypeError):
        pass

    _installed = patched_count > 0
    return _installed
