"""Optional instrumentation for diagnosing NPU distributed deadlocks.

Set ``SGLANG_NPU_COMM_DEBUG=1`` before starting workers to instrument public
``torch.distributed`` communication APIs.  Every NPU call is synchronized and
logged before and after invocation.  CPU-side communication logging is
controlled separately by ``SGLANG_NPU_COMM_DEBUG_CPU=1`` and is disabled by
default.
"""

from __future__ import annotations

import functools
import os
import sys
import threading
import traceback
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

_DEEPEP_OPS = {
    "get_dispatch_layout": "deepep.normal.get_dispatch_layout",
    "dispatch": "deepep.normal.dispatch",
    "combine": "deepep.normal.combine",
    "low_latency_dispatch": "deepep.low_latency.dispatch",
    "low_latency_combine": "deepep.low_latency.combine",
}

_installed = False
_sequence = 0
_sequence_lock = threading.Lock()
_reentrancy = threading.local()


def _is_wrapped(value: Any) -> bool:
    return bool(
        getattr(value, "__sglang_npu_comm_debug__", False)
        or getattr(getattr(value, "__func__", None), "__sglang_npu_comm_debug__", False)
    )


def _enabled() -> bool:
    value = os.getenv("SGLANG_NPU_COMM_DEBUG", "")
    return value.lower() not in ("", "0", "false", "no", "off")


def _cpu_logging_enabled() -> bool:
    value = os.getenv("SGLANG_NPU_COMM_DEBUG_CPU", "")
    return value.lower() not in ("", "0", "false", "no", "off")


def _stack_logging_enabled() -> bool:
    value = os.getenv("SGLANG_NPU_COMM_DEBUG_STACK", "")
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


def _has_npu() -> bool:
    try:
        import torch

        return callable(getattr(getattr(torch, "npu", None), "synchronize", None))
    except Exception:
        return False


def _rank_and_world(group: Any) -> tuple[int, int, tuple[int, ...]]:
    try:
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            return -1, -1, ()
        rank = dist.get_rank(group=group)
        world = dist.get_world_size(group=group)
        group_ranks: tuple[int, ...]
        try:
            get_group_ranks = getattr(dist, "get_process_group_ranks")
            group_ranks = tuple(int(item) for item in get_group_ranks(group))
        except (AttributeError, RuntimeError, TypeError):
            # Older torch versions do not expose group membership.  The
            # default group is still unambiguous, so reconstruct that case.
            group_ranks = tuple(range(world)) if group is None else ()
        return rank, world, group_ranks
    except Exception:
        # Logging must not hide the original communication exception.
        return -1, -1, ()


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


def _contains_npu_tensor(value: Any) -> bool:
    """Whether an argument contains at least one tensor resident on NPU."""
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return getattr(value.device, "type", None) == "npu"
    except Exception:
        return False
    if isinstance(value, dict):
        return any(_contains_npu_tensor(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_npu_tensor(item) for item in value)
    return False


def _should_trace_call(args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
    """Filter CPU-only torch.distributed calls from the default NPU trace."""
    return (
        _cpu_logging_enabled()
        or _contains_npu_tensor(args)
        or _contains_npu_tensor(kwargs)
    )


def _sglang_stack(limit: int = 3) -> str:
    """Return up to ``limit`` caller frames from SGLang source files.

    Frames from torch, DeepEP and other dependencies are deliberately skipped.
    The instrumentation implementation itself is also excluded, so the first
    frame identifies the SGLang communication call site.
    """
    current_file = os.path.normcase(__file__)
    selected: list[str] = []
    for frame in reversed(traceback.extract_stack()[:-1]):
        filename = os.path.normcase(frame.filename)
        if filename == current_file or not _is_sglang_file(filename):
            continue
        selected.append(f"{os.path.basename(frame.filename)}:{frame.lineno}")
        if len(selected) >= limit:
            break
    return " <- ".join(selected) or "unavailable"


def _is_sglang_file(filename: str) -> bool:
    normalized = filename.replace("\\", "/")
    return "/sglang/" in normalized or normalized.endswith("/sglang")


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
    rank, world, group_ranks = _rank_and_world(group)
    global_rank = (
        group_ranks[rank] if 0 <= rank < len(group_ranks) else -1
    )
    members = "|".join(str(item) for item in group_ranks) or "unknown"
    return (
        f"rank={rank},global_rank={global_rank},world={world},"
        f"group_ranks={members},tensors=[{' ; '.join(tensors)}]"
    )


def _log(
    op_name: str, phase: str, sequence: int, details: str, stack: str | None = None
) -> None:
    # A plain print is intentional: logging handlers differ between worker
    # processes, while flushed one-line records are easy to grep and merge.
    import os as _os

    stack_suffix = f" stack={stack}" if stack else ""
    print(
        f"[NPU-COMM] pid={_os.getpid()} seq={sequence} phase={phase} "
        f"op={op_name} {details}{stack_suffix}",
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

        if not _should_trace_call(args, kwargs):
            return original(*args, **kwargs)

        _reentrancy.active = True
        sequence = _next_sequence()
        details = _arguments_summary(
            args, kwargs, process_group_method=process_group_method
        )
        stack = _sglang_stack() if _stack_logging_enabled() else None
        try:
            _log(op_name, "before", sequence, details, stack)
            # Log before synchronizing as well: if a previous NPU kernel is
            # already stuck, the record still tells us which call reached it.
            _sync_npu()
            result = original(*args, **kwargs)
            # This also makes async_op launches observable as completed from
            # the device stream perspective, while preserving the Work object.
            _sync_npu()
            _log(op_name, "after", sequence, details, stack)
            return result
        except BaseException as exc:
            _log(
                op_name,
                "error",
                sequence,
                f"{details},exc={type(exc).__name__}:{exc}",
                stack,
            )
            raise
        finally:
            _reentrancy.active = False

    wrapped.__sglang_npu_comm_debug__ = True  # type: ignore[attr-defined]
    return wrapped


def _wrap_deepep(
    op_name: str, original: Callable[..., Any], *, bound_method: bool = False
) -> Callable[..., Any]:
    """Wrap a DeepEP Buffer communication method."""
    log_name = _DEEPEP_OPS[op_name]

    @functools.wraps(original)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if getattr(_reentrancy, "active", False):
            return original(*args, **kwargs)

        _reentrancy.active = True
        sequence = _next_sequence()
        # Exclude ``self`` from tensor summaries.  DeepEP's Buffer generally
        # stores its process group internally; rank/world still come from the
        # default process group, which is the one used by the dispatcher.
        details = _arguments_summary(args if bound_method else args[1:], kwargs)
        stack = _sglang_stack() if _stack_logging_enabled() else None
        try:
            _log(log_name, "before", sequence, details, stack)
            _sync_npu()
            result = original(*args, **kwargs)
            _sync_npu()
            _log(log_name, "after", sequence, details, stack)
            return result
        except BaseException as exc:
            _log(
                log_name,
                "error",
                sequence,
                f"{details},exc={type(exc).__name__}:{exc}",
                stack,
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
            if _is_wrapped(original):
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


def install_deepep_comm_debug(*buffer_types: type[Any]) -> bool:
    """Instrument DeepEP normal and low-latency Buffer methods.

    The NPU path uses the ``deep_ep`` Buffer class.  Missing packages and
    immutable C extension types are ignored; torch.distributed tracing remains
    available.
    """
    if not _enabled():
        return False

    if not buffer_types:
        candidates = (("deep_ep", "Buffer"),)
        discovered: list[type[Any]] = []
        for module_name, class_name in candidates:
            try:
                module = __import__(module_name, fromlist=[class_name])
                candidate = getattr(module, class_name, None)
                if isinstance(candidate, type):
                    discovered.append(candidate)
            except (ImportError, AttributeError):
                continue
        buffer_types = tuple(discovered)

    patched = False
    for buffer_type in buffer_types:
        for method_name in _DEEPEP_OPS:
            try:
                original = getattr(buffer_type, method_name)
                if _is_wrapped(original):
                    continue
                setattr(buffer_type, method_name, _wrap_deepep(method_name, original))
                patched = True
            except (AttributeError, RuntimeError, TypeError):
                continue
    return patched


def trace_deepep_call(
    op_name: str, fn: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    """Trace one DeepEP call site when class-level patching is unavailable."""
    if not _enabled() or not _has_npu() or _is_wrapped(fn):
        return fn(*args, **kwargs)
    method_name = next(
        (name for name, label in _DEEPEP_OPS.items() if label == op_name), None
    )
    if method_name is None:
        return fn(*args, **kwargs)
    return _wrap_deepep(method_name, fn, bound_method=True)(*args, **kwargs)
