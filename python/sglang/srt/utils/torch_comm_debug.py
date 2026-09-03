"""Optional instrumentation for diagnosing NPU distributed deadlocks.

Set ``SGLANG_NPU_COMM_DEBUG=1`` before starting workers to instrument public
``torch.distributed`` communication APIs.  Every NPU call is synchronized and
logged before and after invocation.  CPU-side communication logging is
controlled separately by ``SGLANG_NPU_COMM_DEBUG_CPU=1`` and is disabled by
default.  DeepEP input capture is controlled by
``SGLANG_NPU_DEEPEP_CAPTURE_DIR`` and
``SGLANG_NPU_DEEPEP_CAPTURE_MODE`` (``mismatch``/``all``/``once``/``seq``/``error``).
For ``seq`` mode, set ``SGLANG_NPU_DEEPEP_CAPTURE_SEQ`` to the local sequence
number to persist.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
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
_deepep_capture_state: dict[str, Any] = {}
_deepep_capture_id = 0


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


def _capture_dir() -> Path | None:
    value = os.getenv("SGLANG_NPU_DEEPEP_CAPTURE_DIR", "")
    if not value:
        return None
    directory = Path(value)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _capture_mode() -> str:
    return os.getenv("SGLANG_NPU_DEEPEP_CAPTURE_MODE", "mismatch").lower()


def _capture_seq() -> int | None:
    value = os.getenv("SGLANG_NPU_DEEPEP_CAPTURE_SEQ", "")
    try:
        return int(value) if value else None
    except ValueError:
        return None


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


def _call_stack() -> str:
    """Return all caller frames as ``filename:line`` entries.

    The instrumentation implementation itself is excluded.  Keeping every
    other frame makes it possible to see transitions through torch, DeepEP and
    application code while retaining a compact, path-independent format.
    """
    current_file = os.path.normcase(__file__)
    selected: list[str] = []
    for frame in reversed(traceback.extract_stack()[:-1]):
        filename = os.path.normcase(frame.filename)
        if filename == current_file:
            continue
        selected.append(f"{os.path.basename(frame.filename)}:{frame.lineno}")
    return " <- ".join(selected) or "unavailable"


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


def _capture_value(value: Any) -> Any:
    """Convert DeepEP call arguments into CPU-serializable debug data."""
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
    except Exception:
        pass
    if isinstance(value, tuple):
        return tuple(_capture_value(item) for item in value)
    if isinstance(value, list):
        return [_capture_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _capture_value(item) for key, item in value.items()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    # Events, handles and process groups are process/device-local and cannot be
    # replayed.  The replay demo creates fresh values from the captured
    # dispatch instead.
    return {"type": type(value).__name__}


def _capture_deepep_inputs(
    op_name: str,
    sequence: int,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    details: str,
    stack: str | None,
    *,
    bound_method: bool = False,
) -> dict[str, Any] | None:
    directory = _capture_dir()
    if directory is None:
        return None
    if _capture_mode() == "mismatch":
        return None
    try:
        import torch

        rank = _int_from_details(details, "global_rank")
        pid = os.getpid()
        if op_name.endswith("get_dispatch_layout"):
            return None
        payload = {
            "op": op_name,
            "seq": sequence,
            "pid": pid,
            "global_rank": rank,
            "details": details,
            "stack": stack,
            "args": _capture_value(args if bound_method else args[1:]),
            "kwargs": _capture_value(kwargs),
        }
        mode = _capture_mode()
        is_dispatch = ".dispatch" in op_name
        is_combine = ".combine" in op_name
        if is_dispatch:
            _deepep_capture_state["last_dispatch"] = payload
        selected = (
            mode == "all"
            or (mode == "seq" and _capture_seq() == sequence)
            or (mode == "seq" and is_combine and _deepep_capture_state.get("armed"))
            or (mode == "once" and not _deepep_capture_state.get("saved"))
            or (mode == "once" and is_combine and _deepep_capture_state.get("armed"))
        )
        if (
            is_dispatch
            and ((mode == "seq" and _capture_seq() == sequence) or mode == "once")
        ):
            _deepep_capture_state["armed"] = True
        if selected:
            _save_capture_payload(directory, payload)
            # Selecting a combine seq also persists the immediately preceding
            # dispatch from this process, producing a replayable pair.
            if is_combine and _deepep_capture_state.get("last_dispatch") is not None:
                _save_capture_payload(
                    directory, _deepep_capture_state["last_dispatch"]
                )
            _deepep_capture_state["saved"] = True
            if is_combine:
                _deepep_capture_state["armed"] = False
        return payload
    except Exception as exc:
        # Capturing must never change the original DeepEP failure mode.
        print(
            f"[NPU-COMM] capture_error op={op_name} "
            f"exc={type(exc).__name__}:{exc}",
            flush=True,
        )
        return None


def _save_capture_payload(directory: Path, payload: dict[str, Any]) -> None:
    import torch

    op_name = str(payload["op"])
    rank = int(payload["global_rank"])
    pid = int(payload["pid"])
    sequence = int(payload["seq"])
    filename = f"{op_name.replace('.', '_')}.rank{rank}.pid{pid}.seq{sequence}.pt"
    torch.save(payload, directory / filename)


def _clone_for_capture(value: Any) -> Any:
    """Clone tensors on their current device; CPU transfer is deferred."""
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().clone()
    except Exception:
        pass
    if isinstance(value, tuple):
        return tuple(_clone_for_capture(item) for item in value)
    if isinstance(value, list):
        return [_clone_for_capture(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _clone_for_capture(item) for key, item in value.items()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return None


def record_deepep_snapshot(
    op_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    details: str = "",
    stack: str | None = None,
    dispatch_tokens: int | None = None,
    topk: int | None = None,
    combine_tokens: int | None = None,
) -> None:
    """Keep one DeepEP input snapshot on device memory for mismatch capture."""
    if _capture_dir() is None or _capture_mode() != "mismatch" or not _has_npu():
        return
    global _deepep_capture_id
    _deepep_capture_id += 1
    key = "last_" + ("dispatch" if ".dispatch" in op_name else "combine")
    global_rank = _int_from_details(details, "global_rank")
    if global_rank < 0:
        try:
            import torch.distributed as dist

            global_rank = dist.get_rank()
        except Exception:
            pass
    _deepep_capture_state[key] = {
        "op": op_name,
        "pid": os.getpid(),
        "seq": _deepep_capture_id,
        "global_rank": global_rank,
        "details": details,
        "stack": stack,
        # Actual Buffer operator inputs are cloned by _wrap_deepep below.  This
        # metadata snapshot only tracks counts until a mismatch is confirmed.
        "args": (),
        "kwargs": {},
        "dispatch_tokens": dispatch_tokens,
        "topk": topk,
        "combine_tokens": combine_tokens,
    }


def _flush_mismatch_capture(
    dispatch: dict[str, Any], combine: dict[str, Any], gathered: list[dict[str, Any]]
) -> None:
    directory = _capture_dir()
    if directory is None or _deepep_capture_state.get("mismatch_saved"):
        return
    for payload in (dispatch, combine):
        cpu_payload = _clone_for_capture(payload)
        cpu_payload = _capture_value(cpu_payload)
        _save_capture_payload(directory, cpu_payload)
    _deepep_capture_state["mismatch_saved"] = True


def maybe_capture_deepep_mismatch(
    *,
    group: Any,
    is_extend_in_batch: bool,
) -> bool:
    """Detect mixed-step token mismatch with a Gloo all-gather and save inputs."""
    if _capture_dir() is None or _capture_mode() != "mismatch" or not _has_npu():
        return False
    import torch.distributed as dist

    if not dist.is_initialized():
        return False
    dispatch = _deepep_capture_state.get("last_dispatch")
    combine = _deepep_capture_state.get("last_combine")
    if dispatch is None or combine is None:
        return False
    gathered: list[dict[str, Any]] = [dict() for _ in range(dist.get_world_size(group))]
    dist.all_gather_object(
        gathered,
        {
            "global_rank": dispatch.get("global_rank", -1),
            "extend": bool(is_extend_in_batch),
            "dispatch_tokens": int(dispatch.get("dispatch_tokens", 0)),
            "topk": int(dispatch.get("topk", 0)),
            "combine_tokens": int(combine.get("combine_tokens", 0)),
        },
        group=group,
    )
    mixed = any(item.get("extend") for item in gathered) and not all(
        item.get("extend") for item in gathered
    )
    dispatch_total = sum(item.get("dispatch_tokens", 0) * item.get("topk", 0) for item in gathered)
    combine_total = sum(item.get("combine_tokens", 0) for item in gathered)
    mismatch = mixed and dispatch_total != combine_total
    if mismatch:
        _deepep_capture_state["mismatch_pending"] = True
        print(
            f"[NPU-COMM] deepep_mismatch mixed=1 dispatch_tokens_topk={dispatch_total} "
            f"combine_tokens={combine_total}",
            flush=True,
        )
    return mismatch


def _int_from_details(details: str, name: str) -> int:
    import re

    match = re.search(rf"(?:^|,){re.escape(name)}=(-?\d+)(?:,|$)", details)
    return int(match.group(1)) if match else -1


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
        stack = _call_stack() if _stack_logging_enabled() else None
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
        stack = _call_stack() if _stack_logging_enabled() else None
        capture_payload = None
        try:
            if _capture_mode() == "mismatch":
                raw_payload = {
                    "op": log_name,
                    "seq": sequence,
                    "pid": os.getpid(),
                    "global_rank": _int_from_details(details, "global_rank"),
                    "details": details,
                    "stack": stack,
                    "args": _clone_for_capture(args if bound_method else args[1:]),
                    "kwargs": _clone_for_capture(kwargs),
                }
                if ".dispatch" in log_name:
                    _deepep_capture_state["last_lowlevel_dispatch"] = raw_payload
                elif ".combine" in log_name:
                    _deepep_capture_state["last_lowlevel_combine"] = raw_payload
                    if _deepep_capture_state.get("mismatch_pending"):
                        dispatch_payload = _deepep_capture_state.get(
                            "last_lowlevel_dispatch"
                        ) or _deepep_capture_state.get("last_dispatch")
                        if dispatch_payload is not None:
                            directory = _capture_dir()
                            if directory is not None:
                                _flush_mismatch_capture(
                                    dispatch_payload, raw_payload, []
                                )
            capture_payload = _capture_deepep_inputs(
                log_name,
                sequence,
                args,
                kwargs,
                details,
                stack,
                bound_method=bound_method,
            )
            _log(log_name, "before", sequence, details, stack)
            _sync_npu()
            result = original(*args, **kwargs)
            _sync_npu()
            _log(log_name, "after", sequence, details, stack)
            return result
        except BaseException as exc:
            if _capture_mode() == "error" and capture_payload is not None:
                directory = _capture_dir()
                if directory is not None:
                    _save_capture_payload(directory, capture_payload)
                    previous_dispatch = _deepep_capture_state.get("last_dispatch")
                    if previous_dispatch is not None:
                        _save_capture_payload(directory, previous_dispatch)
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
