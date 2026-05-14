import inspect
import os

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn_f

_ORIGINAL_FUNCTIONS = {}
_PRINTED_CALL_SITES = set()


def patch_all_all_gathers():
    """开始拦截所有 all_gather 变体"""
    global _ORIGINAL_FUNCTIONS

    # 避免重复 patch 导致死循环
    if _ORIGINAL_FUNCTIONS:
        print("[Warning] Already patched. Skip.")
        return

    def log_meta(api_name, input_tensor, group):
        global _PRINTED_CALL_SITES

        device_id = input_tensor.device.index if input_tensor.is_cuda else "CPU"
        current_group = group if group is not None else dist.group.WORLD
        world_size = dist.get_world_size(current_group)

        this_file = os.path.abspath(__file__)
        stack = inspect.stack()
        caller_frames = []
        for frame_info in stack:
            if os.path.abspath(frame_info.filename) != this_file:
                caller_frames.append(frame_info)
                if len(caller_frames) >= 4:
                    break

        call_site_key = tuple((f.filename, f.lineno) for f in caller_frames)

        if call_site_key and call_site_key in _PRINTED_CALL_SITES:
            return

        if call_site_key:
            _PRINTED_CALL_SITES.add(call_site_key)

        msg = (
            f"[Capture Log] {api_name} -> Device: {device_id}, World Size: {world_size}"
        )
        for i, frame_info in enumerate(caller_frames):
            caller_file = frame_info.filename
            caller_line = frame_info.lineno
            caller_code = (
                frame_info.code_context[0].strip() if frame_info.code_context else "N/A"
            )
            msg += f"\n  L{i}: {caller_file}:{caller_line} -> {caller_code}"

        print(msg)

    # 1. 拦截 dist.all_gather
    _ORIGINAL_FUNCTIONS["dist.all_gather"] = dist.all_gather

    def logged_all_gather(tensor_list, tensor, group=None, async_op=False):
        log_meta("dist.all_gather", tensor, group)
        return _ORIGINAL_FUNCTIONS["dist.all_gather"](
            tensor_list, tensor, group=group, async_op=async_op
        )

    dist.all_gather = logged_all_gather

    # 2. 拦截 dist.all_gather_into_tensor / _all_gather_base
    for attr in ["all_gather_into_tensor", "_all_gather_base"]:
        if hasattr(dist, attr):
            orig_func = getattr(dist, attr)
            _ORIGINAL_FUNCTIONS[f"dist.{attr}"] = orig_func

            # 使用闭包绑定正确的函数名
            def make_logged_func(name):
                return lambda output_tensor, input_tensor, group=None, async_op=False: (
                    log_meta(f"dist.{name}", input_tensor, group),
                    _ORIGINAL_FUNCTIONS[f"dist.{name}"](
                        output_tensor, input_tensor, group=group, async_op=async_op
                    ),
                )[
                    1
                ]  # 返回原本函数的返回值

            setattr(dist, attr, make_logged_func(attr))

    # 3. 拦截 torch.distributed.nn.functional.all_gather
    if hasattr(dist_nn_f, "all_gather"):
        _ORIGINAL_FUNCTIONS["dist_nn_f.all_gather"] = dist_nn_f.all_gather

        def logged_nn_all_gather(tensor, group=None):
            log_meta("dist_nn_f.all_gather", tensor, group)
            return _ORIGINAL_FUNCTIONS["dist_nn_f.all_gather"](tensor, group=group)

        dist_nn_f.all_gather = logged_nn_all_gather

    # 4. 拦截 torch.distributed.nn.functional.all_gather_into_tensor
    if hasattr(dist_nn_f, "all_gather_into_tensor"):
        _ORIGINAL_FUNCTIONS["dist_nn_f.all_gather_into_tensor"] = (
            dist_nn_f.all_gather_into_tensor
        )

        def logged_nn_all_gather_into(tensor, group=None):
            log_meta("dist_nn_f.all_gather_into_tensor", tensor, group)
            return _ORIGINAL_FUNCTIONS["dist_nn_f.all_gather_into_tensor"](
                tensor, group=group
            )

        dist_nn_f.all_gather_into_tensor = logged_nn_all_gather_into

    print("[Patch Success] All all_gather variants are now being monitored.")


def unpatch_all_all_gathers():
    """还原所有被拦截的 all_gather 变体至原本状态"""
    global _ORIGINAL_FUNCTIONS
    global _PRINTED_CALL_SITES

    if not _ORIGINAL_FUNCTIONS:
        print("[Warning] No patches found to restore.")
        return

    # 1. 还原 dist.all_gather
    if "dist.all_gather" in _ORIGINAL_FUNCTIONS:
        dist.all_gather = _ORIGINAL_FUNCTIONS["dist.all_gather"]

    # 2. 还原 dist.all_gather_into_tensor / _all_gather_base
    for attr in ["all_gather_into_tensor", "_all_gather_base"]:
        key = f"dist.{attr}"
        if key in _ORIGINAL_FUNCTIONS:
            setattr(dist, attr, _ORIGINAL_FUNCTIONS[key])

    # 3. 还原 torch.distributed.nn.functional 变体
    if "dist_nn_f.all_gather" in _ORIGINAL_FUNCTIONS:
        dist_nn_f.all_gather = _ORIGINAL_FUNCTIONS["dist_nn_f.all_gather"]
    if "dist_nn_f.all_gather_into_tensor" in _ORIGINAL_FUNCTIONS:
        dist_nn_f.all_gather_into_tensor = _ORIGINAL_FUNCTIONS[
            "dist_nn_f.all_gather_into_tensor"
        ]

    _ORIGINAL_FUNCTIONS.clear()
    _PRINTED_CALL_SITES.clear()
    print("[Unpatch Success] Restored all original all_gather functions.")


def check_size_is_same(size_shape):
    from sglang.srt.distributed.parallel_state import get_tp_group

    size_shape = int(size_shape)
    rank = torch.distributed.get_rank(get_tp_group().cpu_group)
    cpu_group = get_tp_group().cpu_group
    world_size = dist.get_world_size(cpu_group)
    assert world_size == 8
    tensor = torch.tensor(size_shape, device="cpu")
    torch.distributed.all_reduce(tensor, group=cpu_group)
    out_value = tensor.item()
    if out_value != size_shape * world_size:
        print(f"Rank {rank} has size_shape {size_shape}")
        assert False
