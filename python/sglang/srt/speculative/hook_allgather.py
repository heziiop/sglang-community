import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn_f

# 全局字典，用于存储原始函数的引用，方便还原
_ORIGINAL_FUNCTIONS = {}


def patch_all_all_gathers():
    """开始拦截所有 all_gather 变体"""
    global _ORIGINAL_FUNCTIONS

    # 避免重复 patch 导致死循环
    if _ORIGINAL_FUNCTIONS:
        print("[Warning] Already patched. Skip.")
        return

    def log_meta(api_name, input_tensor, group):
        device_id = input_tensor.device.index if input_tensor.is_cuda else "CPU"
        current_group = group if group is not None else dist.group.WORLD
        world_size = dist.get_world_size(current_group)
        print(
            f"[Capture Log] {api_name} -> Device: {device_id}, World Size: {world_size}"
        )

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

    # 清空全局字典，释放引用
    _ORIGINAL_FUNCTIONS.clear()
    print("[Unpatch Success] Restored all original all_gather functions.")
