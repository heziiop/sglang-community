# 适配 Profiling 代码到 PD 分离场景

## 背景

Commit `e41db4ed8e5349dccbac99ebc69b0e9ce25c12db` 在 PD 混部（`event_loop_overlap`）的调度函数中添加了 NPU profiling 采集代码。现在需要将同样的 profiling 逻辑适配到 PD 分离场景下的 prefill 和 decode 调度函数中。

## 原 Commit 的 Profiling 逻辑

原 commit 在 `scheduler.py` 的 `event_loop_overlap` 方法中添加了以下逻辑：

1. **初始化阶段**（while 循环前）：读取环境变量，创建 `torch_npu.profiler.profile` 对象

   * `ENABLE_PROFILING`: 是否启用（仅 tp\_rank==0）

   * `PROFILING_BS`: 触发采集的最小 batch size

   * `PROFILING_STAGE`: 采集哪个阶段（"decode" 或 "prefill"）

   * `PROFILING_step`: 采集步数

2. **采集控制**（batch 存在时）：

   * 判断当前 batch 是否为目标 stage（decode/prefill）

   * 当 batch size >= prof\_bs 且是目标 stage 时，启动 profiler

   * 计数到 prof\_step 时，同步并停止 profiler

   * 在 run\_batch 之后调用 `prof.step()`

## 需要修改的文件和函数

### 1. Prefill 侧 — `python/sglang/srt/disaggregation/prefill.py`

需要在 `SchedulerDisaggregationPrefillMixin` 的两个 event loop 中添加 profiling：

#### a) `event_loop_normal_disagg_prefill` (行 409-437)

* 在 `while True` 循环前添加 profiling 初始化代码

* 在 `if batch:` 分支中，`run_batch` 前后添加 profiling 控制逻辑

* Prefill 侧固定 `is_prof_stage = True`（因为 prefill worker 只做 prefill），简化判断

#### b) `event_loop_overlap_disagg_prefill` (行 440-480)

* 在 `while True` 循环前添加 profiling 初始化代码

* 在 `if batch:` 分支中，`run_batch` 前后添加 profiling 控制逻辑

* 同样，prefill 侧固定 `is_prof_stage = True`

### 2. Decode 侧 — `python/sglang/srt/disaggregation/decode.py`

需要在 `SchedulerDisaggregationDecodeMixin` 的两个 event loop 中添加 profiling：

#### a) `event_loop_normal_disagg_decode` (行 1351-1374)

* 在 `while True` 循环前添加 profiling 初始化代码

* 在 `if batch:` 分支中，`run_batch` 前后添加 profiling 控制逻辑

* Decode 侧固定 `is_prof_stage = True`（因为 decode worker 只做 decode），简化判断

#### b) `event_loop_overlap_disagg_decode` (行 1377-1411)

* 在 `while True` 循环前添加 profiling 初始化代码

* 在 `if batch:` 分支中，`run_batch` 前后添加 profiling 控制逻辑

* 同样，decode 侧固定 `is_prof_stage = True`

### ~~3. PP 模式 — 暂不适配~~

PP 模式（`scheduler_pp_mixin.py`）暂不适配，后续有需求再补充。

## 实现细节

### Profiling 初始化代码模板

```python
import os
enable_profiling: bool = os.getenv("ENABLE_PROFILING", "0") == "1" and self.tp_rank == 0
prof_bs: int = int(os.getenv("PROFILING_BS", 8))
prof_step: int = int(os.getenv("PROFILING_step", 10))
if enable_profiling:
    prof_cnt = 0
    import torch_npu
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
        l2_cache=False,
        data_simplification=False,
    )
    profiling_path = "profiling/"
    prof = torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
            profiling_path
        ),
        schedule=torch_npu.profiler.schedule(wait=1, warmup=1, active=10, repeat=1, skip_first=1),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        with_flops=False,
        with_modules=False,
        experimental_config=experimental_config,
    )
```

### Profiling 控制代码模板（normal 模式）

**在** **`if batch:`** **内、`run_batch`** **前：**

```python
if enable_profiling:
    if len(batch.reqs) >= prof_bs and prof_cnt == 0:
        prof.start()
        prof_cnt += 1
    if prof_cnt > 0:
        prof_cnt += 1
    if prof_cnt == prof_step:
        torch.npu.synchronize()
        prof.stop()
```

**在** **`run_batch`** **后：**

```python
if enable_profiling and prof_cnt > 0 and prof_cnt < prof_step:
    prof.step()
```

### 与原 commit 的差异说明

1. **去掉** **`PROFILING_STAGE`** **环境变量和** **`is_prof_stage`** **判断**：在 PD 分离模式下，prefill worker 只做 prefill，decode worker 只做 decode，不需要通过 stage 判断来过滤 batch 类型。

## 修改步骤

1. 修改 `prefill.py` 的 `event_loop_normal_disagg_prefill`
2. 修改 `prefill.py` 的 `event_loop_overlap_disagg_prefill`
3. 修改 `decode.py` 的 `event_loop_normal_disagg_decode`
4. 修改 `decode.py` 的 `event_loop_overlap_disagg_decode`
