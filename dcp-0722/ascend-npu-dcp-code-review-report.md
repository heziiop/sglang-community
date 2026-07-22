# Ascend NPU DCP 支持代码检视报告

> 检视范围：`bf67a173383dc92f5e90c7377c3cb5e1f64030c4`（含）至当前分支 `HEAD=aac90554ebde39cad58e4b2aab26cba748820058`  
> 基线提交：`e85ef548772e3fc416779eadfd1509f4d31e5db3`  
> 检视日期：2026-07-22  
> 变更规模：15 个文件，新增 516 行，删除 69 行

## 1. 执行摘要

这组修改的核心目标，是把 SGLang 已有的 Decode Context Parallelism（DCP）能力接到 Ascend NPU 的 DeepSeek MLA 推理路径上。

DCP 面向长上下文 decode 中“每个 attention 计算都要扫描完整历史 KV”的瓶颈。它采用按 token 位置交错分片的方式，让每个 rank 的有效 KV ownership 和 attention 读取范围约为全局上下文的 `1 / dcp_size`：

```text
owner(position) = position % dcp_size
```

对单个请求而言，每个 DCP rank 只负责约 `1 / dcp_size` 的历史 KV token，并只针对本地 KV shard 计算局部 attention。需要注意，本次代码没有把 NPU KV pool 的物理预留容量按 `dcp_size` 缩小；降低的是单请求在各 rank 上的有效写入/读取范围和 decode 扫描量，而不是已分配 KV pool 的显存占用。由于每张卡上的 softmax 只覆盖局部 KV，最终不能直接相加，而是需要使用各 rank 输出的 LSE（log-sum-exp）重新归一化，再做跨 rank 汇总。

本次实现大致打通了以下链路：

1. 允许 Ascend NPU 创建和使用 DCP group；
2. 将 NPU 的 KV allocator 调整成 DCP 对齐的“全局逻辑槽位”布局；
3. 在运行时把全局 KV 写入地址转换成当前 rank 的局部地址；
4. 为 NPU paged attention 生成 rank-local sequence length 和 block table；
5. decode 时 all-gather Q，每个 rank 对本地 KV 做局部 attention；
6. 请求 Ascend FIA kernel 返回 LSE，通过 LSE 校正和 reduce-scatter 合并结果；
7. extend/prefill 命中 prefix cache 时，按页取出各 rank 的本地 prefix KV，all-gather 后恢复完整逻辑顺序；
8. 适配 NPU graph capture/replay 的静态 DCP page table 和动态局部长度；
9. 修正 PD 分离场景中因 DCP 放大 allocator page size 导致的页数计算问题。

这不是一个简单的平台开关，而是一次跨越配置校验、并行组、内存分配、KV 写入、page table、attention kernel、collective、图执行和 PD 传输的端到端改造。

同时需要明确：当前改动主要围绕 **Ascend 上的 DeepSeek MLA/MLA 风格 attention 路径**。虽然平台级校验已经允许所有 NPU 配置启用 DCP，但本提交没有给出通用 NPU MHA、所有模型、所有 attention backend 均已支持的证据，也没有新增 Ascend DCP 测试。因此更准确的能力描述应是：

> 初步实现了 Ascend NPU 上 DeepSeek MLA DCP 的核心执行链路，而不是已经证明 Ascend NPU 上任意模型和任意 attention backend 都支持 DCP。

---

## 2. 提交范围与演进

| 提交 | 日期 | 作用 |
|---|---|---|
| `bf67a17338` | 2026-07-20 | 主功能提交：打通 Ascend NPU DCP 的配置、KV 布局、decode、extend、LSE 合并和 graph 路径。 |
| `e45e4ba05f` | 2026-07-21 | PD 分离修复：发送 KV 页数改用 allocator 的真实 page size。 |
| `e62382c101` | 2026-07-21 | prefix all-gather 重构：从逐 token local index 改为基于 DCP page table 的按页读取，并处理部分页。 |
| `aac90554eb` | 2026-07-21 | `.gitignore` 增加本地开发目录，与运行时功能无直接关系。 |

### 2.1 主功能提交 `bf67a17338`

该提交一次性完成了大部分 NPU DCP 链路：

- 放开 NPU DCP 配置和分布式初始化；
- 扩大 NPU allocator 的逻辑容量和 page size；
- 将全局 `out_cache_loc` 转成 rank-local 写入地址；
- 为 Ascend paged attention 构造 DCP 本地长度与 block table；
- 在 DeepSeek MLA decode 中 all-gather Q、执行局部 attention、返回 LSE、合并输出；
- 为 NPU 实现 PyTorch 版本的 LSE correction；
- 为 extend/prefix cache 引入临时的完整 prefix KV buffer；
- 为 NPU graph capture/replay 增加 DCP metadata。

### 2.2 PD 修复 `e45e4ba05f`

DCP 使 allocator 的 page size 从 `P` 变成 `P * dcp_size`。PD bootstrap 原来使用 KV pool 的物理 page size 计算发送页数，会与 allocator 的页语义不一致。本提交将页数计算改成读取：

```python
self.scheduler.token_to_kv_pool_allocator.page_size
```

因此它虽然只改了一个表达式，但属于主功能之后必要的系统兼容修复。

### 2.3 prefix all-gather 修正 `e62382c101`

首版实现先展开 prefix 的逐 token index，再过滤当前 rank 并除以 `dcp_size`。后续提交将它改成：

1. attention backend 统一计算当前 rank 的 prefix local length；
2. 生成 DCP prefix block table；
3. 直接按物理页从 NPU KV pool 读取；
4. 对每个请求的最后一个部分页进行裁剪；
5. 再执行 all-gather 和逻辑顺序恢复。

该改动更符合 NPU paged KV cache 的实际物理布局，也避免为整个 prefix 预先构造逐 token 索引。

### 2.4 `.gitignore` 提交 `aac90554eb`

新增了 `.trae/`、`dcp-npu/`、`cp-npu/` 和 `eagle3-support-quarot/`。这只是本地开发目录清理，不影响 DCP 运行时。

---

## 3. DCP 的基本原理

### 3.1 Token ownership

假设：

- `N = dcp_size`
- `r = dcp_rank`
- token 的逻辑绝对位置为 `p`

则 rank `r` 只拥有：

```text
p % N == r
```

的 KV。

例如 `dcp_size=4`、序列长度为 10：

```text
position:  0 1 2 3 4 5 6 7 8 9
owner:     0 1 2 3 0 1 2 3 0 1
```

各 rank 的本地 KV 长度为：

```text
local_len(r, L) = L // N + int(r < L % N)
```

因此：

| Rank | 拥有位置 | 本地长度 |
|---:|---|---:|
| 0 | 0, 4, 8 | 3 |
| 1 | 1, 5, 9 | 3 |
| 2 | 2, 6 | 2 |
| 3 | 3, 7 | 2 |

通用的布局数学已经存在于 `python/sglang/srt/layers/dcp/layout.py`；本次 Ascend 实现复用了同样的 ownership 语义。

### 3.2 为什么局部 attention 不能直接求和

令 rank `r` 对本地 KV shard 算出的结果为 `O_r`，局部 LSE 为 `LSE_r`。全局 LSE 为：

```text
LSE_global = logsumexp(LSE_0, LSE_1, ..., LSE_N-1)
```

每个局部输出必须按以下比例校正：

```text
scale_r = exp(LSE_r - LSE_global)
```

最终结果为：

```text
O_global = Σ_r O_r * scale_r
```

所以 DCP decode 至少包含：

1. 局部 attention；
2. LSE all-gather；
3. 局部输出校正；
4. all-reduce 或 reduce-scatter。

本次 NPU MLA 路径采用的是 `all-gather LSE + correction + reduce-scatter`。

---

## 4. 整体执行架构

```text
--dcp-size=N
      │
      ▼
ServerArgs / initialize_model_parallel
  在每个 TP group 内创建 DCP subgroup
      │
      ▼
NPU allocator 使用逻辑 page_size = physical_page_size * N
      │
      ▼
Scheduler / req_to_token 保持全局逻辑 slot
      │
      ▼
ForwardBatch 将本 rank 拥有的 global slot 映射为 local slot
  local_slot = global_slot // N
  非本 rank slot = -1
      │
      ├────────────── Extend / Prefill ──────────────┐
      │                                              │
      │   各 rank 从本地 KV pool 按页取 prefix KV    │
      │              │                               │
      │              ▼                               │
      │        all-gather + 去 padding + 重排         │
      │              │                               │
      │              ▼                               │
      │       完整 prefix KV 临时工作 buffer          │
      │                                              │
      └────────────── Decode ────────────────────────┤
                                                     │
          all-gather Q，head 数扩大 N 倍             │
                     │                               │
                     ▼                               │
          每 rank 对本地 KV shard 做 FIA             │
             返回 partial output + LSE               │
                     │                               │
                     ▼                               │
        all-gather LSE + 校正 + reduce-scatter        │
                     │                               │
                     ▼                               │
             恢复本 rank 原始 query heads            │
                     │                               │
                     ▼                               │
                w_vc / o_proj                         │
```

---

## 5. 配置与并行组接入

### 5.1 平台校验

文件：

- `python/sglang/srt/server_args.py`
- `python/sglang/srt/distributed/parallel_state.py`

修改后，`dcp_size > 1` 允许在以下平台启用：

- AMD HIP/ROCm；
- CUDA；
- Ascend NPU。

NPU 上有一个明确限制：

```text
DCP + 任意 speculative algorithm 不支持
```

即如果在 NPU 上同时设置 `--dcp-size > 1` 和 speculative decoding，参数校验会直接报错，而不是进入执行阶段后失败。

此外仍有通用硬约束：

```text
tp_size % dcp_size == 0
```

### 5.2 DCP group 的构造

DCP group 在每个 TP group 内按连续 rank 分段构造。

例如：

```text
TP group:   [0,1,2,3,4,5,6,7]
DCP size:   4
DCP groups: [0,1,2,3], [4,5,6,7]
```

因此 DCP rank 是 DCP subgroup 内的 rank，不是 world rank。

### 5.3 `dcp_enabled()` 放开 NPU

文件：`python/sglang/srt/layers/dcp/comm.py`

原逻辑只把 CUDA 视为可用平台，本次改成：

```python
is_cuda() or is_npu()
```

这使 DeepSeek 模型、planner 和 collective 路径能够在 NPU 上进入 DCP 分支。

---

## 6. KV allocator 与全局/局部地址映射

这是整套实现中最关键、也最容易出错的部分。

### 6.1 保留 scheduler 的全局槽位语义

文件：`python/sglang/srt/model_executor/forward_batch_info.py`

新增 `_maybe_localize_npu_dcp_out_cache_loc(...)`，其设计原则是：

- `ScheduleBatch`、allocator 和 `req_to_token` 继续保存全局逻辑 slot；
- 只在构造运行时 `ForwardBatch` 时产生当前 rank 的局部写入视图；
- 不回写、也不破坏 scheduler 的全局状态。

转换规则：

```python
is_local = out_cache_loc % dcp_size == dcp_rank
local = out_cache_loc // dcp_size
non_local = -1
```

即：

```text
global slot:  0 1 2 3 4 5 6 7 ...
rank 0 view:  0 - - - 1 - - - ...
rank 1 view:  - 0 - - - 1 - - ...
```

这样 NPU KV pool 仍以连续的本地 token slot 写入，不需要让底层存储为每个 rank 保留全局大小的稀疏地址空间。

### 6.2 为什么 allocator 的 page size 要乘 DCP size

文件：`python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`

NPU `NPUPagedTokenToKVPoolAllocator` 的构造改为：

```python
size = max_total_num_tokens * dcp_size
page_size = page_size * dcp_size
```

这里要区分：

- KV pool 的物理 page size：`P`；
- allocator 的全局逻辑 page size：`P * N`。

一个长度为 `P * N` 的全局逻辑 allocator page，恰好给每个 DCP rank 分配 `P` 个 token。映射后：

```text
local_page
= (global_slot // N) // P
= global_slot // (N * P)
= global_page
```

因此全局逻辑 page id 和每个 rank 的本地物理 page id 保持一致。这是后续直接复用 page id、构造 block table、释放 page 和传输 page 的基础。

### 6.3 分配路径使用真实 allocator page size

文件：`python/sglang/srt/mem_cache/common.py`

`_alloc_page_size(...)` 原来只在 HIP/CUDA DCP 下读取 allocator page size，本次把 NPU 也加进去。

否则 scheduler 按 `P` 分配、allocator 却按 `P * N` 管理，会导致：

- 新页数量计算错误；
- prefix cache 页边界不一致；
- page free/evict 粒度错误；
- PD 发送页数错误。

---

## 7. NPU DCP metadata 与 page table

文件：

- `python/sglang/srt/layers/dcp/planner.py`
- `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`

### 7.1 `plan_dcp_kv_metadata_npu(...)`

该函数为从逻辑位置 0 开始的 KV 范围生成两个结果：

1. 当前 rank 实际拥有的 KV 长度；
2. Ascend paged attention 使用的 block table。

本地长度：

```python
dcp_kv_lens = kv_lens // dcp_world_size + (
    dcp_rank < kv_lens % dcp_world_size
)
```

block table 使用 DCP 放大后的逻辑页步长：

```python
dcp_page_size = page_size * dcp_world_size
block_tables = (
    req_to_token[req_pool_indices, :max_len:dcp_page_size]
    // dcp_page_size
)
```

传给 FIA kernel 时：

- `block_table` 是 DCP block table；
- `actual_seq_lengths_kv` 是 rank-local 长度；
- `block_size` 仍是 KV pool 的物理 page size `P`。

这三者组合起来表达的是：“逻辑请求有完整长度 `L`，但本 rank 只读取自己拥有的 `local_len`，每个 block table entry 指向本地物理 KV page”。

### 7.2 普通 eager decode metadata

`AscendAttnBackend.init_forward_metadata(...)` 在 DCP + decode/idle 时构建：

- `dcp_seq_lens_cpu_int`
- `dcp_block_tables`

原有的全局 `seq_lens` 和普通 `block_tables` 仍保留，供非 DCP 路径或其他运行时逻辑使用。

### 7.3 graph 模式 metadata

`init_cuda_graph_state(...)` 额外预分配静态地址的：

```text
dcp_block_tables[max_bs, max_dcp_seq_pages]
```

其中：

```text
max_dcp_seq_pages = ceil(max_graph_seq_len / (P * N))
```

replay 前 `_apply_cuda_graph_metadata(...)`：

1. 根据实时 request 和 sequence length 重算 DCP block table；
2. `copy_` 到 capture 时固定地址的 tensor；
3. 清零未使用列；
4. 更新 `dcp_seq_lens_cpu_int`。

这样既满足 NPU graph 对地址和 shape 稳定的要求，又能在每次 replay 使用真实的 KV 映射。

### 7.4 GraphRunner 的局部长度更新

文件：`python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py`

NPU graph replay 会通过 `graph.update` 更新 kernel 的动态 sequence length 属性。DCP 下，该属性不能继续使用全局长度，而要转换为当前 rank 的局部长度：

```text
local_len = seq_len // N + int(rank < seq_len % N)
```

这与 attention backend 中的 `dcp_seq_lens_cpu_int` 使用相同公式，分别服务于静态 metadata tensor 和 graph 内部动态属性。

---

## 8. Decode 数据流

### 8.1 Q all-gather

文件：

- `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`
- `python/sglang/srt/layers/dcp/comm.py`

DeepSeek MLA 在每个 TP rank 上先得到本地 query heads：

```text
q_nope_out: [B, H_local, D_kv]
q_pe:       [B, H_local, D_rope]
```

DCP rank 只持有一部分历史 KV，但为了让本地 KV shard 对 DCP subgroup 内所有 query heads 计算局部 attention，需要先对 Q 做 all-gather：

```text
[B, H_local, D]
    ↓ all-gather along head dimension
[B, H_local * N, D]
```

`all_gather_q_for_mla_decode(...)` 会先把 Q 转为 head-major，将 `q_pe` 和 `q_nope_out` 拼接成一次 collective，gather 后再拆开，避免两次独立通信。

### 8.2 DCP 专用 attention 对象

仓库已有的 `attn_mqa_for_dcp_decode` 使用：

```text
num_query_heads = num_local_heads * dcp_world_size
num_kv_heads = 1
```

这与 Q all-gather 后的 head 数匹配。扩大的是 query head 轴，不是复制 KV head。

### 8.3 Ascend FIA 局部 attention

`AscendAttnBackend` 在 DCP decode 时改用：

- rank-local `actual_seq_lengths_kv`；
- DCP `block_tables`；
- `softmax_lse_flag=True`。

核心调用是 Ascend 的：

```python
torch_npu.npu_fused_infer_attention_score(...)
```

每个 rank 得到：

```text
partial_out: [B, H_local * N, D_kv]
partial_lse: [B, H_local * N]
```

代码兼容 eager 和 graph 两条 FIA 调用路径，并把 FIA 返回的 `[B, S, H, 1]` 或相近布局压缩成 DCP merge 所需的 `[B, H]`。

DCP 还会强制走 FIA 分支：

```python
if self.use_fia or dcp_enabled():
```

原因是旧的 `_npu_paged_attention_mla` 路径不能提供合并所需的 LSE，且不支持 graph mode。

### 8.4 LSE 校正与 reduce-scatter

`forward_mla_core_npu(...)` 调用：

```python
cp_lse_ag_out_rs_mla(attn_output, lse, get_dcp_group())
```

流程为：

1. all-gather 所有 rank 的 LSE；
2. 计算全局 `logsumexp`；
3. 按 `exp(local_lse - global_lse)` 校正当前 rank 的 partial output；
4. 在 head 维执行 reduce-scatter；
5. 每个 rank 恢复自己的 `H_local` 个 query heads。

最终 tensor 从全 DCP head 视图恢复为：

```text
[B, H_local, D_kv]
```

再进入：

```text
batch_matmul_transpose(..., w_vc)
→ [B, H_local, D_v]
→ o_proj
```

### 8.5 NPU 的 PyTorch correction fallback

文件：`python/sglang/srt/layers/dcp/kernels.py`

原 `correct_attn_out(...)` 依赖 Triton kernel。NPU 不能直接执行这条 Triton 路径，因此新增：

```python
correct_attn_out_torch(...)
```

实现步骤：

```python
global_lse = torch.logsumexp(lses, dim=0)
scale = torch.exp(local_lse - global_lse)
corrected = out * scale.unsqueeze(-1)
```

并用 `torch.nan_to_num` 处理空 shard 或极端数值造成的 NaN/Inf。

函数把输出转成 `[H, B, D]`，以匹配后续 `reduce_scatter_along_dim(dim=0)` 的布局要求。

该 fallback 解决了正确性和可运行性问题，但相对专用融合 kernel 可能有额外算子调度、临时 tensor 和带宽开销，是后续性能优化点。

---

## 9. Extend / Prefix Cache 数据流

DCP 虽然主要优化 decode，但 extend/prefill 命中 prefix cache 时仍需要完整历史 KV。原因是当前 query 必须按原始时间顺序关注整个 prefix，而 prefix KV 已分散在不同 DCP ranks 上。

### 9.1 NPU 专用 extend metadata

文件：

- `python/sglang/srt/layers/dcp/planner.py`
- `python/sglang/srt/models/deepseek_v2.py`

`prepare_npu_dcp_extend_metadata(...)` 不复用 GPU 版完整 DCP workspace planner，而只分配：

```text
dcp_kv_buffer:
[total_prefix_len, 1, kv_lora_rank + qk_rope_head_dim]
```

即 NPU 专用 buffer 只保存 all-gather 后的完整 prefix KV；本轮 extend 的 K/V 仍沿 Ascend backend 原有参数路径传递。

### 9.2 prefix local length 与 page table

`AscendAttnBackend.init_forward_metadata(...)` 在 MLA + extend + prefix 非空时生成：

- `dcp_local_prefix_lens_cpu_int`
- `dcp_prefix_block_tables`
- `prefix_lens`

这样 prefix gather 不再依赖逐 token local index，而是直接使用 paged KV cache 的物理页布局。

### 9.3 按页取本地 prefix KV

`_gather_dcp_prefix_kv(...)` 的步骤：

1. 根据每个请求的 local prefix length 计算需要多少物理页；
2. 对 DCP prefix block table 构造有效 page mask；
3. 得到 `prefix_page_indices`；
4. 直接从 NPU 的 key/value page buffer `index_select`；
5. 展平 page 和 page 内 token 两个维度；
6. 对每个请求的最后一个部分页裁剪掉无效 token。

关键不变量是：

> 一个放大后的 DCP allocator page，在每个 rank 上恰好对应一个物理 KV page，因此从 global slot 转 local slot 后 page id 不变。

这也是为什么最终实现可以直接使用 DCP block table 中的 page id。

### 9.4 all-gather 后恢复逻辑 token 顺序

本地 prefix KV 被拼成：

```text
[k_nope, k_rope]
```

然后调用已有的 `all_gather_kv_cache_for_dcp(...)`。该函数会：

1. 根据各请求 prefix 长度为不同 rank 的本地数据补齐；
2. 执行 DCP all-gather；
3. 将 rank-major 排列转成交错 token 顺序；
4. 去掉 padding；
5. 为每个请求恢复原始 prefix 顺序。

最终完整 prefix 被写入 `forward_batch.attn_dcp_metadata.dcp_kv_buffer`。

### 9.5 Extend attention 使用完整 prefix buffer

`AscendAttnBackend.forward_extend(...)` 中两处 prefix cache 处理分支都增加了 DCP 逻辑：

- DCP：从 `dcp_kv_buffer` 拆出 `kv_cached` 和 `k_rope_cached`；
- 非 DCP：继续根据普通 `flatten_prefix_block_tables` 从本地 token pool 读取。

随后与本轮 extend 的 K/V 拼接，执行完整 causal attention。

### 9.6 RoPE/KV 写入路径调整

`forward_mha_prepare_npu(...)` 原先可使用融合的 `npu_kv_rmsnorm_rope_cache` 同时做 norm、RoPE 和 KV cache 写入。DCP 下关闭该融合分支，改走普通 norm/RoPE 后调用 `set_kv_buffer(...)`。

原因是 DCP 要使用经过 rank-local 化的写入地址，并跳过非本 rank 的 token；原融合写 cache 路径没有表达这套 ownership 和 local-slot 语义。

---

## 10. NPU MLA KV pool 适配

文件：`python/sglang/srt/hardware_backend/npu/memory_pool_npu.py`

新增 `NPUMLATokenToKVPool.get_mla_kv_buffer(...)`，用于把 NPU 的两块 paged buffer 恢复为通用 MLA 所需的 token-index 视图：

```text
cache_k_nope: [T, 1, kv_lora_rank]
cache_k_rope: [T, 1, qk_rope_head_dim]
```

NPU 与 GPU 的存储差异是：

- GPU 通用实现倾向于把 latent KV 和 RoPE K 组织在单个逻辑 buffer 中；
- NPU 将二者放在独立的 paged key/value buffer 中。

该 override：

- 支持空索引；
- 使用 `index_select` 保持 token-index 语义；
- 保留 layer transfer synchronization 和 raw-byte storage 的 dtype view；
- 必要时转换目标 dtype。

首版 prefix gather 使用了该接口。最终的按页 gather 主路径直接读取 key/value page buffer，但这个 override 仍补齐了 NPU MLA pool 对通用 DCP KV 访问契约的支持。

---

## 11. PD 分离适配

文件：`python/sglang/srt/disaggregation/prefill.py`

PD prefill bootstrap 需要根据待发送的 KV token 数初始化 sender 的页数。修改前使用：

```python
self.token_to_kv_pool.page_size
```

这是 KV pool 的物理 page size `P`。DCP 下，scheduler 和 allocator 管理的是放大后的逻辑 page size `P * N`，所以改成：

```python
self.scheduler.token_to_kv_pool_allocator.page_size
```

这个修改只统一了 **bootstrap 的 `num_pages` 计算依据**：页数按 allocator 的 DCP 逻辑 page size `P * N` 计算。PD 传输配置中的 `kv_args.page_size` 仍来自 `token_to_kv_pool.page_size`，即物理 page size `P`；因此不能将该提交解释为 allocator、tree cache 和 PD sender 的全部 page 语义已经统一。

如果 bootstrap 仍按物理 `P` 计算，会把一个 DCP 逻辑分配页误算成多个 bootstrap page，导致 sender 初始化的页数与 allocator 分配粒度不一致。

本提交只修复了 bootstrap 页数计算，并不能单凭这一点证明 DCP + PD 的所有组合、offload 或 HiCache 场景都已完成端到端验证。

---

## 12. 关键文件与职责

| 文件 | 主要职责 |
|---|---|
| `python/sglang/srt/server_args.py` | 允许 NPU DCP；禁止 NPU DCP 与 speculative decoding 组合。 |
| `python/sglang/srt/distributed/parallel_state.py` | 允许 NPU 创建 DCP subgroup；校验 `tp_size % dcp_size == 0`。 |
| `python/sglang/srt/layers/dcp/comm.py` | 将 NPU 纳入 DCP；复用 Q/KV collectives 和 LSE merge。 |
| `python/sglang/srt/layers/dcp/kernels.py` | 为 NPU 增加 PyTorch LSE correction fallback。 |
| `python/sglang/srt/layers/dcp/planner.py` | 构造 NPU DCP local length、block table 和 extend prefix buffer。 |
| `python/sglang/srt/model_executor/forward_batch_info.py` | 将全局 NPU KV 写入地址变成 rank-local 地址。 |
| `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py` | 将 NPU allocator 的容量和 page size 按 DCP size 放大。 |
| `python/sglang/srt/mem_cache/common.py` | NPU DCP 分配使用 allocator 的真实 page size。 |
| `python/sglang/srt/hardware_backend/npu/memory_pool_npu.py` | 补齐 NPU MLA paged KV 到通用 token-index KV 的读取接口。 |
| `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py` | NPU DCP metadata、prefix page table、FIA local attention、LSE 输出和 graph metadata。 |
| `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` | DeepSeek MLA 的 Q gather、prefix KV gather、局部 attention 和结果 merge。 |
| `python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py` | graph replay 时把全局 sequence length 转成 rank-local length。 |
| `python/sglang/srt/models/deepseek_v2.py` | NPU extend 使用专用 DCP prefix metadata。 |
| `python/sglang/srt/disaggregation/prefill.py` | PD bootstrap 页数按 DCP allocator page size 计算。 |
| `.gitignore` | 忽略本地开发目录，不影响功能。 |

---

## 13. 运行约束与能力边界

### 13.1 明确约束

| 组合 | 状态 | 原因/行为 |
|---|---|---|
| `dcp_size < 1` | 不允许 | 参数校验报错。 |
| `dcp_size == 1` | 原路径 | DCP 关闭。 |
| `tp_size % dcp_size != 0` | 不允许 | 无法在 TP group 内整齐构造 DCP subgroup。 |
| Ascend NPU + DCP | 已放开 | 本提交的主要目标。 |
| Ascend NPU + DCP + speculative | 不允许 | `ServerArgs` 提前报错。 |
| DCP + unified memory pool | 不允许 | unified pool 没有 DCP-aware masked/local write。 |

### 13.2 实际支持范围应保守描述

从本提交的代码分布看，完成 DCP local metadata、FIA LSE 输出和跨 rank 结果合并的是以下路径：

- DeepSeek V2/V3 风格 MLA；
- NPU FIA paged attention；
- MLA prefix cache extend；
- eager decode 和 NPU graph decode。

明确的代码能力缺口包括：

- 普通 NPU MHA 分支仍使用全局 `block_tables` / `seq_lens`，没有 DCP rank-local metadata、LSE 输出和跨 rank attention 合并；因此当前不能把通用 NPU MHA 视为受支持的 DCP 路径；
- 其他 Ascend attention backend 没有在本提交中完成同等的 DCP 接入；
- DSA、SWA、cross-attention、mixed chunk 等组合没有获得完整实现或验证证据；
- DCP 与 PD offload、HiCache 等扩展场景是否可用仍未确认；
- 所有 NPU graph 模式和 batch padding 边界均缺少实机回归证据。

因此建议在用户文档或发布说明中使用“支持 DeepSeek MLA 的 Ascend DCP 初版实现”，不要直接写成“Ascend 所有模型支持 DCP”。

---

## 14. 风险与代码检视发现

### 14.1 缺少 Ascend DCP 专项测试

这 4 个提交没有新增或修改任何测试文件。

仓库已有的 DCP 测试主要覆盖：

- CPU 上的 ownership/layout 数学；
- CUDA/H200 上的 DeepSeek MLA DCP；
- CUDA/NCCL reduce-scatter；
- AMD/ROCm Triton DCP。

没有发现可作为本次改动证据的 Ascend NPU DCP 测试或 CI 配置。尤其缺少：

- NPU 多卡 DCP group 启动；
- DCP 与非 DCP 的生成结果/logprob 对比；
- NPU Q all-gather、LSE merge 数值对照；
- prefix cache 命中及部分页；
- 混合长度 batch；
- eager 与 graph replay parity；
- PD prefill/decode 分离；
- `dcp_size=2/4/8`；
- 空 shard、短序列和非整除长度。

### 14.2 平台开关宽于后端实现范围

`ServerArgs` 和 `parallel_state` 是平台级放行，只要是 NPU 就能启用 DCP；但实际 attention 改造主要在 DeepSeek MLA 路径。

如果普通 NPU MHA 或其他未接入的模型启用 `--dcp-size > 1`，`ForwardBatch` 仍会全局应用 rank-local `out_cache_loc` 转换；但普通 MHA attention 分支继续使用全局 `block_tables` / `seq_lens`，也不请求 partial LSE 或执行跨 rank output merge。该组合属于当前代码中的明确未实现路径，可能产生错误结果或运行失败。配置层应按模型/backend capability 拒绝，而不是只按平台放行。

### 14.3 `-1` 非本地写入哨兵依赖 NPU scatter 语义

非 owner token 的局部地址被设置为 `-1`，随后进入 NPU KV pool 的 scatter 更新路径。正确性依赖底层 NPU scatter/cache operator 对 `-1` 索引的约定能够安全跳过，而不是写入最后一个元素或报错。

报告范围内没有看到针对这一契约的单元测试或注释引用，建议用 NPU 小规模测试明确验证。

### 14.4 Prefix 部分页处理复杂

按页读取后，需要按请求分别裁剪最后一个不满页的部分。该逻辑涉及：

- batch 内不同 prefix 长度；
- 每个请求不同页数；
- 展平后的 page offset；
- local prefix length；
- all-gather 前后的 padding。

如果 page offset 或 local length 有一个不一致，就会在请求边界串入其他请求的 KV。建议为多请求、多个部分页专门写可检查 tensor 内容的测试，而不只做最终生成准确率测试。

### 14.5 Graph 路径存在两套局部长度更新

DCP local length 同时用于：

- `AscendAttnBackend` 的 `dcp_seq_lens_cpu_int`；
- `NPUGraphRunner` 的 `graph.update` 属性。

目前两处公式一致。后续如果引入非零 KV 起点、滑窗或 speculative 语义，必须同步修改，最好收敛到同一个 helper，避免 graph 与 eager 产生隐蔽差异。

### 14.6 PyTorch LSE fallback 的性能成本

`correct_attn_out_torch(...)` 会执行：

- `logsumexp`；
- `exp`；
- `nan_to_num`；
- 乘法；
- transpose/copy。

正确性上合理，但在每层、每 decode step 都会发生。它可能成为 DCP 扩展后的新瓶颈，应在 NPU profiler 中检查 collective 与 correction 的占比，再决定是否实现融合 NPU kernel。

### 14.7 每层 Q all-gather 的通信成本

当前 MLA decode 先 all-gather Q，再做局部 attention。随着 DCP size 增大：

- 每个 rank 的 KV 计算下降；
- Q all-gather、LSE all-gather 和 reduce-scatter 成本上升。

DCP 并非 size 越大越快。实际收益取决于 context length、batch size、模型 head 维度、HCCL 拓扑和 kernel 吞吐，需要基准测试确定拐点。

### 14.8 PD 只修复了页数权威来源

PD 提交保证 bootstrap 使用 allocator page size，但没有新增 DCP + PD 的端到端测试。仍需验证：

- prefill 与 decode 两侧 page id 是否使用相同 global/local 语义；
- sender/receiver 是否都理解放大后的逻辑 page；
- prefix cache、offload、HiCache 和失败重试是否保持一致。

---

## 15. 建议的验证计划

### 15.1 纯数学与内存布局单测

1. `local_len`：覆盖 `N=2/4/8`、长度 `0..2N+1`；
2. global slot → local slot：验证 owner mask 和 `// N`；
3. page id 不变量：

   ```text
   (global_slot // N) // P == global_slot // (N * P)
   ```

   对 owner slot 进行随机测试；
4. prefix page table：多请求、不同长度、多个部分页；
5. `_gather_dcp_prefix_kv`：使用可识别 token 值验证 gather 后严格恢复原序列。

### 15.2 NPU 多卡数值测试

建议至少覆盖：

| 项目 | 配置 |
|---|---|
| Baseline parity | TP=N, DCP=1 对比 TP=N, DCP=N |
| DCP size | 2、4、8 |
| Context length | 短、中、长，包含非 DCP size 整除长度 |
| Batch | 1 和混合长度多请求 batch |
| Prefix cache | 无命中、整页命中、部分页命中、多请求部分页 |
| Execution | eager decode、NPU graph decode |
| Accuracy | greedy token 完全一致；逐 token logprob 在明确容差内 |

### 15.3 通信与性能测试

分别统计：

- Q all-gather；
- FIA local attention；
- LSE all-gather；
- correction；
- reduce-scatter；
- prefix KV all-gather。

建议报告：

- TTFT；
- inter-token latency；
- tokens/s；
- 单 rank KV cache 占用；
- 不同 context length 下 DCP=1/2/4/8 的收益曲线。

### 15.4 PD 集成测试

至少验证：

1. P 节点启用 DCP allocator 后 bootstrap 页数正确；
2. D 节点收到的 KV page 与请求位置一致；
3. prefix cache hit 后继续 decode 的结果与非 PD baseline 一致；
4. 多请求、部分页、retract/retry 场景无 page 泄漏或错配。

---

## 16. 总结

这组提交完成的不是一个孤立的 attention kernel，而是一套 Ascend NPU DCP 的系统级初版实现。

其核心设计可以概括为：

```text
全局逻辑 slot 和请求映射保持不变
        ↓
allocator page 按 dcp_size 放大
        ↓
每个 rank 将 owner token 压缩到本地连续 KV slot
        ↓
decode 使用 rank-local length + DCP block table 只读本地 KV
        ↓
Q all-gather 后计算局部 attention
        ↓
利用 LSE 校正并 reduce-scatter，恢复全局等价输出
        ↓
extend 时临时 all-gather prefix KV，恢复完整逻辑上下文
```

实现中最有价值的几个点是：

1. scheduler 保留全局状态、ForwardBatch 提供局部视图，避免污染调度语义；
2. 通过 `allocator_page_size = physical_page_size * dcp_size` 保证 global/local page id 一致；
3. NPU attention 直接使用 rank-local length 和 page table，不需要重建完整 decode KV；
4. 复用通用 DCP 的 Q/KV collective 和 LSE merge 数学，只为 NPU 替换 Triton correction；
5. prefix gather 从逐 token 索引演进为按页读取，更贴合 NPU paged cache；
6. eager、graph 和 PD 页语义均有相应接入。

当前最大不足是没有随提交提供 Ascend 多卡 DCP 测试，也缺少严格的模型/backend capability 限制。下一步应优先补齐正确性矩阵、部分页测试、graph parity 和 PD 集成测试，在此基础上再评估 Q gather 与 PyTorch LSE correction 的性能优化。

---

## 17. 检视与验证说明

本报告基于以下方式完成：

- 逐提交检查 `bf67a17338^..aac90554eb`；
- 检查累计 diff、逐文件改动和最终代码；
- 对 DCP 通用层、NPU attention、KV pool、graph runner 和 PD 路径进行交叉追踪；
- 执行 `git diff --check`，未发现该范围内的 whitespace error。

本次仅进行代码静态检视，没有 Ascend NPU 多卡环境，因此未执行真实 DCP 推理、HCCL collective、NPU graph replay 或性能测试。报告中“已实现”指代码链路已经接入，不等同于已通过实机回归验证。
