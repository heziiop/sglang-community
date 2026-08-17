# DCP for DSA 实现解析

本文解析提交 `9095526083`（`dcp for dsa`）如何在 Ascend NPU 上把 Decode Context Parallelism（DCP）接入 DeepSeek Sparse Attention（DSA）的前向推理流程。

> 阅读范围：以提交 `9095526083f985f9c8f4ffa396af00f572108a63` 为准。提交修改 8 个文件，核心新增文件是 `python/sglang/srt/hardware_backend/npu/attention/dsa_dcp.py`。

## 1. 要解决的问题

### 1.1 DCP 的 KV 布局

DCP 将同一请求的历史 token 按位置轮转分给多个 rank。设 DCP world size 为 `N`，全局 token 位置为 `p`：

```text
owner_rank(p) = p % N
local_index(p) = p // N
```

例如 `N = 3`：

| 全局位置 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| owner rank | 0 | 1 | 2 | 0 | 1 | 2 | 0 | 1 |
| rank 内下标 | 0 | 0 | 0 | 1 | 1 | 1 | 2 | 2 |

因此，每个 rank 只保存约 `1/N` 的 MLA KV cache，并只对自己的局部 KV 做注意力。各 rank 最后根据局部 softmax 的 log-sum-exp（LSE）合并结果，恢复等价于完整 KV 上的注意力输出。

### 1.2 DSA 给 DCP 带来的额外矛盾

DSA 不是在全部历史 token 上计算注意力。它先通过 indexer 为每个 query 选出全局 top-k token，再由 sparse attention 读取这些 token 的 KV。

这产生两个布局冲突：

1. **选点必须是全局的。** 如果 indexer cache 也随 DCP 分片，每个 rank 只能在局部历史里选 top-k，结果不再等价于全局 DSA。
2. **稀疏注意力必须是局部的。** indexer 输出的是全局 token 下标，但每个 rank 的 MLA KV cache 是紧凑的局部布局，NPU sparse attention kernel 不能直接使用全局下标。

本提交采用的核心方案是：

```text
DSA indexer cache：每个 DCP rank 保留完整副本，用于产生相同的全局 top-k
MLA KV cache：继续按 DCP 分片，只计算本 rank 拥有的 top-k 子集
局部 attention 输出：携带 LSE 跨 rank 合并，恢复全局 sparse attention
```

也就是说，它有意让两类 cache 使用不同布局：**indexer cache 复制，真正占大头的 MLA KV cache 分片**。

## 2. 改动总览

| 层次 | 文件 | 本提交的作用 |
| --- | --- | --- |
| 内存预算 | `model_executor/pool_configurator.py` | 把 NPU DSA 的复制式 BF16 indexer cache 计入每 token 内存成本 |
| cache 分配 | `mem_cache/kv_cache_configurator.py` | 为 indexer cache 传入 DCP 放大后的独立容量 `index_size` |
| NPU cache | `hardware_backend/npu/memory_pool_npu.py` | 允许 MLA KV 与 indexer KV 使用不同容量 |
| indexer 写入 | `layers/attention/dsa/dsa_npu_indexer.py` | 按全局 allocator slot 写复制式 indexer cache |
| 下标映射 | `layers/dcp/layout.py` | 将全局 top-k 过滤、映射并稳定压紧为 rank-local 下标 |
| 前向介入点 | `hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` | DSA prepare 阶段 all-gather Q；core 阶段切到 DCP attention 并合并局部结果 |
| 后端元数据 | `hardware_backend/npu/attention/ascend_backend.py` | 为普通、推测和图模式准备 rank-local 长度与 block table；分派 DSA+DCP sparse attention |
| 核心实现 | `hardware_backend/npu/attention/dsa_dcp.py` | indexer 写址、top-k 本地化、局部 sparse attention、LSE 兼容计算 |

## 3. 何时走 DSA+DCP 路径

入口判定是 `deepseek_v2_attention_mla_npu.py::_use_dsa_dcp_partial_attention`：

```python
def _use_dsa_dcp_partial_attention(forward_batch):
    return get_parallel().dcp_enabled and not dsa_use_prefill_cp(forward_batch)
```

`dsa_use_prefill_cp()` 只有同时满足以下条件才为真：

- 当前 batch 有 `attn_cp_metadata`；
- 开启 DSA prefill CP；
- forward mode 是 `context_parallel_extend`。

因此模式选择如下：

| 前向场景 | DCP 已开启 | DSA prefill CP 生效 | 所走路径 |
| --- | --- | --- | --- |
| decode | 是 | 否 | 本提交的 DSA+DCP 局部注意力 |
| target verify | 是 | 否 | 本提交的 DSA+DCP 局部注意力 |
| draft extend v2 | 是 | 否 | 本提交的 DSA+DCP 局部注意力 |
| 普通 extend | 是 | 否 | 本提交的 DSA+DCP 局部注意力 |
| context-parallel prefill | 是 | 是 | 既有 DSA prefill CP 路径，不走本提交的局部合并 |
| 任意模式 | 否 | 任意 | 原始单 rank DSA 路径 |

这个判定比既有 MLA DCP 的 `_use_dcp_mla_partial_attention()` 更宽。后者显式列举 decode、target verify 和 draft extend v2；DSA 版本采用“只排除 prefill CP”的写法，所以普通 extend 也需要 DCP 元数据。本提交相应地把 `AscendAttnBackend.init_forward_metadata()` 中原来只服务 decode 的 `elif is_decode_or_idle()` 改成了 `else`。

## 4. 初始化阶段：先建立两套 cache 容量

这一阶段发生在真正执行前向之前，但它决定了后续写址是否合法。

### 4.1 原理：分片 MLA KV，复制 indexer KV

DCP 下，MLA KV cache 的物理容量可以按 world size 缩小，因为每个 rank 只保存自己拥有的 token。indexer 则必须读取整个序列才能做全局 top-k，所以每个 rank 的 indexer cache 必须覆盖 allocator 的完整全局地址空间。

如果仍让两个 cache 共用相同的 `size`，写全局 indexer slot 时会越界；如果简单地把整个 KV pool 都放大，又会失去 DCP 节省 MLA KV 显存的意义。因此需要为 indexer 单独增加 `index_size`。

### 4.2 代码：内存预算考虑复制开销

`pool_configurator.py::DefaultPoolConfigurator` 在 DSA 模型分支中区分 NPU 与其他设备：

```python
if _is_npu:
    indexer_size_per_token = index_head_dim * dcp_size
    element_size = kv_size
```

含义是：

- NPU indexer cache 使用与 KV cache 相同的存储 dtype；当前实现通常按 BF16 的 `kv_size` 计费；
- 相对于每 rank 分片后的 MLA token 容量，indexer 要保存 `dcp_size` 倍的全局 token，因此每个局部 KV token 对应 `index_head_dim * dcp_size` 个 indexer 元素。

这一步很重要：容量放大不仅要能分配，还必须提前反映在 `_cell_size` 中，否则 `max_total_num_tokens` 会按偏小的单 token 成本估算，最终可能 OOM。

### 4.3 代码：为 indexer 传独立容量

`kv_cache_configurator.py` 创建 `NPUMLATokenToKVPool` 时新增：

```python
index_size = (
    max_total_num_tokens * attn_dcp_size // loc_space_scale
    if is_dsa_model
    else None
)
```

随后 `memory_pool_npu.py::NPUMLATokenToKVPool`：

```python
self.index_size = size if index_size is None else index_size
...
self.index_k_buffer = torch.zeros(
    (
        layer_num,
        self.index_size // self.page_size + 1,
        self.page_size,
        1,
        self.index_head_dim,
    ),
    ...,
)
```

主 `k_buffer` / `v_buffer` 仍按 `self.size` 分配，只有 `index_k_buffer` 使用放大后的 `self.index_size`。末尾多出的 page/slot 仍用于图模式 padding。

## 5. 每个 batch 开始：构造 rank-local paged-KV 元数据

前向进入模型层之前，`AscendAttnBackend.init_forward_metadata()` 会把 scheduler 提供的全局请求状态转换为 attention kernel 可用的元数据。

### 5.1 原理：局部长度和局部 block table 必须配套

全局长度为 `L` 时，rank `r` 拥有的 token 数是：

```text
local_len(L, N, r) = L // N + int(r < L % N)
```

这由 `layout.py::get_dcp_lens()` 实现。

DCP 的一个逻辑 page 跨过全局地址空间中的 `page_size * N` 个位置，因此 block table 不能沿用普通 paged attention 的 `page_size` 步长。`AscendAttnBackend._get_kv_lens_and_block_tables()` 使用：

```python
page_stride = self.page_size * dcp_size
block_tables = (
    self.req_to_token[req_pool_indices, :max_len:page_stride] // page_stride
)
```

得到的 `dcp_block_tables` 指向当前 rank 的紧凑本地 KV page；它必须与 `dcp_seq_lens` 一起传给 sparse attention。

### 5.2 代码：按 forward mode 准备元数据

非图模式中：

- `target_verify` / `draft_extend_v2` 调用 `_get_kv_lens_and_block_tables(..., is_spec=True)`，生成 `dcp_spec_seq_lens_cpu_int` 和 `dcp_spec_block_tables`；
- 其他模式生成 `dcp_seq_lens_cpu_int` 和 `dcp_block_tables`。

推测链不能只用一个长度。`layout.py::get_dcp_chain_spec_lens()` 会为一个请求的每个 draft query 计算逐步增长的可见 KV 前沿，再逐项转为 rank-local 长度。例如最终全局长度为 `L`、每请求有 `T` 个 query 时，可见前沿为：

```text
L - T + 1, L - T + 2, ..., L
```

block table 按每个 speculative query 重复，保持 request-major 排列。

### 5.3 图模式：预分配并原地更新

图捕获不能在 replay 时临时创建形状变化的张量。本提交在 `graph_metadata` 中新增设备侧固定 buffer：

```text
dcp_seq_lens
dcp_spec_seq_lens
dcp_block_tables
dcp_spec_block_tables
```

`_init_cuda_graph_metadata()` 把对应切片挂到 `ForwardMetadata`；`_apply_cuda_graph_metadata()` 每次 replay 重新计算 CPU 侧规划结果，然后 `copy_()` 到已捕获的设备 buffer，并清零未使用的 page。

这也是 `dsa_dcp.py::_get_local_kv_lens()` 优先读取 `dcp_seq_lens` / `dcp_spec_seq_lens` 的原因：图内直接使用设备 tensor，避免 host transfer 和新分配。非图模式则允许把 `*_cpu_int` 转到 NPU；如果没有 CPU 长度，还可直接从 `forward_metadata.seq_lens` 计算。

## 6. 进入模型层：DSA 前向的实际介入点

`deepseek_v2.py::DeepseekV2AttentionMLA.forward_prepare()` 根据 `AttnForwardMethod.DSA_NPU` 调用 `forward_dsa_prepare_npu()`，随后 `forward_core()` 调用 `forward_dsa_core_npu()`。本提交就在这两个既有阶段中插入 DCP 操作。

下面按一次 DSA 层前向的真实顺序展开。

## 7. 步骤一：生成 Q/K，并让 indexer 写完整全局 cache

### 原理

DSA indexer 必须基于完整历史做全局 top-k。DCP 的主 KV allocator 使用分片后的 `out_cache_loc`，它不适合直接作为复制式 indexer cache 的写地址。正确地址应来自请求到全局 token slot 的映射：

```text
request row + token position -> req_to_token[request, position]
```

只要各 DCP rank 对同一 token 使用这个全局 slot 写入，各 rank 上的 `index_k_buffer` 就会形成地址一致的完整副本。

### 代码

`dsa_npu_indexer.py` 原来使用：

```python
set_index_k_buffer(layer_id, forward_batch.out_cache_loc, k)
```

现在改为：

```python
set_index_k_buffer(
    layer_id,
    get_replicated_indexer_cache_loc(forward_batch, positions),
    k,
)
```

`dsa_dcp.py::get_replicated_indexer_cache_loc()` 的执行过程是：

1. 未开启 DCP 或 world size 为 1 时直接返回原 `out_cache_loc`，保持旧行为。
2. 普通 extend 根据每个请求的 `extend_seq_lens` 展开 `req_pool_indices`。
3. decode / speculative 等固定每请求 token 数的场景，根据 `positions.shape[0] // batch_size` 展开请求行。
4. 去掉图 padding，只处理 `num_token_non_padded_cpu` 个有效 token。
5. 使用 `req_to_token[request_rows, positions]` 取得 allocator 的全局 slot。
6. 若图 batch 有 padding，把无效位置补为 0；NPU pool 明确保留 slot 0 给 dummy write。

随后 indexer 从完整 `past_key_states = get_index_k_buffer(layer_id)` 计算 top-k，因此每个 rank 得到相同语义的**全局 token 下标**。

## 8. 步骤二：all-gather Q，让每个 rank 为所有 query head 计算局部贡献

### 原理

KV 按 token 维分散在 DCP ranks 上，而 Q 原本按 rank/head 布局。为了让每个 KV rank 都能对全部需要的 query head 计算其局部 attention 贡献，需要先在 DCP group 内收集 Q。

只收集 Q，不收集巨大的历史 KV，是 DCP 的关键收益。通信量随当前 query batch 变化，而不是随上下文长度增长。

### 代码

`forward_dsa_prepare_npu()` 在 indexer 产生 `topk_indices` 后新增：

```python
if _use_dsa_dcp_partial_attention(forward_batch):
    q_nope_out, q_pe = all_gather_q_for_mla_decode(q_nope_out, q_pe)
```

既有的 `dcp/comm.py::all_gather_q_for_mla_decode()`：

1. 将 `q_pe` 和 `q_nope_out` 从 `[T, H_local, D]` 转成 head-first；
2. 在最后一维拼接 RoPE 与 NoPE 部分，只做一次 collective；
3. 沿 head 维 all-gather，得到 `[T, H_local * N, D]`；
4. 再拆回 `q_pe` / `q_nope_out`。

模型初始化时已有 `attn_mqa_for_dcp_decode`，其 head 数正是：

```python
num_local_heads * attn_dcp_size
```

因此 all-gather 后的 Q 形状与这条 `RadixAttention` 路径匹配。本提交复用了这条原本服务 MLA DCP 的 attention 对象。

## 9. 步骤三：切到 DCP RadixAttention，并先写本地 MLA KV

### 原理

indexer cache 要复制，但真正用于注意力的 `k_nope` / `k_rope` 仍必须按 DCP owner rule 写入分片 MLA KV cache。这样 sparse kernel 读取的是当前 rank 的 paged KV，而不是刚产生的临时 K。

### 代码

`forward_dsa_core_npu()` 根据同一判定选择 attention 对象：

```python
use_dcp = _use_dsa_dcp_partial_attention(forward_batch)
attn = m.attn_mqa_for_dcp_decode if use_dcp else m.attn_mqa
attn_output = attn(..., save_kv_cache=True, topk_indices=topk_indices)
```

调用沿既有路径进入：

```text
RadixAttention
  -> AscendAttnBackend.forward_extend / 对应 forward mode
  -> topk_indices 非空
  -> AscendAttnBackend.forward_sparse
```

`forward_sparse()` 先按原有 cache 写入逻辑保存本 rank 的 MLA K，然后取得 paged `k_nope` / `k_pe` cache。随后本提交新增的分派生效：

```python
if get_parallel().dcp_enabled:
    return forward_dcp_sparse_attention(...)
```

返回值从普通路径的单个 `attn_out` 变为 `(local_attn_out, local_lse)`，供后续跨 rank 精确合并。

## 10. 步骤四：把全局 top-k 映射到当前 rank

### 原理

indexer 输出的 top-k 形如 `[7, 2, -1, 4, 1, 6]`，其中 `-1` 是 padding。对于 `N = 3`、rank 1：

- rank 1 拥有全局位置 `1, 4, 7, ...`；
- 全局位置 `7, 4, 1` 映射为局部位置 `2, 1, 0`；
- 其他 rank 的位置要丢弃。

结果必须是：

```text
[2, 1, 0, -1, -1, -1]
```

不能保留成 `[2, -1, -1, 1, 0, -1]`，因为 sparse attention kernel 要求有效下标连续地位于 `-1` padding 之前。同时不能通过按下标值排序来压紧，因为 indexer 的原顺序按 score 排列；稳定压紧可保留这个顺序。

### 代码

`layout.py::remap_dcp_sparse_indices()` 执行三步：

```python
local_mask = (topk_indices >= 0) & (topk_indices % dcp_size == dcp_rank)
local_indices = where(local_mask, topk_indices // dcp_size, -1)
```

然后分别对 valid 和 invalid 元素做 `cumsum`，构造目标位置：

```text
valid 元素目标区间   = [0, valid_count)
invalid 元素目标区间 = [valid_count, K)
```

最后用一次 `scatter` 得到稳定压紧的结果。这个实现不依赖 sort，适合设备侧张量，也保持 indexer 的得分顺序。

`dsa_dcp.py::forward_dcp_sparse_attention()` 一进入就调用该函数，再把 `[T, K]` 扩成 NPU kernel 所需的 `[T, 1, K]`。

## 11. 步骤五：选择正确的局部长度与 block table

### 原理

局部 sparse index 只能相对于当前 rank 的局部 KV 长度解释。例如全局序列长度为 8、`N = 3` 时，各 rank 长度是 `[3, 3, 2]`。把全局长度 8 传给本地 kernel 会让越界下标看起来合法。

推测解码还要保证每个 query 只能看到自己的 KV 前沿，不能偷看同一 draft chain 后面的 token。

### 代码

`forward_dcp_sparse_attention()`：

- 普通路径使用 `forward_metadata.dcp_block_tables`；
- speculative 路径使用 `dcp_spec_block_tables[::tokens_per_request]`，为每个请求选一份相同 page 映射；
- `_get_local_kv_lens()` 在 speculative 模式把逐 query 的局部长度 reshape 为 `[batch, tokens_per_request]`，取最后一列作为 kernel 的物理 KV 上界；top-k 本身已经为每个 query 编码可见集合。

设备侧长度优先于 CPU 长度，保证图模式可 replay。所有长度最终转换为 NPU 上的 `int32`。

## 12. 步骤六：每个 rank 执行局部 sparse attention

### 原理

每个 rank 现在拥有：

- all-gather 后的完整 Q heads；
- 只属于自己的 paged MLA KV；
- 由全局 top-k 过滤得到的本地 sparse indices；
- 本地 KV 长度和 block table。

因此它可以计算全局 sparse attention 分母与分子的一个分片。设 rank `r` 上选中的 token 集为 `S_r`：

```text
Z_r = sum(i in S_r) exp(score_i)
LSE_r = log(Z_r)
O_r = sum(i in S_r) softmax_r(score_i) * V_i
```

这里的 `O_r` 只按 rank 内的 `Z_r` 归一化，不能直接相加；必须保留 `LSE_r` 才能做全局归一化。

### 代码

调用 `npu_sparse_flash_attention` 时的关键参数是：

```python
layout_query="TND"
layout_kv="PA_BSND"
sparse_block_size=1
attention_mode=2
sparse_mode=0
block_table=dcp_block_tables
actual_seq_lengths_kv=local_lens
```

这里将 `sparse_mode` 从普通 DSA 路径的 `3` 改为 `0`。原因是 causal 可见性已经由全局 indexer/top-k 保证；下标映射为局部坐标以后，kernel 已无法仅凭局部位置恢复全局 causal 关系。如果再让 kernel 按局部下标做 causal 判断，会错误屏蔽 token。

### LSE 的两条兼容路径

1. 若存在自定义 `torch.ops._C_ascend.npu_sparse_flash_attention`，请求 `return_softmax_lse=True`，由 kernel 返回 `softmax_max` 和 `softmax_sum`：

   ```text
   LSE = softmax_max + log(softmax_sum)
   ```

   随后把 LSE 整理成 `[T, H_total]`。

2. 若当前 CANN/torch_npu 接口不能为 paged sparse attention 返回 LSE，先调用标准 kernel 得到 attention output，再由 `_compute_sparse_lse()` 兼容计算 LSE。

兼容计算会：

- 根据 query 累积长度建立每个 query 对应的请求行；
- 检查 sparse local index 是否小于该请求的 `local_kv_lens`；
- 通过 `logical_page = local_index // page_size` 和 block table 找到物理 token；
- gather 对应的 `k_nope` / `k_rope`；
- 计算 `q_nope @ k_nope + q_rope @ k_rope`，乘 scaling 后做 `logsumexp`；
- 每次最多处理 128 个 query，限制中间 gather 和 score tensor 的峰值显存；
- 将图 padding query 的 LSE 置零。

这条 fallback 会重复一部分打分计算，性能不如 kernel 原生返回 LSE，但保证旧接口上也能正确做跨 rank softmax 合并。

## 13. 步骤七：用 LSE 合并各 rank 的局部结果

### 原理

全局 LSE 为：

```text
LSE_global = logsumexp(LSE_0, LSE_1, ..., LSE_(N-1))
```

局部输出应按各自分母占全局分母的比例加权：

```text
O_global = sum_r exp(LSE_r - LSE_global) * O_r
```

所以这里只 all-reduce `O_r` 是错误的；LSE 是恢复全局 softmax 的必要信息。

### 代码

`forward_dsa_core_npu()` 收到 `(attn_output, lse)` 后，先把输出恢复为：

```text
[tokens, num_local_heads * dcp_size, kv_lora_rank]
```

然后调用既有的 `dcp/comm.py::cp_lse_ag_out_rs_mla_npu()`：

1. 将局部 output 转为 FP32，并与 FP32 LSE 在最后一维打包；
2. 按 head 切分，通过一次 `all_to_all_single` 把同一组目标 heads 在不同 KV ranks 上的局部结果汇聚到负责该 head 的 rank；
3. reshape 为 `[world_size, batch * local_heads, head_dim + 1]`；
4. 调用 `torch_npu.npu_attention_update(lse_list, out_list, 0)` 按 LSE 完成数值稳定的 softmax 合并；
5. 返回 `[tokens, num_local_heads, kv_lora_rank]`，并恢复原 dtype。

它等价于“收集 LSE + 按全局 softmax 修正 output + reduce-scatter”，但把 output 和 LSE 打包进一次 all-to-all，减少 collective 次数。

## 14. 步骤八：回到原 DSA 后处理

合并完成后，代码重新进入与非 DCP 路径相同的逻辑：

1. 将 latent attention output reshape 为 `[-1, num_local_heads, kv_lora_rank]`；
2. 与 `w_vc` 做 batch matmul，投影到 `v_head_dim`；
3. flatten heads；
4. 经过 `o_proj`；
5. 按 `next_skip_topk` 决定是否把 `topk_indices` 传给下一层复用。

因此 DCP 的影响被限制在 sparse attention 的输入布局、局部计算和结果归并之内；后续 DSA/MLA 投影与模型层接口保持不变。

## 15. 完整执行时序

```text
服务初始化
  |
  |-- 估算显存：NPU DSA indexer 成本乘 dcp_size
  |-- 分配 MLA KV cache（DCP 分片容量）
  `-- 分配 indexer KV cache（全局复制容量）

每个 batch 初始化 attention metadata
  |
  |-- 从全局 seq_lens 计算当前 rank 的 local KV lens
  |-- 用 page_size * dcp_size 的步长构造 local block table
  `-- 图模式下 copy_ 到固定 device buffers

每个 DSA layer
  |
  |-- forward_dsa_prepare_npu
  |     |-- 生成 q_nope / q_rope / k_nope / k_rope
  |     |-- 用 req_to_token[request, position] 写复制式 indexer cache
  |     |-- indexer 在完整历史上生成全局 top-k
  |     `-- all-gather Q heads
  |
  |-- forward_dsa_core_npu
  |     |-- 选择 attn_mqa_for_dcp_decode
  |     |-- 将本 rank 的 MLA K 写入分片 paged cache
  |     `-- AscendAttnBackend.forward_sparse
  |           |-- 全局 top-k -> owner 过滤 -> rank-local index -> 稳定压紧
  |           |-- 选择 local lens 和 local block table
  |           |-- npu_sparse_flash_attention 计算 local output
  |           `-- kernel 原生或 fallback 计算 natural-log LSE
  |
  |-- cp_lse_ag_out_rs_mla_npu
  |     |-- 打包 FP32 output + LSE
  |     |-- all-to-all 按目标 heads 汇聚各 KV rank 的局部结果
  |     `-- npu_attention_update 做全局 softmax 合并
  |
  `-- w_vc -> o_proj -> 返回原模型前向流程
```

## 16. 一个端到端小例子

假设：

```text
DCP size = 3
某 query 的全局 top-k = [7, 2, 4, 1, 6, -1]
```

各 rank 的本地下标为：

| rank | 拥有的全局位置 | 稳定压紧后的 sparse indices |
| --- | --- | --- |
| 0 | 6 | `[2, -1, -1, -1, -1, -1]` |
| 1 | 7, 4, 1 | `[2, 1, 0, -1, -1, -1]` |
| 2 | 2 | `[0, -1, -1, -1, -1, -1]` |

执行过程是：

1. 三个 rank 的 indexer cache 都包含完整历史，所以都能得到同一个全局 top-k。
2. Q 在三个 rank 间 all-gather，每个 rank 都有完整 query heads。
3. 每个 rank 只读取表中属于自己的局部 KV，计算 `O_r` 与 `LSE_r`。
4. all-to-all 把同一目标 head 的三份 `(O_r, LSE_r)` 放到一起。
5. `npu_attention_update` 按三份 LSE 重新归一化，结果等价于一次性对全局集合 `{7, 2, 4, 1, 6}` 做 sparse attention。

## 17. 正确性与工程上的关键点

### 17.1 indexer 和 MLA KV 不能共用写址语义

`out_cache_loc` 服务于 DCP 分片 KV；复制式 indexer 必须改用 `req_to_token[request, position]` 的 allocator-global slot。容量和写址必须同时改，缺少任一项都会越界或覆盖错误 token。

### 17.2 top-k 压紧必须稳定

压紧不只是为了美观，而是 NPU sparse kernel 的输入约束。稳定性又保证 indexer 的 score 顺序不被破坏。

### 17.3 局部下标不能继续使用 causal sparse mode

局部下标丢失了全局位置信息。因果性由全局 top-k 负责，kernel 使用 `sparse_mode=0`，否则会把局部 index 大小误当作全局先后关系。

### 17.4 局部输出必须和同底数的 LSE 一起合并

本实现返回 natural-log LSE。自定义 kernel 用 `max + log(sum)` 生成，fallback 用 `torch.logsumexp` 生成，二者与 `npu_attention_update` 的合并语义一致。

### 17.5 图模式需要设备侧固定 buffer

`dcp_seq_lens` 与 `dcp_spec_seq_lens` 的新增不只是少一次拷贝；它们保证 graph replay 不依赖动态 host-to-device 构造。block table 同样采用预分配后原地覆盖。

### 17.6 fallback LSE 是兼容方案，不是理想性能路径

标准 sparse kernel 已经计算过 attention score，fallback 又 gather K 并重算一次 `logsumexp`。部署环境若提供能直接返回 paged sparse LSE 的自定义算子，会避开这部分重复工作。

## 18. 测试关注点

与这次实现直接相关的测试应覆盖以下不变量：

- `remap_dcp_sparse_indices()` 对每个 rank 的 owner 过滤、全局到局部映射、稳定压紧和 `dcp_size=1` identity；
- replicated indexer cache loc 在 decode、extend、speculative 和 graph padding 下使用正确的全局 slot；
- 普通与 speculative local KV lens/block table 选择正确；
- 自定义 sparse kernel 的 LSE 形状和 `max + log(sum)` 转换正确；
- fallback LSE 与直接对选中 KV 做 `logsumexp` 一致；
- pool configurator 将 NPU BF16 复制式 indexer 的 `dcp_size` 倍开销计入预算；
- DSA prefill CP 生效时不会误入 DSA+DCP partial attention。

工作区中的 `test/registered/dcp/test_npu_dsa_dcp_unit.py`、`test/registered/dcp/test_dcp_layout_unit.py` 和 `test/registered/unit/model_executor/test_pool_configurator.py` 正是围绕这些边界构造的 CPU 单元测试。

## 19. 总结

这次适配没有重新实现一套完整的 DCP，而是把 DSA 的特殊语义嵌入既有 MLA DCP 主干：

```text
既有能力：KV token 分片、Q all-gather、局部 attention、LSE 跨 rank 合并
DSA 新增：复制式全局 indexer、global top-k -> local top-k 映射、sparse LSE 获取
配套补齐：普通 extend 元数据、speculative 元数据、图模式设备 buffer、显存预算
```

最核心的设计边界是：**选哪些 token 是全局问题，读取和计算这些 token 是局部问题，softmax 归一化再回到全局问题。** 代码分别用复制式 indexer、rank-local sparse attention 和 LSE-aware collective 对应这三个阶段。
