# MiniMax-M3 昇腾 NPU 平台迁移 Spec

## Why
MiniMax-M3 模型当前仅支持 NVIDIA CUDA 和 AMD ROCm 平台，需要将其迁移到华为昇腾 NPU 平台，并使用 ModelScope 提供的 W8A8 量化权重 (`Eco-Tech/MiniMax-M3-w8a8`) 进行推理，以降低部署成本并提升国产硬件利用率。

## 参考实现
vllm-ascend (`/root/project/vllm-ascend`, commit `75a2c88d`) 已完成 MiniMax-M3 在 Ascend 上的迁移，以下组件可参考/复用：

| 组件 | vllm-ascend 文件 | 复用方式 |
|------|-----------------|---------|
| Block score decode Triton kernel | `vllm_ascend/ops/triton/flash_block_score_decode.py` | 适配 SGLang Triton 框架后复用，含 streaming topk、融合 score+attn kernel |
| Topk sparse decode Triton kernel | `vllm_ascend/ops/triton/topk_sparse_decode.py` | 适配 SGLang Triton 框架后复用，含 GQA share sparse decode、split-topk 合并 |
| SwiGLU-OAI 激活函数 | `vllm_ascend/ops/activation.py` `AscendSwigluOAIAndMul` | 直接复用，纯 PyTorch 实现 |
| 稀疏 block 选择/合并逻辑 | `vllm_ascend/models/minimax_m3_vl.py` `_select/merge/expand_sparse_blocks` | 直接复用，纯 PyTorch 逻辑 |
| ModelSlim 量化配置映射 | `vllm_ascend/quantization/modelslim_config.py` | 参考 `packed_modules_model_mapping` 中 `minimax_m3_vl` 的配置 |
| MoE SwiGLU-OAI 量化路径 | `vllm_ascend/ops/fused_moe/moe_mlp.py` | 参考 `use_swigluoai` 分支：gmm → swigluoai → dynamic_quant |

**关键设计决策（来自 vllm-ascend）**：
1. **MXFP8 在 NPU 上反量化到 BF16 执行**：NPU 不支持原生 FP8 执行，MXFP8 权重在加载时反量化为 BF16
2. **SwiGLU-OAI 不能使用 `npu_dequant_swiglu_quant` 融合 kernel**：数学形式不同，需拆分为 gmm → swigluoai_forward → dynamic_quant
3. **Triton kernel 的 Ascend 适配要点**：不使用 `tl.make_block_ptr`、不使用 3D reshape、`CHUNK_SIZE_T >= 2`、不计算 `real_topk`（用 -1 sentinel）、使用 streaming topk 替代 bitonic sort
4. **KV cache 格式**：K cache 中拼接 main K + index K（`head_size + index_head_dim`），V cache 保持 `head_size_v`

## What Changes
- 新增 NPU 兼容的 MiniMax 稀疏注意力后端（`AscendMiniMaxSparseAttnBackend`），基于参考 vllm-ascend 的 Triton kernel 适配实现
- 新增 NPU 版 `MiniMaxSparseKVPool`（`NPUMiniMaxSparseKVPool`），使用 `NPUMHATokenToKVPool` 作为子池
- 适配 ModelSlim W8A8 量化方案，确保 `quant_model_description.json` 中的层名映射与 MiniMax-M3 权重命名一致
- 适配 SwiGLU-OAI 激活函数在 NPU W8A8 MoE 路径中的支持（参考 vllm-ascend 的拆分路径）
- 适配 per-head RMSNorm + partial RoPE 的 NPU 融合路径（参考 ROCm Triton 实现）
- 修改注意力后端注册逻辑，在 NPU 平台上为 MiniMax-M3 使用 Ascend 专用后端
- 修改 KV 池创建路径，在 NPU 上使用 `NPUMiniMaxSparseKVPool`
- 禁用 NPU 上不可用的功能：MSA (fmha_sm100)、共享专家融合、CUDA JIT 内核、MXFP8 原生执行
- 新增 NPU Graph 兼容性适配
- 新增端到端测试用例

## Impact
- Affected specs: 注意力后端系统、量化系统、KV 缓存池系统、MoE 系统
- Affected code:
  - `python/sglang/srt/models/minimax_m3.py` — 模型文件，需添加 NPU 平台分支
  - `python/sglang/srt/layers/attention/attention_registry.py` — 注意力后端注册
  - `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py` — KV 池创建
  - `python/sglang/srt/layers/attention/minimax_sparse_backend.py` — 稀疏注意力后端
  - `python/sglang/srt/mem_cache/memory_pool.py` — MiniMaxSparseKVPool
  - `python/sglang/srt/hardware_backend/npu/` — NPU 硬件后端（新增文件）
  - `python/sglang/srt/layers/quantization/modelslim/modelslim.py` — ModelSlim 量化配置
  - `python/sglang/srt/layers/moe/topk.py` — MoE 路由（MultiPlatformOp）

## ADDED Requirements

### Requirement: NPU 兼容的 MiniMax 稀疏注意力后端
系统 SHALL 在 NPU 平台上为 MiniMax-M3 提供功能完整的稀疏注意力实现，包括 index head topk block 选择和 main head sparse attention 两步流程。

#### Scenario: NPU 上 MiniMax-M3 稀疏层 decode
- **WHEN** MiniMax-M3 在 NPU 上执行 decode 阶段的稀疏注意力层
- **THEN** 系统使用 `torch_npu.npu_sparse_flash_attention` 或等效 NPU 算子执行 index head topk 选择和 main head sparse attention，结果与 CUDA 平台数值等价（允许精度差异）

#### Scenario: NPU 上 MiniMax-M3 稀疏层 prefill
- **WHEN** MiniMax-M3 在 NPU 上执行 prefill 阶段的稀疏注意力层
- **THEN** 系统使用 NPU 兼容算子完成 index head topk 选择和 main head sparse attention

### Requirement: NPU 版 MiniMaxSparseKVPool
系统 SHALL 在 NPU 平台上提供 `NPUMiniMaxSparseKVPool`，使用 `NPUMHATokenToKVPool` 作为子池，支持 index KV 缓存的读写。

#### Scenario: NPU 上创建 MiniMax 稀疏 KV 池
- **WHEN** MiniMax-M3 模型在 NPU 上初始化 KV 缓存池
- **THEN** 系统创建 `NPUMiniMaxSparseKVPool`，其内部 buffer 布局符合 NPU FIA 要求，index KV 缓存使用 `torch_npu.npu_scatter_nd_update_` 写入

### Requirement: ModelSlim W8A8 量化适配
系统 SHALL 支持加载 ModelScope `Eco-Tech/MiniMax-M3-w8a8` 权重，正确解析 `quant_model_description.json` 配置，并为 Linear 和 MoE 层分别应用 `ModelSlimW8A8Int8` 和 `ModelSlimW8A8Int8MoE` 量化方案。

#### Scenario: 加载 W8A8 量化权重
- **WHEN** 用户指定 `Eco-Tech/MiniMax-M3-w8a8` 模型路径在 NPU 上启动推理
- **THEN** 系统自动检测 `quant_model_description.json`，为 Linear 层应用 `NPUW8A8Int8LinearMethod`，为 MoE 层应用 `NPUW8A8Int8DynamicMoEMethod`

#### Scenario: MoE 层 W8A8 动态量化推理
- **WHEN** MoE 层在 NPU 上执行 W8A8 动态量化推理
- **THEN** 系统使用 `npu_dynamic_quant` + `npu_quant_matmul` + `npu_dequant_swiglu_quant` 完成计算

### Requirement: SwiGLU-OAI 激活函数 NPU 适配
系统 SHALL 在 NPU 的 W8A8 MoE 路径中支持 MiniMax-M3 的 SwiGLU-OAI 激活函数（带 `swiglu_alpha` 和 `swiglu_limit` 参数）。

#### Scenario: MoE 层使用 SwiGLU-OAI 激活
- **WHEN** MiniMax-M3 的 MoE 层使用 `swigluoai` 激活函数在 NPU 上执行
- **THEN** 系统正确计算带 alpha 和 limit 参数的 SwiGLU，或在 `npu_dequant_swiglu_quant` 不支持时提供 fallback 实现

### Requirement: per-head RMSNorm + RoPE NPU 适配
系统 SHALL 在 NPU 上为 MiniMax-M3 的 per-head GemmaRMSNorm + partial RoPE 提供高效实现或 fallback 路径。

#### Scenario: QK normalization + RoPE 计算
- **WHEN** MiniMax-M3 在 NPU 上执行 QK normalization + RoPE
- **THEN** 系统使用 NPU 兼容的实现（参考 ROCm Triton 内核或使用 PyTorch native fallback），确保数值正确

### Requirement: NPU 平台分支与功能禁用
系统 SHALL 在 NPU 平台上正确禁用不可用的 CUDA 专用功能，并使用 NPU 替代实现。

#### Scenario: 禁用 CUDA 专用功能
- **WHEN** MiniMax-M3 在 NPU 上运行
- **THEN** 以下功能被禁用或替换：
  - MSA (fmha_sm100) → 使用 NPU sparse attention fallback
  - CUDA JIT fused qknorm+rope → 使用 PyTorch native 或 Triton fallback
  - 共享专家融合 → 禁用（`disable_shared_experts_fusion = True`）
  - CUDA JIT store_kv_index → 使用 PyTorch native scatter 写入
  - CUDA JIT per_token_quant_ue8m0 → 使用 NPU 量化算子
  - DeepGEMM scale 布局 → 使用 NPU 量化方案替代

### Requirement: 注意力后端注册与 KV 池创建集成
系统 SHALL 在 NPU 平台上自动为 MiniMax-M3 选择正确的注意力后端和 KV 池实现。

#### Scenario: NPU 上自动选择 MiniMax 注意力后端
- **WHEN** MiniMax-M3 模型在 NPU 平台上初始化
- **THEN** 系统创建 `AscendMiniMaxSparseAttnBackend` 作为稀疏注意力后端，并与 `AscendAttnBackend` 组合为 `MiniMaxHybridAttnBackend`

#### Scenario: NPU 上自动创建 MiniMax KV 池
- **WHEN** MiniMax-M3 模型在 NPU 平台上创建 KV 缓存池
- **THEN** 系统创建 `NPUMiniMaxSparseKVPool` 而非 `MiniMaxSparseKVPool`

### Requirement: NPU Graph 兼容性
系统 SHALL 确保 MiniMax-M3 的 decode 路径在 NPU Graph 模式下可正确捕获和回放。

#### Scenario: NPU Graph 捕获与回放
- **WHEN** MiniMax-M3 在 NPU 上启用 Graph 模式执行 decode
- **THEN** 稀疏注意力的 topk 选择和 sparse attention 计算都能在 NPU Graph 中正确执行

### Requirement: 端到端测试
系统 SHALL 提供 MiniMax-M3 W8A8 在 NPU 上的端到端测试用例。

#### Scenario: NPU 上运行 MiniMax-M3 W8A8 推理
- **WHEN** 执行 NPU 上的 MiniMax-M3 W8A8 测试
- **THEN** 模型能正确加载权重、完成推理，并在 GSM8K 等基准上达到可接受的精度（参考 ModelScope 报告的 96.89%）

## MODIFIED Requirements

### Requirement: MiniMax-M3 模型文件平台分支
原 `minimax_m3.py` 中的 `_is_cuda` / `_is_hip` 条件分支需扩展为 `_is_cuda` / `_is_hip` / `_is_npu` 三路分支，确保 NPU 平台走正确的代码路径。具体包括：
- `_use_fused_qknorm_rope`：NPU 上设为 False
- `_fuse_qkv_index_enabled`：NPU 上需评估是否启用（取决于 NPU 量化方案是否支持 fused GEMM）
- `determine_num_fused_shared_experts`：NPU 上禁用共享专家融合
- `MiniMaxM3Attention.forward` 中的 CUDA JIT 调用：NPU 上走 fallback 路径

### Requirement: MiniMaxHybridAttnBackend 平台分发
原 `attention_registry.py` 中 MiniMax 稀疏注意力后端的创建逻辑需增加 NPU 平台判断，在 NPU 上使用 `AscendMiniMaxSparseAttnBackend` 替代 `MiniMaxSparseAttnBackend`。

### Requirement: KV 池创建路径
原 `model_runner_kv_cache_mixin.py` 中 `is_minimax_sparse` 分支直接创建 `MiniMaxSparseKVPool`，需增加 NPU 平台判断，在 NPU 上创建 `NPUMiniMaxSparseKVPool`。

## REMOVED Requirements
无（所有现有功能保留，NPU 上不支持的走 fallback 或禁用）
