# Tasks

> **工作方式**：每个 SubTask 执行完后暂停，向用户讲解修改的代码和设计思路，等待用户确认后再继续下一个 SubTask。

- [ ] Task 1: 修改 `minimax_m3.py` 模型文件添加 NPU 平台分支
  - [ ] SubTask 1.1: 添加 `_is_npu = is_npu()` 全局变量，导入 `is_npu`
  - [ ] SubTask 1.2: 修改 `_use_fused_qknorm_rope` 条件，NPU 上设为 False
  - [ ] SubTask 1.3: 修改 `_fuse_qkv_index_enabled` 条件，NPU 上设为 False（MXFP8 在 NPU 上反量化为 BF16，不需要 fused GEMM）
  - [ ] SubTask 1.4: 修改 `determine_num_fused_shared_experts`，NPU 上禁用共享专家融合
  - [ ] SubTask 1.5: 修改 `MiniMaxM3Attention.forward` 中的 CUDA JIT 调用，NPU 上走 PyTorch native fallback（参考 vllm-ascend 的 `_qk_norm_rope` 路径）
  - [ ] SubTask 1.6: 修改 `MiniMaxM3DecoderLayer.forward` 中的 `_is_hip` AITER 条件，增加 NPU 分支
  - [ ] SubTask 1.7: 修改 `MiniMaxM3Model.forward` 中 `check_cuda_graph_backend` 调用，NPU 上使用 NPU Graph 条件

- [ ] Task 2: 移植 vllm-ascend 的 Triton 稀疏注意力 kernel 到 SGLang
  - [ ] SubTask 2.1: 将 `vllm_ascend/ops/triton/flash_block_score_decode.py` 适配为 SGLang 的 Triton kernel 格式，放在 `python/sglang/srt/layers/attention/minimax_sparse_ops/npu/` 目录下
  - [ ] SubTask 2.2: 将 `vllm_ascend/ops/triton/topk_sparse_decode.py` 适配为 SGLang 的 Triton kernel 格式
  - [ ] SubTask 2.3: 适配 Triton kernel 的 Ascend 特定要点：不使用 `tl.make_block_ptr`、`CHUNK_SIZE_T >= 2`、streaming topk 替代 bitonic sort、-1 sentinel 替代 `real_topk` 计算
  - [ ] SubTask 2.4: 适配 `_choose_num_kv_chunks` 和 `_choose_num_topk_chunks` 到 SGLang 的 target grid 规则
  - [ ] SubTask 2.5: 编写 NPU 版 prefill 稀疏注意力 kernel（参考 SGLang 现有的 `prefill/flash_with_topk_idx.py` 和 `prefill/topk_sparse.py`，适配 Ascend Triton 要求）

- [ ] Task 3: 实现 `NPUMiniMaxSparseKVPool`
  - [ ] SubTask 3.1: 在 `python/sglang/srt/hardware_backend/npu/memory_pool_npu.py` 中创建 `NPUMiniMaxSparseKVPool` 类
  - [ ] SubTask 3.2: 使用 `NPUMHATokenToKVPool` 替代 `MHATokenToKVPool` 作为主 KV 子池
  - [ ] SubTask 3.3: 实现 index KV 缓存的 buffer 管理和读写（使用 `torch_npu.npu_scatter_nd_update_`）
  - [ ] SubTask 3.4: 参考 vllm-ascend 的 KV cache 格式：K cache 拼接 main K + index K（`head_size + index_head_dim`），V cache 保持 `head_size_v`
  - [ ] SubTask 3.5: 实现 `set_fused_kv_index_buffer` / `get_index_k_buffer` / `get_index_kv_buffer` 等 NPU 兼容接口

- [ ] Task 4: 实现 `AscendMiniMaxSparseAttnBackend`
  - [ ] SubTask 4.1: 在 `python/sglang/srt/hardware_backend/npu/attention/` 下创建 `ascend_minimax_sparse_backend.py`
  - [ ] SubTask 4.2: 实现 decode 路径：调用 Task 2 移植的 Triton kernel（`flash_decode_bnsd_with_topk_idx` + `flash_decode_bnsd_with_gqa_share_sparse`）
  - [ ] SubTask 4.3: 实现 prefill 路径：调用 Task 2 移植的 prefill Triton kernel
  - [ ] SubTask 4.4: 参考 vllm-ascend 的三种 decode 路径（Triton → batched PTA → per-request PTA），实现 fallback 层级
  - [ ] SubTask 4.5: 禁用 MSA (fmha_sm100) 路径
  - [ ] SubTask 4.6: 适配 NPU Graph 模式（`init_forward_metadata` / `forward_decode_graph`）

- [ ] Task 5: 修改注意力后端注册逻辑
  - [ ] SubTask 5.1: 在 `attention_registry.py` 的 `attn_backend_wrapper()` 中增加 NPU 平台判断
  - [ ] SubTask 5.2: NPU 上为 MiniMax-M3 创建 `AscendMiniMaxSparseAttnBackend` + `MiniMaxHybridAttnBackend`

- [ ] Task 6: 修改 KV 池创建路径
  - [ ] SubTask 6.1: 在 `model_runner_kv_cache_mixin.py` 的 `is_minimax_sparse` 分支增加 NPU 判断
  - [ ] SubTask 6.2: NPU 上创建 `NPUMiniMaxSparseKVPool` 替代 `MiniMaxSparseKVPool`

- [ ] Task 7: 适配 ModelSlim W8A8 量化方案
  - [ ] SubTask 7.1: 验证 `Eco-Tech/MiniMax-M3-w8a8` 权重的 `quant_model_description.json` 格式
  - [ ] SubTask 7.2: 参考 vllm-ascend 的 `modelslim_config.py` 中 `minimax_m3_vl` 的 `packed_modules_model_mapping`，确认 SGLang 的 ModelSlim 配置正确映射 MiniMax-M3 的权重名
  - [ ] SubTask 7.3: 确认 `block_sparse_moe` → `mlp` 的 name remapping 在 ModelSlim 层名匹配前完成
  - [ ] SubTask 7.4: 验证 `NPUW8A8Int8DynamicMoEMethod` 对 MiniMax-M3 MoE 结构的兼容性
  - [ ] SubTask 7.5: 处理 MXFP8 checkpoint：在 NPU 上检测到 MXFP8 量化时，将 `quantization` 设为 None 并反量化权重到 BF16（参考 vllm-ascend 的 `patch_minimax_m3.py`）

- [ ] Task 8: 适配 SwiGLU-OAI 激活函数
  - [ ] SubTask 8.1: 直接复用 vllm-ascend 的 `AscendSwigluOAIAndMul.swiglu_oai_forward` 实现，添加到 SGLang 的 `activation.py` 或 `MiniMaxM3MLP` 中
  - [ ] SubTask 8.2: 在 `NPUW8A8Int8DynamicMoEMethod` 中添加 SwiGLU-OAI 分支：`npu_grouped_matmul` → `swigluoai_forward` → `npu_dynamic_quant`（参考 vllm-ascend 的 `moe_mlp.py` `use_swigluoai` 路径）
  - [ ] SubTask 8.3: 在 `MiniMaxM3MLP` 中为 NPU 平台添加 `swigluoai` 激活函数的 NPU 兼容实现

- [ ] Task 9: 适配 per-head RMSNorm + RoPE
  - [ ] SubTask 9.1: 参考 ROCm 的 Triton 实现，为 NPU 提供 `qk_gemma_rmsnorm_rope` 的等价实现
  - [ ] SubTask 9.2: 在 `MiniMaxM3Attention.__init__` 中添加 NPU 的 `_can_use_npu_qk_norm_rope_static` 条件
  - [ ] SubTask 9.3: 在 `_qk_norm_rope` / `_index_qk_norm_rope` / `_sparse_qk_index_norm_rope` 中添加 NPU 分支
  - [ ] SubTask 9.4: 如 Triton 不可用，使用 PyTorch native fallback（`MultiHeadRMSNorm.forward` + `rotary_emb`）

- [ ] Task 10: 适配 MoE 路由 TopK
  - [ ] SubTask 10.1: 验证 `npu_moe_gating_top_k` 对 sigmoid scoring + correction_bias 的支持
  - [ ] SubTask 10.2: 在 `TopK` 的 `MultiPlatformOp.forward_npu` 中添加 MiniMax-M3 sigmoid 路由支持
  - [ ] SubTask 10.3: 验证 `npu_moe_compute_slope` 对 `e_score_correction_bias` 的处理

- [ ] Task 11: NPU Graph 兼容性适配
  - [ ] SubTask 11.1: 在 `npu_graph_runner.py` 中添加 MiniMax-M3 sparse decode 的 graph 捕获支持
  - [ ] SubTask 11.2: 预分配 topk_indices buffer 和 workspace
  - [ ] SubTask 11.3: 验证 `NPUGraph.update()` 对 sparse attention 输入的更新

- [ ] Task 12: 端到端测试
  - [ ] SubTask 12.1: 创建 `test/registered/ascend/llm_models/test_ascend_minimax_m3_w8a8.py` 测试文件
  - [ ] SubTask 12.2: 测试 BF16 非量化模式的基本推理（先验证模型结构正确）
  - [ ] SubTask 12.3: 测试 W8A8 量化模式的推理精度
  - [ ] SubTask 12.4: 测试 NPU Graph 模式
  - [ ] SubTask 12.5: 测试长上下文（稀疏注意力核心场景）

# Task Dependencies
- [Task 2] 可与 [Task 1] 并行（Triton kernel 移植不依赖模型文件修改）
- [Task 3] depends on [Task 1] (需要先确定 NPU 平台分支逻辑)
- [Task 4] depends on [Task 2, Task 3] (稀疏注意力后端依赖 Triton kernel 和 KV 池)
- [Task 5] depends on [Task 4] (注册需要后端实现完成)
- [Task 6] depends on [Task 3] (KV 池创建路径依赖 NPUMiniMaxSparseKVPool 实现)
- [Task 7] 无强依赖，可与 Task 1-4 并行
- [Task 8] depends on [Task 7] (SwiGLU-OAI 适配依赖量化方案确认)
- [Task 9] 无强依赖，可与 Task 1-4 并行
- [Task 10] 无强依赖，可与 Task 1-4 并行
- [Task 11] depends on [Task 4] (Graph 适配依赖稀疏注意力后端)
- [Task 12] depends on [Task 1-11] (端到端测试依赖所有组件完成)
