# AReaL 适配 Qwen3.5 VL RLVR — 适配报告

**日期**: 2026-04-23 → 2026-04-27
**作者**: zengbw
**状态**:
- Qwen3.5-0.8B ✅ 已验证
- **Qwen3.5-35B-A3B ✅ 已验证**(v17 6 节点上 60+ 次稳定 update_weights, 21+ 小时长跑无显存泄漏)
- **超参对齐 8B benchmark ✅** (v18: n_samples 2→8 显存不变; v20: batch 4→32 显存不变)
- **序列长度推进 ✅** (v21: seq 4K, actor 41 GB; v22: seq 16K @ PP=4 验证中)
- **xccl 优化路径根因定位 ✅** (GDN conv1d shape mismatch, 修复方案 D 待实施)

本报告分两个阶段记录 AReaL 框架对 Qwen3.5 系列 VL 模型的端到端适配工作:

- **第一阶段**(§§1–4): Qwen3.5-0.8B 稠密模型上线——建立基础配方
  (pad_to_maximum、disk 模式、output-gate 处理、fusion 禁用)。
- **第二阶段**(§§5.1–5.5): Qwen3.5-35B-A3B (MoE) 模型上线——把基础配方扩展到
  MoE + VL 包装结构, 修复一系列在小规模稠密模型上不会出现的问题
  (ref colocate OOM、expert export 转换、keepalive ghost ALLREDUCE、
  mbridge buffer 重置)。
- **第三阶段**(§§5.6–5.8): 上线后的优化迭代——超参对齐 8B benchmark
  (v18-v20)、序列长度推进至 16K (v21-v22)、update_weights 性能瓶颈分析、
  CP 可行性调研 (硬阻断)、xccl 优化的根因定位 (GDN conv1d shape mismatch)。

两个/三个阶段共用同一个 docker 镜像、同一套环境变量前置, 启动脚本基本一致;
差异仅在源码级修复和 YAML 集群布局。

## 1. 背景

Qwen3.5 是 Qwen 推出的新一代视觉语言模型家族, 采用混合注意力架构:
softmax + Gated Delta Net (GDN) 线性注意力。AReaL 既有的 Qwen2.5/3 集成无法
开箱即用。本报告记录全链路问题以及在该家族上跑通 RL 训练
(基于 dapo_math_17k_processed 的 Math RLVR) 的最终可用配方。

- **目标模型**: Qwen3.5-35B-A3B (MoE, 35B 总参数 / 3B 激活)、Qwen3.5-0.8B (稠密, 验证用)
- **基础镜像**: `fuyao:areal-qwen3_5-megatron-v21` (在 veRL 的 `verl-qwen3_5-v9-latest` 基础上改造)
  - torch 2.10 / vLLM 0.17.0 / Megatron-core 0.16.0 / mbridge 0.15.1 / TransformerEngine 2.12 / transformers 4.57.1
  - 加上 AReaL 特有依赖: `aiofiles tensorboardX math_verify`
- **数据集**: dapo_math_17k_processed (纯文本数学 RLVR)

## 2. 架构层面的约束

### 2.1 Qwen3.5 模型结构
- **混合注意力层**:
  - 大多数层: `GatedDeltaNet` (线性注意力, 不支持 THD/packed sequences)
  - 每 4 层一次: `Qwen3_5VLSelfAttention` (带 output gate 的标准 softmax 注意力)
- **Output gate**: `config.text_config.attn_output_gate=true`——Q 投影包含融合的 gate,
  导致 `q_proj.weight` shape 为 `[2 * num_q_heads * head_dim, hidden]` (沿 head_dim 维 q || gate 拼接)
- **MRoPE**: 多模态旋转 (3 轴: text/vision-h/vision-w)。对于纯文本输入,
  mbridge 通过 `get_rope_index` 从 attention_mask 内部计算 position_ids。
- **嵌套 VL 配置**: HF config 把 LM 标量嵌套在 `hf_config.text_config.{vocab_size,
  num_attention_heads, ...}`下——直接访问 `hf_config.vocab_size` 会失败。
- **MoE 特有** (仅 35B-A3B): 256 experts/层, `num_experts_per_tok=8`,
  `moe_intermediate_size=512`, EP=8 / ETP=1。
- **VL 包装的 HF key** (仅 35B-A3B): docker 中的 mbridge bridge 在 HF 侧暴露
  `model.language_model.layers.N.*` 形式的 key, 即使 `hf_config.model_type ==
  "qwen3_5_moe"` (本身是文本-only model_type 但带 VL 风格 key 前缀)。

### 2.2 veRL 与 AReaL 集成方式的差异 (关键发现)

veRL 的 Qwen3.5 工作路径:
1. `mbridge.bridge.export_weights()` → 生成 (name, tensor) 的 generator
2. ZMQ + CUDA IPC 传输
3. **`vllm.model_runner.model.load_weights(weights)` —— vLLM 原生 loader** 处理
   Qwen3.5 的 q||gate 拼接格式

AReaL 的 `weight_update_mode=xccl` 路径:
1. mbridge 导出 → `convert_to_hf` → 自定义 HTTP `/areal_update_weights_xccl` →
   NCCL 广播**直接写入 vLLM 内部 buffer**
2. 绕过 vLLM 原生 loader → 对 q||gate 格式 shape convention 不匹配

后果: AReaL 的 xccl 路径开箱不兼容 Qwen3.5。**Disk 模式**
(`save_weights_to_hf_with_mbridge_fast` → NFS safetensor → vLLM 重新加载) 和
veRL 概念上依赖的路径一致。

## 3. 第一阶段——0.8B 稠密模型验证 Bug 链

下面的每一项 fix 都是必需的, 缺一不可。0.8B 和 35B-A3B 都需要这些 fix。

| # | 现象 | 根因 | 修复 | 文件 |
|---|---------|-----------|-----|------|
| 1 | preprocess_packed_seqs 中 `AttributeError 'NoneType'.sum()` | mbridge 的 Qwen3.5 GDN 需要 `attention_mask`; AReaL 的 `pack_tensor_dict` 把它移除了 | 设置 `actor.pad_to_maximum: true` 跳过 packing, 保留 attention_mask (BSHD 格式) | YAML |
| 2 | `KeyError: 'max_seqlen'`, `AssertionError: cu_seqlens key`, `AttributeError .to() on None` | 下游代码假设 packed 格式 (即 cu_seqlens、max_seqlen 存在) | `_prepare_mb_list` 增加 pad_to_maximum 分支: 预设 `_max_seqlen`, `old_cu_seqlens_list=None`, 守卫 `max_seqlen` 访问 | `megatron_engine.py:1500-1530` |
| 3 | `First dimension of the tensor should be divisible by tensor parallel size` | Megatron TP sequence parallelism 要求 seq 维对齐到 `tp_size` (或 `tp_size*cp_size*2`) | `_pad_seq_dim` 对所有 micro-batch 中 2D+ tensor 的 seq 维做 padding | `megatron_engine.py:1540+` |
| 4 | rope_utils 中 `too many indices for tensor of dimension 2` | Qwen3.5 MRoPE 期望 `[3, B, S]` 形状的 position_ids, 而非 `[B, S]` | `forward_step` 在 `pad_to_maximum` 时设 `position_ids=None`, 让 mbridge 通过 `get_rope_index` 计算 | `megatron_engine.py:614-630` |
| 5 | `_apply_output_gate` 处 `gate.view(*x.shape)` shape 2× 不匹配 | Megatron 的 `get_query_key_value_tensors` 在 `num_kv_heads < TP` 时把 `num_query_groups` 提到 `world_size` → gate 维变成 Q 维的 2× | **Actor TP ≤ num_kv_heads**。0.8B: TP=2 (num_kv_heads=2)。35B-A3B: TP 按 num_kv_heads 设。 | YAML backend |
| 6 | shape 推断时 `torch._dynamo.exc.TorchRuntimeError` | Megatron 的 `_apply_output_gate` 被 `@torch.compile` 装饰, Dynamo 的 fake-tensor shape inference 对 gated attention 失败 | 设环境变量 `TORCHDYNAMO_DISABLE=1`、`TORCH_COMPILE_DISABLE=1` | `fuyao_areal_run.sh:154` |
| 7 | mbridge 的各种 compile/fusion 错误 | Megatron 的 fusion kernel 与 mbridge 的 Qwen3.5 gated attention 不兼容 | 在模型构建之前在 tf_config 上禁用 5 个 fusion: `apply_rope_fusion`、`masked_softmax_fusion`、`bias_activation_fusion`、`bias_dropout_fusion`、`gradient_accumulation_fusion` | `megatron_utils/deterministic.py:11-35` |
| 8 | Ref.compute_logp 报 `NoneType.sum()` (与 #1 同) | Ref 引擎配置漏了 `pad_to_maximum` | YAML 中 ref block 也加 `pad_to_maximum: true` | YAML |
| 9 | update_weights 中 `'Qwen3_5Config' object has no attribute 'vocab_size'` | VL config 把 LM 标量嵌套在 `text_config` 下 | `remove_padding` 调用 fallback 到 `hf_config.text_config.vocab_size` | `megatron_engine.py:1203-1213` |
| 10 | `Unknown parameter name ... vision_model.patch_embed.proj.weight` | mbridge 的 qwen3_5 converter 没有 vision tower 的映射 | weight update 循环里跳过 `.vision_model.` 参数 (纯文本训练; 视觉部分冻结) | `megatron_engine.py:1407+, 1462+` |
| 11 | `Unknown parameter name ... language_model.embedding.word_embeddings.weight` | AReaL 的 `convert_qwen3_5_to_hf` fallback 链不识别 VL 的 `language_model.` 前缀 | 对 Qwen3.5 模型直接调 mbridge 原生 `self.bridge._weight_to_hf_format()` (匹配 veRL 的 `vanilla_mbridge` 路径) | `megatron_engine.py:1223+` |
| 12 | vLLM `Failed to update parameter! expanded (4) must match (8) at dim 2` | xccl 直接写入路径与 Qwen3.5 q\|\|gate 在 vLLM 内部 buffer 的拼接布局不兼容 | **`weight_update_mode: disk`**——保存 HF safetensor, vLLM 通过原生 `load_weights` 重新加载 | YAML |
| 13 | 真正的 AttributeError 被 RPC 误报为 "method not found" | `rpc_server.py` 笼统 catch 了 AttributeError | 用 `hasattr` 判断 → 仅当方法真的不存在时才报 "method not found" | `rpc/rpc_server.py:759` |

## 4. 0.8B 可用配置

### 4.1 YAML (`fuyao_examples/math/qwen3_5_0_8b_rlvr_vllm.yaml`)

关键配置 (全部强制):

```yaml
actor:
  backend: "megatron:d2p1t2"       # TP=2 (0.8B 的 num_kv_heads=2)
  pad_to_maximum: true              # BSHD 格式, GDN 强制要求
  weight_update_mode: disk          # 绕开 xccl 的 q||gate bug
  megatron:
    use_deterministic_algorithms: true

ref:
  backend: ${actor.backend}
  pad_to_maximum: true              # 必须与 actor 一致

rollout:
  backend: "vllm:d4t1"              # TP=1 (规避 vLLM GQA replicate 边界 case)
```

### 4.2 启动脚本环境变量 (`fuyao_examples/fuyao_areal_run.sh`)

```bash
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export VLLM_USE_V1=1
export VLLM_DISABLE_CUSTOM_ALL_REDUCE=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
```

### 4.3 性能数据 (0.8B, 1 节点 × 8 GPU A100-80G)

| 指标 | 数值 |
|--------|-------|
| Step time (稳态) | ~29-34 s |
| Throughput | 148-150 tok/gpu/s |
| MFU | 6% |
| 训练 compute | 235-248 tok/gpu/s |
| Rollout (vLLM) | 6.8M-10.7M tok/gpu/s |
| 权重同步开销 | 每次迭代 ~6 s (disk 模式, 0.8B) |

权重同步占 step 时间 ~20%。35B 上这个比例更小 (compute 增长比 IO 快)。

---

## 5. 第二阶段——35B-A3B (MoE) 模型上线

第一阶段的稠密路径已验证 Qwen3.5 的混合注意力。第二阶段把它扩展到
MoE + VL 包装结构。下面这部分工作是当时第一阶段计划中
("§5.4 已知风险/未决项")乐观假设可以平滑放大的——结果实际上需要
4 个额外的源码级修复。

### 5.1 原计划 vs 实际情况

第一阶段原计划列出 4 项上 35B 之前的未决项。第二阶段又冒出 4 类计划没有
预料到的问题。

| # | 原计划 | 实际结果 |
|---|---------------|----------------|
| 1 | "Disk 模式可扩展性"——预期每次 70 GB, ~25 s 同步 | 验证通过; NFS 上往返 ~3.5 分钟 (含 vLLM reload)。可接受。 |
| 2 | "EP + pad_to_maximum 验证"——预期能跑 | 代码路径正常。问题在别处 (expert HF name 转换)。 |
| 3 | "`attn_output_gate` 验证" | 35B-A3B 上确认为 `true`; 沿用 0.8B 配方。 |
| 4 | "xccl 对齐"——列为后续工作 | 跳过 (仍然用 disk 模式)。仍是后续工作。 |
| **5** | _(计划外)_ | 第一次 ppo_update 时 **Actor + Ref colocate OOM** (§5.3.1)。 |
| **6** | _(计划外)_ | mbridge 对 VL 风格名字 **MoE expert disk 导出返回空** (§5.3.2)。 |
| **7** | _(计划外)_ | 第一次 save 后 **`update_weights` 死锁 2 小时** (§5.3.3)。 |
| **8** | _(计划外)_ | 第二次 `update_weights` 在 **重复 expert index 上 AssertionError** (§5.3.4)。 |

#5–8 这些问题只在集成调试中才浮现。其中 #7、#8 用了多次迭代加上 NCCL Flight
Recorder dump 才定位到根因。

### 5.2 35B-A3B 架构约束

**通过 `inspect_qwen3_5_35b_config.sh` 于 2026-04-23 验证**:

| 参数 | 数值 |
|-----------|-------|
| num_attention_heads | 16 |
| **num_key_value_heads** | **2** (与 0.8B 相同) |
| head_dim | 256 |
| hidden_size | 2048 |
| num_hidden_layers | 40 |
| attn_output_gate | **true** |
| num_experts | 256 |
| num_experts_per_tok | 8 |
| moe_intermediate_size | 512 |
| HF key 前缀 | `model.language_model.layers.N.*` |

**对并行度的约束**:
- Actor TP ≤ 2 (与 0.8B 相同)
- vLLM TP ≤ 2 (强制: bf16 35B ≈ 70 GB 装不下单张 A100-80G)
- MoE: EP=8、ETP=1、EDP=1 (唯一可行布局, 见 §5.4)

### 5.3 问题与修复

#### 5.3.1 Ref 引擎 Offload 生命周期 (commit `a774cde`)

**现象**: 第一次 `ppo_update` OOM, 显存 79+/80 GB; `fused_adam._initialize_state` 失败。

**根因**: Ref 引擎和 actor 共享物理 GPU
(`scheduling_strategy: colocation target: actor`) 但作为独立的 Ray actor 运行。
`enable_offload: true` 给 actor 注入了 `torch_memory_saver` 环境 hook,
但 **trainer 从未在 `ref.initialize()` 之后调用过 `ref.offload()`**——
导致 ref 权重 (~17 GB/rank) 在 actor ppo_update 显存峰值期间还驻留在 GPU 上。

**设计**: 让 AReaL 的 ref 引擎显存生命周期与 veRL 的
`param_offload`/`optimizer_offload`/`grad_offload` 模式对齐 (ref 只需 param,
没有 optimizer/grad 要 offload)。在 `TrainController` 上加 `offload()` 和
`onload()` 接口, 在 RL trainer 中只在唯一需要 ref 的地方 (`compute_logp`)
前后调用它们。

**实现**:
1. `areal/infra/controller/train_controller.py`——通过 `_custom_function_call`
   暴露 `offload()` / `onload()`。
2. `areal/trainer/rl_trainer.py`:
   - `ref.initialize()` 之后, 如果 `config.enable_offload`, 立即 `ref.offload()`。
   - 把 `ref.compute_logp(rollout_batch)` 包在 `onload()` … `try:` … `finally: offload()` 里。

**结果**: 4 节点布局下 ppo_update 显存峰值从 79 GB → 73 GB。

#### 5.3.2 VL 包装名字下 Expert 导出的 Fallback (commits `8847c53`, `a9be24a`, `e49abe3`)

**现象**: 第一次迭代 disk 保存崩溃, 报
`ValueError: state_dict has 0 keys, but n_shards=2`,
位置在 expert shard 集合的 `split_state_dict_into_shards` 内。

**根因** (通过加诊断 logger 验证): docker 中的 mbridge `qwen3_5_moe` bridge
能接收 MoE expert 名字 (如
`language_model.decoder.layers.N.mlp.experts.linear_fc1.weightK`),
但 `bridge._weight_to_hf_format()` 对它们返回空的
`(converted_names, converted_params)`。诊断 (rank-0) 显示
`n_expert_specs=1280, n_empty_conversions=1280` (每条 spec 都返回空)。

mbridge 的纯文本 `qwen2_moe._weight_name_mapping_mlp()` 用
`layer_number = name.split(".")[2]` 取层号。对于 `decoder.layers.0.mlp...`,
index 2 是 `"0"` (正确)。对于 `language_model.decoder.layers.0.mlp...`,
index 2 是 `"layers"` (错误)。docker 的 bridge 对 expert MLP 名字有同样的缺陷,
即使它能正确处理非 expert 权重。

**设计**: 在 `save_weights_to_hf_with_mbridge_fast` 中实现本地 fallback——当
`model_type == "qwen3_5_moe"` 且 mbridge 返回空时, 使用 AReaL 已有的
Megatron→HF 命名约定写出 expert 权重。输出 key 必须匹配 vLLM 重新加载时的
期望: `model.language_model.layers.N.mlp.experts.M.{gate,up,down}_proj.weight`。

第一次尝试用了纯文本格式 (`model.layers.N...`), 导致 vLLM 在
`/areal_update_weights` 上返回 HTTP 400。`vllm_worker_extension.py` 中
`_summarize_checkpoint_keys` 的诊断显示
`sample_keys=['model.visual.blocks.0.attn.proj.bias', ...]`——确认是 VL HF 布局。
前缀已修正为 `model.language_model.layers.N...` (commit `a9be24a`)。

**实现** (`areal/models/mcore/hf_save.py` 节选):

```python
def _qwen3_5_moe_fallback_expert_export(global_name, merged_param):
    pattern = (
        r"(?:(?:module\.module|language_model)\.)?decoder\.layers\.(\d+)\."
        r"mlp\.experts\.(linear_fc[12])\.weight(\d+)"
    )
    match = re.fullmatch(pattern, global_name)
    if match is None:
        return [], []
    layer_idx, linear_name, expert_idx = match.groups()
    if linear_name == "linear_fc1":
        gate_weight, up_weight = merged_param.chunk(2, dim=0)
        return ([
            f"model.language_model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight",
            f"model.language_model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight",
        ], [gate_weight, up_weight])
    if linear_name == "linear_fc2":
        return ([
            f"model.language_model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight"
        ], [merged_param])
    return [], []
```

Fallback 仅在 `bridge._weight_to_hf_format` 返回空 **且**
`model_type == "qwen3_5_moe"` 时调用, 命中时打印诊断 log,
mbridge 与 fallback 都失败时硬抛 `ValueError`。

**结果**: 每个 rank `n_fallback_conversions=1280`; disk 模式 + vLLM reload
往返成功。

#### 5.3.3 default_pg 上的 Keepalive Ghost ALLREDUCE (commit `4958453`)

**现象 (v10–v15)**: 第一次 `update_weights` 完成后约 30 秒, 所有 actor worker
停止前进。两小时后 NCCL watchdog 触发, 每次都是同样的 fingerprint:

```
[Rank 0]  WorkNCCL(SeqNum=11, OpType=ALLREDUCE, NumelIn=1, NumelOut=1)
          ran for 7200001 ms before timing out
          last enqueued NCCL work: 11, last completed NCCL work: 10
[Rank 9]  last enqueued: 10, last completed: 10
```

Rank 0 在 `default_pg` 上比其他 rank **多入队了一个** ALLREDUCE。这个孤立 op
没有对端, 永远卡住整条队列。在 PP=2、PP=4、4 节点、6 节点布局下都复现。

**迭代调试**: 在找到真正的根因之前, 试过几个 "workaround", 单独都不能解决:

- `dc4ddea`、`7936fc9`: 在 `_update_weights_from_disk` 和
  `_save_hf` / `_save_recover_checkpoint` / `_evaluate_fn` / `_evaluate` /
  `_export_and_commit_stats` 的 gloo barrier 之后移除
  `current_platform.synchronize()`。让 trainer 越过
  "synchronize 卡在 pending NCCL 上"的 hang, 但孤立 op 仍会让下一次 NCCL
  collective 在别处死锁 (例如 `RayRPCServer.broadcast_tensor_container` 在
  `CONTEXT_AND_MODEL_PARALLEL_GROUP` 上)。

这些 workaround 保留, 因为它们让现象在真正的故障点暴露 (而不是在下游
synchronize), 而且 gloo CPU barrier 之后的 synchronize 本就冗余——
之前的调用已经在它自己出口处 synchronize 过了。

**根因排查** (commit `fa68ae0`): 给启动脚本加上 NCCL Flight Recorder:

```bash
export TORCH_NCCL_TRACE_BUFFER_SIZE=20000   # 后续重命名为 TORCH_FR_BUFFER_SIZE
export TORCH_NCCL_DESYNC_DEBUG=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
```

下一次故障 (v15) 输出了失败 collective 的 Python 调用栈:

```
#0 all_reduce              torch/distributed/distributed_c10d.py:3007
#1 wrapper                 torch/distributed/c10d_logger.py:83
#2 check                   areal/utils/timeutil.py:120
#3 _keepalive_thread_run   areal/utils/name_resolve.py:1190
#4 run                     threading.py:1010   ← 后台线程
```

**根因**: `name_resolve.add(name, value, keepalive_ttl=N)` 在
`_update_weights_from_disk` 内部仅 rank 0 调用。它注册了一个
`FrequencyControl(frequency_seconds=keepalive_ttl/3)` 作为 keeper, 并启动
后台 keepalive 线程, 周期性地调 `keeper.check()`。`check()` 里:

```python
if self.frequency_seconds is not None:
    if dist.is_initialized():
        interval_seconds = torch.tensor(...)
        dist.all_reduce(interval_seconds, op=MAX, group=self.group)  # group=None ⇒ default_pg!
```

`group=None` 解析为 **default_pg**, 包含全部 32 个 actor rank。但 keepalive
线程只在 rank 0 上存在, 其他 rank 永远不会入队对应的 op; rank 0 的 allreduce
会阻塞之后所有 default_pg collective。

这能解释故障 fingerprint 的每一个细节:

| 现象 | 机制 |
|---|---|
| `default_pg` 而非模型并行组 | `FrequencyControl(group=None)` ⇒ `WORLD` |
| `NumelIn=1 NumelOut=1` | `interval_seconds` 是标量 tensor |
| `OpType=ALLREDUCE` | 字面调用 |
| `SeqNum=11` 复现 | rank-0 每次训练都恰好多入队一个 default_pg op |
| `last_enqueued (rank 0) = last_completed + 1` | 单个未配对 op |
| `last_enqueued (其他 rank) = last_completed` | 它们从未入队这个 op |
| `name_resolve.add(keepalive_ttl=120)` 之后 ~30 s 触发 | 第一次 keepalive tick 在 `keepalive_ttl/3 = 40 s` |
| 在所有 PP/EP 布局下复现 | 与拓扑无关 |

**修复**:

1. `areal/utils/timeutil.py`——给 `FrequencyControl.__init__` 添加
   `disable_dist_sync: bool = False`。设为 True 时, `check()` 跳过
   `all_reduce`, 直接用本地的 `__interval_seconds`。
2. `areal/utils/name_resolve.py`——构造 keepalive 用的 `FrequencyControl` 时
   传 `disable_dist_sync=True`。Keepalive 是本地 NFS 租约刷新, 跨 rank 时间
   同步本来就毫无意义。

#### 5.3.4 mbridge `export_weights_buff` 重置 (commit `e225f80`)

**现象** (在 §5.3.3 修复之后): 第二次调用
`save_weights_to_hf_with_mbridge_fast` 在每个 rank 都崩溃:

```
File mbridge/models/qwen3_5/base_bridge.py:261, in _weight_to_hf_format
    assert experts_idx not in self.export_weights_buff[experts_key]
AssertionError
```

**根因**: docker 中的 `qwen3_5` mbridge bridge 把 expert tensor 累积到
`self.export_weights_buff[experts_key][experts_idx]` 里, 并对重复的
`experts_idx` assert。Buffer 在两次调用之间从未清空, 所以第二次调用时
每个 `(key, idx)` 都是重复值。

**修复**: 在 `save_weights_to_hf_with_mbridge_fast` 入口处, 防御性地清空
所有看起来像 buffer state 的 dict 类型属性, 不依赖 mbridge 版本:

```python
for buf_name in ("export_weights_buff", "_export_weights_buff"):
    buf = getattr(bridge, buf_name, None)
    if isinstance(buf, dict):
        buf.clear()
```

### 5.4 集群布局升级——4 节点 (PP=2) → 6 节点 (PP=4)

第一阶段原计划是 4 节点 PP=2。实测 PP=2 下 `ppo_update` 显存峰值达到
75.25/79.25 GB (94.9%), 留给 MoE 负载不均和权重更新瞬态的余量只有 ~4 GB。
增加 2 个节点改成 PP=4 是第一个真正能把峰值降下来的非配置类调整。
其他选项均被否决:

| 选项 | 否决原因 |
|---|---|
| `ppo_n_minibatches` 2→4 | 总样本数 = `batch_size×n_samples = 4×2 = 8`。切 4 个 minibatch 每个只有 2 < `DP=4`, 触发 `balanced_greedy_partition` 报错 ("Number of items must be >= K")。 |
| `max_tokens_per_mb` 4096→2048 | 已经是每微批 1 个序列。再砍半要切序列, 与 `pad_to_maximum=true` 不兼容 (Qwen3.5 GDN 强制 BSHD)。 |
| Adam `dtype=bf16` | 用户原则: optimizer states 保持 fp32 以保证精度。 |
| `vllm.gpu_memory_utilization 0.72→0.68` | 仅在 disk 模式 reload 窗口期内略有帮助。作为安全垫保留 (commit `83a83cb`)。 |
| Activation offload | 需要改 Megatron pipeline 调度器, 风险不低。 |
| Context Parallel (CP=2) | AReaL 上 Qwen3.5 GDN 与 CP 兼容性未测。 |

**最终 MoE 并行 mesh**:

```
World = 32 GPU = DP×PP×TP   (attention)   = 4×4×2  (最终 6 节点布局)
                = EP×ETP×PP×EDP (MoE)     = 8×1×4×1
                  EP=8 ⇒ 256 experts / 8 = 32 experts/rank, 负载均衡
                  ETP=1: moe_intermediate_size=512 太小, TP 切了不划算
                  EDP=1: 由 EP×ETP×PP = world 推出 ⇒ expert 不复制
```

`EP=16` 需要 `DP=1` (data parallelism 没了); `ETP>1` 把已经很小的
`moe_intermediate_size=512` 再砍半; `EDP>1` 让 expert 显存翻倍。

**最终 6 节点配置** (`fuyao_examples/math/qwen3_5_35b_a3b_rlvr_vllm_6node.yaml`):

```yaml
cluster: { n_nodes: 6, n_gpus_per_node: 8 }
enable_offload: true                              # 必须 true (§5.3.1)
actor:
  backend: "megatron:(attn:d4p4t2|ffn:e8t1)"     # 32 GPU
  pad_to_maximum: true                            # GDN 要求 BSHD
  weight_update_mode: disk                        # xccl 不兼容
  ppo_n_minibatches: 2                            # DP=4 + batch=4×n_samples=2 下的最大值
ref:
  backend: ${actor.backend}
  scheduling_strategy: { type: colocation, target: actor }
  pad_to_maximum: true
rollout:
  backend: "vllm:d8t2"                            # 16 GPU vLLM
vllm:
  max_num_seqs: 8
  max_model_len: 4096
  gpu_memory_utilization: 0.68
  enforce_eager: true
gconfig:
  n_samples: 2
  max_new_tokens: 2048
train_dataset: { batch_size: 4 }
```

### 5.5 验证结果

**最终任务**: `bifrost-2026042618245400-zengbw1` (label `qwen3_5-35b-a3b-v17-bufclr`)

| 阶段 | 4 节点 PP=2 基线 | **6 节点 PP=4 最终版** |
|---|---|---|
| `ppo_update` allocated | 44.25 GB | **21.27 GB** |
| `ppo_update` reserved | 47.30 GB | **24.94 GB** |
| 设备显存使用 | 75.25 GB (94.9%) | **38.58 GB (48.7%)** |
| Headroom | ~4 GB | **~40 GB** |
| `update_weights` step time | ~3.5 min | ~3.5 min (NFS 瓶颈, 不变) |
| 总 step time | ~6 min | ~6 min |
| 连续成功 `update_weights` 次数 | 0 (sync hang) | **20+ 次稳定** |

对比之下, 之前每个版本 (v10–v16) 都在 §5.3 描述的四类 bug 之一上挂掉,
最多撑过 1–2 次 `update_weights`。

### 5.6 上线后的超参对齐 + 序列长度推进 (v18–v22)

v17 跑通后, 围绕"对齐 8B benchmark 的超参风格"+"把序列长度推到 32K"两条
线推进。设计原则: **每次只动一个变量, 用上一版的实测显存数据指导下一版决策,
避免一步多改导致归因困难**。

#### 5.6.1 实验配置矩阵

所有版本均使用 `bifrost-2026042618245400-zengbw1` 验证过的 v17 基础设施
(6 节点 32+16 GPU, PP=4, EP=8, weight_update_mode=disk, enable_offload=true,
optimizer=adam fp32, pad_to_maximum=true)。变化仅在数据/序列维度:

| 版本 | yaml | batch | n_samples | seq (gen) | ppo_n_mb | max_concurrent | vllm.max_num_seqs | 对齐目标 |
|---|---|---|---|---|---|---|---|---|
| **v17** (baseline) | `_6node.yaml` | 4 | 2 | 2K | 2 | 8 | 8 | 显存验证 |
| **v18** (aligned) | `_6node_aligned.yaml` | 4 | **8** | 2K | **4** | **32** | **32** | n_samples → 8B 风格 |
| **v19** (16k+batch32) | `_6node_aligned_16k.yaml` | **32** | 8 | **16K** | **8** | **256** | **256** | 一步到位 (失败: OOM 风险高 + ray 集群多次超时) |
| **v20** (batch32) | `_6node_batch32.yaml` | **32** | 8 | 2K | **32** | **256** | **256** | batch 完全对齐 8B |
| **v21** (seq8k) | `_6node_seq8k.yaml` | 32 | 8 | **8K** (max_new_tokens=4K) | 32 | **128** | **128** | 序列 sanity check |
| **v22** (seq16k) | `_6node_seq16k.yaml` | 32 | 8 | **16K** | 32 | **32** | **32** | 中间步骤 (PP=4 不变) |

每 PPO 内每 rank 每 minibatch 的样本数始终保持在 v18 的等价水平
(`total_samples / ppo_n_minibatches / DP = 2 sample/rank`), 只通过同步放大
ppo_n_minibatches 来抵消 batch_size 放大, 这是显存稳定的关键技巧。

#### 5.6.2 实测显存数据 (PP=4 / 6 节点不变)

| 版本 | seq | total samples/step | actor 显存 (实测) | vLLM 显存 (实测) | 备注 |
|---|---|---|---|---|---|
| v17 | 2K | 8 | **38.58 GB / 80 GB** (48.7%) | ~40 GB | 60 step soak 零增长 |
| v18-r2 | 2K | 32 | **38.56 GB** | ~40 GB | n_samples 4× 不增显存 ✓ |
| v20 | 2K | **256** | **40 GB** | **56 GB** | batch 8× → vLLM KV cache 涨 (max_num_seqs=256) |
| **v21** | **8K** | 256 | **41 GB** | **55.5 GB** | seq 4× 几乎不涨 (与 v20 等价) ⭐ |
| v22 | 16K | 256 | (验证中) | (验证中) | 预估 actor 45-55 GB |

**关键洞察 (v21 数据颠覆原假设)**:

之前 megatron-engine-expert 调研时预测 "MoE alltoall dispatch buffer 随 token 数线性增长,
是 32K 路径的真正杀手"。v21 (8K, 4× v20 token) 实测显示 actor 显存仅
+1 GB, **MoE buffer 实际是 sublinear 的** (可能 EP=8 alltoall 实现有
fixed-size cudagraph buffer, 或 token routing 的 kernel 复用)。

这个发现直接简化了 32K 路径规划:

- **专家原方案**: v22 必须切到 PP=8 (DP=2, layers 40/8=5 整除), v23 32K
  必须 9 节点 (actor 64 + vLLM 8)
- **基于 v21 实测的新方案**: v22 直接 16K @ PP=4 不变; 若 v22 实测 actor
  < 50 GB, 则 v23 32K @ PP=4 也可能行, 不需要切 PP=8 / 加节点

#### 5.6.3 故障与基础设施问题

| 任务 | 故障 | 类型 | 处理 |
|---|---|---|---|
| v18 r1 | Ray 集群 4/6 节点超时 (400s) | 基础设施 | retry r2 成功 |
| v19 r1/r2/r3 | Ray 集群 4-5/6 节点超时 (400-900s) | 基础设施 | 多次 retry |
| 所有 6 节点任务 | 30% 概率出现节点未加入 Ray | fuyao 集群调度 | 暂以 retry 应对 |

`fuyao_areal_run.sh` 的 `MAX_WAIT=900s` 已经放宽过一次, 但仍有节点完全死掉
的情况。可考虑后续加申请 +1 容错节点的逻辑 (但需要绕过
`SLURM_JOB_NUM_NODES != CONFIG_CLUSTER_NODES` 强校验)。

### 5.7 update_weights 性能瓶颈分析 + 优化路径

#### 5.7.1 v17 实测 timeline

```
08:55:08  Starting actor.update_weights (step 58)
08:55:19  HF save synchronized                 ← _save_model_to_hf 仅 ~11s
08:58:50  actor.update_weights completed       ← 总 3:42
08:59:13  next step rollout pause              ← 训练步骤 ~23s (async overlap)
```

**update_weights 占整个 step time ~85%** (3:42 / 4:05)。拆分:

| 阶段 | 耗时 | 占比 |
|---|---|---|
| `_save_model_to_hf` (32 actor ranks 写 HF safetensor 到 NFS) | ~11 s | ~5% |
| vLLM reload (8 instance × 35 GB safetensor 从 NFS load) | **~226 s** | **~93%** |
| barrier + cleanup | ~5 s | ~2% |

瓶颈: **NFS 出口带宽被 8 个 vLLM instance × ~35GB 全量 reload 打死**, 不是
vLLM 内部串行加载。

#### 5.7.2 优化方向调研结论

调用 megatron-engine-expert + 后续 0.8B xccl debug 实测后, 完整 ROI 表:

| 方向 | 收益 | 工作量 | 状态 |
|---|---|---|---|
| **方向 1**: xccl + batch load_weights | 3.5min → **5-15s** | **2-4 小时** (本次实测后修正) | 根因已定位, 待实施 |
| 方向 2: vLLM 内并发优化 | ≈0 (NFS 瓶颈不在 vLLM) | – | 否决 |
| 方向 3-A: 更快共享存储 | 3.5min → 30-60s | 30 min (问运维) | 待问 |
| 方向 4: ZMQ+IPC | ≈方向 1 (实际部署不 colocate) | 1-2 周 | 否决 (跨节点 IPC 不可用) |
| 方向 5: 增量 sync | N/A (PPO 全量更新) | – | 否决 |
| 方向 6: update_weights 与下一 step prepare_batch 重叠 | 3.5min → 30-60s | 2-3 天 | 备选 |
| 方向 8: skip vision_model in disk save | -10 to -15s | 30 min | 顺手 |

#### 5.7.3 xccl 失败的精确根因 (本次会话核心发现)

**早期误判** (历史文档 §5 / megatron-engine-expert 第一轮):
> "xccl sender 直接写 vLLM 内部 buffer, 与 q||gate fused tensor 不兼容,
> 修复需要理解 vLLM 内部 buffer layout"

**实测驳斥** ([vllm_worker_extension.py:181](areal/engine/vllm_ext/vllm_worker_extension.py#L181)
xccl 路径**也是**通过 `model.load_weights` 注入, 不是直写 buffer。)

**精确定位** (0.8B xccl debug r2, `bifrost-2026042714355601-zengbw1`):

加 debug log 后捕获到失败的具体 tensor:

```
[xccl debug] load_weights FAILED at #8
  name='model.language_model.layers.0.linear_attn.conv1d.weight'
  shape=(3072, 1, 8) dtype=torch.bfloat16
RuntimeError: The expanded size of the tensor (4) must match the existing
              size (8) at non-singleton dimension 2.
              Target sizes: [2048, 1, 4]. Tensor sizes: [2048, 1, 8]

last_ok = 'model.language_model.layers.0.linear_attn.in_proj_a.weight'
```

**真正失败的不是 q_proj** (softmax attention 的 q||gate fused), **而是 GDN 层的
`conv1d.weight`**:

- mbridge 输出: `(3072, 1, 8)` — fused QKV (3072 = q_size + k_size + v_size),
  kernel-padded from 4 to 8 (Megatron causal-conv1d 内核对齐要求)
- vLLM 期望: `(2048, 1, 4)` — q-shard 单 stream, kernel=4

**根因 (megatron-engine-expert 第二轮调研, agent a532c61950e9f9c1d)**:

mbridge 的 `bridge.export_weights()` 和 `bridge._weight_to_hf_format()` per-tensor
**输出完全一致** (`bridge.py:497` 内部就是调 `_weight_to_hf_format`)。所以
"换 mbridge API" 解决不了问题。

真正差异在 **vLLM 端的调用方式**:

| 路径 | vLLM API | 调用粒度 | GDN conv1d 处理 |
|---|---|---|---|
| **AReaL disk** | `model_loader.load_weights(model, model_config=...)` | 通过 model_config 看完整 state_dict | model_loader 内部带 `packed_modules_mapping` 处理 fused tensor ✓ |
| **AReaL xccl** | `model.load_weights(weights=[(name, t)])` | **单个 tensor** | 没机会触发 packed loader 拆解, shape 直接 mismatch ❌ |
| **veRL** | `model.load_weights(weights=batch_iter)` | **批量** generator | vLLM 看到所有分量后按 packed_modules_mapping 派发 ✓ |

`vllm_worker_extension.py:184-227` 的 for-loop 每次只喂一个 tensor 给
`model.load_weights`, **绕开了 vLLM `packed_modules_mapping` 的 fused 派发机制**。
GDN conv1d 这种需要 vLLM 端 packed loader 拆给三个 sub-module 的张量, 在
单 tensor 调用下没法触发 fused loader 钩子。

**veRL 的 ZMQ+IPC 路径之所以能跑**, 不是因为 mbridge 输出格式特殊 (输出和
我们一致), 而是因为 vLLM 端调用粒度是 batch 的。

#### 5.7.4 推荐修复方案 (方向 1, 最像 veRL)

`vllm_worker_extension.py:184-227` 的 single-tensor for-loop:

```python
# 当前 (失败)
for name, dtype, shape in zip(names, dtypes, shapes):
    tensor = torch.empty(...)
    torch.distributed.broadcast(tensor, ...)
    self.model_runner.model.load_weights(weights=[(name, tensor)])  # ← 单个

# 修复 (D 方案, 对齐 veRL)
weights_buffer = []
for name, dtype, shape in zip(names, dtypes, shapes):
    tensor = torch.empty(...)
    torch.distributed.broadcast(tensor, ...)
    weights_buffer.append((name, tensor))
self.model_runner.model.load_weights(weights=weights_buffer)  # ← 一次性
```

**预期收益**: update_weights 3.5 min → 5-15s (NCCL broadcast 35 GB on IB 几秒;
免去 vLLM disk reload), step time 6 min → ~3 min, **~50% wall-clock 加速**。

**显存代价**: vLLM worker 端瞬态多持一份完整 weights (35B/TP=2 ≈ 35 GB)。
缓解: 按 `packed_modules_mapping` group 分批 broadcast (例如每个 layer 一批),
峰值 << 35 GB。

**工作量重估** (随调研推进收敛):
- 最初专家估计: 1-2 天 (基于 q_proj 误判)
- r2 实测后: 3-5 天 (基于"GDN tensor 全要 sender reshape"的中间假设)
- **本次根因清晰后: 2-4 小时** (只改 vllm_worker_extension.py 一个 for 循环)

### 5.8 CP (Context Parallel) 可行性调研

为了支持 32K 序列, 调研过 CP=2 (沿 sequence 维度切半 activation) 是否可行。

**结论: 硬阻断 (HARD BLOCK)**, 详见 megatron-engine-expert (agent
a6081ad957786433c) 的完整分析。

#### 5.8.1 决定性证据

[`areal/engine/megatron_engine.py:1643-1647`](areal/engine/megatron_engine.py#L1643-L1647)
的入口断言:

```python
if self.config.pad_to_maximum and cp_size > 1:
    raise ValueError(
        "pad_to_maximum=True is incompatible with context_parallel_size>1; "
        "CP split logic in packed_context_parallel_forward requires cu_seqlens."
    )
```

#### 5.8.2 完整因果链

1. AReaL 的 CP 实现 (`packed_context_parallel.py`) **完全依赖 packed (THD)
   格式**——用 `cu_seqlens` 切 sequence chunks 给每个 CP rank。
2. Qwen3.5 GDN 强制 `pad_to_maximum=true` (§3 表 #1: mbridge 的 GDN 需要
   `attention_mask`, AReaL 的 `pack_tensor_dict` 会移除它)。
3. GDN 是 stateful recurrent linear attention (`S_t = S_{t-1} + k_t v_t^T`),
   跨 rank 切 sequence 必须显式传递 recurrent state。docker 里的 mbridge
   0.15.1 GDN 模块**没有 CP-aware recurrent state ring-pass 实现**。
4. 32K 显存预估 (基于 v21 实测斜率): actor ~62-68 GB, 不切 PP 也接近
   80 GB 上限。**CP 是不加节点上 32K 的唯一方案**, 但被以上三点封死。

#### 5.8.3 替代方案 ROI 排序

| 方案 | 收益 | 工作量 | 备注 |
|---|---|---|---|
| **PP=4 → PP=8** (同 6 节点) | params/optim 减半 + activation/rank 减半 | 1h (改 backend 字符串) | 40 layers / 8 = 5 ✓ 整除 |
| 加节点到 8/9 (PP=8 + DP=4) | 同上 + DP 翻倍 | 中 (重新调度) | 9 节点 = 64 actor + 8 vLLM 是干净布局 |
| actor params/optim CPU offload | -16 GB | 1-2 周 | 长期工作, 不是 32K 应急 |
| ETP > 1 | MoE FFN 跨 TP 分摊 | 中 | `moe_intermediate_size=512`, ETP=2 后 256, 效率差 |

实际上 v21 实测显示 **MoE buffer sublinear**, 让上面"32K 必须切 PP"的预估
变得过于悲观——v22/v23 跑出实测后再确定是否真的需要切 PP=8 或加节点。

## 6. 代码改动汇总

所有提交在 `zbw-ai/AReaL-xpeng-from-zjh` 的 `main` 分支。

### 6.1 第一阶段 (0.8B 稠密)

| 文件 | 用途 |
|------|---------|
| `areal/engine/megatron_engine.py` | pad_to_maximum 核心 pipeline、VL position_ids 处理、VL vocab_size fallback、weight update 中跳过 vision_model、mbridge native conversion |
| `areal/engine/megatron_utils/deterministic.py` | `disable_qwen3_5_incompatible_fusions` 函数 |
| `areal/engine/megatron_utils/megatron.py` | conversion registry 注册 qwen3_5 model_type (非 VL fallback 路径) |
| `areal/utils/data.py` | `amend_position_ids` 的 masked_fill inplace 修复 |
| `areal/infra/rpc/rpc_server.py` | 改善 AttributeError 报告以便 debug |
| `fuyao_examples/fuyao_areal_run.sh` | 关 torch.compile、vLLM V1、no custom all-reduce 等环境变量 |
| `fuyao_examples/math/qwen3_5_0_8b_rlvr_vllm.yaml` | 0.8B 生产配置 |
| `fuyao_examples/math/qwen3_5_35b_a3b_rlvr_vllm.yaml` | 35B 4 节点配置 (保留作参考) |
| `fuyao_examples/inspect_mbridge.sh` | 集群上 inspect mbridge/Megatron 源码的脚本 |

### 6.2 第二阶段 (35B-A3B MoE)

| 分组 | 文件 | 改动 | Commit |
|---|---|---|---|
| §5.3.1 Ref offload | `areal/infra/controller/train_controller.py` | 通过 custom RPC 暴露 `offload()` / `onload()` | `a774cde` |
| | `areal/trainer/rl_trainer.py` | init 后 offload ref; 在 `compute_logp` 前后 onload/offload | `a774cde` |
| §5.3.2 Expert export | `areal/models/mcore/hf_save.py` | `_qwen3_5_moe_fallback_expert_export` + 诊断 logger | `8847c53`, `a9be24a` |
| | `areal/models/mcore/hf_save.py` | mbridge 空 conversion 的诊断 logger | `e49abe3` |
| §5.3.3 Keepalive sync | `areal/utils/timeutil.py` | `FrequencyControl` 加 `disable_dist_sync` 标志 | `4958453` |
| | `areal/utils/name_resolve.py` | keepalive `FrequencyControl` 传 `disable_dist_sync=True` | `4958453` |
| | `areal/engine/megatron_engine.py` | `_update_weights_from_disk` 里去掉 gloo barrier 之后冗余的 `synchronize()` | `7936fc9` |
| | `areal/trainer/rl_trainer.py` | `_save_hf` / `_save_recover_checkpoint` / `_evaluate*` / `_export_and_commit_stats` 同样处理 | `dc4ddea` |
| | `fuyao_examples/fuyao_areal_run.sh` | NCCL Flight Recorder 环境变量 (默认开) | `fa68ae0` |
| §5.3.4 mbridge buffer | `areal/models/mcore/hf_save.py` | 函数入口处重置 `export_weights_buff` | `e225f80` |
| §5.4 布局 | `fuyao_examples/math/qwen3_5_35b_a3b_rlvr_vllm_6node.yaml` | 6 节点 PP=4 生产配置 | `13d3c15` |
| | `fuyao_examples/math/qwen3_5_35b_a3b_rlvr_vllm.yaml` | vLLM mem 0.72→0.68 安全垫 | `83a83cb`, `0950848` |

第二阶段总计: ~12 commit、~7 个生产源文件、1 个新 YAML、env 更新。
所有改动**与 Qwen3.5 无强绑定**——它们是受益于任何后续 MoE/VL 模型上线的
通用 fix (尤其是 §5.3.1 和 §5.3.3)。

### 6.3 第三阶段 (上线后优化迭代)

| 分组 | 文件 | 改动 | Commit |
|---|---|---|---|
| **§5.6 超参对齐 8B** | `fuyao_examples/math/qwen3_5_35b_a3b_rlvr_vllm_6node_aligned.yaml` | v18: n_samples 2→8, ppo_n_minibatches 2→4, weight_decay 0.01→0.1 | `62c8c97` |
| | `fuyao_examples/math/deploy_qwen3_5_35b_aligned.sh` | v18 一键启动脚本 | `62c8c97` |
| | `fuyao_examples/math/qwen3_5_35b_a3b_rlvr_vllm_6node_batch32.yaml` | v20: batch 4→32, ppo_n_minibatches 4→32, max_concurrent/max_num_seqs 32→256 | `3929744` |
| **§5.6 序列推进** | `fuyao_examples/math/qwen3_5_35b_a3b_rlvr_vllm_6node_aligned_16k.yaml` | v19: batch=32 + seq=16K (一步到位, OOM 风险) | `573b9ab` |
| | `fuyao_examples/math/qwen3_5_35b_a3b_rlvr_vllm_6node_seq8k.yaml` | v21: 8K seq sanity check, max_num_seqs 256→128 | `63c4f6e` |
| | `fuyao_examples/math/qwen3_5_35b_a3b_rlvr_vllm_6node_seq16k.yaml` | v22: 16K seq @ PP=4 不变 (基于 v21 实测乐观推进) | `ee829d3` |
| **§5.7 xccl 调研** | `areal/engine/vllm_ext/vllm_worker_extension.py` | xccl 路径加 `[xccl debug]` 逐 tensor trace + `_summarize_checkpoint_keys` (disk 路径) | `44b4615` |
| | `fuyao_examples/math/qwen3_5_0_8b_rlvr_vllm_xccl_debug.yaml` | 0.8B xccl 复现配置 | `44b4615` |
| | `areal/engine/vllm_ext/vllm_worker_extension.py` | 修复 `logger.info(..., flush=True)` TypeError (logging.Logger 不接受 flush kwarg) | `0995f59` |

第三阶段总计: ~6 commit, 全部聚焦在 yaml 配置 + 1 处 vllm extension debug。
**xccl 实际修复 (方向 1, 方案 D 批量 load_weights) 待实施**, 详见 §9.1。

## 7. 复现命令

```bash
# 0.8B 验证 (第一阶段)
fuyao deploy --disable-fault-tolerance \
    --docker-image=infra-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/data-infra/fuyao:areal-qwen3_5-megatron-v21 \
    --project=rc-ai-infra --experiment=zengbw1/llm_rl \
    --gpu-type a100 --gpus-per-node 8 --node=1 \
    --label=qwen3_5-0_8b-rlvr \
    --site=fuyao_b1 --queue=rc-llmrl-a100 \
    SWANLAB_API_KEY=<key> \
    bash fuyao_examples/fuyao_areal_run.sh \
        --run-type math_rlvr \
        --config fuyao_examples/math/qwen3_5_0_8b_rlvr_vllm.yaml

# 35B-A3B v17 baseline (验证稳定, batch=4 n_samples=2 seq=2K)
fuyao deploy --disable-fault-tolerance \
    --docker-image=infra-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/data-infra/fuyao:areal-qwen3_5-megatron-v21 \
    --project=rc-ai-infra --experiment=zengbw1/llm_rl \
    --gpu-type a100 --gpus-per-node 8 --node=6 \
    --label=qwen3_5-35b-a3b-rlvr \
    --site=fuyao_b1 --queue=rc-llmrl-a100 \
    SWANLAB_API_KEY=<key> \
    bash fuyao_examples/fuyao_areal_run.sh \
        --run-type math_rlvr \
        --config fuyao_examples/math/qwen3_5_35b_a3b_rlvr_vllm_6node.yaml

# 35B-A3B v20 batch=32 (对齐 8B benchmark batch 维度, seq 仍 2K, 推荐)
fuyao deploy --disable-fault-tolerance \
    --docker-image=infra-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/data-infra/fuyao:areal-qwen3_5-megatron-v21 \
    --project=rc-ai-infra --experiment=zengbw1/llm_rl \
    --gpu-type a100 --gpus-per-node 8 --node=6 \
    --label=qwen3_5-35b-a3b-batch32 \
    --site=fuyao_b1 --queue=rc-llmrl-a100 \
    SWANLAB_API_KEY=<key> \
    bash fuyao_examples/fuyao_areal_run.sh \
        --run-type math_rlvr \
        --config fuyao_examples/math/qwen3_5_35b_a3b_rlvr_vllm_6node_batch32.yaml

# 35B-A3B v22 长上下文 (16K seq + batch=32, PP=4 不变)
fuyao deploy --disable-fault-tolerance \
    --docker-image=infra-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/data-infra/fuyao:areal-qwen3_5-megatron-v21 \
    --project=rc-ai-infra --experiment=zengbw1/llm_rl \
    --gpu-type a100 --gpus-per-node 8 --node=6 \
    --label=qwen3_5-35b-a3b-seq16k \
    --site=fuyao_b1 --queue=rc-llmrl-a100 \
    SWANLAB_API_KEY=<key> \
    bash fuyao_examples/fuyao_areal_run.sh \
        --run-type math_rlvr \
        --config fuyao_examples/math/qwen3_5_35b_a3b_rlvr_vllm_6node_seq16k.yaml
```

## 8. 经验教训

1. **新模型上线时, NCCL Flight Recorder 应当默认开启**。§5.3.3 的
   ghost ALLREDUCE 仅看日志根本看不出来——只表现为通用的 2 小时 barrier
   timeout, 可归因到十几种可能。Flight Recorder 的调用栈把数小时的猜测变成
   30 秒的诊断。

2. **`dist.all_reduce` 中的 `group=None` 在可能仅由部分 rank 执行的代码里
   是正确性陷阱** (后台线程、rank-0-only 路径)。任何不显式带 `group=` 的
   `dist.*` 调用都要审计 "这是谁在调用?" 再假设全 rank 都参与。

3. **gloo CPU barrier 之后的 `current_platform.synchronize()` 是冗余的且
   会掩盖 bug**。如果某 rank 上有 pending NCCL op, synchronize 会在那里
   静默卡住, 死锁会在另一个位置才暴露。

4. **不要假设 mbridge 在多次调用之间是无状态的**。§5.3.4 的 buffer
   累积没有任何文档记载, 只能从 assert 中看出。在 AReaL 的调用点做防御性
   reset 是廉价的保险。

5. **VL 包装影响的不只是命名**。HF key 前缀差异
   (`model.layers.N.*` vs `model.language_model.layers.N.*`) 会通过 mbridge
   的 `_weight_name_mapping_*` (它用 `name.split(".")[2]`) 级联放大, 决定整个
   fallback 路径。新模型上, 通过 `_summarize_checkpoint_keys` 取一个非 expert
   权重 key 来确认格式是最快的方法。

6. **35B 以上规模 PP 升级是值得的**。PP=2→PP=4 不仅把 per-rank 显存砍半,
   还把 per-rank expert 数量砍半 (1280 → 640 specs), 进而成比例缩小 save 时
   NCCL 流量。

7. **超参对齐时, 同步放大 ppo_n_minibatches 是显存稳定的关键**。v18-v20
   把 batch_size × n_samples 从 8 升到 256 (32×) 时, ppo_n_minibatches
   也从 2 升到 32, 维持 `total_samples / ppo_n_minibatches / DP` 不变,
   每 rank 每 minibatch 仍是 2 sample, **显存峰值实测和 v17 一致**
   (38.56 GB). 这个技巧让 batch 维度的对齐不带显存代价。

8. **MoE alltoall buffer 实测是 sublinear 的, 不要被理论预估吓住**。
   megatron-engine-expert 第一轮预测 "32K 必然爆炸", v21 (8K, 4× v20
   token) 实测 actor 显存仅 +1 GB。EP=8 alltoall 实现可能有 fixed-size
   cudagraph buffer 或 token routing kernel 复用, 实际系数远低于理论
   linear。**新模型/大序列实测优先于纯理论估算**。

9. **debug log 自身的 bug 会掩盖真实信号**。0.8B xccl debug r1 在
   `logger.info(..., flush=True)` 触发 `TypeError: Logger._log() got an
   unexpected keyword argument 'flush'`, 这个 TypeError 被外层
   try/except 吞了, 让我们一开始误以为是真实 load_weights 失败。修
   debug log 时**避免和被诊断代码混入异常路径**, 或至少用 `logging`
   而非 `print` 风格签名。

10. **xccl 失败的真正根因不一定是 sender 端的 tensor 形状**。AReaL 文档
    最初记述 "xccl 直接写 vLLM 内部 buffer" 是不准确的。xccl 也走
    `model.load_weights`, 真正问题是**调用粒度**: 单 tensor 调用绕开了
    vLLM 的 packed_modules_mapping fused 派发, 让需要 vLLM 端拆解的
    fused tensor (q_proj+attn_gate, gate_up_proj, GDN conv1d) 直接撞进
    sub-module 的单 tensor weight_loader。修复方案是 **batch
    load_weights** (对齐 veRL), 不是 sender 端 reshape。

## 9. 后续工作

1. **🔥 xccl 优化 (方案 D 批量 load_weights)** (2-4 小时, 高 ROI):
   `vllm_worker_extension.py:184-227` 的 single-tensor for-loop 改成
   "先收齐所有 (name, tensor) 再一次性 `model.load_weights(weights=
   batch)`", 让 vLLM 的 packed_modules_mapping 处理 GDN conv1d /
   q||gate 等 fused tensor 的拆解。**预期 update_weights 3.5 min →
   5-15s, step time 6 min → ~3 min, 50% wall-clock 加速**。
   先在 0.8B (`qwen3_5_0_8b_rlvr_vllm_xccl_debug.yaml`) 验证, 通过后
   迁到 35B-A3B。详见 §5.7.

2. **update_weights 与下一 step prepare_batch 重叠** (方向 6, 备选,
   2-3 天): 若 xccl 修复有困难, 可改 `_update_weights_from_disk` 拆成
   `_save_and_dispatch` (同步 11s) + `_await_pending_reload` (异步),
   trainer 主循环在 save 完成后立即开始下一 step, 让 vLLM reload 的
   226s 与 prepare_batch 重叠。**不依赖 xccl, 收益 ~50%**。

3. **MoE expert 导出贡献回上游 mbridge**: 把
   `_qwen3_5_moe_fallback_expert_export` 语义贡献回 mbridge, 让后续
   mbridge 版本原生处理 VL 前缀。

4. **ZMQ+IPC 路径** (1-2 周): 把 veRL 的 ZMQ/IPC 权重传输移植到 AReaL.
   注意: actor/vLLM 在当前部署下**不在同一节点 colocate**, CUDA IPC
   不跨节点, 实际收益等价方案 1 (xccl batch), 工作量 5-10×。
   **当前不推荐**, 除非未来改 colocate 部署。

5. **32K 序列推进 (v23)**: 待 v22 (16K @ PP=4) 实测后决定。
   - v22 actor < 50 GB → v23 32K @ PP=4 也可能行, 不需要切 PP=8
   - v22 actor 50-60 GB → v23 必须切 PP=8 (40 layers / 8 = 5 整除)
   - v22 actor > 60 GB → 需要加节点到 8/9 (actor 64 + vLLM 8 干净布局)

6. **Eval 限流**: `evaluator.freq_steps=20` 触发整个 `valid_dataset`
   一次, eval 边界处占用大量 wall-clock (v17 实测 evaluator 占 15+
   小时中的 ~10 小时)。增加 `evaluator.max_samples` 或用 256 样本
   切片做训练中 eval。

7. **长跑稳定性**: v17 已验证 60+ step / 21+ 小时, 接近 production-ready。
   建议跑 24 小时 soak (≥200 step) 才能向第三方用户宣称稳定。

8. **CP=2 + GDN 兼容性** (硬阻断, 见 §5.8): 当前不可行, 需要 mbridge
   GDN 模块加 CP-aware recurrent state ring-pass + AReaL 加 BSHD CP
   path。工作量等同新 feature, 不在 short-term 计划。

9. **Ray 集群启动健壮性**: fuyao 6 节点任务约 30% 概率出现节点未加入,
   `MAX_WAIT=900s` 下仍超时。可改 `fuyao_areal_run.sh` 容错: 申请
   N+1 节点, 等到 N 个 join 即开始, 留 1 节点冗余。但需要绕过
   `SLURM_JOB_NUM_NODES != CONFIG_CLUSTER_NODES` 强校验。

10. **Tree attention + Qwen3.5**: 未测试; 如启用很可能需要额外修复。

11. **FP8 训练**: `attn_output_gate` + FP8 当前不兼容 (代码里 assert
    死); 需要 mbridge 上游支持。
