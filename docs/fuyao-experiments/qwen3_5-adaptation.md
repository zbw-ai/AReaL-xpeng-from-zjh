# AReaL 适配 Qwen3.5 VL RLVR — 适配报告

**日期**: 2026-04-23 → 2026-04-26
**作者**: zengbw
**状态**: Qwen3.5-0.8B ✅ 已验证 · **Qwen3.5-35B-A3B ✅ 已验证**(6 节点上 20+ 次稳定 update_weights)

本报告分两个阶段记录 AReaL 框架对 Qwen3.5 系列 VL 模型的端到端适配工作:

- **第一阶段**(§§1–4): Qwen3.5-0.8B 稠密模型上线——建立基础配方
  (pad_to_maximum、disk 模式、output-gate 处理、fusion 禁用)。
- **第二阶段**(§§5–8): Qwen3.5-35B-A3B (MoE) 模型上线——把基础配方扩展到
  MoE + VL 包装结构, 修复一系列在小规模稠密模型上不会出现的问题
  (ref colocate OOM、expert export 转换、keepalive ghost ALLREDUCE、
  mbridge buffer 重置)。

两个阶段共用同一个 docker 镜像、同一套环境变量前置, 启动脚本基本一致;
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

# 35B-A3B 生产 (第二阶段, 推荐)
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

## 9. 后续工作

1. **xccl Qwen3.5 对齐** (1-2 天): 改 AReaL xccl sender 在广播进 vLLM 内部
   buffer 之前先拆 q||gate。消除每次 `update_weights` ~3.5 分钟的 disk 模式
   往返。
2. **MoE expert 导出贡献回上游**: 把 `_qwen3_5_moe_fallback_expert_export`
   语义贡献回 mbridge, 让后续 mbridge 版本原生处理 VL 前缀。
3. **ZMQ+IPC 路径** (1-2 周): 把 veRL 的 ZMQ/IPC 权重传输移植到 AReaL,
   完全对齐 veRL 跑通的路径; 消除 disk IO。
4. **Eval 限流**: `evaluator.freq_steps=20` 触发整个 `valid_dataset` 一次,
   eval 边界处占用大量 wall-clock。增加 `evaluator.max_samples` 或用
   256 样本切片做训练中 eval。
5. **长跑稳定性**: 已验证 20 次 update; 建议跑 24 小时 soak (≥200 次)
   才能向第三方用户宣称生产可用。
6. **CP=2 + GDN 兼容性**: 能再砍 ~50% activation, 让 4 节点 PP=2 也有
   充足 headroom。
7. **Tree attention + Qwen3.5**: 未测试; 如启用很可能需要额外修复。
8. **FP8 训练**: `attn_output_gate` + FP8 当前不兼容 (代码里 assert 死);
   需要 mbridge 上游支持。
