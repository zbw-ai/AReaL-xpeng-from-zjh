# Qwen3-8B 后训练路线图：Math RLVR → Code → Agentic RL
> 版本：v1.1 | 作者：zengbw | 日期：2026-04-09
> 实验模型：Qwen3-8B | 集群：2 节点 × 8 GPU (A100 80G) | 聚焦：math 单任务先行

---

## 1. 需求内容
### 1.1 需求目标
基于 Qwen3-8B（已完成 SFT 的 checkpoint），通过 RLVR → Agentic RL 多阶段后训练，复现业界已验证的训练路线，实现模型通用推理、代码生成和工具使用能力的提升。
### 1.2 验证目标
- Phase 1（核心验证）：使用 Qwen3-8B SFT checkpoint，在 math 任务上跑通 RLVR 全流程，reward 曲线稳步上升，GSM8K/MATH/AIME 有可观测提升
- Phase 2（能力扩展）：在 Phase 1 checkpoint 基础上扩展到代码执行（Code DAPO）和搜索增强（Search R1）agentic 场景
- Phase 3（通用对齐）：可选，通过 DPO/RLHF 对齐通用偏好，提升安全性和有用性
### 1.3 参考方案
| 方案 | 来源 | 核心贡献 | 链接 |
|---|---|---|---|
| DAPO | ByteDance Seed + 清华 | Clip-Higher / Dynamic Sampling / Token-Level Loss / Overlong Mask | [arXiv:2503.14476](https://arxiv.org/abs/2503.14476) |
| POLARIS | ByteDance Seed | 700 步 RL 让 4B 模型逼近 235B 数学推理 | [GitHub](https://github.com/ChenxinAn-fdu/POLARIS) |
| Skywork-OR1 | 昆仑万维 | MAGIC entropy 调度解决 entropy collapse | [GitHub](https://github.com/SkyworkAI/Skywork-OR1) |
| DeepSeek-R1 | DeepSeek | 纯 RL 涌现推理 + 四阶段训练流水线 | [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) |
| Search-R1 | 密歇根大学 | 推理中自主搜索的 RL 训练 | [GitHub](https://github.com/PeterGriffinJin/Search-R1) |
| VAPO | ByteDance Seed | Value-augmented PPO，AIME24 达 60.4 | [arXiv:2504.05118](https://arxiv.org/abs/2504.05118) |

---

## 2. 需求分析
### 2.1 技术背景
**RLVR（Reinforcement Learning with Verifiable Rewards）** 是当前 LLM 推理能力提升的主流方案。核心思路是：用可自动验证的信号（数学答案正确性、代码执行结果）作为奖励，通过 RL 训练将模型的 pass@k 能力压缩到 pass@1。

DeepSeek-R1 证明了纯 RL 可以从 base model 涌现推理能力（R1-Zero），但也指出先做 SFT 再做 RL 效果更好——SFT 提供格式和基础推理链，RL 负责优化策略。DAPO 进一步在 GRPO 基础上解决了四个核心问题（entropy collapse、长度偏差、截断噪声、采样低效），用 50% 步数超越 R1-Zero。

**关于 math-only 是否影响全局训练**：业界证据表明，在数学上训练的推理能力可以泛化到其他领域。POLARIS 发现逻辑谜题上训练可以迁移到数学；DeepSeek-R1 的数学 RL 训练也提升了代码和 STEM 能力。因此，先在 math 上验证再扩展到多领域，是一条被广泛验证的正确路径——不会"浪费"训练，反而是效率最高的验证方式。
### 2.2 技术挑战
| 挑战 | 说明 |
|---|---|
| Entropy Collapse | GRPO/DAPO 训练中策略 entropy 急剧下降到 ~0，模型失去探索能力，reward 停滞。这是 RL 训练失败的最常见原因 |
| SFT 充分性判断 | SFT 不充分会导致 RL 初期格式混乱、reward 极低；SFT 过度会导致模型过拟合、RL 难以跳出局部最优。需要明确的停止标准 |
| 奖励函数正确率区间 | 奖励函数在训练集上的正确率需要在 20%-80% 范围：太低无学习信号，太高无挑战。需要在训练前评估 |
| 从 math 到 agentic 的迁移 | 单轮 math RLVR 到多轮 agentic RL，workflow 架构不同（RLVRWorkflow vs OpenAIProxyWorkflow），reward 归因方式不同，需要验证 checkpoint 兼容性 |
| Base model 能力上限 | 最新研究表明 RLVR 并非凭空创造推理能力，而是压缩 pass@k → pass@1。如果 base model 的 pass@64 本身很低，RL 增益有限 |
### 2.3 约束条件
- 部署环境：fuyao 集群，A100/H800 80G GPU
- 基础框架：AReaL（当前仓库），训练后端 Megatron + SGLang
- 实验跟踪：SwanLab
- 提交入口：`fuyao_examples/fuyao_areal_run.sh`
- 实验模型：**Qwen3-8B-Base + SFT**（已有 SFT checkpoint：`qwen3_8b_base_ot3_sft_0105/global_step_4450`）
- 选择 Base 而非 Instruct 的理由：更高的探索多样性 + 无 RLHF alignment tax（详见 3.3 节）
- 集群规模：2 节点 × 8 GPU = 16 GPU（A100 80G）

---

## 3. 方案设计
### 3.1 整体训练路线

![Training Roadmap](images/01_training_roadmap.png)

### 3.2 "先 math 后扩展" 策略的合理性
| 维度 | math-only 先行 | 多任务同时训练 |
|---|---|---|
| 验证速度 | 快：1 节点 8GPU，无外部服务依赖 | 慢：需要搜索服务、代码沙箱 |
| 调试难度 | 低：单轮生成，reward 可解释 | 高：多轮交互，reward 归因复杂 |
| 能力迁移 | 有：数学推理可泛化到代码/逻辑 | — |
| 资源开销 | 低：dapo_math_17k 仅 17K 样本 | 高：多数据集、多服务 |
| 风险 | 低：可在 1-2 天内验证完整流程 | 高：任何环节出错都阻塞全局 |

结论：**math-only 是正确的第一步**。POLARIS 证明 4B 模型仅 700 步 RL 就能在 AIME24 达到 81.2%，说明 math 任务足以验证 RL pipeline 的有效性。后续扩展到 agentic 场景时，math RLVR checkpoint 是更好的起点（比 base model 有更强的推理基础）。
### 3.3 起始模型选择：Base vs Instruct

**推荐：Qwen3-8B-Base + SFT → RLVR**（使用已有 SFT checkpoint）

| 维度 | Qwen3-8B-Base (+ SFT) | Qwen3-8B (instruct) |
|---|---|---|
| 数学基线 | GSM8K 89.84, MATH 60.80 (base) | MATH-500 95.16% (thinking), 43.55% (non-thinking) |
| 探索多样性 | 高：未经 RLHF 压缩，entropy 保持完整 | 低：RLHF 导致 40-79% 问题只产生单一语义簇 |
| KL penalty | 可完全去掉 (kl_ctl=0.0)，训练更简洁 | 通常需要保留，防止偏移过大 |
| pass@k 上限 | 更高：base 模型在大 k 下反而超过 RL 后的 instruct | 较低：RL 后 pass@k 反而可能收窄 |
| 格式稳定性 | 需要 SFT 解决（已有 checkpoint） | 原生支持 `<think>` 模式 |
| 社区验证 | 数学推理 RL 的主流选择 | POLARIS (4B instruct) 验证可行 |

**选择 Base 的三个核心理由：**

1. **探索空间更大**：Instruct 模型经过 RLHF/DPO 后，输出多样性被严重压缩（研究表明单一语义簇比例从 base 的 1% 升到 instruct 的 28.5%）。RLVR 本质是搜索压缩（pass@k → pass@1），起点的 pass@k 越高，RL 能压缩出的增益越大。Base 模型的 pass@k 上限更高。

2. **训练更简洁**：Base 模型做 RL 可以完全移除 KL penalty（kl_ctl=0.0），减少一个需要调的超参数。Instruct 模型因为已有 RLHF 对齐，通常需要保留 KL 约束防止灾难性遗忘，增加了调参复杂度。

3. **已有 SFT checkpoint 补齐格式短板**：Base 模型唯一的劣势是格式不稳定，但我们已有内部 SFT checkpoint（`qwen3_8b_base_ot3_sft_0105/global_step_4450`），这恰好提供了"base 模型的探索潜力 + SFT 的格式稳定性"的最佳组合，同时没有 RLHF 的 alignment tax。

**备选方案**：如果 SFT checkpoint 质量不满意（答案提取成功率 < 90% 或格式混乱），可以回退到 Qwen3-8B（instruct），跳过 SFT 直接 RL。POLARIS 验证了从 instruct 直接 RL 同样有效。

**参考数据**：社区从 Qwen3-8B-Base 出发做 SFT+GRPO 的实验结果：
| 阶段 | GSM8K | AIME24 |
|---|---|---|
| Base | 61.92 | 10.0 |
| +SFT | 63.48 | 13.3 |
| +SFT+GRPO | 83.59 | 16.7 |

（注：上述为 LoRA 消费级 GPU 结果，full finetune + 更多 GPU 预期更好）

### 3.4 其他关键设计决策
| 决策 | 选择 | 理由 |
|---|---|---|
| 起始模型 | Qwen3-8B-Base + SFT checkpoint | 高探索空间 + 格式稳定，无 RLHF alignment tax |
| RL 算法 | GRPO（kl_ctl=0.001，当前 8B 配置） | 先用已验证的 GRPO 跑通，再考虑切换 DAPO |
| 数据集 | dapo_math_17k | 已在仓库中集成，数据量适中，覆盖多难度等级 |
| KL 惩罚 | kl_ctl=0.001（保守起步）→ 0.0（跑稳后尝试） | 因为用的是 base+SFT，理论上可以完全去掉 KL |
| 采样数 | n_samples=8 | 8B 配置已设定，平衡效率和多样性 |
| 集群 | 2 节点 16 GPU | Actor 8GPU (node 0) + SGLang 8GPU (node 1) |

---

## 4. 需求实现
### 4.1 已有基础设施
当前仓库已有完整的 math RLVR 训练基础设施，可直接复用：

**训练 pipeline：**
| 组件 | 路径 | 状态 |
|---|---|---|
| 训练入口 | `fuyao_examples/math/train_math_rlvr.py` | 已完成 |
| 启动脚本 | `fuyao_examples/fuyao_areal_run.sh --run-type math_rlvr` | 已完成 |
| 数据集加载 | `fuyao_examples/dataset/dapo_math.py` | 已完成 |
| 奖励函数 | `fuyao_examples/reward.py` (math_reward_fn) | 已完成 |
| Workflow | `areal/workflow/rlvr/RLVRWorkflow` | 框架内置 |
| Trainer | `areal/trainer/rl_trainer.py` (PPOTrainer) | 框架内置 |

**已有配置（三个模型规模）：**
| 配置 | 路径 | GPU | 节点 |
|---|---|---|---|
| Qwen3-4B | `fuyao_examples/math/qwen3_4b_rlvr.yaml` | 8 | 1 |
| Qwen3-8B | `fuyao_examples/math/qwen3_8b_rlvr.yaml` | 16 | 2 |
| Qwen3-30B-A3B | `fuyao_examples/math/qwen3_30b_a3b_rlvr.yaml` | 32 | 4 |

**Agentic 扩展（已实现，Phase 3 使用）：**
| 组件 | 路径 | 状态 |
|---|---|---|
| Code DAPO Agent | `fuyao_examples/code_dapo/code_exec_agent.py` | 已完成 |
| Search R1 Agent | `fuyao_examples/search_r1/search_r1_agent.py` | 已完成 |
| Code DAPO 配置 | `fuyao_examples/code_dapo/code_dapo_qwen3_4b.yaml` | 已完成 |
| Search R1 配置 | `fuyao_examples/search_r1/search_r1_qwen3_4b.yaml` | 已完成 |
### 4.2 需要新增/调整的工作
| 工作项 | 说明 | 优先级 |
|---|---|---|
| 启动 8B RLVR 快速验证 | 使用 `qwen3_8b_rlvr.yaml`，先跑 2 epochs 观察指标 | P0 |
| Entropy 监控 | 在 SwanLab 中确认 entropy 指标追踪正常，及时发现 entropy collapse | P0 |
| Reward 正确率基线 | 用 SFT checkpoint 在 dapo_math_17k 上跑 pass@1，确认在 20%-80% 区间 | P0 |
| 独立验证集评估 | 确保 eval 用了独立数据（非训练集），监控泛化而非记忆 | P1 |
| 高奖励样本回灌 | RLVR 跑稳后，筛选高 reward 样本回灌 SFT，作为下一轮 RL 更好的起点 | P2 |
| Code DAPO 8B 适配 | 基于 RLVR checkpoint，适配 code_dapo 配置到 8B 并行策略 | P2 |

---

## 5. 实验配置
### 5.1 Phase 0/1: 跳过（已有 SFT checkpoint）
当前使用的 Qwen3-8B 已经完成了 SFT 阶段（checkpoint: `qwen3_8b_base_ot3_sft_0105/global_step_4450`），**直接进入 Phase 2 RLVR**。

如未来需要从 base model 重新训练，Phase 0/1 流程参见 Decision Tree：

![Decision Tree](images/04_decision_tree.png)

### 5.2 Phase 2: Math RLVR — Qwen3-8B（核心阶段）
**输入**: Qwen3-8B SFT checkpoint
**输出**: 数学推理能力显著提升的 checkpoint
**训练框架**: AReaL `PPOTrainer` + `RLVRWorkflow`
**配置文件**: `fuyao_examples/math/qwen3_8b_rlvr.yaml`

#### 5.2.1 完整实验配置（基于 qwen3_8b_rlvr.yaml）
**模型与数据：**
| 参数 (YAML path) | 值 | 说明 |
|---|---|---|
| `actor.path` | `/dataset_rc_b1/.../qwen3_8b_base_ot3_sft_0105/global_step_4450/huggingface` | SFT checkpoint |
| `train_dataset.type` | dapo_math | 17K 数学题，parquet 格式 |
| `train_dataset.batch_size` | 16 | 每步 16 个 prompt |
| `total_train_epochs` | 10 | 约 10870 步/epoch |

**生成配置：**
| 参数 (YAML path) | 值 | 说明 |
|---|---|---|
| `gconfig.n_samples` | 8 | 每个 prompt 生成 8 条回复 |
| `gconfig.max_new_tokens` | 8192 | 最大生成长度 |
| `gconfig.temperature` | 0.99 | 高温采样促进多样性 |
| `gconfig.top_p` | 0.99 | nucleus sampling |
| `gconfig.top_k` | 100 | top-k 采样 |

**训练核心超参数：**
| 参数 (YAML path) | 值 | 说明 |
|---|---|---|
| `actor.optimizer.lr` | 1.0e-6 | 学习率 |
| `actor.optimizer.weight_decay` | 0.1 | DAPO 推荐，防止参数极端值 |
| `actor.optimizer.beta1` | 0.9 | Adam 动量 |
| `actor.optimizer.beta2` | 0.999 | Adam 二阶矩 |
| `actor.optimizer.lr_scheduler_type` | cosine | 余弦退火 |
| `actor.optimizer.gradient_clipping` | 1.0 | 梯度裁剪 |
| `actor.optimizer.warmup_steps_proportion` | 0.01 | 暖启比例 |
| `actor.eps_clip` | 0.2 | PPO clip ratio |
| `actor.kl_ctl` | 0.001 | 轻量 KL 约束（GRPO 模式） |
| `actor.reward_scaling` | 1.0 | 不缩放 reward |
| `actor.reward_bias` | 0.0 | 不偏置 reward |
| `actor.behave_imp_weight_cap` | 2.0 | importance weight 裁剪 |
| `actor.behave_imp_weight_mode` | token_mask | token 级别 IS 修正 |
| `actor.use_decoupled_loss` | true | token-level loss（符合 DAPO 要求） |
| `actor.adv_norm` | batch-level mean+std | advantage 归一化 |
| `actor.reward_norm` | group-level (group_size=8) | 组内 reward 归一化 |

**并行策略：**
| 参数 (YAML path) | 值 | 说明 |
|---|---|---|
| `actor.backend` | megatron:d2t2p2 | DP2 × TP2 × PP2 = 8 GPU (node 0) |
| `rollout.backend` | sglang:d4p1t2 | DP4 × TP2 = 8 GPU (node 1) |
| `ref.scheduling_strategy` | colocation with actor | Ref 与 Actor 共置 |
| `sglang.mem_fraction_static` | 0.85 | SGLang 静态显存 |
| `sglang.disable_custom_all_reduce` | true | 避免 TP2 下 custom_all_reduce 崩溃 |

![AReaL RLVR Architecture](images/02_rlvr_architecture.png)

#### 5.2.2 集群资源配置（2 节点 × 8 GPU）
| 角色 | GPU 数 | 节点 | 并行策略 | 说明 |
|---|---|---|---|---|
| Actor (Megatron) | 8 | node 0 | DP2 × TP2 × PP2 | PP2 减半单卡显存，为 Ref 腾空间 |
| SGLang Rollout | 8 | node 1 | DP4 × TP2 | 静态显存 85%，context=16384 |
| Ref Model | 0 | node 0 | colocation with actor | 与 Actor 共置，冻结权重 |

#### 5.2.3 Reward 设计（渐进式）
**第一版 reward（先跑通）：**
- 答案正确 = 1，错误 = 0（二值 reward）
- 使用现有 `fuyao_examples/reward.py` 的 `math_reward_fn`
- 通过 math_verify 库验证 `\boxed{...}` 中提取的答案

**第二版 reward（跑稳后可选加入）：**
| 辅助项 | 权重 | 说明 |
|---|---|---|
| 格式合法奖励 | 0.1 | 鼓励 `<think>` 标签 |
| 过长惩罚 | -0.1 × (len/max_len) | 防止输出无限增长 |

注意：一开始不要把 reward 做复杂，否则不知道到底是哪一项起作用。

#### 5.2.4 数据集要求
dapo_math_17k 包含多难度等级的数学题。需要确保训练集里有足够比例的"模型会一部分但做不稳"的题（即 reward 在 20%-80% 区间）。

如果模型在训练集上已经 > 80% 正确率，说明题目太简单，reward 没梯度，RLVR 不会明显提升。需要补充更难的题目。

#### 5.2.5 DAPO 进阶调整（效果不好时再考虑）

![DAPO vs GRPO](images/03_dapo_vs_grpo.png)

当前 8B 配置使用 GRPO（kl_ctl=0.001），这是更保守稳定的选择。如果跑通后想进一步提升，可以尝试切换到 DAPO：
| DAPO 技术 | 对应 AReaL 配置 | 说明 |
|---|---|---|
| Clip-Higher | 需新增 `eps_clip_high` = 0.28 | **[待确认]** AReaL 是否支持双 clip 参数 |
| Dynamic Sampling | rollout 层过滤 reward std=0 的样本组 | **[待确认]** AReaL 是否支持 |
| Token-Level Loss | `actor.use_decoupled_loss: true` (已开启) | 已符合 DAPO 要求 |
| 移除 KL | `actor.kl_ctl: 0.0` | 从 0.001 降到 0.0 |

### 5.3 RLVR 跑稳后的优化闭环
RLVR 跑稳后有两条可选优化路径，优先考虑回灌 SFT：

**路径 A: 高奖励样本回灌 SFT（推荐）**
```
RLVR checkpoint → 用 RL 后的 policy 在训练集上 rollout
  → 筛选 reward=1 的高质量样本
  → 回灌做新一轮 SFT（数据质量 > 数据数量）
  → 再用新 SFT checkpoint 做第二轮 RLVR
```
这是 DeepSeek-R1 验证过的路线：R1-Zero → 收集数据做 SFT → 再 RLVR，效果好于单纯延长 RL。

**路径 B: 轻量 DPO（RL 不稳时优先补）**
DPO 在这里的作用不是增强数学能力，而是：
- 拉齐输出风格，减少废话
- 提高格式稳定性
- 降低 RLVR 的 reward 方差

如果发现 RL 不稳（reward 震荡、格式混乱），最先补的通常就是一层轻量 DPO，而不是猛调 RL 超参。

### 5.4 后续扩展路线（Phase 2 跑稳后）
Phase 2 跑稳的标志：reward 曲线在 100 步内开始上升，独立验证集准确率持续提升，entropy 不崩溃。

跑稳后按以下顺序扩展，**每次只加一个任务**：

**Step 2: 加 code（最自然的扩展，reward 同样可验证）**
```
math-stabilized checkpoint
  → code SFT（少量高质量代码解题数据）
  → code RLVR（unit test / compile / execution 作为 verifier）
  → 混合 math + code 训练
```
配置：`fuyao_examples/code_dapo/code_dapo_qwen3_4b.yaml`（需适配 8B）

**Step 3: 加 tool use**
```
math+code checkpoint
  → tool-use SFT
  → short-horizon tool RL
```
配置：`fuyao_examples/search_r1/search_r1_qwen3_4b.yaml`（需适配 8B）

**Step 4: Agentic RL（最后）**
```
multi-turn environment → short horizon → longer horizon
```

---

![Monitoring Dashboard](images/05_monitoring.png)

---

## 6. 实验结果
### 6.1 已有基线数据
**Qwen3-4B infra 测试参考**（来自 infra-reports/20260407，4B 单节点）：
| 指标 | 4B 值 | 8B 预期 |
|---|---|---|
| 任务完成 | 通过（46 步无崩溃） | 待验证 |
| 单步耗时 | ~19s | 预计更长（TP2+PP2 通信开销） |
| GPU 利用率 | 44.52% | 待验证 |
| sample_staleness | 1.8-2.0 | 预计相近（相同 offpolicyness=2） |

**Qwen3-8B 预估**：2 节点 16GPU，模型参数量是 4B 的 2 倍，TP2+PP2 通信增加，预计单步耗时 30-50s。建议先用 `total_train_epochs=2` 快速验证。

### 6.2 业界对标基线
| 模型 | 算法 | AIME24 | MATH-500 | GSM8K | 训练步数 | 来源 |
|---|---|---|---|---|---|---|
| Qwen2.5-32B + DAPO | DAPO | 50.0 | — | — | ~5000 | DAPO 论文 |
| Qwen2.5-32B + VAPO | VAPO | 60.4 | — | — | ~5000 | VAPO 论文 |
| Qwen3-4B + POLARIS | DAPO 变体 | 81.2 | — | — | 700 | POLARIS |
| Qwen2.5-32B + Skywork-OR1 | GRPO+MAGIC | — | 72.8 (avg) | — | — | Skywork-OR1 |
| DeepSeek-R1-Zero-Qwen-32B | GRPO | 47.0 | — | — | — | R1 论文 |

### 6.3 效果验证计划
**[待补充]** 以下指标将在实验执行后填入：

**Phase 2 (Math RLVR) 预期追踪指标：**
| 指标 | 健康范围 | 异常信号 |
|---|---|---|
| reward_mean | 持续上升，最终 > 0.3 | 停滞或下降 |
| entropy | 缓慢下降，始终 > 0.5 | 急剧降到 ~0 = entropy collapse |
| response_length_mean | 稳定或缓慢增长 | 无限增长 = reward hacking |
| clip_fraction | 0.1-0.3 | > 0.5 = 学习率过大 |
| pass@1 (eval) | 逐步提升 | 上升后大幅回落 = 过拟合 |

---

## 7. 实验结论
**[待补充]** 实验执行后填入具体结论。

### 7.1 预期验证项（Qwen3-8B Math RLVR）
| 验证项 | 成功标准 | 状态 |
|---|---|---|
| 8B RLVR pipeline 跑通 | 2 节点 16GPU 训练无崩溃，完成 2 epochs | 待验证 |
| reward 曲线上升 | reward_mean 在 100 步内开始上升 | 待验证 |
| Entropy 不崩溃 | 训练全程 entropy > 0.5 | 待验证 |
| 独立验证集准确率提升 | eval accuracy 相比 SFT baseline 提升 > 5% | 待验证 |
| 输出格式稳定 | 答案可从 `\boxed{}` 中提取，提取成功率 > 95% | 待验证 |

---

## 8. 复现指南
### 8.1 实验执行步骤（Qwen3-8B Math RLVR）

#### Step 1: 快速验证（2 epochs）
```bash
# 确认模型和数据路径可访问
ls /dataset_rc_b1/llm_train_sft/lijl42/SFT_Qwen3_8B_Base/qwen3_8b_base_ot3_sft_0105/global_step_4450/huggingface
ls /workspace/zhangjh37@xiaopeng.com/data/dapo_math_17k_processed

# 启动 8B Math RLVR (2 节点, 16 GPU)
# 先改 total_train_epochs=2 做快速验证
bash fuyao_examples/fuyao_areal_run.sh \
    --run-type math_rlvr \
    --config fuyao_examples/math/qwen3_8b_rlvr.yaml \
    --swanlab-api-key $SWANLAB_API_KEY
```

#### Step 2: 观察 SwanLab 指标（前 100 步）
```
必须看的 4 个指标:
1. reward_mean      — 是否上升？（100 步内应开始上升）
2. entropy          — 是否崩溃？（不应急降到 ~0）
3. response_length  — 是否爆炸？（不应无限增长）
4. eval accuracy    — 独立验证集准确率是否提升？

训练 reward 在升 ≠ 模型真的变强，一定要看验证集！
```

#### Step 3: 根据观察决定下一步
```
情况 A: reward 上升 + entropy 稳定 + eval 准确率提升
  → 成功！调整 total_train_epochs=10，完整训练
  → 训练完成后，筛选高奖励样本，回灌做新一轮 SFT（可选）

情况 B: reward 停滞不动
  → 检查 reward 函数：单独测试 math_reward_fn，确认正确率在 20%-80%
  → 如果正确率 > 80%：题目太简单，需要补充更难的题
  → 如果正确率 < 20%：SFT checkpoint 太弱，考虑更多 SFT
  → 增加 n_samples: 8 → 16

情况 C: entropy 急剧下降到 ~0
  → 降低 lr: 1e-6 → 5e-7
  → 增大 gradient_clipping: 1.0 → 0.1（更激进裁剪）
  → 如果仍崩溃，考虑切换到 DAPO (见 5.2.5)

情况 D: response_length 无限增长
  → reward hacking 信号，加入长度惩罚
  → 或增加 max_new_tokens 给模型更多空间
```

#### Step 4: 完整训练后扩展（Phase 2 跑稳后）
```bash
# 后续扩展到 code（需要先适配 8B 配置）
# 1. 复制 code_dapo_qwen3_4b.yaml → code_dapo_qwen3_8b.yaml
# 2. 修改 actor.path 指向 Phase 2 checkpoint
# 3. 调整并行策略为 d2t2p2 / sglang:d4p1t2
bash fuyao_examples/fuyao_areal_run.sh \
    --run-type code_dapo \
    --config fuyao_examples/code_dapo/code_dapo_qwen3_8b.yaml \
    --swanlab-api-key $SWANLAB_API_KEY
```

### 8.2 超参调优优先级

![Hyperparameter Tuning Priority](images/06_hyperparam_priority.png)
当训练效果不好时，按以下优先级调整：

**第一优先级（影响最大）：**
| 参数 | 8B 当前值 | 调整方向 | 预期效果 |
|---|---|---|---|
| `gconfig.n_samples` | 8 | 8 → 16 | 增加采样多样性，reward 方差更小 |
| `actor.optimizer.gradient_clipping` | 1.0 | 1.0 → 0.1 | 稳定训练，防止梯度爆炸 |
| `actor.optimizer.lr` | 1e-6 | 1e-6 → 5e-7 | 更保守的更新 |

**第二优先级（精细调整）：**
| 参数 | 8B 当前值 | 调整方向 | 预期效果 |
|---|---|---|---|
| `actor.optimizer.beta2` | 0.999 | 0.999 → 0.99 | 适合 RL 噪声梯度 |
| `actor.kl_ctl` | 0.001 | 0.001 → 0.0 | 移除 KL（DAPO 风格），允许更大策略更新 |
| `actor.behave_imp_weight_cap` | 2.0 | 2.0 → 1.5 | 更严格的 importance weight 裁剪 |

**第三优先级（架构调整）：**
| 参数 | 8B 当前值 | 调整方向 | 预期效果 |
|---|---|---|---|
| `gconfig.max_new_tokens` | 8192 | 8192 → 16384 | 允许更长推理链 |
| `rollout.max_head_offpolicyness` | 2 | 2 → 1 | 牺牲吞吐换取更 on-policy 的训练 |

### 8.3 最容易踩的三个坑
1. **题目太简单**：大部分题模型已经会了，reward 没梯度。需要分层（easy/medium/hard），确保训练集有足够"会一部分但做不稳"的题。
2. **只看训练 reward**：训练 reward 在升不代表模型真的变强，可能只是学会了模板化输出。**一定要看独立验证集准确率。**
3. **过度追求长 CoT**：不是越长越好。需要的是有效推理 + 可验证结果 + 稳定格式，不是把输出拉得特别长。

### 8.4 关键监控指标
| 指标 | 健康范围 | 异常信号 | 应对措施 |
|---|---|---|---|
| reward_mean | 持续上升 | 100 步内无上升 | 检查奖励函数；增加 n_samples |
| entropy | 缓慢下降，> 0.5 | 急降到 ~0 | 增大 eps_clip_high（如支持）；降低 lr |
| response_length_mean | < max_new_tokens 的 80% | 持续增长接近上限 | 增加 max_new_tokens；检查是否 reward hacking |
| clip_fraction | 0.1-0.3 | > 0.5 | 降低 lr；增加 gradient_clipping |
| sample_staleness | < max_head_offpolicyness | 持续接近上限 | 增加 rollout 并发；减少 train batch |
| pass@1 (eval) | 逐步提升 | 上升后大幅回落 | 降低 lr；提前停止 |

---

## 附录 A: 业界训练路线对比

### A.1 已验证的成功路线
| 路线 | 起始模型 | 阶段 | 成果 | 资源 |
|---|---|---|---|---|
| POLARIS | Qwen3-4B instruct | 直接 DAPO RL (700 步) | AIME24: 81.2% | 8 GPU |
| Skywork-OR1 | Qwen2.5-32B base | SFT (10 epoch) + GRPO+MAGIC | 72.8% avg | 多节点 |
| DeepSeek-R1 | DeepSeek-V3 base | R1-Zero(GRPO) + SFT + RLVR + RLHF | AIME24: 79.8% | 大规模 |
| Open-R1 | Qwen2.5-32B | SFT(OpenR1-Math) + GRPO | 匹配 R1 蒸馏模型 | 多节点 |

### A.2 当前选定路线（Qwen3-8B，聚焦 math 单任务打通闭环）
```
Qwen3-8B SFT checkpoint (已完成)
  → Math RLVR (dapo_math_17k, GRPO, 2 节点 16GPU)
  → 验证：reward 上升 + eval 准确率提升 + entropy 稳定
  → 高奖励样本回灌 SFT（可选，提升下一轮 RL 起点）
  → 扩展到 code RLVR（可验证 reward：unit test / compile）
  → 扩展到 tool use（短 horizon）
  → 最后进 agentic RL（多轮，长 horizon）
```

核心原则：**每次只加一个任务，跑稳后再扩展。先用 math 验证训练闭环。**

---

## 附录 B: 核心参考资源

### 论文
| 论文 | 关键内容 | 链接 |
|---|---|---|
| DAPO | Clip-Higher, Dynamic Sampling, Token-Level Loss, Overlong Mask | [arXiv:2503.14476](https://arxiv.org/abs/2503.14476) |
| DeepSeek-R1 | 纯 RL 涌现推理，GRPO 算法详解 | [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) |
| VAPO | Value-augmented PPO，AIME24 SOTA | [arXiv:2504.05118](https://arxiv.org/abs/2504.05118) |
| Kimi K1.5 | 128K 上下文 RL，Partial Rollouts | [arXiv:2501.12599](https://arxiv.org/abs/2501.12599) |
| Entropy Mechanism | RL 训练中 entropy 与 reward 的数学关系 | [arXiv:2505.22617](https://arxiv.org/abs/2505.22617) |
| Search-R1 | 推理中自主搜索的 RL 训练 | [arXiv:2503.09516](https://arxiv.org/abs/2503.09516) |

### 开源项目
| 项目 | 特点 | 链接 |
|---|---|---|
| POLARIS | 4B 模型 700 步达 AIME24 81.2% | [GitHub](https://github.com/ChenxinAn-fdu/POLARIS) |
| Skywork-OR1 | MAGIC entropy 调度 + 开源数据集 | [GitHub](https://github.com/SkyworkAI/Skywork-OR1) |
| Open-R1 | DeepSeek-R1 系统性复现 | [GitHub](https://github.com/huggingface/open-r1) |
| DAPO 代码 | 基于 veRL 的 DAPO 实现 | [GitHub](https://github.com/BytedTsinghua-SIA/DAPO) |
| veRL | 最成熟的开源 RL 训练框架 | [GitHub](https://github.com/volcengine/verl) |

### 调优指南
| 指南 | 内容 | 链接 |
|---|---|---|
| PPO → GRPO → DAPO 全参数解读 | 每个超参数的作用和推荐值 | [Blog](https://blog.softmaxdata.com/from-ppo-to-grpo-to-dapo-understanding-rl-for-llms-and-every-training-parameter-explained/) |
| 16 个框架深度对比 | 异步 RL 框架选型 | [HuggingFace](https://huggingface.co/blog/async-rl-training-landscape) |
| veRL GRPO+LoRA 工程手册 | 工程实践经验 | [HuggingFace](https://huggingface.co/blog/Weyaxi/engineering-handbook-grpo-lora-with-verl) |
| Awesome-RLVR | RLVR 论文合集 | [GitHub](https://github.com/opendilab/awesome-RLVR) |
| Awesome-RL-for-LRMs | RL for LLM 论文合集 | [GitHub](https://github.com/TsinghuaC3I/Awesome-RL-for-LRMs) |
