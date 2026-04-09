# Qwen3-8B Math RLVR 实验记录
> 创建日期：2026-04-09 | 状态：P0 基线评测中 | 负责人：zengbw

---

## 1. 实验概述
### 1.1 目标
在 Qwen3-8B 上打通 Math SFT → RLVR → 轨迹回灌 完整后训练闭环，验证 RL 能持续提升数学推理能力。详见 [roadmap.md](roadmap.md)。
### 1.2 核心路线
```
Qwen3-8B Instruct → Math SFT → RLVR → 轨迹筛选回灌 → 二轮 RLVR → code 扩展
```

---

## 2. 模型资产

### 2.1 待评测模型
| 编号 | 名称 | 路径 | 说明 |
|---|---|---|---|
| M1 | qwen3-8b-base | `/publicdata/huggingface.co/Qwen/Qwen3-8B-Base` | 预训练基座，无任何后训练 |
| M2 | qwen3-8b-instruct | `/publicdata/huggingface.co/Qwen/Qwen3-8B` | 官方 post-trained 版本，支持 thinking/non-thinking 双模式 |
| M3 | qwen3-8b-sft3905 | [待确认路径] | 内部 SFT checkpoint (global_step_3905) |
| M4 | qwen3-8b-sft4450 | `/dataset_rc_b1/llm_train_sft/lijl42/SFT_Qwen3_8B_Base/qwen3_8b_base_ot3_sft_0105/global_step_4450/huggingface` | 内部 SFT checkpoint (global_step_4450)，当前 RLVR 配置默认模型 |

### 2.2 模型关系
```
Qwen3-8B-Base (M1)
  │
  ├── 官方后训练 ──→ Qwen3-8B Instruct (M2)
  │                  (SFT + RLHF + GRPO，官方完整流水线)
  │
  └── 内部 SFT ────→ sft3905 (M3) ──→ sft4450 (M4)
                     (中间 checkpoint)    (最终 checkpoint)
```

---

## 3. 数据资产

### 3.1 训练数据
| 数据集 | 路径 | 规模 | 说明 |
|---|---|---|---|
| deepmath_math_rule_20k | `/workspace/lijl42@xiaopeng.com/datasets/hard_prompts/20251203_for_rl/data/deepmath_math_rule_20k.parquet` | ~20K | RL 训练主题池，硬数学题 |
| dapo_math_17k | `/workspace/zhangjh37@xiaopeng.com/data/dapo_math_17k_processed` | ~17K | 已集成到 AReaL 的训练数据 |

### 3.2 评测数据
| 数据集 | 路径/来源 | 题数 | 难度 | 角色 |
|---|---|---|---|---|
| aime_2024 | `/workspace/lijl42@xiaopeng.com/data_process/20251205_rl_data/aime_2024_rule.parquet` | 30 | 高 | 训练验证集 + 业界对标指标 |
| aime_2025 | [待确认] | 30 | 高 | 防 2024 过拟合 |
| math_500 | [待确认] | 500 | 中 | 主指标，综合数学能力 |
| gsm8k | [待确认] | ~1319 | 低 | 基础能力不退化的底线 |
| gpqa_diamond | [待确认] | [待确认] | 很高 | 推理泛化到科学领域 |
| livecodebench_v5 | [待确认] | [待确认] | 中高 | code 不退化监控 |

### 3.3 评测数据集用途分配
| 用途 | 数据集 | 频率 |
|---|---|---|
| 训练中 eval（每 N 步） | aime_2024 | 每 20 步 |
| 阶段结束全量评测 | 全部 6 个 | P0 / P1 / P3 / P4 / P5 各跑一次 |
| 核心对标指标 | math_500 + aime_2024 | 所有报告必须包含 |
| 基础能力退化监控 | gsm8k | 如果掉分说明"偏科" |

---

## 4. P0 基线评测

### 4.1 评测矩阵
| DataSet \ Model | qwen3-8b-base (M1) | qwen3-8b-instruct (M2) | qwen3-8b-sft3905 (M3) | qwen3-8b-sft4450 (M4) |
|---|---|---|---|---|
| gsm8k | | | | |
| math_500 | | | | |
| aime_2024 | | | | |
| aime_2025 | | | | |
| gpqa_diamond | | | | |
| livecodebench_v5 | | | | |

### 4.2 评测设置
| 参数 | 值 | 说明 |
|---|---|---|
| 推理模式 | thinking (M2); standard (M1/M3/M4) | M2 有 thinking 模式，需测两种 |
| temperature | 0.0 (greedy) | 基线用 greedy，pass@k 用高温 |
| max_new_tokens | 8192 | 与 RLVR 配置一致 |
| 答案提取 | `\boxed{}` | 使用 math_verify 验证 |

### 4.3 关键问题（评测后回答）
1. **哪个模型做 RL 起点最好？** 看 aime_2024 + math_500 的综合表现
2. **SFT 是否有效？** 比较 M1 vs M3/M4，SFT 带来多少提升
3. **Instruct vs SFT？** 比较 M2 vs M4，谁更适合做 RL 起点
4. **SFT 训练是否饱和？** 比较 M3 vs M4，step 3905→4450 是否还有提升
5. **格式合规率？** 各模型 `\boxed{}` 出现率是否 > 95%
6. **基础能力如何？** GSM8K 作为底线，所有模型应该 > 80%

---

## 5. RLVR 训练配置

### 5.1 AReaL 配置
| 配置项 | 值 |
|---|---|
| 配置文件 | `fuyao_examples/math/qwen3_8b_rlvr.yaml` |
| 训练入口 | `fuyao_examples/math/train_math_rlvr.py` |
| 启动命令 | `bash fuyao_examples/fuyao_areal_run.sh --run-type math_rlvr --config fuyao_examples/math/qwen3_8b_rlvr.yaml` |
| 模型路径 | 由 P0 评测结果决定（M2 或 M4） |
| 训练数据 | deepmath_math_rule_20k 或 dapo_math_17k |
| 验证集 | aime_2024 (30 题，每 20 步 eval) |

### 5.2 集群资源
| 角色 | GPU | 节点 | 并行策略 |
|---|---|---|---|
| Actor (Megatron) | 8 | node 0 | DP2 × TP2 × PP2 |
| SGLang Rollout | 8 | node 1 | DP4 × TP2 |
| Ref Model | 0 | node 0 | colocation with actor |

### 5.3 核心超参数
| 参数 | 值 |
|---|---|
| `gconfig.n_samples` | 8 |
| `gconfig.max_new_tokens` | 8192 |
| `gconfig.temperature` | 0.99 |
| `actor.optimizer.lr` | 1.0e-6 |
| `actor.optimizer.weight_decay` | 0.1 |
| `actor.eps_clip` | 0.2 |
| `actor.kl_ctl` | 0.001 |
| `actor.behave_imp_weight_cap` | 2.0 |
| `total_train_epochs` | 2 (快速验证) → 10 (完整训练) |

---

## 6. 实验日志

### Run 0: P0 基线评测
- **日期**：2026-04-09
- **状态**：进行中
- **内容**：4 个模型 × 6 个数据集 = 24 组评测
- **结果**：[待填入]

---

## 7. 决策记录

| 日期 | 决策 | 理由 | 状态 |
|---|---|---|---|
| 2026-04-09 | 聚焦 Qwen3-8B，math 单任务先行 | 最快打通闭环，math reward 最干净 | 确认 |
| 2026-04-09 | 默认从 Instruct 起步 | POLARIS 验证可行；格式稳定性好；调试范围小 | 待 P0 数据确认 |
| 2026-04-09 | aime_2024 作为训练验证集 | 30 题，eval 成本低，业界对标指标 | 确认 |
| 2026-04-09 | 6 数据集评测体系 | gsm8k(底线) + math_500(主指标) + aime(对标) + gpqa(泛化) + livecode(code 监控) | 确认 |

---

## 8. 文件索引

| 文件 | 说明 |
|---|---|
| [roadmap.md](roadmap.md) | P0-P6 分阶段训练路线图 |
| [experiment-record.md](experiment-record.md) | 本文件：实验记录与数据汇总 |
| [01_training_roadmap.png](01_training_roadmap.png) | 训练路线全景图 |
| [02_rlvr_architecture.png](02_rlvr_architecture.png) | AReaL RLVR 训练架构图 |
| [03_dapo_vs_grpo.png](03_dapo_vs_grpo.png) | DAPO vs GRPO 算法对比图 |
| [04_decision_tree.png](04_decision_tree.png) | SFT 必要性决策树 |
| [05_monitoring.png](05_monitoring.png) | 训练监控指标看板 |
| [06_hyperparam_priority.png](06_hyperparam_priority.png) | 超参调优优先级图 |
| [generate_roadmap_diagrams.py](generate_roadmap_diagrams.py) | 图表生成脚本 |
