# AReaL Fuyao Examples

AReaL 框架在 fuyao 环境下的训练配置，支持三个实验场景。
训练在本地执行（本机 GPU），fuyao SDK 仅用于按需部署远程沙盒服务。

## 快速开始

### 场景 1: Math RLVR (Qwen3 4B)

单轮数学推理 RL，使用 dapo_math_17k 数据集。无需沙盒。

```bash
bash fuyao_examples/fuyao_areal_run.sh \
    --run-type math_rlvr \
    --config fuyao_examples/math/qwen3_4b_rlvr.yaml
```

### 场景 2: Search R1 Agentic RL (Qwen3 4B Base)

多轮搜索增强 RL，使用 NQ search 数据集。**需要检索服务。**

```bash
# 方式 A: 自动通过 fuyao SDK 部署检索服务
bash fuyao_examples/fuyao_areal_run.sh \
    --run-type search_r1 \
    --config fuyao_examples/agentic/search_r1_qwen3_4b.yaml

# 方式 B: 使用已有检索服务
RETRIEVAL_ENDPOINT=http://10.1.x.x:8001/retrieve \
bash fuyao_examples/fuyao_areal_run.sh \
    --run-type search_r1 \
    --config fuyao_examples/agentic/search_r1_qwen3_4b.yaml \
    --skip-deploy
```

### 场景 3: Code DAPO Agentic RL (Qwen3 4B)

多轮代码执行 RL，使用 dapo_math_17k 数据集。默认 local subprocess 执行。

```bash
# 默认模式: local subprocess
bash fuyao_examples/fuyao_areal_run.sh \
    --run-type code_dapo \
    --config fuyao_examples/agentic/code_dapo_qwen3_4b.yaml

# 可选: 使用远程 execd 沙盒
EXECD_ENDPOINT=http://10.1.x.x:39524 \
bash fuyao_examples/fuyao_areal_run.sh \
    --run-type code_dapo \
    --config fuyao_examples/agentic/code_dapo_qwen3_4b.yaml
```

## 环境变量

| 变量 | 用途 | 必需 |
|------|------|------|
| `SWANLAB_API_KEY` | SwanLab 实验追踪 API key | 推荐 |
| `RETRIEVAL_ENDPOINT` | 检索服务地址 (search_r1) | search_r1 必需 |
| `EXECD_ENDPOINT` | 代码执行服务地址 (code_dapo) | 可选 |
| `AUTH_USER` | fuyao SDK 用户身份 | 部署沙盒时必需 |

## 数据集

| 场景 | 训练集 | 验证集 |
|------|--------|--------|
| Math RLVR | `/workspace/.../data/dapo_math_17k_processed` | 同上 |
| Search R1 | `/workspace/.../code/Search-R1/data/nq_search/` | `/workspace/.../data/HotpotQA` |
| Code DAPO | `/workspace/.../data/dapo_math_17k_processed` | 同上 |

## 沙盒部署

手动部署沙盒服务（通过 fuyao SDK）：

```bash
# 部署检索服务
python fuyao_examples/deploy_sandboxes.py --sandbox search --export

# 部署代码执行服务
python fuyao_examples/deploy_sandboxes.py --sandbox code --export

# 部署并获取 endpoint
eval $(python fuyao_examples/deploy_sandboxes.py --sandbox search --export)
echo $RETRIEVAL_ENDPOINT
```

## Dockerfile

Fuyao 适配对应的镜像文件是仓库根目录下的 `areal_fuyao.dockerfile`。

```bash
docker build -f areal_fuyao.dockerfile .
```

当前 Dockerfile 按 **vLLM-first** 路径准备运行时依赖，包含：

- `uv sync --extra cuda`
- `pip install --ignore-installed blinker`
- `pip3 install -U fuyao-all --extra-index-url http://nexus-wl.xiaopeng.link:8081/repository/ai_infra_pypi/simple --trusted-host nexus-wl.xiaopeng.link`
- `swanlab`
- `httpx`

## 目录结构

```
fuyao_examples/
├── fuyao_areal_run.sh              # 统一启动脚本
├── deploy_sandboxes.py             # fuyao SDK 沙盒部署
├── math/
│   └── qwen3_4b_rlvr.yaml         # 场景1 配置
├── agentic/
│   ├── search_r1_qwen3_4b.yaml    # 场景2 配置
│   └── code_dapo_qwen3_4b.yaml    # 场景3 配置
└── README.md                       # 本文件
```

## SwanLab 指标

训练指标自动映射到 DeepInsight 命名空间：

| AReaL 原始指标 | DeepInsight 指标 |
|----------------|-----------------|
| `ppo_actor/task_reward/avg` | `deepinsight_algorithm/reward` |
| `ppo_actor/pg_loss` | `deepinsight_algorithm/policy_loss` |
| `timeperf/step/total` | `deepinsight_infra/step_time` |

Agentic 场景额外指标：`tool_use_count`, `tool_use_success`, `num_turns`, `search_latency_ms`
