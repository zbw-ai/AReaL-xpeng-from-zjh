# Fuyao Patch Changelog

AReaL fuyao 适配改造，支持三个实验场景在本地 GPU 环境运行。

## 新增文件

### 基础设施 (M1)

| 文件 | 功能 |
|------|------|
| `fuyao_examples/fuyao_areal_run.sh` | 统一启动脚本，支持 math_rlvr / search_r1 / code_dapo |
| `fuyao_examples/deploy_sandboxes.py` | 通过 fuyao SDK 部署远程沙盒 (search / code) |
| `areal/utils/tracking_patch.py` | DeepInsight SwanLab 指标映射 |

### 场景 1: Math RLVR (M2)

| 文件 | 功能 |
|------|------|
| `areal/dataset/dapo_math.py` | dapo_math_17k 数据加载器 (Parquet → AReaL RL 格式) |
| `fuyao_examples/math/qwen3_4b_rlvr.yaml` | Qwen3-4B Math RLVR 训练配置 |

### 场景 2: Search R1 (M3)

| 文件 | 功能 |
|------|------|
| `areal/workflow/search_r1.py` | SearchR1Workflow — 多轮搜索增强 RL |
| `examples/agentic/train_search_r1.py` | 训练入口脚本 |
| `examples/agentic/configs.py` | AgenticConfig 扩展配置类 |
| `fuyao_examples/agentic/search_r1_qwen3_4b.yaml` | Qwen3-4B-Base Search R1 训练配置 |

### 场景 3: Code DAPO (M4)

| 文件 | 功能 |
|------|------|
| `areal/workflow/code_exec.py` | CodeExecWorkflow — 多轮代码执行 RL (local + execd) |
| `examples/agentic/train_code_dapo.py` | 训练入口脚本 |
| `fuyao_examples/agentic/code_dapo_qwen3_4b.yaml` | Qwen3-4B Code DAPO 训练配置 |

### 文档 (M5)

| 文件 | 功能 |
|------|------|
| `fuyao_examples/README.md` | 使用说明 |
| `docs/fuyao_patch_changelog.md` | 本文件 |
| `areal_fuyao.dockerfile` | vLLM-first 的完整训练环境镜像 |

## 修改文件

| 文件 | 变更 |
|------|------|
| `areal/utils/stats_logger.py` | 集成 DeepInsight 指标映射 (`apply_metric_mapping`) |
| `areal/dataset/__init__.py` | 注册 `dapo_math` 数据集类型 |

## 新增依赖

| 库 | 版本 | 用途 | 来源 |
|----|------|------|------|
| `fuyao` + `xbigdata` | (集群预装) | fuyao SDK 沙盒管理 | M1 |
| `blinker` | ignore-installed 重装 | 修复当前 base image 的包冲突 | Docker runtime |
| `fuyao-all` | latest from nexus | 提供 `fuyao.py` / `xbigdata` 运行时 | Docker runtime |
| `swanlab` | latest | 实验追踪 | M1 (AReaL 已支持) |
| `httpx` | >=0.24 | Search R1 HTTP 请求 | M3 |
| `math_verify` | (AReaL 已含) | 数学答案验证 | M2/M4 |

## 已知限制

1. Search R1 场景必须有 RETRIEVAL_ENDPOINT，需要先部署检索服务
2. vLLM stop_strings 支持已接入实现，仍需结合真实 rollout 进一步验证（`</search>`, `</code>`）
3. 多节点训练暂未支持（仅单节点 8 GPU）
4. dapo_math_17k validation 使用同一数据集（可后续切分）
