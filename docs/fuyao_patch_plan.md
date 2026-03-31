# AReaL Fuyao Patch 改造计划 v2

> 目标：在 AReaL 框架上引入 fuyao 特有的启动命令与规整化启动流程，支持在 fuyao 集群上运行以下三个实验场景：
>
> 1. **Qwen3 4B Math RLVR** — 单轮数学推理 RL（dapo_math_17k 数据集）
> 2. **Qwen3 4B Search R1 Agentic RL** — 多轮搜索增强 RL（NQ search + HotpotQA 验证）
> 3. **Qwen3 4B Code DAPO Agentic RL** — 多轮代码执行 RL（dapo_17k_with_python_code）

---

## 一、确认事项汇总

| # | 问题 | 确认结果 |
|---|------|---------|
| 1 | 数学数据集 | **dapo_math_17k**，严格对齐 ROLL 配置，含 validation |
| 2 | Search R1 数据 | 训练: `/workspace/.../Search-R1/data/nq_search/` (question/golden_answers)；验证: `/workspace/.../data/HotpotQA` (problem/answer) |
| 3 | Code DAPO 数据 | `/workspace/.../data/dapo_math_17k_processed` (prompt/solution) |
| 4 | RETRIEVAL_ENDPOINT | **无默认地址**，必须跟着起检索服务器，参考 launch_agentic.sh |
| 5 | 代码执行模式 | 默认 local subprocess，同时实现 execd sandbox（可选） |
| 6 | fuyao SDK | **需要安装**，沙盒管理依赖 fuyao SDK |
| 7 | Qwen3-4B 版本 | 场景1/3: Instruct，场景2: Base |
| 8 | SwanLab | **需要**，含 DeepInsight 指标映射 + agentic 指标 (tool_use_count, tool_use_success) |
| 9 | 节点规模 | 先单节点 8 GPU |

---

## 二、目标文件结构

```
AReaL_xpeng/
├── fuyao_examples/                             # [新增] fuyao 配置目录
│   ├── fuyao_areal_run.sh                      # [M1] 统一启动脚本
│   ├── deploy_sandboxes.py                     # [M1] sandbox 部署 (复用 llm_train 适配)
│   ├── math/
│   │   └── qwen3_4b_rlvr.yaml                 # [M2] 场景1 配置
│   ├── agentic/
│   │   ├── search_r1_qwen3_4b.yaml             # [M3] 场景2 配置
│   │   └── code_dapo_qwen3_4b.yaml             # [M4] 场景3 配置
│   └── README.md                               # [M5] 使用说明
├── examples/
│   ├── math/gsm8k_rl.py                        # [复用] 场景1 入口（需微调支持 dapo_17k）
│   ├── agentic/                                # [已存在，补齐]
│   │   ├── train_search_r1.py                  # [M3] 场景2 入口
│   │   ├── train_code_dapo.py                  # [M4] 场景3 入口
│   │   └── configs.py                          # [M3] AgenticConfig 扩展
│   └── ...
├── areal/
│   ├── workflow/
│   │   ├── search_r1.py                        # [已存在，M3补齐] Search R1 多轮 Workflow
│   │   └── code_exec.py                        # [已存在，M4补齐] Code Execution 多轮 Workflow
│   ├── reward/
│   │   ├── math_rule.py                        # [可选] 仅在现有 MathVerify 无法覆盖时新增
│   │   └── code_exec.py                        # [暂不新增] 优先复用现有 MathVerify
│   ├── dataset/
│   │   └── dapo_math.py                        # [已存在，M2补齐] dapo_math_17k 数据加载器
│   └── utils/
│       └── tracking_patch.py                   # [已存在，M1补齐] DeepInsight SwanLab 指标映射
├── areal_fuyao.dockerfile                      # [M5] 最终镜像构建（三个场景全部跑通后统一生成）
└── docs/
    └── fuyao_patch_changelog.md                # [M5] 改动记录
```

---

## 三、Milestones

### M1: 基础设施层

**交付物**：启动脚本 + sandbox 部署 + SwanLab 指标映射（Dockerfile 统一在 M5 生成）

> 注意：M1 不再假设 SGLang 是默认推理后端。当前 A100 环境优先使用 **vLLM**，SGLang 仅作为后续可选项保留。

#### M1.1 `fuyao_areal_run.sh` — 统一本地启动脚本

> **关键架构**：训练在当前环境本地运行（本机 GPU），fuyao SDK 仅用于按需部署沙盒服务（search retrieval / code execd）。

参考 ROLL 的 `fuyao_roll_run.sh`，但简化为本地执行模式，并以 **vLLM** 为默认 rollout backend：

1. 解析命令行参数：`--run-type` (math_rlvr / search_r1 / code_dapo) + `--config <yaml>`
2. 清理本脚本记录的残留进程（ray / inference server / trainer），**不做全局 `pkill python`**
3. 校验参数（SWANLAB_API_KEY 可选）
4. 配置 NCCL 环境变量
5. **按需部署沙盒**：search_r1 类型自动通过 fuyao SDK 部署检索服务
6. 本地启动 Ray 集群（单节点 head）
7. 根据 `--run-type` 选择 Python 入口，本地执行

```bash
# 场景1: Math RLVR (纯本地，无需沙盒)
bash fuyao_examples/fuyao_areal_run.sh \
    --run-type math_rlvr \
    --config fuyao_examples/math/qwen3_4b_rlvr.yaml

# 场景2: Search R1 (自动通过 fuyao SDK 部署检索服务)
bash fuyao_examples/fuyao_areal_run.sh \
    --run-type search_r1 \
    --config fuyao_examples/agentic/search_r1_qwen3_4b.yaml

# 场景2: 跳过部署，使用已有的检索服务
RETRIEVAL_ENDPOINT=http://10.1.x.x:8001/retrieve \
bash fuyao_examples/fuyao_areal_run.sh \
    --run-type search_r1 \
    --config fuyao_examples/agentic/search_r1_qwen3_4b.yaml \
    --skip-deploy

# 场景3: Code DAPO (默认 local subprocess，无需沙盒)
bash fuyao_examples/fuyao_areal_run.sh \
    --run-type code_dapo \
    --config fuyao_examples/agentic/code_dapo_qwen3_4b.yaml
```

**run-type 映射**：

| run-type | Python 入口 | 沙盒需求 |
|----------|------------|---------|
| `math_rlvr` | `examples/math/gsm8k_rl.py` | 无（纯本地） |
| `search_r1` | `examples/agentic/train_search_r1.py` | fuyao 部署 search sandbox → 获得 RETRIEVAL_ENDPOINT |
| `code_dapo` | `examples/agentic/train_code_dapo.py` | 默认 local subprocess；可选 fuyao 部署 execd sandbox |

**推理后端策略**：

| backend | 用途 | 当前策略 |
|---------|------|---------|
| `vllm` | rollout 推理 | **默认**，A100 当前已验证可运行 |
| `sglang` | rollout 推理 | 可选，不作为本计划主路径 |

#### M1.2 `deploy_sandboxes.py` — 远程沙盒部署（fuyao SDK）

复用 llm_train 的 `scripts/deploy_sandboxes.py`，适配 AReaL 路径。
**仅用于部署沙盒服务，训练本身在本地跑。**

- Search sandbox: 通过 fuyao SDK 提交 GPU job → xpeng_retriever 服务 (端口 8001)
- Code sandbox: 通过 fuyao SDK 提交 CPU job → execd daemon (端口 39524)

fuyao SDK 核心调用：
```python
import fuyao
fuyao.etl.init(fuyao_api_uri="http://fuyao-v2-api.xiaopeng.link", ...)
result = fuyao.etl.deploy_run(deploy_run_args)  # 提交沙盒 job
run_info = fuyao.etl.get_run_by_name(job_name)  # 查询状态
pod_info = fuyao.etl.search_run_pods(run_name)   # 获取 host_ip
# → endpoint = f"http://{host_ip}:{port}/retrieve"
```

在 `fuyao_areal_run.sh` 中集成：
```bash
if [[ "$RUN_TYPE" == "search_r1" ]] && [[ -z "$RETRIEVAL_ENDPOINT" ]]; then
    echo "Deploying search sandbox via fuyao SDK..."
    eval $(python3 fuyao_examples/deploy_sandboxes.py --sandbox search --export)
    # 输出: export RETRIEVAL_ENDPOINT=http://10.1.x.x:8001/retrieve
fi
```

#### M1.3 `areal/utils/tracking_patch.py` — DeepInsight SwanLab 指标映射

对齐 ROLL 的 `fuyao_patch/tracking_patch.py`，补齐 AReaL 现有 `StatsLogger` 中的指标重命名。

> 现状：`areal/utils/tracking_patch.py` 与 `areal/utils/stats_logger.py` 已存在接入，本 Milestone 主要做映射核对和补测试，而不是从零新建。

```python
# AReaL 原始指标 → DeepInsight 指标名
METRIC_MAPPING = {
    # infra 指标
    "timeperf/ref/compute_log_probs/total": "deepinsight_infra/ref_logp_time",
    "timeperf/actor/compute_log_probs/total": "deepinsight_infra/logp_time",
    "timeperf/actor/train_step/total": "deepinsight_infra/backward_step_time",
    "timeperf/rollout/total": "deepinsight_infra/rollout_step_time",
    "timeperf/actor/model_update/total": "deepinsight_infra/sync_weight_time",
    "timeperf/step/total": "deepinsight_infra/step_time",
    # 需确认 AReaL 的 throughput 指标名

    # 算法指标
    "ppo_actor/task_reward/avg": "deepinsight_algorithm/reward",
    "ppo_actor/pg_loss": "deepinsight_algorithm/policy_loss",
    "ppo_actor/kl_loss": "deepinsight_algorithm/kl_loss",
    "ppo_actor/clip_frac": "deepinsight_algorithm/clip_ratio",
    "ppo_actor/seq_len/avg": "deepinsight_algorithm/response_length",
    "ppo_actor/grad_norm": "deepinsight_algorithm/grad_norm",
}
```

**Agentic 场景额外指标**（通过 `stats_tracker` 在 Workflow 中上报）：

| 指标名 | 类型 | 描述 |
|--------|------|------|
| `tool_use_count` | per-sequence | 每条轨迹的工具调用次数 |
| `tool_use_success` | per-sequence | 工具调用成功率 |
| `num_turns` | per-sequence | 交互轮次 |
| `search_latency_ms` | scalar | 搜索 API 平均延迟 (ms) |

实现方式：在 Workflow 的 `arun_episode()` 中通过 `stats_tracker.get(workflow_context.stat_scope()).scalar(...)` 上报。

#### M1.4 依赖跟踪（Dockerfile 在 M5 统一生成）

> Dockerfile 不在 M1 生成。M1-M4 实现过程中，新增的依赖实时记录到下方表格。
> M5 时根据此表生成最终完整的 `areal_fuyao.dockerfile`。

**新增依赖跟踪表**（实现过程中持续更新）：

| 库 | 版本 | 用途 | 来源 Milestone | 安装方式 | 是否已含 |
|----|------|------|---------------|---------|---------|
| `swanlab` | latest | 实验追踪 | M1 | `pip install swanlab` | AReaL 已支持，需确认镜像中是否预装 |
| `bifrost-fuyao` | TBD | fuyao SDK / sandbox 管理 | M1 | `pip install bifrost-fuyao` [待确认包名] | 否 |
| `httpx` | >=0.24 | Search R1 HTTP 请求 | M3 | 已含在 sglang 依赖 | 是 |
| `datasets` | >=2.14 | 数据集加载 | M2 | 已含在 AReaL 依赖 | 是 |
| _(待补充)_ | | | | | |

> **规则**：每个 Milestone 实现时，如果 `pip install` 或 `import` 了新库，必须更新此表。

#### M1 验收标准
- [ ] `fuyao_areal_run.sh` 可解析参数并启动 Ray（无需实际训练）
- [ ] SwanLab 指标映射逻辑可独立测试
- [ ] deploy_sandboxes.py 可运行（需 fuyao 集群环境验证）

---

### M2: 场景1 — Qwen3 4B Math RLVR (dapo_math_17k)

**交付物**：数据加载器补齐 + 奖励路径核对 + 配置文件

#### M2.1 `areal/dataset/dapo_math.py` — 数据加载器补齐

ROLL 的 dapo_17k 数据格式：
```
路径: /workspace/zhangjh37@xiaopeng.com/data/dapo_math_17k_processed
格式: Parquet
字段: prompt (str), solution (str)
```

AReaL 期望的 RL 数据格式：
```python
{"messages": [{"role": "user", "content": "..."}], "answer": "\\boxed{...}"}
```

当前仓库已有 `dapo_math.py`，本 Milestone 改为补齐以下缺口：

- 校验 `split` 语义，不再固定读 `train`
- 明确 train / valid 的真实数据来源
- 补充长度过滤和字段校验测试

需要的转换层目标仍然是：
```python
def dapo_math_dataset(split, dataset_config, tokenizer):
    """加载 dapo_math_17k 并转为 AReaL 格式。"""
    ds = load_dataset("parquet", data_files=..., split="train")
    # 字段映射: prompt → messages, solution → answer
    # 从 solution 中提取 \\boxed{} 答案
    return ds.map(convert_fn)
```

#### M2.2 数学奖励路径

优先复用 AReaL 现有 `MathVerifyWorker`，仅在以下情况新增 `areal/reward/math_rule.py`：

- 现有 `MathVerifyWorker` 无法覆盖 ROLL 的答案等价规则
- 需要引入额外的 boxed 提取或字符串后处理逻辑

目标仍然是对齐 ROLL 的 math 环境奖励逻辑：
- 提取 `\boxed{answer}`
- 与 ground truth 比对（支持数值/字符串/LaTeX 等价判断）
- 返回 1.0 (正确) / 0.0 (错误)

现有基线：`areal/reward/gsm8k.py` 和 `areal/reward/__init__.py` 中的 `MathVerifyWorker`。

#### M2.3 `fuyao_examples/math/qwen3_4b_rlvr.yaml`

以 ROLL 配置为参考，但默认 rollout backend 改为 **vLLM**：

```yaml
experiment_name: qwen3-4b-math-rlvr
trial_name: trial0
seed: 42
total_train_epochs: 10
tokenizer_path: ${actor.path}

cluster:
  n_nodes: 1
  n_gpus_per_node: 8
  fileroot: ${oc.env:CPFS_DIR,/tmp/areal}/areal_output/${experiment_name}
  name_resolve:
    type: nfs
    nfs_record_root: ${cluster.fileroot}/name_resolve

scheduler:
  type: null

actor:
  backend: "fsdp:d8p1t1"
  path: /publicdata/huggingface.co/Qwen/Qwen3-4B    # Instruct
  dtype: bfloat16
  gradient_checkpointing: true
  optimizer:
    type: adam
    lr: 1.0e-6
    weight_decay: 0
    lr_scheduler_type: constant
    gradient_clipping: 1.0
    warmup_steps_proportion: 0.01
  eps_clip: 0.2
  kl_ctl: 0.0
  reward_scaling: 10.0
  ppo_n_minibatches: 1
  recompute_logprob: true
  use_decoupled_loss: true
  adv_norm:
    mean_level: batch
    std_level: batch
  mb_spec:
    max_tokens_per_mb: 16384

ref:
  backend: ${actor.backend}
  path: ${actor.path}
  dtype: ${actor.dtype}
  scheduling_strategy:
    type: colocation
    target: actor

rollout:
  backend: "vllm:d8p1t1"
  max_concurrent_rollouts: 128
  consumer_batch_size: ${train_dataset.batch_size}
  max_head_offpolicyness: 2

gconfig:
  n_samples: 4
  max_new_tokens: 4096
  temperature: 0.99
  top_p: 0.99

vllm:
  model: ${actor.path}
  dtype: ${actor.dtype}
  gpu_memory_utilization: 0.8
  max_model_len: 32768

train_dataset:
  batch_size: 128
  shuffle: true
  path: /workspace/zhangjh37@xiaopeng.com/data/dapo_math_17k_processed
  type: dapo_math

valid_dataset:
  batch_size: 128
  path: /workspace/zhangjh37@xiaopeng.com/data/dapo_math_17k_processed
  type: dapo_math
  # TODO(agent): 若无独立 validation split，需在实现阶段明确切分策略或单独验证集来源

# SwanLab tracking
stats_logger:
  experiment_name: ${experiment_name}
  trial_name: ${trial_name}
  fileroot: ${cluster.fileroot}
  wandb:
    mode: disabled
  swanlab:
    mode: online
    project: areal-experiments
    name: ${experiment_name}-${trial_name}
    # api_key via SWANLAB_API_KEY env

saver:
  experiment_name: ${experiment_name}
  trial_name: ${trial_name}
  fileroot: ${cluster.fileroot}
  freq_epochs: 1
  freq_steps: 10000

evaluator:
  experiment_name: ${experiment_name}
  trial_name: ${trial_name}
  fileroot: ${cluster.fileroot}
  freq_epochs: 1
  freq_steps: 10
```

#### M2 验收标准
- [ ] dapo_math_17k 数据成功加载并转为 AReaL 格式
- [ ] 数学奖励路径通过单元测试
- [ ] 配置文件语法校验通过
- [ ] `python examples/math/gsm8k_rl.py --config fuyao_examples/math/qwen3_4b_rlvr.yaml` 可启动（可以在无 GPU 时验证到配置解析阶段）

---

### M3: 场景2 — Qwen3 4B Search R1 Agentic RL

**交付物**：SearchR1Workflow 补齐 + 训练入口补齐 + 配置文件

#### M3.1 `areal/workflow/search_r1.py` — Search R1 Workflow 补齐

```python
class SearchR1Workflow(RolloutWorkflow):
    """
    多轮 Search-R1 风格 Workflow。

    交互协议 (XML标签)：
    - LLM 输出 <search>query</search> → 触发检索
    - 检索结果以 <result>...</result> 拼回上下文
    - LLM 输出 <answer>xxx</answer> → 结束并评分
    - 达到 max_turns 无答案 → reward=0

    奖励：EM (Exact Match) on golden_answers
    """
    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer: str | PreTrainedTokenizerFast,
        retrieval_endpoint: str,          # 必需，RETRIEVAL_ENDPOINT
        max_turns: int = 10,
        max_tool_uses: int = 2,
        max_total_tokens: int = 12800,
    ):
        ...

    async def _retrieve(self, query: str) -> str:
        """调用 xpeng_retriever 检索服务"""
        # POST retrieval_endpoint, json={"query": query}
        # 返回格式化的检索结果文本
        ...

    async def arun_episode(self, engine, data) -> list[dict] | None:
        """多轮生成-检索循环"""
        # 1. 构造初始 prompt (system + question)
        # 2. 循环:
        #    a. engine.agenerate() with stop_strings=["</search>"]
        #    b. 解析输出，检查 <search> 或 <answer> 标签
        #    c. 如果 <search>: 调用 _retrieve(), 拼接结果继续
        #    d. 如果 <answer>: 提取答案，计算 EM 奖励，结束
        # 3. 拼接所有 turns 为单序列
        # 4. 构建 loss_mask (仅 LLM 生成的 token)
        # 5. 上报 stats: tool_use_count, tool_use_success, num_turns
        ...
```

现状：仓库中已存在 `areal/workflow/search_r1.py` 和 `examples/agentic/train_search_r1.py`，本 Milestone 重点是补 stop 条件、指标、测试和配置。

**数据集**：
- 训练: `/workspace/zhangjh37@xiaopeng.com/code/Search-R1/data/nq_search/`
  - 字段: `question`, `golden_answers`
  - 需要写适配加载器（golden_answers 是 list[str]）
- 验证: `/workspace/zhangjh37@xiaopeng.com/data/HotpotQA`
  - 字段: `problem`, `answer`

#### M3.2 `examples/agentic/train_search_r1.py`

```python
def main(args):
    config, _ = load_expr_config(args, AgenticConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    train_dataset = load_search_dataset(config.train_dataset.path, split="train")
    valid_dataset = load_search_dataset(config.valid_dataset.path, split="train")

    workflow_kwargs = dict(
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        retrieval_endpoint=config.retrieval_endpoint,
        max_turns=config.max_turns,
        max_total_tokens=config.max_total_tokens,
    )

    with PPOTrainer(config, train_dataset, valid_dataset) as trainer:
        trainer.train(
            workflow="areal.workflow.search_r1.SearchR1Workflow",
            workflow_kwargs=workflow_kwargs,
        )
```

#### M3.3 `fuyao_examples/agentic/search_r1_qwen3_4b.yaml`

对齐 ROLL 的 search_r1.yaml：
- 模型: `Qwen3-4B-Base` (非 Instruct)
- 数据: nq_search (train) + HotpotQA (val)
- sequence_length: 12800
- max_turns: 10, max_tool_uses: 2
- 训练: lr=1e-6, cosine, warmup=10 steps
- 推理: temperature=0.99, top_p=0.99, top_k=100, stop_strings=["</search>"]
- rollout backend: **vllm**

#### M3 验收标准
- [ ] SearchR1Workflow 单元测试（mock retrieval endpoint）
- [ ] NQ 数据集成功加载
- [ ] 完整训练循环可启动（需 GPU + 检索服务）
- [ ] SwanLab 可看到 tool_use_count, tool_use_success 指标

---

### M4: 场景3 — Qwen3 4B Code DAPO Agentic RL

**交付物**：CodeExecWorkflow 补齐 + 训练入口补齐 + 配置文件

#### M4.1 `areal/workflow/code_exec.py` — Code Execution Workflow 补齐

```python
class CodeExecWorkflow(RolloutWorkflow):
    """
    多轮 Code DAPO Workflow。

    交互协议：
    - LLM 输出 <code>python_code</code> → 执行代码
    - 执行结果以 <output>stdout/stderr</output> 拼回上下文
    - LLM 输出 \\boxed{answer} → 结束并评分
    - 达到 max_turns 无答案 → reward=0

    两种执行模式：
    - local: subprocess.run(["python3", "-c", code], timeout=10)
    - execd: HTTP POST to EXECD_ENDPOINT (可选)
    """
    def __init__(
        self,
        gconfig, tokenizer,
        sandbox_type: str = "local",      # "local" | "execd"
        execd_endpoint: str | None = None,
        code_timeout: int = 10,
        max_turns: int = 10,
        max_tool_uses: int = 5,
        max_total_tokens: int = 8192,
    ):
        ...

    async def _execute_code(self, code: str) -> str:
        """执行 Python 代码，返回 stdout 或 error"""
        if self.sandbox_type == "local":
            # subprocess with timeout, tempdir isolation
            ...
        elif self.sandbox_type == "execd":
            # HTTP POST to execd_endpoint
            ...

    async def arun_episode(self, engine, data):
        # 类似 SearchR1Workflow，但：
        # - stop_strings: ["</code>"]
        # - 解析 <code> 标签执行代码
        # - 检查 \\boxed{} 答案正确性
        # - 上报 tool_use_count, tool_use_success
        ...
```

现状：仓库中已存在 `areal/workflow/code_exec.py` 和 `examples/agentic/train_code_dapo.py`，本 Milestone 重点是补安全边界、测试和配置。

**安全措施**（local 模式）：
- `subprocess.run(timeout=code_timeout)`
- 临时目录执行 `tempfile.mkdtemp()`
- 限制 stdout 长度（截断到 1024 字符）

#### M4.2 配置对齐 ROLL code_dapo.yaml

关键差异 vs Search R1：
- 模型: `Qwen3-4B` (Instruct)
- sequence_length: 8192（vs 12800）
- stop_strings: `["</code>"]`（vs `["</search>"]`）
- rollout backend: **vllm**
- lr_scheduler: constant（vs cosine）
- warmup: 0（vs 10）
- gradient_accumulation: 8（vs 16，需映射到 AReaL 的 ppo_n_minibatches）
- 数据: dapo_17k_with_python_code（同路径，不同环境配置）
- system_prompt: "Please reason step by step, and put your final answer within '\\boxed{}'"

#### M4 验收标准
- [ ] CodeExecWorkflow 单元测试（local subprocess 模式）
- [ ] 代码执行安全限制测试（timeout, 长输出截断）
- [ ] 完整训练循环可启动（需 GPU）
- [ ] execd 模式可选启用

---

### M5: Dockerfile + 文档 + 最终验证

**交付物**：完整 Dockerfile + README + changelog + 端到端验证

> Dockerfile 放在最后一个 Milestone，确保 M1-M4 过程中所有新增依赖都被收录。

#### M5.1 `areal_fuyao.dockerfile` — 最终完整镜像

> 此镜像用于**复现本地训练环境**：包含 AReaL 核心 + 三个场景的所有依赖 + fuyao SDK（沙盒管理）。
> 训练在本地（或任何有 GPU 的机器）执行，沙盒通过 fuyao SDK 远程部署。

三个场景全部跑通后，根据 **M1.4 依赖跟踪表** 生成最终 Dockerfile：

```dockerfile
FROM infra-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/data-infra/nvidia-pytorch:25.06-py3

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV MAX_JOBS=1
ENV TZ=Asia/Shanghai
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /code/AReaL_xpeng
COPY . .

# ── 核心依赖 ──
RUN pip install uv && \
    uv sync --extra cuda

# ── M1-M4 过程中新增的依赖（以下为模板，实际内容在 M4 完成后确定）──
# fuyao SDK (M1, 远程沙盒部署)
RUN pip install bifrost-fuyao    # [待确认包名]

# SwanLab tracking (M1)
RUN pip install swanlab

# （M2-M4 过程中如有新增依赖，追加到此处）
# RUN pip install <lib_from_M2>
# RUN pip install <lib_from_M3>
# RUN pip install <lib_from_M4>

# ── 环境变量 ──
ENV PYTHONPATH="/code/AReaL_xpeng:${PYTHONPATH}"
ENV CUDA_DEVICE_MAX_CONNECTIONS="1"
```

**生成流程**：
1. M1-M4 每次 `pip install` 新库时，更新 M1.4 依赖跟踪表
2. M5 时，汇总依赖跟踪表，生成最终 Dockerfile
3. 构建镜像，验证三个场景均可在有 GPU 的环境中启动

#### M5.2 `fuyao_examples/README.md`

内容：
- 快速开始（三个场景的一键启动命令）
- 环境变量说明
- 数据集准备
- Sandbox 部署指南
- SwanLab 配置
- Dockerfile 构建说明

#### M5.3 `docs/fuyao_patch_changelog.md`

内容：
- 完整的新增/修改文件清单
- 每个文件的功能描述
- **完整依赖库变更清单**（从 M1.4 跟踪表汇总）
- 已知限制和后续计划

#### M5.4 端到端验证

| 场景 | 验证方法 | 预期结果 |
|------|---------|---------|
| Math RLVR | 启动训练 1 epoch | reward 上升，SwanLab 有数据 |
| Search R1 | 启动训练 + mock retrieval | 工具调用日志正常 |
| Code DAPO | 启动训练 + local 代码执行 | 代码执行结果正确拼接 |
| **Dockerfile** | `docker build -f areal_fuyao.dockerfile .` | 镜像构建成功，三场景可启动 |

#### M5 验收标准
- [ ] Dockerfile 包含 M1-M4 全部新增依赖，镜像构建成功
- [ ] 镜像内三个场景各运行 1 step 成功
- [ ] README 包含完整使用说明
- [ ] changelog 列出所有变更和依赖

---

## 四、关键技术决策

### 决策 1：Workflow 架构 → 方案 A（AReaL 原生 RolloutWorkflow）

在 `arun_episode()` 内实现多轮交互循环，不修改 PPOTrainer 核心。
参考 `tongyi_deepresearch` 的 `ArealOpenAI.export_interactions(style="concat")` 实现序列拼接。

### 决策 2：推理引擎 → vLLM

当前 A100 环境主路径使用 vLLM。需验证：

- `stop_strings` 在现有 AReaL + vLLM 接口中的行为
- `</search>` / `</code>` 是否保留在输出中
- 与多轮拼接逻辑是否一致

SGLang 保留为后续可选后端，不纳入本计划的关键路径验收。

### 决策 3：训练引擎 → FSDP2

4B 模型 FSDP2 足够，`backend: "fsdp:d8p1t1"`。

### 决策 4：数据格式转换

ROLL 数据 → AReaL 格式需要适配层：
- dapo_math_17k: `prompt` → `messages`, `solution` → `answer`
- nq_search: `question` → `messages`, `golden_answers` → `answer` (list)
- HotpotQA: `problem` → `messages`, `answer` → `answer`

在 `areal/dataset/` 下新增专用加载器，或在训练脚本中直接使用 `datasets.load_dataset()` + `map()`。

### 决策 5：SwanLab 指标映射

AReaL **已内置** SwanLab 支持（`StatsLogger` + `SwanlabConfig`），只需：
1. 在 YAML 中启用 `swanlab.mode: online`
2. 添加 DeepInsight 指标重命名逻辑（参考 ROLL tracking_patch.py）
3. 在 Workflow 中通过 `stats_tracker` 上报 agentic 指标

---

## 五、风险与注意事项

1. **vLLM stop_strings 验证**：需在 M1 阶段验证 vLLM 是否正确支持 `stop_strings: ["</search>"]` 和 `["</code>"]`，包括停止字符串是否包含在输出中。

2. **多轮序列长度**：Search R1 (12800) 和 Code DAPO (8192) 序列较长，需确保 `vllm.max_model_len` 和 `actor.mb_spec.max_tokens_per_mb` 足够大。

3. **检索服务依赖**：Search R1 场景**必须**有 RETRIEVAL_ENDPOINT，无法脱机运行。M3 的单元测试需 mock。

4. **数据格式兼容**：dapo_math_17k 的 `solution` 字段可能包含完整推理过程，需要从中提取 `\boxed{}` 答案作为 ground truth。

5. **代码执行安全**：subprocess 模式需要限制 timeout 和输出长度，避免训练进程被恶意代码阻塞。

---

## 六、依赖库清单（持续更新 → M5 汇总到 Dockerfile）

> **工作流**：M1-M4 实现过程中新增依赖 → 更新此表 + M1.4 跟踪表 → M5 汇总生成最终 Dockerfile。

| 库 | 版本 | 用途 | 来源 | 安装方式 | 状态 |
|----|------|------|------|---------|------|
| swanlab | latest | 实验追踪 | M1 | `pip install swanlab` | AReaL 已支持，需确认镜像中是否预装 |
| bifrost-fuyao | TBD | fuyao SDK / sandbox 管理 | M1 | `pip install bifrost-fuyao` | 待确认包名 |
| httpx | >=0.24 | Search R1 HTTP 请求 | M3 | 已含在 sglang 依赖 | 无需额外安装 |
| datasets | >=2.14 | 数据集加载 | M2 | 已含在 AReaL 依赖 | 无需额外安装 |
| _(M2-M4 新增)_ | | | | | |
