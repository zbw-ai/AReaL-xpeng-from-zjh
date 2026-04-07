# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# AReaL

## WHAT: Project Overview

AReaL is a distributed RL training framework for LLM alignment via reinforcement
learning.

**Tech Stack**: Python 3.12+ | PyTorch | FSDP2/Megatron | SGLang/vLLM

**Core Directories**:

- `areal/` - Core package
  - `api/` - Config dataclasses, workflow/engine contracts
  - `engine/` - FSDP2, Megatron, SGLang/vLLM adapters
    - `fsdp_utils/` - FSDP2-specific utilities (checkpoint, grad, optimizer, parallel)
    - `megatron_utils/` - Megatron/FP8 utilities (checkpoint, pipeline, quantization)
    - `core/` - Engine-shared utilities (distributed, lock, model, offload)
  - `infra/` - Infrastructure (launcher, scheduler, RPC)
    - `utils/` - Infrastructure utilities (launcher, proc, http, concurrent, slurm, ray)
  - `workflow/` - RolloutWorkflow implementations
  - `reward/` - Reward functions
  - `dataset/` - Dataset loaders
  - `utils/` - Cross-cutting utilities (logging, data, checkpoints, network, RL
    functional)
- `examples/` - Training scripts and configs
- `docs/` - Jupyter Book source

## Architecture: Execution Flow

Understanding how the layers connect requires reading multiple files:

```
Entry point (e.g. examples/math/gsm8k_rl.py)
  ├── load_expr_config()              # CLI + YAML → GRPOConfig (OmegaConf merge)
  ├── get_custom_dataset()            # if/elif dispatch in areal/dataset/__init__.py
  └── PPOTrainer(config, datasets)    # areal/trainer/ppo_trainer.py
        ├── _create_train_engine()    # FSDPPPOActor or MegatronPPOActor
        ├── _init_rollout()           # RemotevLLMEngine or RemoteSGLangEngine
        └── trainer.train(workflow=..., reward_fn=...)
              for each step:
                actor.prepare_batch()       # async/sync bridge
                  └── InferenceEngine.submit() × N
                      └── WorkflowExecutor → workflow.arun_episode(engine, data)
                          ├── engine.agenerate(req)          # LLM generation
                          └── AsyncRewardWrapper(reward_fn)  # reward scoring
                  └── InferenceEngine.wait()  # collect trajectories
                actor.train_batch()           # gradient update
                actor.update_weights()        # push weights to inference server
```

**Key abstractions** (`areal/api/`):

- `RolloutWorkflow`: Single method `async arun_episode()` — the main extension point
- `TrainEngine`: Training loop contract (FSDP2 or Megatron implementations)
- `InferenceEngine`: Async generation client (vLLM or SGLang backends)
- Reward functions: Plain callables, no base class — wrapped via `AsyncRewardWrapper`

**Plugin system**: No decorator registry. Workflows, rewards, and engines are resolved at
runtime via string import paths (e.g. `workflow="areal.workflow.rlvr.RLVRWorkflow"`),
using `import_from_string()`.

**Config hierarchy**: `GRPOConfig` → `PPOConfig` → `BaseExperimentConfig`, with nested
dataclass fields (`actor`, `rollout`, `gconfig`, `scheduler`). CLI overrides use
Hydra-style dotted paths (e.g. `scheduler.type=local`).

## HOW: Core Commands

```bash
# Check environment
python --version              # Requires 3.12+
uv --version                  # Install: https://docs.astral.sh/uv/

# Sync dependencies
uv sync --extra cuda          # CUDA + SGLang inference (default)
uv sync --extra cuda-vllm     # Alternative: CUDA + vLLM inference
uv sync --group dev           # Include dev/test packages
uv run python3 areal/tools/validate_installation.py  # Validate installation

# Pre-commit hooks
pre-commit install --install-hooks  # Set up hooks (run once)
pre-commit run --all-files    # Format and lint

# Run a single-node training (quickstart)
python3 examples/math/gsm8k_rl.py --config examples/math/gsm8k_grpo.yaml scheduler.type=local

# Run tests
# First check GPU availability (many tests require GPU)
python -c "import torch; print('GPU available:', torch.cuda.is_available())"
uv run pytest tests/test_<topic>.py
uv run pytest -sv --sw --lf tests/  # step-wise debug + rerun last failed

# Generate CLI docs
uv run python docs/generate_cli_docs.py

# Build docs (canonical, release-aligned)
./docs/build_all.sh
# Do NOT use `jupyter-book build docs/en|docs/zh` directly for final preview/release,
# because it skips AReaL-specific static setup and output packaging.
```

## Boundaries

### Constraints

- Designed for distributed GPU clusters; assume containerized execution
- Integration tests require multi-node hardware; explain skips when unavailable
- Secrets and endpoints are managed outside the repo

### Always Do

- Read relevant files before modifying code
- Run `pre-commit run --all-files` before committing
- Follow existing code patterns in the same module
- Add tests for new functionality

### Ask First

- Modifying config structures in `areal/api/cli_args.py`
- Adding new dependencies
- Changing launcher or scheduler logic
- Deleting or renaming public APIs
- Running GPU/distributed tests (check GPU first:
  `python -c "import torch; print('GPU available:', torch.cuda.is_available())"`)

### Never Do

- Hardcode secrets, paths, or endpoints
- Skip pre-commit hooks
- Guess cluster configs or rebuild CUDA/driver stacks
- Use wildcard imports (`from x import *`)

## Progressive Disclosure: Detailed Guides

| Task                   | Reference                                                     |
| ---------------------- | ------------------------------------------------------------- |
| Add Workflow           | `docs/customization/agent.md`, `areal/workflow/multi_turn.py` |
| Add Dataset            | `docs/customization/`, `areal/dataset/gsm8k.py`               |
| Add Reward             | `areal/api/reward_api.py`, `areal/reward/geometry3k.py`       |
| Add Archon Model       | `areal/experimental/models/archon/qwen2/`, `qwen3/`           |
| Algorithm Details      | `docs/algorithms/*.md`                                        |
| Quickstart             | `docs/tutorial/quickstart.md`                                 |
| Architecture Deep Dive | `docs/tutorial/gsm8k_grpo.md`                                 |
| CLI Reference          | `docs/cli_reference.md`                                       |

## Git Workflow

- **Commits**: Conventional Commits (`feat:`, `fix:`, `docs:`), ~72 chars subject,
  imperative voice, reasoning in body
- **Squash**: Squash WIP commits before opening PR
- **PR requirements**: Run pre-commit, document test coverage, note hardware limitations

## Extended Configuration

See `.claude/agents/`, `.claude/skills/`, `.claude/commands/`, and `.claude/rules/` for
specialized instructions.

### Agents

| Agent                       | Purpose                                   | Activation Trigger                                                  |
| --------------------------- | ----------------------------------------- | ------------------------------------------------------------------- |
| `planner`                   | Implementation planning                   | Before multi-file changes, new features, or architectural decisions |
| `simple-code-reviewer`      | Quick code quality checks                 | After code changes, before committing                               |
| `code-verifier`             | Formatting/linting/tests                  | After code changes, before committing                               |
| `fsdp-engine-expert`        | FSDPEngine implementation                 | FSDPEngine code changes or questions                                |
| `archon-engine-expert`      | ArchonEngine implementation               | ArchonEngine code changes or questions                              |
| `megatron-engine-expert`    | MegatronEngine implementation             | MegatronEngine code changes or questions                            |
| `algorithm-expert`          | RL algorithms                             | GRPO/PPO/DAPO questions                                             |
| `launcher-scheduler-expert` | Cluster launching and resource scheduling | Launcher/scheduler code changes or configuration questions          |

**Stage-by-Stage Agent Guidance**:

1. **Planning Stage** (Before coding): Use `planner` for architecture design and
   implementation planning
1. **Code Formatting & Linting** (After coding): Use `code-verifier` to automatically
   run formatting, linting, and tests, catching syntax errors and style issues quickly
1. **Code Quality Check** (After formatting): Use `simple-code-reviewer` for quick code
   quality checks, focusing on logic issues and code smells

### Skills (Guided Development Workflows)

Skills provide step-by-step guides for common development tasks:

- `/add-dataset` - Dataset loader creation guide
- `/add-workflow` - Workflow implementation guide
- `/add-reward` - Reward function guide
- `/add-archon-model` - Archon engine model architecture guide
- `/debug-distributed` - Distributed debugging guide
- `/add-unit-tests` - Test development guide (NEW)

### Commands (User-invoked Actions)

Commands perform specific actions when invoked:

- `/create-pr` - Rebase, squash commits, and create/update PR with intelligent messages
- `/gen-commit-msg` - Generate commit messages from staged changes
- `/review-pr` - Intelligent PR code review with dynamic agent allocation
- `/translate-doc-zh` - Translate English documentation to Chinese

### Rules (Code Quality Standards)

Project-wide standards enforced across all code changes:

- `api-config.md` - Configuration dataclass design patterns
- `code-style.md` - Coding conventions beyond pre-commit hooks
- `distributed.md` - Distributed training patterns and constraints
- `testing.md` - Testing strategy and coverage requirements
