# Qwen3.5-35B-A3B Megatron Adaptation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run Qwen3.5-35B-A3B RLVR training on AReaL using MegatronEngine with EP+TP+PP, on 4×8 A100-80GB (32 GPUs).

**Architecture:** Upgrade mbridge to main branch (contains Qwen3.5 support from PR#83), upgrade transformers to ≥5.3.0 (qwen3_5_moe model type), and create a Megatron YAML config following the proven Qwen3-30B-A3B pattern. The MegatronEngine already integrates with mbridge via `AutoBridge.from_pretrained()` — once mbridge knows Qwen3.5, the engine works automatically. Key constraint from veRL: GDN linear attention requires `pad_to_maximum=true` (no remove_padding/THD format).

**Tech Stack:** mbridge (main branch), megatron-core, transformers ≥5.3.0, SGLang 0.5.9

**Reference:** veRL's Qwen3.5 Megatron PR (#5381) — `examples/grpo_trainer/run_qwen3_5-35b-megatron.sh`

---

## File Structure

| File | Action | Purpose |
|------|--------|---------|
| `areal_fuyao_qwen3_5.dockerfile` | Modify | Add mbridge upgrade + transformers upgrade |
| `fuyao_examples/math/qwen3_5_35b_a3b_rlvr.yaml` | Modify | Switch from fsdp:d16 to megatron MoE backend |
| `areal/models/mcore/registry.py` | Modify | Register `Qwen3_5MoeForConditionalGeneration` as VLM architecture |
| `pyproject.toml` | Modify | Bump mbridge version |

**No new files needed.** mbridge handles all model conversion, weight loading, and layer spec building internally.

---

### Task 1: Dockerfile — upgrade mbridge + transformers

The Docker image needs two packages upgraded:
- **mbridge**: from 0.15.1 (PyPI) to main branch (contains Qwen3.5 PR#83)
- **transformers**: from 4.57.1 to ≥5.3.0 (qwen3_5_moe in CONFIG_MAPPING)

Both are pure Python — CPU builder can handle them.

**Files:**
- Modify: `areal_fuyao_qwen3_5.dockerfile`

- [ ] **Step 1: Update Dockerfile**

```dockerfile
FROM infra-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/data-infra/fuyao:zhangjh37-260325-0644
ENV MAX_JOBS=1

# Qwen3.5 Megatron support:
# 1) mbridge main branch contains Qwen3.5 bridge (PR#83)
# 2) transformers>=5.3.0 has qwen3_5_moe model type
RUN pip install --upgrade --target /AReaL/.venv/lib/python3.12/site-packages \
    transformers tokenizers \
    git+https://github.com/ISEEKYAN/mbridge.git
```

- [ ] **Step 2: Build and push image**

```bash
fuyao docker --site=fuyao_b1 --push \
    --dockerfile=areal_fuyao_qwen3_5.dockerfile \
    --image-name=areal-qwen3_5-megatron-v1
```

Expected: Image builds successfully (pure Python, no CUDA compilation needed).

- [ ] **Step 3: Commit**

```bash
git add areal_fuyao_qwen3_5.dockerfile
git commit -m "feat(docker): upgrade mbridge + transformers for Qwen3.5 Megatron"
```

---

### Task 2: Registry — register Qwen3.5 VLM architecture

The MegatronEngine loads models via `mbridge.AutoBridge.from_pretrained()` which auto-detects architecture. When `bridge is not None`, `make_hf_and_mcore_config()` at line 105 returns `bridge.config` directly — bypassing the architecture switch. So technically no registry change is needed for the mbridge path.

However, if someone tries the non-mbridge path (bridge=None), the architecture `Qwen3_5MoeForConditionalGeneration` is not registered and will raise ValueError. Add it for robustness.

**Files:**
- Modify: `areal/models/mcore/registry.py:115-127`

- [ ] **Step 1: Add Qwen3.5 VLM architectures to registry**

In `make_hf_and_mcore_config()`, the `bridge is not None` path already handles Qwen3.5 (line 105-108). For the else branch, add a note that Qwen3.5 requires mbridge:

```python
# In make_hf_and_mcore_config(), after line 123, add:
        elif architecture in (
            "Qwen3_5MoeForConditionalGeneration",
            "Qwen3_5ForConditionalGeneration",
        ):
            raise ValueError(
                f"Architecture '{architecture}' requires mbridge with Qwen3.5 support. "
                f"Install: pip install -U git+https://github.com/ISEEKYAN/mbridge.git"
            )
```

And same for `make_mcore_layer_specs()` after line 140.

- [ ] **Step 2: Verify the mbridge path works**

The mbridge path in `make_hf_and_mcore_config()` (line 105-108) returns `bridge.config` directly. No architecture dispatch needed. Verify by reading the code:

```python
def make_hf_and_mcore_config(hf_path, dtype, bridge=None):
    if bridge is not None:                    # <-- mbridge path, Qwen3.5 goes here
        hf_config = bridge.hf_config
        return hf_config, bridge.config       # mbridge handles everything
    else:                                      # <-- non-mbridge path
        ...                                    # architecture switch
```

- [ ] **Step 3: Commit**

```bash
git add areal/models/mcore/registry.py
git commit -m "feat(mcore): register Qwen3.5 VLM architectures in registry"
```

---

### Task 3: YAML config — Megatron MoE for Qwen3.5-35B-A3B

Create the training config based on the working `qwen3_30b_a3b_rlvr.yaml` and veRL's `run_qwen3_5-35b-megatron.sh`.

**Key differences from Qwen3-30B:**
- `pad_to_maximum: true` — GDN linear attention requires bshd format (no THD/remove_padding)
- Same parallelism: `megatron:(attn:d2p2t4|ffn:e8t1)` with PP=2

**Files:**
- Modify: `fuyao_examples/math/qwen3_5_35b_a3b_rlvr.yaml`

- [ ] **Step 1: Rewrite YAML for Megatron**

```yaml
# Qwen3.5-35B-A3B (MoE + GDN) Math RLVR — dapo_math_17k
#   - Training: Megatron with Expert Parallelism (mbridge)
#   - Inference: SGLang (TP4, DP4)
#   - Algorithm: GRPO
#   - Data: dapo_math_17k_processed
#   - Cluster: 4 nodes × 8 GPU = 32 GPU (A100 80G)
#   - Sequence: 8K max response
#
# GPU layout (32 GPUs total):
#   Actor  (16 GPU): megatron:(attn:d2p2t4|ffn:e8t1) = 16 GPU
#   SGLang (16 GPU): sglang:d4t4 = DP4 × TP4 = 16 GPU
#   Ref: colocated with actor
#
# NOTE: Qwen3.5 GDN linear attention requires bshd format:
#   pad_to_maximum=true (no THD/remove_padding support yet)
#
# Usage:
#   bash fuyao_examples/fuyao_areal_run.sh \
#       --run-type math_rlvr \
#       --config fuyao_examples/math/qwen3_5_35b_a3b_rlvr.yaml

experiment_name: ${oc.env:BIFROST_JOB_NAME,default}-qwen3_5-35b-a3b-math-rlvr
trial_name: trial0

seed: 42
enable_offload: false
total_train_epochs: 10
tokenizer_path: ${actor.path}

cluster:
  n_nodes: 4
  n_gpus_per_node: 8
  fileroot: /dataset_rc_llmrl/zengbw1/areal_experiments/${experiment_name}/files
  name_resolve:
    type: nfs
    nfs_record_root: /dataset_rc_llmrl/zengbw1/areal_experiments/${experiment_name}/name_resolve

scheduler:
  type: local

# ── Rollout (Inference) — SGLang TP4 × DP4 = 16 GPU ──
rollout:
  backend: "sglang:d4t4"
  experiment_name: ${experiment_name}
  trial_name: ${trial_name}
  setup_timeout: 600.0
  max_concurrent_rollouts: 64
  queue_size: null
  consumer_batch_size: ${train_dataset.batch_size}
  max_head_offpolicyness: 2
  enable_rollout_tracing: false
  scheduling_spec: ${actor.scheduling_spec}
  fileroot: ${cluster.fileroot}
  tokenizer_path: ${tokenizer_path}
  dump_to_file: true

# ── Generation Config ──
gconfig:
  n_samples: 8
  min_new_tokens: 0
  max_new_tokens: 8192
  greedy: false
  temperature: 0.9
  top_p: 1.0
  top_k: -1

# ── Actor (Training) — Megatron MoE: 16 GPU ──
# attn: d2 × p2 × t4 = 16 GPU
# ffn:  dp(auto) × e8 × etp1 × p2 = 16 GPU
actor:
  backend: "megatron:(attn:d2p2t4|ffn:e8t1)"
  experiment_name: ${experiment_name}
  trial_name: ${trial_name}
  path: /dataset_rc_b1/models/Qwen3.5-35B-A3B
  init_from_scratch: false
  disable_dropout: true
  gradient_checkpointing: true
  dtype: bfloat16
  grad_reduce_dtype: float32
  mb_spec:
    n_mbs: 1
    granularity: 1
    max_tokens_per_mb: 10240
    n_mbs_divisor: 1
  pad_to_maximum: true   # REQUIRED: Qwen3.5 GDN requires bshd format (no THD support)
  optimizer:
    type: adam
    lr: 1.0e-6
    weight_decay: 0.01
    beta1: 0.9
    beta2: 0.999
    eps: 1e-8
    lr_scheduler_type: cosine
    gradient_clipping: 1.0
    warmup_steps_proportion: 0.01
  megatron:
    wrap_with_ddp: true
    ddp:
      grad_reduce_in_fp32: true
      overlap_grad_reduce: false
      overlap_param_gather: false
      use_distributed_optimizer: true
    use_deterministic_algorithms: true
    recompute_granularity: full
    recompute_method: uniform
    recompute_num_layers: 1
    moe_router_dtype: fp32
    moe_token_dispatcher_type: alltoall
  eps_clip: 0.2
  temperature: ${gconfig.temperature}
  reward_scaling: 1.0
  reward_bias: 0.0
  kl_ctl: 0.001
  ppo_n_minibatches: 1
  recompute_logprob: true
  use_decoupled_loss: true
  behave_imp_weight_cap: 2.0
  behave_imp_weight_mode: token_mask
  adv_norm:
    mean_level: batch
    std_level: batch
  reward_norm:
    mean_level: group
    std_level: group
    group_size: ${gconfig.n_samples}
  weight_update_mode: xccl
  max_new_tokens: ${gconfig.max_new_tokens}
  scheduling_spec:
    - task_type: worker
      port_count: 2
      gpu: 1
      mem: 32
      cmd: python3 -m areal.infra.rpc.rpc_server
      env_vars: {}

# ── Reference Model — Megatron colocated with actor ──
ref:
  backend: ${actor.backend}
  experiment_name: ${experiment_name}
  trial_name: ${trial_name}
  path: ${actor.path}
  init_from_scratch: false
  disable_dropout: true
  dtype: ${actor.dtype}
  mb_spec:
    n_mbs: 1
    granularity: 1
    max_tokens_per_mb: 10240
    n_mbs_divisor: 1
  optimizer: null
  scheduling_strategy:
    type: colocation
    target: actor
  scheduling_spec: ${actor.scheduling_spec}

# ── SGLang ──
sglang:
  model_path: ${actor.path}
  random_seed: ${seed}
  skip_tokenizer_init: false
  dtype: ${actor.dtype}
  max_running_requests: null
  context_length: 10240
  mem_fraction_static: 0.8
  disable_custom_all_reduce: true

# ── Datasets — dapo_math_17k ──
train_dataset:
  batch_size: 8
  shuffle: true
  pin_memory: true
  num_workers: 4
  path: /workspace/zhangjh37@xiaopeng.com/data/dapo_math_17k_processed
  type: dapo_math

valid_dataset:
  batch_size: 8
  pin_memory: true
  num_workers: 4
  path: /workspace/zhangjh37@xiaopeng.com/data/dapo_math_17k_processed
  type: dapo_math

# ── Utilities ──
saver:
  experiment_name: ${experiment_name}
  trial_name: ${trial_name}
  fileroot: ${cluster.fileroot}
  freq_epochs: 1
  freq_steps: 10000
  freq_secs: null

recover:
  mode: disabled
  experiment_name: ${experiment_name}
  trial_name: ${trial_name}
  fileroot: ${cluster.fileroot}
  freq_epochs: null
  freq_steps: null
  freq_secs: null

evaluator:
  experiment_name: ${experiment_name}
  trial_name: ${trial_name}
  fileroot: ${cluster.fileroot}
  freq_epochs: 1
  freq_steps: 20
  freq_secs: null

stats_logger:
  experiment_name: ${experiment_name}
  trial_name: ${trial_name}
  fileroot: ${cluster.fileroot}
  wandb:
    mode: disabled
  swanlab:
    mode: online
    project: areal_train
    name: ${experiment_name}-${trial_name}
    api_key: ${oc.env:SWANLAB_API_KEY,}

perf_tracer:
  experiment_name: ${experiment_name}
  trial_name: ${trial_name}
  fileroot: ${cluster.fileroot}
  enabled: false
  session_tracer:
    enabled: false
```

- [ ] **Step 2: Validate config parses**

```bash
python -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('fuyao_examples/math/qwen3_5_35b_a3b_rlvr.yaml')
print('backend:', cfg.actor.backend)
print('pad_to_maximum:', cfg.actor.pad_to_maximum)
print('model_path:', cfg.actor.path)
"
```

Expected: `backend: megatron:(attn:d2p2t4|ffn:e8t1)`, `pad_to_maximum: True`

- [ ] **Step 3: Commit**

```bash
git add fuyao_examples/math/qwen3_5_35b_a3b_rlvr.yaml
git commit -m "feat(fuyao): Qwen3.5-35B-A3B Megatron MoE RLVR config"
```

---

### Task 4: Update pyproject.toml mbridge version

Update the pinned mbridge version to ensure new installations get Qwen3.5 support.

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Update mbridge dependency**

Change line containing `mbridge==0.15.1` to install from git:

```
"mbridge @ git+https://github.com/ISEEKYAN/mbridge.git; sys_platform == 'linux' and platform_machine == 'x86_64'",
```

Or if a new PyPI release is available (check `pip index versions mbridge`), pin to that version.

- [ ] **Step 2: Commit**

```bash
git add pyproject.toml
git commit -m "chore(deps): upgrade mbridge to main branch for Qwen3.5 support"
```

---

### Task 5: Remove runtime pip install from launch script

With the new Docker image containing upgraded packages, remove the runtime install logic that was causing 15+ minute delays on cluster nodes.

**Files:**
- Modify: `fuyao_examples/fuyao_areal_run.sh`

- [ ] **Step 1: Clean up runtime deps section**

Replace the Section 3.5 block with a simple check:

```bash
# ========================== 3.5 Qwen3.5 依赖检查 ==========================
if python -c "from transformers.models.auto.configuration_auto import CONFIG_MAPPING; 'qwen3_5_moe' in CONFIG_MAPPING" 2>/dev/null; then
    :
else
    echo "[qwen3.5-deps] WARNING: transformers does not support qwen3_5_moe. Use areal-qwen3_5 Docker image."
fi
```

- [ ] **Step 2: Commit**

```bash
git add fuyao_examples/fuyao_areal_run.sh
git commit -m "fix(fuyao): remove runtime pip install, rely on baked Docker image"
```

---

### Task 6: E2E deployment and validation

Deploy the Qwen3.5 Megatron training job on the cluster.

- [ ] **Step 1: Build and push Docker image (Task 1)**

Wait for `fuyao docker --push` to complete. Note the output image name.

- [ ] **Step 2: Deploy training job**

```bash
fuyao deploy --disable-fault-tolerance \
    --docker-image=<新镜像完整名> \
    --project=rc-ai-infra --experiment=zengbw1/llm_rl \
    --gpu-type a100 --gpus-per-node 8 --node=4 \
    --label=qwen3_5-35b-a3b-megatron-v1 \
    --site=fuyao_b1 --queue=rc-llmrl-a100 \
    SWANLAB_API_KEY=sQOWAKdHZlG94Q8BSTnCM \
    bash fuyao_examples/fuyao_areal_run.sh \
        --run-type math_rlvr \
        --config fuyao_examples/math/qwen3_5_35b_a3b_rlvr.yaml
```

- [ ] **Step 3: Monitor startup**

Check logs for these milestones:
1. `"Using mbridge to create models"` — mbridge recognized Qwen3.5
2. `"Loading HF checkpoint"` — weight loading started
3. Model loading progress bar completes
4. SGLang server starts and connects
5. First training step completes with non-NaN loss

- [ ] **Step 4: If GDN grad buffer error occurs**

veRL hit a GDN grad buffer size mismatch in `megatron_utils.py`. If AReaL hits similar:

```
RuntimeError: ... grad_data.storage().resize_ ... size mismatch
```

The fix (from veRL): wrap the `grad_data.storage().resize_()` call with a size check. Location depends on where AReaL manages grad buffers — search for `grad_data` in the megatron engine code.

---

## Summary

| Task | Files | Lines Changed | Time |
|------|-------|---------------|------|
| 1. Dockerfile | `areal_fuyao_qwen3_5.dockerfile` | ~5 | 10 min + build time |
| 2. Registry | `areal/models/mcore/registry.py` | ~8 | 10 min |
| 3. YAML config | `fuyao_examples/math/qwen3_5_35b_a3b_rlvr.yaml` | full rewrite | 15 min |
| 4. pyproject.toml | `pyproject.toml` | 1 | 5 min |
| 5. Launch script | `fuyao_examples/fuyao_areal_run.sh` | ~10 | 5 min |
| 6. E2E validation | — | — | 2-3 hrs |
| **Total** | | **~30 lines code** | **半天** |

**Critical path:** Task 1 (Docker image build) — everything else depends on the image having the right packages.
