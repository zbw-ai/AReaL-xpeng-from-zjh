# Qwen3.5 VL RLVR on AReaL — Adaptation Report

**Date**: 2026-04-23 → 2026-04-26
**Author**: zengbw
**Status**: Qwen3.5-0.8B ✅ validated · **Qwen3.5-35B-A3B ✅ validated** (20+ stable update_weights cycles on 6 nodes)

This report documents the end-to-end adaptation of AReaL to the Qwen3.5
family of VL models, performed in two phases:

- **Phase 1** (§§1–4): Qwen3.5-0.8B dense bring-up — establishes the baseline
  recipe (pad_to_maximum, disk mode, output-gate handling, fusion disables).
- **Phase 2** (§§5–8): Qwen3.5-35B-A3B (MoE) bring-up — extends the recipe
  to MoE + VL packaging, fixes a class of issues that don't appear at
  smaller dense scale (colocate OOM, expert-export conversion, keepalive
  ghost ALLREDUCE, mbridge buffer reset).

Both phases share the same docker image, same env-var preamble, and roughly
the same launcher; the differences are isolated to source-level fixes and
the YAML cluster layout.

## 1. Background

Qwen3.5 is a new Vision-Language model family from Qwen with hybrid attention:
softmax + Gated Delta Net (GDN) linear attention. AReaL's existing Qwen2.5/3
integration didn't work out of the box. This report records the full chain of
issues and the final working recipe for running RL training
(Math RLVR with dapo_math_17k_processed) on this family.

- **Target models**: Qwen3.5-35B-A3B (MoE, 35B total / 3B active), Qwen3.5-0.8B (dense, validation)
- **Base image**: `fuyao:areal-qwen3_5-megatron-v21` (derived from veRL's `verl-qwen3_5-v9-latest`)
  - torch 2.10 / vLLM 0.17.0 / Megatron-core 0.16.0 / mbridge 0.15.1 / TransformerEngine 2.12 / transformers 4.57.1
  - Plus AReaL-specific: `aiofiles tensorboardX math_verify`
- **Dataset**: dapo_math_17k_processed (text-only math RLVR)

## 2. Architecture-Specific Constraints

### 2.1 Qwen3.5 model structure
- **Hybrid attention layers**:
  - Most layers: `GatedDeltaNet` (linear attention; does not support THD/packed sequences)
  - Every 4th layer: `Qwen3_5VLSelfAttention` (standard softmax attention with output gate)
- **Output gate**: `config.text_config.attn_output_gate=true` — Q projection includes a fused gate
  producing `q_proj.weight` with shape `[2 * num_q_heads * head_dim, hidden]` (q || gate along head_dim)
- **MRoPE**: multimodal rotary (3 axes text/vision-h/vision-w). For text-only inputs, mbridge
  computes position_ids internally from attention_mask via `get_rope_index`.
- **Nested VL config**: HF config has `hf_config.text_config.{vocab_size, num_attention_heads, ...}` —
  direct `hf_config.vocab_size` fails.
- **MoE-specific** (35B-A3B only): 256 experts/layer, `num_experts_per_tok=8`,
  `moe_intermediate_size=512`, EP=8 / ETP=1.
- **VL-wrapped HF keys** (35B-A3B only): the docker mbridge bridge exposes
  `model.language_model.layers.N.*` on the HF side even though
  `hf_config.model_type == "qwen3_5_moe"` (text-only model type but VL-style key prefix).

### 2.2 veRL vs AReaL integration differences (key finding)

veRL's Qwen3.5 works via:
1. `mbridge.bridge.export_weights()` → generator of (name, tensor)
2. ZMQ + CUDA IPC transport
3. **`vllm.model_runner.model.load_weights(weights)` — vLLM's native loader** handles
   Qwen3.5's q||gate concat format

AReaL's `weight_update_mode=xccl` path:
1. mbridge export → `convert_to_hf` → custom HTTP `/areal_update_weights_xccl` → NCCL broadcast
   **directly into vLLM's internal buffers**
2. Bypasses vLLM's native loader → shape convention mismatch for q||gate format

Consequence: AReaL's xccl path incompatible with Qwen3.5 out of the box. **Disk mode**
(`save_weights_to_hf_with_mbridge_fast` → NFS safetensor → vLLM reload) uses the same
path veRL conceptually relies on.

## 3. Phase 1 — Bug Chain (0.8B Dense Validation)

Every fix here is required for both 0.8B and 35B-A3B. Removing any one
reintroduces a failure mode.

| # | Symptom | Root cause | Fix | File |
|---|---------|-----------|-----|------|
| 1 | `AttributeError 'NoneType'.sum()` in preprocess_packed_seqs | mbridge's Qwen3.5 GDN requires `attention_mask`; AReaL's `pack_tensor_dict` removes it | Set `actor.pad_to_maximum: true` to skip packing, preserve attention_mask (BSHD) | YAML |
| 2 | `KeyError: 'max_seqlen'`, `AssertionError: cu_seqlens key`, `AttributeError .to() on None` | Downstream code assumed packed format (cu_seqlens, max_seqlen exist) | `_prepare_mb_list` pad_to_maximum branch: pre-set `_max_seqlen`, set `old_cu_seqlens_list=None`, guard `max_seqlen` access | `megatron_engine.py:1500-1530` |
| 3 | `First dimension of the tensor should be divisible by tensor parallel size` | Megatron TP seq parallelism requires seq dim aligned to `tp_size` (or `tp_size*cp_size*2`) | `_pad_seq_dim` pads seq dim of all 2D+ tensors in micro-batches | `megatron_engine.py:1540+` |
| 4 | `too many indices for tensor of dimension 2` in rope_utils | Qwen3.5 MRoPE expects `[3, B, S]` position_ids, not `[B, S]` | `forward_step` sets `position_ids=None` when `pad_to_maximum`, lets mbridge compute via `get_rope_index` | `megatron_engine.py:614-630` |
| 5 | `_apply_output_gate` `gate.view(*x.shape)` 2× shape mismatch | Megatron's `get_query_key_value_tensors` bumps `num_query_groups` to `world_size` when `num_kv_heads < TP` → gate dim 2× vs Q dim | **Actor TP ≤ num_kv_heads**. For 0.8B: TP=2 (num_kv_heads=2). For 35B-A3B: TP according to num_kv_heads. | YAML backend |
| 6 | `torch._dynamo.exc.TorchRuntimeError` during shape tracing | Megatron's `_apply_output_gate` is `@torch.compile`-decorated, Dynamo fake-tensor shape inference fails for gated attention | `TORCHDYNAMO_DISABLE=1`, `TORCH_COMPILE_DISABLE=1` env vars | `fuyao_areal_run.sh:154` |
| 7 | Various compile/fusion errors with mbridge | Megatron fusion kernels incompatible with mbridge's Qwen3.5 gated attention | Disable 5 fusions on tf_config BEFORE model build: `apply_rope_fusion`, `masked_softmax_fusion`, `bias_activation_fusion`, `bias_dropout_fusion`, `gradient_accumulation_fusion` | `megatron_utils/deterministic.py:11-35` |
| 8 | Ref.compute_logp fails with `NoneType.sum()` (same as #1) | Ref engine config missed `pad_to_maximum` | Add `pad_to_maximum: true` to ref block in YAML | YAML |
| 9 | `'Qwen3_5Config' object has no attribute 'vocab_size'` in update_weights | VL config nests LM scalars under `text_config` | `remove_padding` call falls back to `hf_config.text_config.vocab_size` | `megatron_engine.py:1203-1213` |
| 10 | `Unknown parameter name ... vision_model.patch_embed.proj.weight` | mbridge's qwen3_5 converter has no mapping for vision tower | Skip `.vision_model.` params in weight update loops (text-only training; vision frozen) | `megatron_engine.py:1407+, 1462+` |
| 11 | `Unknown parameter name ... language_model.embedding.word_embeddings.weight` | AReaL's `convert_qwen3_5_to_hf` fallback chain doesn't understand VL `language_model.` prefix | Call mbridge's native `self.bridge._weight_to_hf_format()` directly for Qwen3.5 models (matches veRL's `vanilla_mbridge` path) | `megatron_engine.py:1223+` |
| 12 | vLLM `Failed to update parameter! expanded (4) must match (8) at dim 2` | xccl direct-write path incompatible with Qwen3.5's q\|\|gate concat layout in vLLM's internal buffers | **`weight_update_mode: disk`** — save HF safetensor, vLLM reloads via its native `load_weights` | YAML |
| 13 | RPC error misreports "method not found" for real AttributeErrors inside method body | `rpc_server.py` caught AttributeError generically | Distinguish `hasattr` check → only report "method not found" when truly absent | `rpc/rpc_server.py:759` |

## 4. Working Configuration (0.8B)

### 4.1 YAML (`fuyao_examples/math/qwen3_5_0_8b_rlvr_vllm.yaml`)

Critical settings (all mandatory):

```yaml
actor:
  backend: "megatron:d2p1t2"       # TP=2 (num_kv_heads=2 for 0.8B)
  pad_to_maximum: true              # BSHD format, mandatory for GDN
  weight_update_mode: disk          # bypasses xccl q||gate bug
  megatron:
    use_deterministic_algorithms: true

ref:
  backend: ${actor.backend}
  pad_to_maximum: true              # MUST match actor

rollout:
  backend: "vllm:d4t1"              # TP=1 (avoids vLLM GQA replicate edge case)
```

### 4.2 Launch script env vars (`fuyao_examples/fuyao_areal_run.sh`)

```bash
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export VLLM_USE_V1=1
export VLLM_DISABLE_CUSTOM_ALL_REDUCE=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
```

### 4.3 Performance (0.8B, 1 node × 8 GPU A100-80G)

| Metric | Value |
|--------|-------|
| Step time (steady state) | ~29-34 s |
| Throughput | 148-150 tok/gpu/s |
| MFU | 6% |
| Train compute | 235-248 tok/gpu/s |
| Rollout (vLLM) | 6.8M-10.7M tok/gpu/s |
| Weight sync overhead | ~6 s per iteration (disk mode, 0.8B) |

Weight sync is ~20% of step time. For 35B this ratio drops further (compute grows faster than IO).

---

## 5. Phase 2 — 35B-A3B (MoE) Bring-Up

The dense Phase 1 path validated against Qwen3.5's hybrid attention. Phase 2
extends it to MoE + VL packaging. The work below is what the original Phase 1
plan ([§5.4 of an earlier draft, replicated as §5.1 here] called "known
risks / open items") assumed would be a smooth scale-up — and which actually
required four additional source-level fixes.

### 5.1 Initial Plan vs. Reality

The original Phase 1 plan listed four open items before going to 35B. Phase 2
exposed a fifth class of issues that the plan didn't anticipate.

| # | Original plan | Actual outcome |
|---|---------------|----------------|
| 1 | "Disk mode scalability" — 70 GB/iter, ~25 s sync expected | Confirmed; ~3.5 min round-trip on NFS (incl. vLLM reload). Acceptable. |
| 2 | "EP + pad_to_maximum verification" — expected to work | Code path was fine. Issue was elsewhere (expert HF name conversion). |
| 3 | "`attn_output_gate` verification" | Confirmed `true` on 35B-A3B; same recipe as 0.8B. |
| 4 | "xccl alignment" — listed as future work | Skipped (still using disk mode). Still future work. |
| **5** | _(not in plan)_ | **Actor + Ref colocate OOM** on first ppo_update (§5.3.1). |
| **6** | _(not in plan)_ | **mbridge MoE expert disk-export returns empty** for VL-style names (§5.3.2). |
| **7** | _(not in plan)_ | **`update_weights` deadlocks** 2 hours after first save (§5.3.3). |
| **8** | _(not in plan)_ | **Second `update_weights`** AssertionError on duplicate expert index (§5.3.4). |

Items 5–8 only surfaced during integration. Items 7 and 8 in particular took
multiple iterations and an NCCL Flight Recorder dump to diagnose.

### 5.2 35B-A3B Architectural Constraints

**Verified via `inspect_qwen3_5_35b_config.sh` on 2026-04-23**:

| Parameter | Value |
|-----------|-------|
| num_attention_heads | 16 |
| **num_key_value_heads** | **2** (same as 0.8B) |
| head_dim | 256 |
| hidden_size | 2048 |
| num_hidden_layers | 40 |
| attn_output_gate | **true** |
| num_experts | 256 |
| num_experts_per_tok | 8 |
| moe_intermediate_size | 512 |
| HF key prefix | `model.language_model.layers.N.*` |

**Consequence for parallelism**:
- Actor TP ≤ 2 (same as 0.8B)
- vLLM TP ≤ 2 (mandatory: bf16 35B ≈ 70 GB doesn't fit a single A100-80G)
- MoE: EP=8, ETP=1, EDP=1 (only viable layout, see §5.4)

### 5.3 Issues & Fixes

#### 5.3.1 Ref-Engine Offload Lifecycle (commit `a774cde`)

**Symptom**: First `ppo_update` OOM at 79+/80 GB; `_initialize_state` of
`fused_adam` failing.

**Root cause**: Ref engine and actor share the same physical GPU
(`scheduling_strategy: colocation target: actor`) but live in separate Ray
actors. `enable_offload: true` injects `torch_memory_saver` env hooks for the
actor, but **the trainer never called `ref.offload()` after `ref.initialize()`**,
so ref weights (~17 GB/rank) stayed on GPU during the actor's ppo_update peak.

**Design**: Bring AReaL's ref-engine memory lifecycle in line with veRL's
`param_offload`/`optimizer_offload`/`grad_offload` pattern (ref-only — no
optimizer/grad to offload). Add explicit `offload()` and `onload()` plumbing
on `TrainController`, and call them in the RL trainer around the only place
where the ref engine is needed — `compute_logp`.

**Implementation**:
1. `areal/infra/controller/train_controller.py` — expose `offload()` /
   `onload()` that dispatch through `_custom_function_call`.
2. `areal/trainer/rl_trainer.py`:
   - After `ref.initialize()`, immediately `ref.offload()` if `config.enable_offload`.
   - Wrap `ref.compute_logp(rollout_batch)` in `onload()` … `try:` … `finally: offload()`.

**Result**: ppo_update peak dropped from 79 GB → 73 GB on the 4-node layout.

#### 5.3.2 Expert Export Fallback for VL-Wrapped Names (commits `8847c53`, `a9be24a`, `e49abe3`)

**Symptom**: First-iteration disk save crashed with
`ValueError: state_dict has 0 keys, but n_shards=2` inside
`split_state_dict_into_shards` for the expert shard set.

**Root cause** (verified by adding a diagnostic logger): The docker mbridge
`qwen3_5_moe` bridge accepts MoE expert names like
`language_model.decoder.layers.N.mlp.experts.linear_fc1.weightK` but
`bridge._weight_to_hf_format()` returns empty
`(converted_names, converted_params)` for them. The diagnostic on rank-0
showed `n_expert_specs=1280, n_empty_conversions=1280` (every spec empty).

mbridge's text-only `qwen2_moe._weight_name_mapping_mlp()` does
`layer_number = name.split(".")[2]`. For `decoder.layers.0.mlp...`, index 2
is `"0"` (correct). For `language_model.decoder.layers.0.mlp...`, index 2 is
`"layers"` (broken). The docker bridge has the same flaw for expert MLP
names, even though it handles non-expert weights correctly.

**Design**: Implement a local fallback in `save_weights_to_hf_with_mbridge_fast`
that detects the empty-conversion case for `model_type == "qwen3_5_moe"` and
writes the expert weights using AReaL's existing Megatron→HF naming semantics.
Output keys must match what vLLM expects on reload:
`model.language_model.layers.N.mlp.experts.M.{gate,up,down}_proj.weight`.

The first attempt used the text-only key format (`model.layers.N...`) and
caused vLLM HTTP 400 on `/areal_update_weights`. The diagnostic
`_summarize_checkpoint_keys` in `vllm_worker_extension.py` showed
`sample_keys=['model.visual.blocks.0.attn.proj.bias', ...]` — confirming a VL
HF layout. The prefix was corrected to `model.language_model.layers.N...`
(commit `a9be24a`).

**Implementation** (excerpt from `areal/models/mcore/hf_save.py`):

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

The fallback is called only when `bridge._weight_to_hf_format` returns empty
**and** `model_type == "qwen3_5_moe"`, with diagnostic logging on success
and a hard `ValueError` when both mbridge and the fallback fail.

**Result**: Per-rank `n_fallback_conversions=1280`; disk-mode round-trip with
vLLM reload succeeds.

#### 5.3.3 Keepalive Ghost ALLREDUCE on default_pg (commit `4958453`)

**Symptom (v10–v15)**: ~30 s after the first `update_weights` finished, all
actor workers stopped progressing. Two hours later NCCL watchdog fired with
the same fingerprint every time:

```
[Rank 0]  WorkNCCL(SeqNum=11, OpType=ALLREDUCE, NumelIn=1, NumelOut=1)
          ran for 7200001 ms before timing out
          last enqueued NCCL work: 11, last completed NCCL work: 10
[Rank 9]  last enqueued: 10, last completed: 10
```

Rank 0 enqueued **one more** ALLREDUCE on `default_pg` than other ranks. That
extra op had no peer and blocked the queue forever. Reproduced on PP=2, PP=4,
4-node, and 6-node layouts.

**Iterative debug**: Several "workarounds" were tried before the real cause
was found, all unsuccessful in isolation:

- `dc4ddea`, `7936fc9`: removed `current_platform.synchronize()` after gloo
  barriers in `_update_weights_from_disk` and `_save_hf` /
  `_save_recover_checkpoint` / `_evaluate_fn` / `_evaluate` /
  `_export_and_commit_stats`. Let the trainer past the
  synchronize-blocks-on-pending-NCCL hang, but the unmatched op still
  deadlocked the next NCCL collective elsewhere
  (e.g. `RayRPCServer.broadcast_tensor_container` on
  `CONTEXT_AND_MODEL_PARALLEL_GROUP`).

These workarounds are kept because they make the symptom appear at the actual
offending site (instead of a downstream synchronize), and because synchronize
after a CPU gloo barrier is genuinely redundant — the prior call site already
synchronized once at its own exit.

**Root-cause hunt** (commit `fa68ae0`): added NCCL Flight Recorder to the
launch script:

```bash
export TORCH_NCCL_TRACE_BUFFER_SIZE=20000   # later renamed TORCH_FR_BUFFER_SIZE
export TORCH_NCCL_DESYNC_DEBUG=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
```

The next failure (v15) dumped a Python stack trace for the failed collective:

```
#0 all_reduce              torch/distributed/distributed_c10d.py:3007
#1 wrapper                 torch/distributed/c10d_logger.py:83
#2 check                   areal/utils/timeutil.py:120
#3 _keepalive_thread_run   areal/utils/name_resolve.py:1190
#4 run                     threading.py:1010   ← background thread
```

**Root cause**: `name_resolve.add(name, value, keepalive_ttl=N)` is called by
rank 0 only inside `_update_weights_from_disk`. It registers a per-entry
`FrequencyControl(frequency_seconds=keepalive_ttl/3)` and starts a background
keepalive thread that periodically calls `keeper.check()`, which does:

```python
if self.frequency_seconds is not None:
    if dist.is_initialized():
        interval_seconds = torch.tensor(...)
        dist.all_reduce(interval_seconds, op=MAX, group=self.group)  # group=None ⇒ default_pg!
```

`group=None` resolves to **default_pg**, containing all 32 actor ranks. But
the keepalive thread only exists on rank 0. Other ranks never queue a
matching op; rank 0's allreduce blocks every subsequent default_pg collective.

This explains every detail of the failure fingerprint:

| Observation | Mechanism |
|---|---|
| `default_pg`, not the model-parallel groups | `FrequencyControl(group=None)` ⇒ `WORLD` |
| `NumelIn=1 NumelOut=1` | `interval_seconds` is a torch scalar |
| `OpType=ALLREDUCE` | Literal call site |
| `SeqNum=11` reproducible | rank-0 enqueues exactly one extra default_pg op per training run |
| `last_enqueued (rank 0) = last_completed + 1` | Single unmatched op |
| `last_enqueued (other ranks) = last_completed` | They never queued it |
| Triggered ~30 s after `name_resolve.add(keepalive_ttl=120)` | First keepalive tick at `keepalive_ttl/3 = 40 s` |
| Reproduces on every PP/EP layout | Not topology-dependent |

**Fix**:

1. `areal/utils/timeutil.py` — add `disable_dist_sync: bool = False` to
   `FrequencyControl.__init__`. When set, `check()` skips the `all_reduce`
   and uses local `__interval_seconds` directly.
2. `areal/utils/name_resolve.py` — pass `disable_dist_sync=True` when
   constructing the keepalive `FrequencyControl`. Cross-rank time
   synchronization is meaningless for keepalive (it's a local NFS lease
   refresher).

#### 5.3.4 mbridge `export_weights_buff` Reset (commit `e225f80`)

**Symptom** (after §5.3.3 fix): Second invocation of
`save_weights_to_hf_with_mbridge_fast` crashed on every rank:

```
File mbridge/models/qwen3_5/base_bridge.py:261, in _weight_to_hf_format
    assert experts_idx not in self.export_weights_buff[experts_key]
AssertionError
```

**Root cause**: The docker `qwen3_5` mbridge bridge accumulates expert
tensors into `self.export_weights_buff[experts_key][experts_idx]` and asserts
on duplicate `experts_idx`. The buffer is never cleared between calls, so
on the second invocation every `(key, idx)` pair is a duplicate.

**Fix**: At the entry of `save_weights_to_hf_with_mbridge_fast`, defensively
clear all dict-typed attributes that look like buffer state, regardless of
mbridge version:

```python
for buf_name in ("export_weights_buff", "_export_weights_buff"):
    buf = getattr(bridge, buf_name, None)
    if isinstance(buf, dict):
        buf.clear()
```

### 5.4 Cluster Layout Upgrade — 4 Nodes (PP=2) → 6 Nodes (PP=4)

The original Phase 1 plan targeted 4 nodes with PP=2. Empirically the
`ppo_update` peak under PP=2 reached 75.25/79.25 GB (94.9%) with only ~4 GB
headroom against MoE load imbalance and weight-update transients. Adding two
nodes and going to PP=4 was the first non-config knob that meaningfully
moved the peak. Other options were ruled out:

| Option | Why rejected |
|---|---|
| `ppo_n_minibatches` 2→4 | Total samples = `batch_size×n_samples = 4×2 = 8`. At 4 minibatches that's 2/minibatch < `DP=4`, breaking `balanced_greedy_partition` ("Number of items must be >= K"). |
| `max_tokens_per_mb` 4096→2048 | Already at 1 sequence per microbatch. Halving forces seq-split, incompatible with `pad_to_maximum=true` (Qwen3.5 GDN requires BSHD). |
| Adam `dtype=bf16` | User policy: keep fp32 optimizer states for precision. |
| `vllm.gpu_memory_utilization 0.72→0.68` | Helps slightly during the disk-mode reload window only. Kept as a safety cushion (commit `83a83cb`). |
| Activation offload | Would require Megatron pipeline-scheduler changes; not low-risk. |
| Context Parallel (CP=2) | Untested with Qwen3.5 GDN on AReaL. |

**MoE parallelism mesh** under the final layout:

```
World = 32 GPU = DP×PP×TP   (attention)   = 4×4×2  (final 6-node layout)
                = EP×ETP×PP×EDP (MoE)     = 8×1×4×1
                  EP=8 ⇒ 256 experts / 8 = 32 experts per rank, balanced
                  ETP=1: moe_intermediate_size=512 too small to TP
                  EDP=1: implied by EP×ETP×PP = world ⇒ no expert replication
```

`EP=16` would require `DP=1` (kills data parallelism); `ETP>1` halves an
already-small `moe_intermediate_size=512`; `EDP>1` doubles expert memory.

**Final 6-node config** (`fuyao_examples/math/qwen3_5_35b_a3b_rlvr_vllm_6node.yaml`):

```yaml
cluster: { n_nodes: 6, n_gpus_per_node: 8 }
enable_offload: true                              # MUST be true (§5.3.1)
actor:
  backend: "megatron:(attn:d4p4t2|ffn:e8t1)"     # 32 GPU
  pad_to_maximum: true                            # GDN requires BSHD
  weight_update_mode: disk                        # xccl incompatible
  ppo_n_minibatches: 2                            # max under DP=4 with batch=4×n_samples=2
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

### 5.5 Validation Results

**Final job**: `bifrost-2026042618245400-zengbw1` (label
`qwen3_5-35b-a3b-v17-bufclr`)

| Stage | 4-node PP=2 baseline | **6-node PP=4 final** |
|---|---|---|
| `ppo_update` allocated | 44.25 GB | **21.27 GB** |
| `ppo_update` reserved | 47.30 GB | **24.94 GB** |
| device memory used | 75.25 GB (94.9%) | **38.58 GB (48.7%)** |
| Headroom | ~4 GB | **~40 GB** |
| `update_weights` step time | ~3.5 min | ~3.5 min (unchanged; NFS-bound) |
| Total step time | ~6 min | ~6 min |
| Successive `update_weights` runs | 0 (sync hang) | **20+ stable** |

For comparison, every prior version (v10–v16) failed at one of the four bug
classes in §5.3 within the first one or two `update_weights` cycles.

## 6. Code Changes Summary

All commits on `main` of `zbw-ai/AReaL-xpeng-from-zjh`.

### 6.1 Phase 1 (0.8B dense)

| File | Purpose |
|------|---------|
| `areal/engine/megatron_engine.py` | Core pad_to_maximum pipeline, VL position_ids handling, VL vocab_size fallback, skip vision_model in weight update, mbridge native conversion |
| `areal/engine/megatron_utils/deterministic.py` | `disable_qwen3_5_incompatible_fusions` function |
| `areal/engine/megatron_utils/megatron.py` | Register qwen3_5 model type in conversion registry (fallback path for non-VL) |
| `areal/utils/data.py` | `amend_position_ids` masked_fill inplace fix |
| `areal/infra/rpc/rpc_server.py` | Clarify AttributeError reporting for debuggability |
| `fuyao_examples/fuyao_areal_run.sh` | Env vars for torch.compile off, vLLM V1, no custom all-reduce |
| `fuyao_examples/math/qwen3_5_0_8b_rlvr_vllm.yaml` | 0.8B production config |
| `fuyao_examples/math/qwen3_5_35b_a3b_rlvr_vllm.yaml` | 35B 4-node config (kept for reference) |
| `fuyao_examples/inspect_mbridge.sh` | On-cluster inspection script |

### 6.2 Phase 2 (35B-A3B MoE)

| Group | File | Change | Commit |
|---|---|---|---|
| §5.3.1 Ref offload | `areal/infra/controller/train_controller.py` | Expose `offload()` / `onload()` via custom RPC | `a774cde` |
| | `areal/trainer/rl_trainer.py` | Offload ref after init; onload/offload around `compute_logp` | `a774cde` |
| §5.3.2 Expert export | `areal/models/mcore/hf_save.py` | `_qwen3_5_moe_fallback_expert_export` + diagnostic logger | `8847c53`, `a9be24a` |
| | `areal/models/mcore/hf_save.py` | Diagnostic logger for empty mbridge conversions | `e49abe3` |
| §5.3.3 Keepalive sync | `areal/utils/timeutil.py` | Add `disable_dist_sync` flag to `FrequencyControl` | `4958453` |
| | `areal/utils/name_resolve.py` | Pass `disable_dist_sync=True` for keepalive `FrequencyControl` | `4958453` |
| | `areal/engine/megatron_engine.py` | Drop redundant `synchronize()` after gloo barrier in `_update_weights_from_disk` | `7936fc9` |
| | `areal/trainer/rl_trainer.py` | Same in `_save_hf` / `_save_recover_checkpoint` / `_evaluate*` / `_export_and_commit_stats` | `dc4ddea` |
| | `fuyao_examples/fuyao_areal_run.sh` | NCCL Flight Recorder env vars (kept on by default) | `fa68ae0` |
| §5.3.4 mbridge buffer | `areal/models/mcore/hf_save.py` | Reset `export_weights_buff` at function entry | `e225f80` |
| §5.4 Layout | `fuyao_examples/math/qwen3_5_35b_a3b_rlvr_vllm_6node.yaml` | 6-node, PP=4 production config | `13d3c15` |
| | `fuyao_examples/math/qwen3_5_35b_a3b_rlvr_vllm.yaml` | vLLM mem 0.72→0.68 cushion | `83a83cb`, `0950848` |

Phase 2 totals: ~12 commits, ~7 production source files, 1 new YAML, env-var
update. None of the changes are Qwen3.5-specific — they are general fixes
that benefit any future MoE/VL bring-up on AReaL (especially §5.3.1 and
§5.3.3).

## 7. Reproduction Recipe

```bash
# 0.8B validation (Phase 1)
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

# 35B-A3B production (Phase 2; recommended)
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

## 8. Lessons Learned

1. **NCCL Flight Recorder should be on by default for any new model
   bring-up**. The §5.3.3 ghost ALLREDUCE was invisible from logs alone — it
   manifested as a generic 2-hour barrier timeout, attributable to a dozen
   plausible causes. The Flight Recorder stack trace turned hours of guessing
   into a 30-second diagnosis.

2. **`group=None` in `dist.all_reduce` is a correctness footgun in code that
   may run on a subset of ranks** (background threads, rank-0-only paths).
   Audit any `dist.*` call without an explicit `group=` for "who calls this?"
   before assuming all ranks participate.

3. **`current_platform.synchronize()` after a gloo CPU barrier is redundant
   and can hide bugs**. If pending NCCL ops exist on one rank, the synchronize
   blocks there silently and deadlock manifests later at a different site.

4. **Don't trust mbridge to be stateless across calls**. The §5.3.4 buffer
   accumulation isn't documented anywhere; defensive resets at AReaL's call
   site are cheap insurance.

5. **VL packaging changes more than naming**. The HF key prefix difference
   (`model.layers.N.*` vs `model.language_model.layers.N.*`) cascades through
   mbridge's `_weight_name_mapping_*` (which uses `name.split(".")[2]`) and
   determines the entire fallback path needed. Inspect one non-expert
   weight key from the saved checkpoint (via `_summarize_checkpoint_keys`)
   to confirm format on a new model.

6. **PP scaling pays for itself at the 35B+ tier**. PP=2→PP=4 not only halves
   per-rank memory but also halves per-rank expert count (1280 → 640 specs),
   which proportionally shrinks save-time NCCL traffic.

## 9. Future Work

1. **xccl Qwen3.5 alignment** (1-2 days): patch AReaL's xccl sender to split
   q||gate before broadcasting into vLLM's internal buffers. Eliminates the
   ~3.5 min disk-mode round trip per `update_weights`.
2. **MoE expert export upstream**: contribute `_qwen3_5_moe_fallback_expert_export`
   semantics back to mbridge so future mbridge releases handle the VL prefix
   natively.
3. **ZMQ+IPC path** (1-2 weeks): port veRL's ZMQ/IPC weight transport to
   AReaL. Full alignment with veRL's working path; eliminates disk IO.
4. **Eval throttling**: `evaluator.freq_steps=20` triggers a full
   `valid_dataset` pass and dominates wall-clock time at the eval boundaries.
   Add `evaluator.max_samples` or use a held-out 256-sample slice.
5. **Long-run stability**: validated 20 update cycles; recommend a 24-hour
   soak run (≥200 cycles) before declaring the path production-ready for
   third-party users.
6. **CP=2 + GDN compatibility**: would unlock another ~50% activation
   reduction, allowing 4-node PP=2 to run with ample headroom.
7. **Tree attention + Qwen3.5**: currently untested; will likely need
   additional fixes if used.
8. **FP8 training**: `attn_output_gate` + FP8 currently incompatible
   (asserted in code). Needs mbridge upstream support.
