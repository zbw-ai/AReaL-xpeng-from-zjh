# Qwen3.5-35B-A3B Long-Context via Megatron 0.17 GDN-CP

**Branch**: `feat/cp-gdn-megatron-017`

**Goal**: Unblock 32K context on Qwen3.5-35B-A3B by upgrading megatron-core 0.16.0 → 0.17.0 to pick up native GDN context-parallel support, then plumbing CP through mbridge + AReaL.

**Trigger**: v22 (PP=4, 16K) and v22-pp8-r2 (PP=8 EP=4, 16K) both OOM at compute_logp. The 15.31 GB allocation is the LM-head logit buffer at the last PP stage; PP doesn't help (logit lives only on the last stage), TP=4 is blocked (`num_kv_heads=2`), so the only structural lever for 16K/32K is CP.

---

## Day 1 Research Findings (2026-04-28)

### Megatron upstream status

Verified via NVIDIA/Megatron-LM PRs:

| PR | Title | Merged to dev | Merged to main | Target release | Notes |
|---|---|---|---|---|---|
| [#2614](https://github.com/NVIDIA/Megatron-LM/pull/2614) | GDN context parallel | 2025-12-19 | — | 0.16 milestone | head-parallel via all-to-all (Mamba scheme) |
| [#2642](https://github.com/NVIDIA/Megatron-LM/pull/2642) | #2614 main-branch port | — | **2026-04-13** | **0.17.0** | the actually-released CP code |
| [#2644](https://github.com/NVIDIA/Megatron-LM/pull/2644) | GDN THD packed | 2026-04-07 | — | 0.16 milestone | Qwen3.5+THD has NaN; not useful |

**Decision**: Use BSHD-CP (PR #2614/#2642). Avoid THD path (PR #2644) — it requires cuDNN ≥9.19 and has known numerical instability with Qwen3.5 fused attention.

### Implementation surface (PR #2614/#2642)

Only 3 files modified upstream:
- `megatron/core/ssm/gated_delta_net.py` — CP logic embedded in module forward
- `megatron/core/transformer/transformer_config.py` — config additions
- `tests/unit_tests/ssm/test_gated_delta_net.py` — test refactor

**Key API contract** (verified by reading raw 0.17 main GDN module):
- GDN `__init__` reads `pg_collection: ProcessGroupCollection`, picks up `pg_collection.cp.size()`
- All-to-all functions: `tensor_a2a_cp2hp` (CP → head-parallel), `tensor_a2a_hp2cp` (back to CP)
- `sharded_state_dict` updated to include `'dp_cp_group'`
- Caller activates via `--context-parallel-size N` (or equivalent config)
- **No external wrapper required** — caller just sets CP size, GDN handles the rest internally

### PyPI version timeline

| Version | Released | Has GDN CP | Notes |
|---|---|---|---|
| 0.15.0 | 2025-12-18 | No | |
| 0.15.3 | 2026-02-06 | No | |
| 0.16.0 | 2026-02-26 | **No** (PR #2642 not yet merged) | **what we're using** |
| 0.16.1 | 2026-03-20 | No (still pre-0.17) | |
| **0.17.0** | **2026-04-16** | **YES** (PR #2642 merged 2026-04-13) | upgrade target |

Only breaking change in 0.17.0: Python 3.10 deprecated. We're on 3.12 — OK.

### mbridge compatibility risks (the main unknown)

- **Latest mbridge**: 0.15.1 (2025-09-22). Released **before** megatron-core 0.16+, so no claim of compat with 0.17.
- mbridge upstream README explicitly says it's in maintenance mode, redirects users to NVIDIA's [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) for advanced features.
- mbridge `main` branch (post-release) has new `qwen3_5/`, `qwen3_omni_moe/` directories with conditionally-imported `Qwen3_5VlBridge`, `Qwen3_5MoeVlBridge` — but these are **not in 0.15.1**.

AReaL's existing Qwen3.5 patches (verified by grep):
- [areal/models/mcore/registry.py:125-150](areal/models/mcore/registry.py) registers `Qwen3_5MoeForConditionalGeneration` and `Qwen3_5ForConditionalGeneration`
- [areal/models/mcore/hf_save.py:204-624](areal/models/mcore/hf_save.py) has `_qwen3_5_moe_fallback_expert_export` workaround for mbridge returning empty expert names
- [areal/engine/megatron_engine.py:1265,1340](areal/engine/megatron_engine.py) special path: `"qwen3_5" in model_type` triggers `_mbridge_convert_to_hf` (vanilla mbridge fallback)
- [areal/engine/megatron_engine.py:283,346](areal/engine/megatron_engine.py) calls `disable_qwen3_5_incompatible_fusions` (5 fusions disabled)

**Implication**: mbridge 0.15.1 partially handles Qwen3.5 (recognizes config, has expert name routing) but the pipeline is stitched together with AReaL workarounds. Whether it instantiates GDN with a properly-configured `pg_collection` (with `.cp` group) is **not yet verified**.

### AReaL CP code path

[areal/engine/megatron_utils/packed_context_parallel.py](areal/engine/megatron_utils/packed_context_parallel.py):
- Current logic: split sequence by 2*CP zigzag using `cu_seqlens` (THD only)
- BSHD branch (cu_seqlens=None): just calls `model(...)` directly, no preprocess, **no postprocess all-gather**

[areal/engine/megatron_engine.py:1643-1647](areal/engine/megatron_engine.py#L1643-L1647) hard-fails when `pad_to_maximum and cp_size > 1`:
```python
if self.config.pad_to_maximum and cp_size > 1:
    raise ValueError("pad_to_maximum=True is incompatible with context_parallel_size>1; ...")
```

**This guard becomes obsolete once Megatron 0.17 GDN CP works** — but we still need to:
1. Remove the guard
2. Verify the BSHD postprocess all-gather (logits must be gathered across CP ranks before loss/logp computation)

### Strategic conclusion

Path forward is **Megatron upgrade + minor AReaL plumbing + mbridge compat verification**.
**Worst-case fallback**: if mbridge 0.15.1 ⊥ megatron 0.17, switch Qwen3.5 path to NVIDIA Megatron-Bridge (larger refactor but more future-proof).

---

## Architecture: 32K Memory Plan

| Layer | 16K (current OOM) | 16K + CP=2 | 32K + CP=4 |
|---|---|---|---|
| Per-rank seq tokens | 18432 | **9216** | 8192 |
| LM-head logit alloc | 15.31 GB | **7.6 GB** | 6.8 GB |
| compute_logp peak (actor + ref) | OOM (84 GB) | ~70 GB | ~70 GB |
| Topology | d4p4t2 \| e8t1 (32 GPU) | **d2p4t2cp2 \| e8t1** (32 GPU) | **d1p4t2cp4 \| e8t1** (32 GPU) |
| Net actor world | 32 | 32 (DP=2 → DP=2 still, CP added) | 32 (DP=1) |

CP doesn't change actor GPU count — it eats DP. With CP=4 and DP=1, single-batch gradient noise increases; will need to scale `ppo_n_minibatches` to compensate.

---

## Tech Stack
- megatron-core: 0.16.0 → **0.17.0**
- mbridge: 0.15.1 (verify compat; fallback NVIDIA Megatron-Bridge)
- AReaL: patch [packed_context_parallel.py](areal/engine/megatron_utils/packed_context_parallel.py) + remove [megatron_engine.py:1643-1647](areal/engine/megatron_engine.py#L1643-L1647) guard

---

## Tasks

### Task 1: megatron-core 0.17 + mbridge 0.15.1 sanity check (no CP)

**Files:**
- Modify: `pyproject.toml` (megatron-core==0.16.0 → 0.17.0)
- Modify: `uv.lock` (regenerated)
- Test: existing 0.8B yaml on this branch

- [ ] **Step 1: Bump megatron-core in pyproject.toml**

```diff
-megatron-core==0.16.0
+megatron-core==0.17.0
```

- [ ] **Step 2: Lock-file refresh**

Run: `uv sync --extra cuda --extra megatron`
Expected: lock file updates; install succeeds with mbridge 0.15.1 + megatron 0.17.0; **if pip resolution fails** (e.g. mbridge pins megatron-core<0.17), fall to Task 2 fork path.

- [ ] **Step 3: Build container image with new pin**

Commit pyproject + lock file changes.
Build new fuyao image: `areal-qwen3_5-megatron-v22` (next image tag in series).

- [ ] **Step 4: Smoke test on 0.8B (no CP)**

Submit existing `qwen3_5_0_8b_rlvr_vllm.yaml` job on this branch. **No yaml changes** — just verify Megatron 0.17 doesn't break the existing flow.
- Expected pass: same throughput / output as v17, no init crash, no NaN in first train step
- Expected risk: mbridge.AutoBridge.from_pretrained may fail if mbridge ↔ Megatron 0.17 ABI diverged
- Output: pass/fail + any traceback

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: upgrade megatron-core 0.16.0 → 0.17.0 (GDN CP support)"
```

### Task 2: mbridge compat decision (only if Task 1 fails)

**Triage**:
- If failure is API rename / minor signature mismatch → patch mbridge in-tree (vendor a small mbridge fork)
- If failure is fundamental (Qwen3.5 model construction broken) → switch to NVIDIA Megatron-Bridge for Qwen3.5 only (separate code path from non-Qwen3.5 models)

This task is conditional and detailed only if Task 1 fails.

### Task 3: Plumb pg_collection.cp through model construction

**Files:**
- Read: AReaL's mbridge AutoBridge.from_pretrained call site at [areal/engine/megatron_engine.py:245](areal/engine/megatron_engine.py#L245)
- Read: mbridge's Qwen3.5 model factory (find which file in mbridge constructs the GDN-bearing TransformerLayer)
- Possibly modify: AReaL or mbridge to propagate `pg_collection.cp = mpu.get_context_parallel_group()`

- [ ] **Step 1: Trace pg_collection construction**

Read AReaL's `MegatronEngine.initialize` from [areal/engine/megatron_engine.py:140-300](areal/engine/megatron_engine.py#L140-L300). Document where ProcessGroupCollection is built (likely from `mpu.get_*_group()` calls) and whether `cp` is included.

- [ ] **Step 2: If cp group not propagated, add it**

Patch the construction site to set `pg_collection.cp = mpu.get_context_parallel_group()` before passing to mbridge.

- [ ] **Step 3: Verify by debug-print**

Add temporary log inside GDN module init (via Megatron source patch) to confirm `pg_collection.cp.size()` equals YAML's `context_parallel_size`. Remove after verification.

### Task 4: Remove BSHD-CP fail-fast guard

**Files:**
- Modify: [areal/engine/megatron_engine.py:1640-1647](areal/engine/megatron_engine.py#L1640-L1647)

- [ ] **Step 1: Replace guard with a comment + version check**

```python
# Megatron 0.17+ supports BSHD CP for GDN (PR #2614/#2642).
# The earlier fail-fast (pad_to_maximum + cp_size > 1) is now obsolete.
# AReaL's packed_context_parallel_forward BSHD branch passes through the
# Megatron model directly; GDN's internal CP all-to-all handles the
# sequence shard transparently.
```

- [ ] **Step 2: Commit**

```bash
git commit -m "feat(megatron): allow CP with pad_to_maximum (Megatron 0.17 GDN CP)"
```

### Task 5: Fix BSHD postprocess to all-gather logits across CP

**Files:**
- Modify: [areal/engine/megatron_utils/packed_context_parallel.py](areal/engine/megatron_utils/packed_context_parallel.py)

- [ ] **Step 1: Locate BSHD output handling**

In `postprocess_packed_seqs_context_parallel`, current logic at [packed_context_parallel.py:80-81](areal/engine/megatron_utils/packed_context_parallel.py#L80-L81):
```python
if cp_size <= 1 or cu_seqlens is None:
    return output.squeeze(0)
```
This skips all-gather for BSHD + CP > 1, which is wrong post Megatron 0.17.

- [ ] **Step 2: Add BSHD CP postprocess**

When `cp_size > 1` and `cu_seqlens is None`:
- Output shape on each rank is `[B, S/cp_size, ...]` after GDN's internal CP processing
- Need `dist.all_gather` across CP group + `torch.cat(dim=1)` to reconstruct `[B, S, ...]`
- (Alternative: rely on Megatron's built-in `gather_from_context_parallel_region` if applicable to non-pipeline-last stages)

- [ ] **Step 3: Test on 0.8B (cp_size=2, pad_to_maximum=true)**

Add a unit test that compares `cp_size=1` output vs `cp_size=2` output for the same input — must match within bf16 tolerance.

- [ ] **Step 4: Commit**

```bash
git commit -m "fix(megatron): BSHD CP postprocess all-gather for Megatron 0.17 GDN CP"
```

### Task 6: 0.8B numerical validation

**Files:**
- Create: `tests/test_qwen3_5_cp_correctness.py`

- [ ] **Step 1: Write the failing test**

```python
def test_cp2_matches_cp1_logits():
    # Run forward with cp_size=1 and cp_size=2 on same inputs
    # Assert max abs diff < 1e-2 (bf16) for logits
```

- [ ] **Step 2: Run on 8 GPU 0.8B job (one node, TP=2 CP=2)**

If correctness fails → debug post-process gather logic.

### Task 7: 35B 16K + CP=2 (memory validation)

**Files:**
- Create: `fuyao_examples/math/qwen3_5_35b_a3b_rlvr_vllm_6node_seq16k_cp2.yaml`

- [ ] **Step 1: Write yaml**

Backend: `megatron:(attn:d2p4t2c2|ffn:e8t1)` (DP=2 PP=4 TP=2 CP=2; ffn ep*tp*pp=32 ✓)

- [ ] **Step 2: Deploy and verify**

- compute_logp passes (no OOM at LM head)
- actor peak < 60 GB; ref onload + activation < 80 GB total

### Task 8: 35B 32K + CP=4 (target)

**Files:**
- Create: `fuyao_examples/math/qwen3_5_35b_a3b_rlvr_vllm_6node_seq32k_cp4.yaml`

Backend: `(attn:d1p4t2c4|ffn:e8t1)` (DP=1 PP=4 TP=2 CP=4)
- [ ] Bump `max_new_tokens: 16384 → 32768`, `max_tokens: 18432 → 34816`
- [ ] Validate end-to-end run

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| mbridge 0.15.1 ⊥ megatron 0.17 ABI | High | Block Task 1 | Switch to NVIDIA Megatron-Bridge for Qwen3.5 only |
| pg_collection.cp not plumbed by mbridge | Medium | Block Task 3 | Patch AReaL to inject cp group post-construction |
| BSHD CP all-gather has subtle order bug | Medium | Wrong logits at training time | Numerical test in Task 6 must catch |
| Qwen3.5 GDN CP merged but disabled by default | Low | Soft fail (no speedup) | Verify via Megatron debug print |
| AReaL's 13 Qwen3.5 patches conflict with 0.17 | Medium | Various crashes | Re-validate with smoke test in Task 1 Step 4 |

---

## Self-Review

- [x] Spec coverage: Tasks 1-8 cover upgrade, plumbing, correctness, memory validation, scaling.
- [x] Placeholder scan: No TBD/TODO; conditional task (Task 2) flagged.
- [x] Type consistency: pg_collection, cp_size used consistently; backend strings follow existing AReaL syntax `(attn:dXpYtZcN|ffn:eMtK)`.

## Execution

Tasks 1-2 are sequential (env setup). Task 3 depends on Task 1 success. Tasks 4-5 are independent. Task 6 depends on 4-5. Tasks 7-8 depend on 6.

This branch is independent of `feat/weight-update-bucket` — both can advance in parallel and merge to `main` separately.
