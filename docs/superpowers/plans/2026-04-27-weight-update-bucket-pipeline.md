# Weight Update Bucket Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Branch:** `feat/weight-update-bucket`
**Goal:** Replace per-tensor xccl broadcast (broken on Qwen3.5 GDN, and disk fallback ~3.5 min) with veRL-style bucketed NCCL broadcast pipeline. Target: `update_weights` 3.5 min → 5-15 s on Qwen3.5 35B, while fixing 0.8B GDN shape mismatch as a side effect (whole-bucket `model.load_weights` lets vLLM's `packed_modules_mapping` dispatch correctly).
**ETA:** 3-5 days
**Architecture:** Sender-side NCCL bucket broadcast (cupy-style uint8 buffer) + receiver-side `model.load_weights(weights=bucket_list)` whole-bucket loader. Single-hop only (skip veRL's hop-2 ZMQ+IPC because AReaL `vllm_worker_extension` already runs inside the vLLM worker process).
**Tech Stack:** PyTorch distributed (NCCL), torch.uint8 buffer, AReaL existing RPC channel for metadata.

---

## 0. Design Decisions (locked before coding)

| # | Decision | Choice | Rationale |
|---|---|---|---|
| D1 | Bucket size default | **512 MB** for MVP, configurable up to 2 GB | veRL uses 2 GB but allocates `2 × bucket_size` per rank; on 80 GB GPU with Megatron PP+TP+EP this is too tight. 512 MB × 2 = 1 GB headroom, still amortizes NCCL launch over hundreds of tensors. Expose `actor.weight_update_bucket.bucket_size_bytes`. |
| D2 | Buffer dtype | **`torch.uint8` on CUDA, no cupy** | veRL uses cupy at master to dodge a `expandable_segments` bug in PyTorch's caching allocator. AReaL doesn't set `expandable_segments:True` by default. Avoid the cupy dep; if we hit the alloc bug later, swap rank-0 buffers to cupy with a one-liner. |
| D3 | Metadata channel | **Reuse existing AReaL RPC** | AReaL's trainer rank-0 already calls `rollout_engine.update_weights_from_distributed(meta, param_specs)` per bucket (`megatron_engine.py:1143`). Adding ZMQ doubles the failure surface for no win. |
| D4 | Buffering | **Single-buffer MVP first, double-buffer second** | Single-buffer = simpler debug, validates correctness on 0.8B and 35B. Double-buffer only after MVP runs end-to-end; expected speedup 1.5-1.8× (overlap pack with broadcast). |
| D5 | Sender side topology | **Trainer rank-0 only sends; other trainer ranks consume generator without packing** | Mirrors veRL `send_weights` rank<0 branch. Avoids extra TP all-gather round trips beyond what `_collect_param` already does. |
| D6 | Receiver side topology | **Every vLLM worker receives directly via NCCL** (no hop-2 IPC) | AReaL's `vllm_worker_extension` runs inside the vLLM worker process, so we can broadcast straight into the worker. veRL's CE-process is unnecessary. |
| D7 | Backward compat | **`weight_update_mode` enum gains `xccl_bucket`**; `disk` and `xccl` keep working unchanged | Disk mode stays as fallback. Old `xccl` (per-tensor) stays for diff/regression purposes; deprecate after 35B passes. |
| D8 | Unit of `model.load_weights` | **Whole bucket as a list** (this is what fixes 0.8B GDN) | veRL: `self.model_runner.model.load_weights(weights)` where `weights = list[(name, tensor)]`. AReaL today: `load_weights(weights=[(name, tensor)])` per tensor. The list form lets vLLM's `packed_modules_mapping` see all 3 of `gate/up/down` (or `q/k/v`, or `conv1d` shards) at once and dispatch the fused-loader path. **This is the actual root cause of the GDN `(3072,1,8)` vs `(2048,1,4)` mismatch.** |

---

## 1. Task Breakdown (5 days, half-day granularity)

### Day 1 AM — Config + scaffolding
- [ ] Step 1.1 — Add `xccl_bucket` enum value
- [ ] Step 1.2 — Add `WeightUpdateBucketConfig` dataclass
- [ ] Step 1.3 — Wire config into `MegatronEngineConfig` and PPO trainer
- [ ] Step 1.4 — Stub a new `BucketBroadcaster` class file

### Day 1 PM — Sender-side bucket packer (single buffer)
- [ ] Step 1.5 — Implement `BucketBroadcaster.__init__` (allocate uint8 buffer)
- [ ] Step 1.6 — Implement `pack_and_broadcast(named_tensor_iter)` in single-buffer mode
- [ ] Step 1.7 — Wire `_update_bucket_weights_from_distributed` to use it when `mode == xccl_bucket`

### Day 2 AM — Receiver-side bucket unpacker
- [ ] Step 2.1 — Add `areal_set_bucket_meta` RPC handler in `vllm_worker_extension.py`
- [ ] Step 2.2 — Add `areal_update_weight_xccl_bucket` worker method
- [ ] Step 2.3 — Plumb `param_specs` (extended with `offset`/`nbytes`) end-to-end

### Day 2 PM — End-to-end 0.8B smoke test
- [ ] Step 2.4 — Write `qwen3_5_0_8b_rlvr_vllm_xccl_bucket_debug.yaml`
- [ ] Step 2.5 — Run smoke test on 1 node, fix obvious bugs
- [ ] Step 2.6 — Capture timing baseline (single buffer)

### Day 3 AM — 35B validation (single buffer)
- [ ] Step 3.1 — Derive `qwen3_5_35b_a3b_rlvr_v17_xccl_bucket.yaml`
- [ ] Step 3.2 — Run on production cluster, verify reward curve unchanged
- [ ] Step 3.3 — Profile: capture step-time breakdown, memory peak

### Day 3 PM — Double-buffer upgrade
- [ ] Step 3.4 — Refactor `BucketBroadcaster` to swap two buffers
- [ ] Step 3.5 — Move the broadcast to a background `asyncio` executor task
- [ ] Step 3.6 — Add `wait_previous()` + `synchronize()` ordering points

### Day 4 AM — Double-buffer 35B run + tuning
- [ ] Step 4.1 — Re-benchmark 35B with double buffer
- [ ] Step 4.2 — Tune `bucket_size_bytes` (try 256/512/1024 MB)
- [ ] Step 4.3 — Compare against disk fallback (3.5 min) and old xccl

### Day 4 PM — Robustness
- [ ] Step 4.4 — Handle "no PP head" and "MoE expert all-gather" interactions
- [ ] Step 4.5 — Handle weight tying / shared embeddings (lm_head)
- [ ] Step 4.6 — Error path: timeout, NCCL group rebuild, fallback to disk

### Day 5 — Tests + docs + PR
- [ ] Step 5.1 — Unit tests: `test_bucket_broadcaster.py`
- [ ] Step 5.2 — Update CLI docs via `docs/generate_cli_docs.py`
- [ ] Step 5.3 — Write feature note in `docs/algorithms/`
- [ ] Step 5.4 — Open PR (`/create-pr`), squash WIP commits

---

## 2. Detailed Steps with File Diffs

### Step 1.1 — Add `xccl_bucket` enum value

**File:** `areal/api/cli_args.py`

Find the existing `weight_update_mode` field.

```python
# Old
weight_update_mode: Literal["disk", "xccl"] = field(
    default="disk",
    metadata={"help": "..."}
)

# New
weight_update_mode: Literal["disk", "xccl", "xccl_bucket"] = field(
    default="disk",
    metadata={"help": "disk|xccl|xccl_bucket. xccl_bucket pipelines packed buckets via NCCL."}
)
```

### Step 1.2 — Add `WeightUpdateBucketConfig` dataclass

**File:** `areal/api/cli_args.py`

```python
@dataclass
class WeightUpdateBucketConfig:
    """Bucket pipeline config for xccl_bucket weight update mode."""
    bucket_size_bytes: int = field(
        default=512 * 1024 * 1024,
        metadata={"help": "Bytes per NCCL bucket. Memory cost = 2 * bucket_size on each rank."},
    )
    double_buffer: bool = field(
        default=True,
        metadata={"help": "Overlap pack with broadcast using two send buffers."},
    )
    sync_every_bucket: bool = field(
        default=False,
        metadata={"help": "Force torch.cuda.synchronize() after each bucket. Debug only."},
    )
```

### Step 1.3 — Wire into `MegatronEngineConfig`

```python
weight_update_bucket: WeightUpdateBucketConfig = field(
    default_factory=WeightUpdateBucketConfig
)
```

### Step 1.4 — Create `BucketBroadcaster` skeleton

**New file:** `areal/engine/megatron_utils/bucket_broadcast.py`

```python
"""Bucket-pipelined NCCL broadcaster for trainer→rollout weight updates.

Mirrors veRL's NCCLCheckpointEngine.send_weights single-direction (rank-0 send,
multiple receivers). Receiver-side lives in vllm_worker_extension.areal_update_weight_xccl_bucket.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Iterator

import torch
import torch.distributed as dist

from areal.utils.logging import getLogger

logger = getLogger("BucketBroadcaster")


@dataclass
class BucketTensorMeta:
    name: str
    shape: tuple[int, ...]
    dtype: str          # "bfloat16" etc.
    offset: int         # byte offset inside bucket
    nbytes: int


class BucketBroadcaster:
    """Single-direction (src=0) bucket broadcaster on a torch.distributed group."""

    def __init__(
        self,
        process_group: dist.ProcessGroup,
        bucket_size_bytes: int,
        device: torch.device,
        double_buffer: bool = True,
        is_sender: bool = True,
    ):
        self.pg = process_group
        self.bucket_size = bucket_size_bytes
        self.device = device
        self.double_buffer = double_buffer
        self.is_sender = is_sender

        n = 2 if double_buffer else 1
        self._bufs = [
            torch.empty(bucket_size_bytes, dtype=torch.uint8, device=device)
            for _ in range(n)
        ]
        self._buf_idx = 0
        self._pending: dist.Work | None = None
        self._pending_meta: list[BucketTensorMeta] | None = None

    def _next_buf(self) -> torch.Tensor:
        buf = self._bufs[self._buf_idx]
        self._buf_idx = (self._buf_idx + 1) % len(self._bufs)
        return buf

    def _wait_pending(self) -> None:
        if self._pending is not None:
            self._pending.wait()
            self._pending = None

    def close(self) -> None:
        self._wait_pending()
        self._bufs = []
        torch.cuda.empty_cache()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
```

### Step 1.6 — `pack_and_broadcast` (single buffer first)

**Same file** — sender meat:

```python
def pack_and_broadcast(
    self,
    named_tensors: Iterator[tuple[str, torch.Tensor]],
    on_bucket_ready: Callable[[list[BucketTensorMeta]], None],
) -> None:
    """Pack tensors into buckets and broadcast each from rank 0.

    on_bucket_ready is called BEFORE the broadcast launches, with the
    bucket's metadata. Caller is responsible for sending that metadata
    to the receiver out-of-band (we use AReaL RPC).
    """
    assert self.is_sender, "Only sender calls pack_and_broadcast"

    buf = self._next_buf()
    metas: list[BucketTensorMeta] = []
    offset = 0
    n_buckets = 0
    t_start = time.time()

    for name, tensor in named_tensors:
        nbytes = tensor.numel() * tensor.element_size()
        if nbytes > self.bucket_size:
            raise RuntimeError(
                f"Tensor {name} ({tensor.shape}, {tensor.dtype}, {nbytes} B) "
                f"exceeds bucket_size {self.bucket_size}"
            )
        if offset + nbytes > self.bucket_size:
            self._flush(buf, metas, on_bucket_ready, is_last=False)
            n_buckets += 1
            buf = self._next_buf()
            metas, offset = [], 0

        # in-place pack: reinterpret tensor bytes into uint8 view
        flat = tensor.detach().contiguous().view(-1).view(torch.uint8)
        buf[offset : offset + nbytes].copy_(flat, non_blocking=True)
        metas.append(BucketTensorMeta(
            name=name,
            shape=tuple(tensor.shape),
            dtype=str(tensor.dtype).split(".")[-1],
            offset=offset,
            nbytes=nbytes,
        ))
        offset += nbytes

    if metas:
        self._flush(buf, metas, on_bucket_ready, is_last=True)
        n_buckets += 1

    self._wait_pending()
    logger.info(
        f"sender done: {n_buckets} buckets in {time.time() - t_start:.2f}s"
    )

def _flush(
    self,
    buf: torch.Tensor,
    metas: list[BucketTensorMeta],
    on_bucket_ready: Callable[[list[BucketTensorMeta]], None],
    is_last: bool,
) -> None:
    self._wait_pending()
    torch.cuda.synchronize()        # ensure pack copies finished
    on_bucket_ready(metas)          # tell receiver via RPC what's coming
    self._pending = dist.broadcast(buf, src=0, group=self.pg, async_op=True)
    self._pending_meta = metas
```

### Step 1.7 — Wire into `_update_bucket_weights_from_distributed`

**File:** `areal/engine/megatron_engine.py:1123-1159`

```python
def _update_bucket_weights_from_distributed(
    self,
    meta: WeightUpdateMeta,
    converted_named_tensors: list[tuple[str, nn.Parameter | torch.Tensor]],
) -> None:
    if not converted_named_tensors:
        return
    self.engine_lock.acquire()
    try:
        if self.config.weight_update_mode == "xccl_bucket":
            self._broadcast_bucket(meta, converted_named_tensors)
        else:
            # legacy per-tensor xccl path (unchanged)
            self._broadcast_per_tensor(meta, converted_named_tensors)
    finally:
        converted_named_tensors.clear()
        self.engine_lock.release()


def _broadcast_bucket(self, meta, named_tensors):
    if self._bucket_broadcaster is None:
        cfg = self.config.weight_update_bucket
        self._bucket_broadcaster = BucketBroadcaster(
            process_group=self.weight_update_group,
            bucket_size_bytes=cfg.bucket_size_bytes,
            device=current_platform.current_device(),
            double_buffer=cfg.double_buffer,
            is_sender=True,
        )

    def _notify(metas: list[BucketTensorMeta]):
        param_specs = [
            ParamSpec(name=m.name, shape=m.shape, dtype=m.dtype,
                      offset=m.offset, nbytes=m.nbytes)
            for m in metas
        ]
        self._pending_rpc = self.rollout_engine.update_weights_bucket(
            meta, param_specs
        )

    self._bucket_broadcaster.pack_and_broadcast(
        iter(named_tensors), on_bucket_ready=_notify
    )
    if getattr(self, "_pending_rpc", None):
        self._pending_rpc.result()
```

### Step 2.1 — Extend `ParamSpec`

**File:** `areal/api/engine_api.py` (or wherever `ParamSpec` is defined)

```python
@dataclass
class ParamSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str
    offset: int = 0       # NEW
    nbytes: int = 0       # NEW
```

Default values keep legacy callers working unchanged.

### Step 2.2 — Receiver `areal_update_weight_xccl_bucket`

**File:** `areal/engine/vllm_ext/vllm_worker_extension.py`

Add after `areal_update_weight_xccl`:

```python
def areal_set_bucket_meta(
    self, param_specs: list[dict], group_name: str
):
    """Receive next bucket's metadata. Called per-bucket via RPC just before
    the corresponding NCCL broadcast lands."""
    self._areal_bucket_specs = param_specs
    self._areal_bucket_group_name = group_name
    if not hasattr(self, "_areal_bucket_recv_buf"):
        bucket_size = max(
            spec["offset"] + spec["nbytes"] for spec in param_specs
        )
        bucket_size = getattr(self, "_areal_bucket_size", bucket_size)
        self._areal_bucket_recv_buf = torch.empty(
            bucket_size, dtype=torch.uint8, device=self.model_runner.device
        )
    return True, "Success"


def areal_update_weight_xccl_bucket(self):
    """Receive one bucket and load its tensors as a list (fixes packed_modules
    fused-loader dispatch that the per-tensor path bypassed)."""
    specs = self._areal_bucket_specs
    group = self.weight_update_groups[self._areal_bucket_group_name]
    buf = self._areal_bucket_recv_buf

    # 1. receive packed bucket from rank 0
    torch.distributed.broadcast(buf, src=0, group=group, async_op=False)

    # 2. carve views per-tensor (zero copy)
    weights: list[tuple[str, torch.Tensor]] = []
    for spec in specs:
        name = spec["name"]
        shape = tuple(spec["shape"])
        dtype = getattr(torch, spec["dtype"])
        offset = spec["offset"]
        nbytes = spec["nbytes"]
        view = buf[offset : offset + nbytes].view(dtype).view(shape)
        weights.append((name, view))

    # 3. THE KEY DIFFERENCE: feed the whole bucket as a list so vLLM's
    # packed_modules_mapping sees gate||up, q||k||v, and conv1d shards
    # together and dispatches the correct fused loader.
    try:
        self.model_runner.model.load_weights(weights=weights)
    except Exception as e:
        names = [w[0] for w in weights]
        logger.error(
            "[xccl_bucket] load_weights failed on bucket of %d tensors. "
            "first=%r last=%r err=%s",
            len(weights), names[0], names[-1], e,
        )
        raise
    return True, "Success"
```

### Step 2.3 — RPC plumbing on RolloutEngine side

**File:** `areal/engine/sglang_remote.py` or `vllm_remote.py`

Add `update_weights_bucket(meta, param_specs)` that:
1. POSTs `areal_set_bucket_meta(specs, group_name)` to all workers
2. POSTs `areal_update_weight_xccl_bucket()` to all workers
3. Returns a Future the trainer can wait on

Keep `update_weights_from_distributed` untouched — that's the legacy `xccl` per-tensor path.

### Step 2.4 — 0.8B test YAML

**New file:** `fuyao_examples/math/qwen3_5_0_8b_rlvr_vllm_xccl_bucket_debug.yaml`

Derive from existing `qwen3_5_0_8b_rlvr_vllm_xccl_debug.yaml`:

```yaml
actor:
  weight_update_mode: xccl_bucket
  weight_update_bucket:
    bucket_size_bytes: 268435456    # 256 MB for 0.8B (small model)
    double_buffer: false             # MVP: single buffer
    sync_every_bucket: true          # debug
```

### Step 3.1 — 35B test YAML

**New file:** `fuyao_examples/math/qwen3_5_35b_a3b_rlvr_v17_xccl_bucket.yaml`

Copy from v17 baseline; override:
```yaml
actor:
  weight_update_mode: xccl_bucket
  weight_update_bucket:
    bucket_size_bytes: 536870912    # 512 MB
    double_buffer: false             # still MVP
```

### Step 3.4 — Double-buffer refactor

```python
def _flush(self, buf, metas, on_bucket_ready, is_last):
    # Wait the broadcast that was using THIS buf (not the just-completed one).
    # With 2 buffers: at flush N we wait broadcast N-2.
    if self._pending is not None:
        self._pending.wait()
        self._pending = None
    torch.cuda.synchronize()
    on_bucket_ready(metas)
    self._pending = dist.broadcast(buf, src=0, group=self.pg, async_op=True)
```

### Step 3.6 — Ordering points

```python
# Trainer side, end of update_weights:
self._bucket_broadcaster._wait_pending()
torch.cuda.synchronize()                      # ensure all NCCL work done
dist.barrier(group=self.weight_update_group)  # gate before next training step
```

> Per `.claude/rules/distributed.md`: barrier "Avoid unless necessary". This one IS necessary — we must guarantee all rollout workers finished `load_weights` before the next training step uses them.

### Step 4.6 — Fallback path

```python
try:
    self._bucket_broadcaster.pack_and_broadcast(...)
except (RuntimeError, dist.DistBackendError) as e:
    logger.error(f"xccl_bucket failed: {e}; falling back to disk mode for this step")
    self._fallback_disk_update(meta, named_tensors)
```

### Step 5.1 — Unit test

**New file:** `tests/test_bucket_broadcaster.py`

Test on 2-rank single-node `gloo` backend (no GPU required):
- Pack 100 random-shape tensors, verify metas
- Round-trip on rank 0 → rank 1, verify byte-identical
- Edge cases: empty input, single huge tensor exceeding bucket, exactly-fitting bucket

---

## 3. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| `dist.broadcast` on uint8 buffer corrupts subsequent bf16 reinterpret | Low | High | Already covered — uint8 is a re-view, NCCL doesn't care about dtype |
| **Bucket boundary splits a "fused" group** (e.g. q\|\|k\|\|v lands in different buckets) → vLLM fused loader still fails on 0.8B | **Medium** | High | **Group fused-mapped tensors before bucketing**. Read `vllm.model_executor.models.qwen3.*.packed_modules_mapping` and emit fused groups atomically. Add a unit test that forces gate/up into separate buckets and confirms failure, then enable grouping. |
| Memory pressure: 2 × 512 MB on rank 0 + already-tight 35B activations | Medium | Med | Make bucket_size adaptive: `min(configured, 0.5% * free_mem)` |
| RPC `update_weights_bucket` per-bucket overhead dominates on small models | Low | Low | Batch metas: send all bucket metas in one RPC, then issue N broadcasts. (Defer to v2.) |
| NCCL group rebuild required after rollout restart | Low | Med | Already handled by existing `weight_update_group` plumbing |
| Tied weights duplicated in bucket → vLLM warns/errors | Medium | Low | Step 4.5 dedupe |
| MoE EP all-gather peaks memory before pack | Medium | Med | Do not change EP path; bucket starts AFTER all-gather |
| `pre-commit` reformats config defaults | Low | Trivial | Run `pre-commit run --all-files` before commit |
| 35B run hits cluster's 2-hour quota before producing baseline | Med | Med | Test at iter=10 first; full run only after MVP green |

---

## 4. Rollback Strategy (per-step)

| If this step fails | Roll back to |
|---|---|
| Step 1.7 (megatron_engine wiring) | Revert that single file; `weight_update_mode=xccl_bucket` becomes a no-op that hits an `else` branch |
| Step 2.5 (0.8B smoke) — wrong outputs | Keep `xccl_bucket` code, switch yaml to `weight_update_mode=disk`. Continue iterating on bucket code on a side branch. |
| Step 3.2 (35B run) — diverges | Switch production yaml back to `weight_update_mode=disk` (no code change). Investigate offline. |
| Step 3.4 (double buffer) — race condition | Set `double_buffer: false` in yaml, keep code. Single buffer is still 5-10× faster than disk. |
| Step 4.6 (fallback path) — never triggers | Remove the try/except wrapper before merge; nice-to-have only. |
| Whole feature | `git checkout main -- areal/engine/megatron_engine.py areal/engine/vllm_ext/vllm_worker_extension.py areal/api/cli_args.py` and delete new files. Branch is feature-isolated; main is unaffected. |

---

## 5. Testing Matrix

| Test | Hardware | Validation |
|---|---|---|
| Unit (bucket pack round-trip) | CPU only, 2 procs | Byte-identical reconstruction |
| 0.8B GDN smoke | 1 × A100 | No `(3072,1,8)` vs `(2048,1,4)` shape error; `update_weights` < 1s |
| 0.8B end-to-end | 1 × A100 | Reward curve matches disk-mode baseline at step 50 |
| 35B v17 baseline | Production cluster | `update_weights` 5-15 s; reward at step 100 within 0.5% of disk-mode |
| Fallback to disk on simulated NCCL fail | 35B cluster | Step succeeds via disk path; clear log message |
| MoE expert weight update | 35B (Qwen3.5 has experts) | No regression in expert tensor count or correctness |

> **GPU dependency:** Steps 2.5, 3.2, and the 35B/MoE tests require GPU. Per CLAUDE.md "Ask First" rule, request user confirmation before running them. Unit tests (5.1) are CPU-safe.

---

## 6. Files Touched (summary)

| File | Action |
|---|---|
| `areal/api/cli_args.py` | Modify — add `xccl_bucket` enum, `WeightUpdateBucketConfig` |
| `areal/api/engine_api.py` (or wherever ParamSpec lives) | Modify — add `offset`, `nbytes` fields with defaults |
| `areal/engine/megatron_utils/bucket_broadcast.py` | **Create** — `BucketBroadcaster` class |
| `areal/engine/megatron_engine.py` (lines 1123-1159 + new `_broadcast_bucket`) | Modify — branch on `weight_update_mode`, lazy-init broadcaster |
| `areal/engine/vllm_ext/vllm_worker_extension.py` (after line 236) | Modify — add `areal_set_bucket_meta` + `areal_update_weight_xccl_bucket` |
| `areal/engine/sglang_remote.py` or `vllm_remote.py` | Modify — add `update_weights_bucket` RPC method |
| `fuyao_examples/math/qwen3_5_0_8b_rlvr_vllm_xccl_bucket_debug.yaml` | **Create** |
| `fuyao_examples/math/qwen3_5_35b_a3b_rlvr_v17_xccl_bucket.yaml` | **Create** |
| `tests/test_bucket_broadcaster.py` | **Create** |
| `docs/algorithms/xccl_bucket_weight_update.md` | **Create** |
| `docs/cli_reference.md` | Auto-regenerated |

---

## 7. Patterns to Follow

- **veRL sender** (best reference): `/Users/zengbw/Codebase/for_llm_train_070/llm_train_sft_0402/verl/verl/checkpoint_engine/nccl_checkpoint_engine.py:223-294` — copy the bucket fill loop and last-bucket-flush pattern verbatim, just swap `cp.asarray` → `tensor.view(torch.uint8)` and `collective.broadcast` → `dist.broadcast`.
- **veRL receiver**: same file, lines 296-362 — copy the buffer-swap-after-yield pattern. We yield directly into `model.load_weights(weights)` instead of yielding a generator.
- **veRL whole-bucket load_weights call**: `/Users/zengbw/Codebase/for_llm_train_070/llm_train_sft_0402/verl/verl/workers/rollout/vllm_rollout/utils.py:208-230` — note `self.model_runner.model.load_weights(weights)` takes the full list.
- **Existing AReaL per-tensor xccl** (the thing we're replacing): `areal/engine/vllm_ext/vllm_worker_extension.py:156-227` — keep this code path alive as fallback for diff/regression.
- **AReaL trainer-side dispatch we're branching on**: `areal/engine/megatron_engine.py:1123-1159` and `:1244-1279`.

---

## 8. Open Questions for the User

Before starting Day 1 coding, please confirm:

1. **Is the trainer-side `weight_update_group` already created with NCCL backend?** Looking at `vllm_worker_extension.py:349-374` it looks like it; please confirm it is reused (not rebuilt) across update steps.
2. **Bucket size 512 MB OK for the 35B v17 production GPUs?** This is 1 GB/rank for double buffer. If the cluster is memory-tight, drop to 256 MB.
3. **Should the new `weight_update_mode=xccl_bucket` be the new default, or stay opt-in until 35B passes a full epoch?** Per "must keep disk as fallback" leaving default = `disk`; the v17 yaml opts in.

---

## 9. Execution Handoff

**Plan saved to:** `docs/superpowers/plans/2026-04-27-weight-update-bucket-pipeline.md` (this file)

Two execution options:

**1. Subagent-Driven (recommended)** — Dispatch a fresh subagent per task on `feat/weight-update-bucket` branch, review between tasks, fast iteration. Use `superpowers:subagent-driven-development`.

**2. Inline Execution** — Execute tasks in current session using `superpowers:executing-plans`, batch execution with checkpoints for review.
