# Qwen3.5 VL RLVR on AReaL — Adaptation Report

**Date**: 2026-04-23
**Author**: zengbw
**Status**: Qwen3.5-0.8B ✅ validated, Qwen3.5-35B-A3B 🚧 config ready

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

## 3. Full Bug Chain (0.8B validation)

Every fix here is required. Removing any one reintroduces a failure mode.

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

Weight sync is ~20% of step time. For 35B this ratio will be smaller (compute grows faster than IO).

## 5. 35B-A3B Plan

### 5.1 Config file

`fuyao_examples/math/qwen3_5_35b_a3b_rlvr_vllm.yaml` (production-ready, updated
with verified parameters below).

**Verified via `inspect_qwen3_5_35b_config.sh` on 2026-04-23**:

| Parameter | Value |
|-----------|-------|
| num_attention_heads | 16 |
| **num_key_value_heads** | **2** (same as 0.8B!) |
| head_dim | 256 |
| hidden_size | 2048 |
| num_hidden_layers | 40 |
| attn_output_gate | **true** (same as 0.8B) |
| num_experts | 256 |
| num_experts_per_tok | 8 |
| moe_intermediate_size | 512 |

**Consequence**: same TP constraints as 0.8B apply.
- Actor TP must be ≤ 2
- vLLM TP must be ≤ 2 (for safety with disk mode; stricter under xccl)

### 5.2 Final parallelism (32 GPU, 4 nodes)

```yaml
actor:
  backend: "megatron:(attn:d4p2t2|ffn:e8t1)"
  # attn: DP=4 × PP=2 × TP=2 = 16 GPU
  # ffn:  EP=8 × ETP=1 = 256 experts sharded 32-per-rank
  pad_to_maximum: true
  weight_update_mode: disk

ref:
  backend: ${actor.backend}  # colocated
  pad_to_maximum: true

rollout:
  backend: "vllm:d8t2"       # DP=8 × TP=2 = 16 GPU
  # TP=1 infeasible: 35B bf16 weights (~70 GB) exceed A100-80G capacity.
```

Aligned with veRL's `TP=2 EP=8 ETP=1` pattern. Only difference: 32 GPU total
(veRL uses 8 GPU) allows 2 DP replicas + 2 PP stages + 16 GPU for non-colocated vLLM.

### 5.3 Pre-flight checklist before running 35B

1. **Disk space on NFS**:
   35B-A3B bf16 safetensor ≈ 70 GB. Disk-mode writes this every iteration.
   Ensure NFS has several GB/s write bandwidth and ≥ 200 GB free on
   `${cluster.fileroot}`.

2. **Memory sanity**:
   - Actor per-GPU: ~2 GB (1/16 of 35B params) + optimizer states (fp32 adam = 8× params) + activations
   - MoE router/shared expert + EP'd experts per rank: ~4 GB experts per GPU
   - Total per GPU: ~16-20 GB weights + activations → fits in 80 GB ✓
   - vLLM per TP=2 rank: ~35 GB weights + KV cache → fits in 80 GB ✓

3. **Expected step time**:
   Estimated 60-120s per step (conservative). Disk-mode weight sync ~25s
   (70 GB / ~3 GB/s NFS write), plus ~35-95s compute.

### 5.4 Known risks / open items

- **Disk mode scalability**: 35B writes ~70 GB per iteration. At ~3 GB/s NFS → ~25s weight sync,
  comparable or smaller than compute. Acceptable but can be improved.
- **Future xccl alignment**: once we understand vLLM's internal buffer layout for Qwen3.5 q||gate,
  add a special-case in AReaL's xccl sender. Expected ~1 day of focused work.
- **EP + pad_to_maximum**: 0.8B validation is dense (no EP). 35B has `ep=8`; ensure our
  pad_to_maximum path works with MoE. Expected to work (no code depends on dense-only) but
  must be verified on first 35B run.
- **`attn_output_gate`**: still not confirmed whether 35B-A3B sets this to true. If true, same
  path as 0.8B. If false, some 0.8B-specific workarounds (TP limit) don't apply.

## 6. Code Changes Summary

All changes on `main` branch of `zbw-ai/AReaL-xpeng-from-zjh`.

| File | Purpose |
|------|---------|
| `areal/engine/megatron_engine.py` | Core pad_to_maximum pipeline, VL position_ids handling, VL vocab_size fallback, skip vision_model in weight update, mbridge native conversion |
| `areal/engine/megatron_utils/deterministic.py` | `disable_qwen3_5_incompatible_fusions` function |
| `areal/engine/megatron_utils/megatron.py` | Register qwen3_5 model type in conversion registry (fallback path for non-VL) |
| `areal/utils/data.py` | `amend_position_ids` masked_fill inplace fix |
| `areal/infra/rpc/rpc_server.py` | Clarify AttributeError reporting for debuggability |
| `fuyao_examples/fuyao_areal_run.sh` | Env vars for torch.compile off, vLLM V1, no custom all-reduce |
| `fuyao_examples/math/qwen3_5_0_8b_rlvr_vllm.yaml` | 0.8B production config |
| `fuyao_examples/math/qwen3_5_35b_a3b_rlvr_vllm.yaml` | 35B production config |
| `fuyao_examples/inspect_mbridge.sh` | On-cluster inspection script for mbridge/Megatron source |

## 7. Reproduction Recipe

```bash
# 0.8B validation
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

# 35B production (after config verification)
fuyao deploy --disable-fault-tolerance \
    --docker-image=infra-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/data-infra/fuyao:areal-qwen3_5-megatron-v21 \
    --project=rc-ai-infra --experiment=zengbw1/llm_rl \
    --gpu-type a100 --gpus-per-node 8 --node=4 \
    --label=qwen3_5-35b-a3b-rlvr \
    --site=fuyao_b1 --queue=rc-llmrl-a100 \
    SWANLAB_API_KEY=<key> \
    bash fuyao_examples/fuyao_areal_run.sh \
        --run-type math_rlvr \
        --config fuyao_examples/math/qwen3_5_35b_a3b_rlvr_vllm.yaml
```

## 8. Future Work

1. **xccl Qwen3.5 alignment** (1-2 days): patch AReaL's xccl sender to split q||gate before
   broadcasting into vLLM's internal buffers. Eliminates disk mode overhead.
2. **ZMQ+IPC path** (1-2 weeks): port veRL's ZMQ/IPC weight transport to AReaL. Full alignment
   with veRL's working path, eliminates disk IO.
3. **Tree attention + Qwen3.5**: currently untested. Will likely need additional fixes if used.
4. **FP8 training**: `attn_output_gate` + FP8 currently incompatible (asserted in code). Needs
   mbridge upstream support.
