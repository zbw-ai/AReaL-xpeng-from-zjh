# veRL image: torch 2.10, vllm 0.17, mbridge@dc1321b, transformers@d64a6d6, megatron-core 0.16
# Add AReaL-specific deps + upgrade megatron-core / mbridge / transformers to support
# GDN + Context Parallel for Qwen3.5-35B-A3B 32K long-context training.
#
# Approach mirrors veRL's verl-qwen35-gdn-cp image (see
# /Users/zengbw/Codebase/for_llm_train_070/llm_train_sft_0402/docker/Dockerfile.qwen35-gdn-cp
# and docs/qwen35_long_context_cp.md). veRL has battle-tested this exact stack on
# 32-card CP=2 multi-step SFT with monotonic loss decrease.
FROM infra-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/data-infra/fuyao:verl-qwen3_5-v9-latest
ENV MAX_JOBS=1

# ---------------------------------------------------------------------------
# AReaL-only deps not in veRL image
# ---------------------------------------------------------------------------
RUN pip install aiofiles tensorboardX math_verify

# ---------------------------------------------------------------------------
# 1. Megatron-LM commit 0f6fcb0 (dev branch, 2026-04-13).
#
# Includes:
#   - PR #2614 (GDN + CP, Dec 2025): head-parallel via all-to-all
#   - PR #2644 (GDN + THD, Apr 2026): packed sequence support
#   - PR #4230 (GDN packed-seq + CP padding align fix, Apr 13 2026)
#
# Why dev (not main): PR #4230 is on dev branch. The main branch's 0.17.0
# release tag does NOT include PR #2614/#2642/#4230 — verified by inspecting
# core_v0.17.0:megatron/core/ssm/gated_delta_net.py which still has
# `# TODO: Implement GatedDeltaNetContextParallel`.
#
# uninstall+install (NOT --upgrade) because pip's git+url resolver may
# preserve a previous install when version strings tie.
# --no-cache-dir prevents stale wheel/layer reuse.
# ---------------------------------------------------------------------------
RUN pip uninstall -y megatron-core mbridge \
    && pip install --no-deps --no-cache-dir --force-reinstall \
       "megatron-core @ git+https://github.com/NVIDIA/Megatron-LM.git@0f6fcb0c5778327868e6866447a58b5568059ae1" \
    && pip install --no-deps --no-cache-dir --force-reinstall \
       "mbridge @ git+https://github.com/ISEEKYAN/mbridge.git@310e8fb35ccf4fcd4419d32973e563a6d43ee5fb"

# ---------------------------------------------------------------------------
# 2. Patch megatron's param_and_grad_buffer.py to bypass _coalescing_manager.
#
# PyTorch's NCCL backend never implemented `reduce_scatter_tensor_coalesced` /
# `allgather_into_tensor_coalesced` — only Gloo has them. Megatron's
# `start_grad_sync` and `start_param_sync` wrap per-bucket ops in
# `with _coalescing_manager(...)`, which raises on __exit__:
#
#   RuntimeError: Backend nccl does not support reduce_scatter_tensor_coalesced
#
# This crashes step 1's optimizer phase. Upstream issues unfixed:
#   NVIDIA/Megatron-LM#1369, pytorch/pytorch#134833
#
# Fix: shadow `_coalescing_manager` in this module with a function that returns
# `nullcontext()`. The inner per-bucket reduce_scatter_tensor / all_gather_into_tensor
# calls are NCCL-supported and execute fine without the coalescing wrapper —
# just slightly less efficient (acceptable trade-off for correctness).
#
# Safe because we force overlap_grad_reduce=False / overlap_param_gather=False
# in our recipes (handle is None either way).
# ---------------------------------------------------------------------------
RUN python3 - <<'PY'
import os, sys
import megatron.core.distributed.param_and_grad_buffer as pgb_mod
target = pgb_mod.__file__
with open(target) as f:
    src = f.read()
if "PATCHED_FOR_NCCL_COALESCING_BUG" in src:
    print(f"[OK] Patch already applied at {target}")
    sys.exit(0)
needle = "from torch.distributed import _coalescing_manager"
if needle not in src:
    print(f"[FAIL] Cannot find import in {target}", file=sys.stderr)
    sys.exit(1)
replacement = (
    "from torch.distributed import _coalescing_manager as _orig_coalescing_manager  # noqa: F401  PATCHED_FOR_NCCL_COALESCING_BUG\n"
    "from contextlib import nullcontext as _nullctx\n"
    "def _coalescing_manager(*_args, **_kwargs):\n"
    "    # PATCHED: NCCL backend lacks reduce_scatter_tensor_coalesced / allgather_into_tensor_coalesced.\n"
    "    # Bypass with nullcontext so per-bucket NCCL ops run individually.\n"
    "    return _nullctx()\n"
)
with open(target, "w") as f:
    f.write(src.replace(needle, replacement, 1))
print(f"[OK] Patched {target}")
PY

# ---------------------------------------------------------------------------
# 3. Build-time verification: fail fast if any of the upgrades did not stick.
# ---------------------------------------------------------------------------
RUN echo "=== Verifying installed versions ===" \
    && pip show megatron-core | head -3 \
    && pip show mbridge | head -5 \
    && python3 -c "import megatron.core; print('megatron.core:', megatron.core.__file__, 'version:', megatron.core.__version__)" \
    && python3 -c "from megatron.core.ssm import gated_delta_net; import inspect; src=inspect.getsource(gated_delta_net); assert 'self.cp_size = self.pg_collection.cp.size()' in src, 'GDN CP code missing!'; print('[OK] GDN CP code present')" \
    && python3 -c "from megatron.core.ssm import gated_delta_net; import inspect; src=inspect.getsource(gated_delta_net); assert '_resolve_cu_seqlens' in src, 'PR #4230 fix missing!'; print('[OK] PR #4230 padding alignment fix present')" \
    && python3 -c "from mbridge.core.llm_bridge import LLMBridge; import inspect; src=inspect.getsource(LLMBridge._build_base_config); assert 'async_tensor_model_parallel_allreduce' not in src, 'mbridge has deprecated kwarg!'; print('[OK] mbridge clean')" \
    && python3 -c "import megatron.core.distributed.param_and_grad_buffer as pgb; assert 'PATCHED_FOR_NCCL_COALESCING_BUG' in open(pgb.__file__).read(), 'param_and_grad_buffer not patched'; print('[OK] _coalescing_manager bypass patched')"
