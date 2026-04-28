# veRL image: has torch 2.10, vllm 0.17, mbridge@dc1321b, transformers@d64a6d6, megatron-core 0.16
# Add AReaL-specific deps + upgrade megatron-core / mbridge to pick up GDN BSHD-CP
# (Megatron-LM PR #2614/#2642).
FROM infra-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/data-infra/fuyao:verl-qwen3_5-v9-latest
ENV MAX_JOBS=1

# AReaL-only deps not in veRL image
RUN pip install aiofiles tensorboardX math_verify

# Upgrade megatron-core to 0.17.0 (Apr 16, 2026) — bundles native GDN BSHD context-parallel
# (PR #2614/#2642). Required for Qwen3.5-35B-A3B 16K/32K long-context training.
# mbridge bumped to 2026-04-24 main commit ("adapt to new mcore for qwen35 mtp").
#
# Use uninstall+reinstall (NOT --upgrade) because:
#   1. veRL base may pin mbridge by exact version; --upgrade with same version
#      string skips reinstall.
#   2. pip's git+url install can preserve a previous PyPI-cached build.
# --no-cache-dir prevents stale layer/wheel reuse on rebuild.
RUN pip uninstall -y megatron-core mbridge \
    && pip install --no-deps --no-cache-dir megatron-core==0.17.0 \
    && pip install --no-deps --no-cache-dir \
       "mbridge @ git+https://github.com/ISEEKYAN/mbridge.git@310e8fb35ccf4fcd4419d32973e563a6d43ee5fb"

# Build-time verification: print resolved versions so smoke-test logs show truth
RUN echo "=== Verifying installed versions ===" \
    && pip show megatron-core | head -3 \
    && pip show mbridge | head -5 \
    && python -c "import mbridge.models.qwen3_5.qwen3_5_vl_bridge as m; print('Qwen3_5VlBridge file:', m.__file__)" \
    && python -c "from mbridge.core.llm_bridge import LLMBridge; import inspect; src=inspect.getsource(LLMBridge._build_base_config); print('async_tensor_model_parallel_allreduce in _build_base_config:', 'async_tensor_model_parallel_allreduce' in src)"
