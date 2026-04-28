# veRL image: has torch 2.10, vllm 0.17, mbridge@dc1321b, transformers@d64a6d6, megatron-core 0.16
# Add AReaL-specific deps + upgrade megatron-core / mbridge to pick up GDN BSHD-CP
# (Megatron-LM PR #2614/#2642).
FROM infra-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/data-infra/fuyao:verl-qwen3_5-v9-latest
ENV MAX_JOBS=1

# AReaL-only deps not in veRL image
RUN pip install aiofiles tensorboardX math_verify

# Upgrade megatron-core to a main-branch commit that contains GDN BSHD context-parallel
# (Megatron-LM PR #2614/#2642). The 0.17.0 PyPI release tag does NOT include this PR
# (release branch was cut before Apr-13 main merge); only commit 20ba03f or later has CP.
#
# Pinned commit 20ba03fec03ebaec050c6bc7e79b77a4b4b5c000 (Apr 13, 2026):
#   - the merge commit of PR #2642 itself (CI-validated atomic state)
#   - internal package_info.py reports 0.18.0 (main is already on next-cycle)
#   - BEFORE Apr-19 MambaModel→HybridModel rename (avoids mbridge import break)
#   - BEFORE Apr-22 DDP refactoring (#3812) and Mamba inference optimisations
#   - smallest diff surface that gives us full GDN CP
#
# mbridge pinned to 2026-04-24 main commit ("adapt to new mcore for qwen35 mtp").
#
# Use uninstall+reinstall (NOT --upgrade) because:
#   1. veRL base may pin mbridge by exact version; --upgrade with same version
#      string skips reinstall.
#   2. pip's git+url install can preserve a previous PyPI-cached build.
# --no-cache-dir prevents stale layer/wheel reuse on rebuild.
RUN pip uninstall -y megatron-core mbridge \
    && pip install --no-deps --no-cache-dir \
       "megatron-core @ git+https://github.com/NVIDIA/Megatron-LM.git@20ba03fec03ebaec050c6bc7e79b77a4b4b5c000" \
    && pip install --no-deps --no-cache-dir \
       "mbridge @ git+https://github.com/ISEEKYAN/mbridge.git@310e8fb35ccf4fcd4419d32973e563a6d43ee5fb"

# Build-time verification: print resolved versions + check that GDN CP code is present.
# If 'self.cp_size' is missing from gated_delta_net.py, build fails fast (before image push).
RUN echo "=== Verifying installed versions ===" \
    && pip show megatron-core | head -3 \
    && pip show mbridge | head -5 \
    && python -c "import megatron.core; print('megatron.core:', megatron.core.__file__, 'version:', megatron.core.__version__)" \
    && python -c "from megatron.core.ssm import gated_delta_net; import inspect; src=inspect.getsource(gated_delta_net); assert 'self.cp_size = self.pg_collection.cp.size()' in src, 'GDN CP code missing!'; print('GDN CP code present: OK')" \
    && python -c "from mbridge.core.llm_bridge import LLMBridge; import inspect; src=inspect.getsource(LLMBridge._build_base_config); assert 'async_tensor_model_parallel_allreduce' not in src, 'mbridge still has deprecated kwarg!'; print('mbridge clean: OK')"
