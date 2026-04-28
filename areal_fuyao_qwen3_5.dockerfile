# veRL image: has torch 2.10, vllm 0.17, mbridge@dc1321b, transformers@d64a6d6, megatron-core 0.16
# Add AReaL-specific deps that veRL image doesn't have, and upgrade
# megatron-core / mbridge to pick up GDN BSHD-CP (Megatron-LM PR #2614/#2642).
FROM infra-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/data-infra/fuyao:verl-qwen3_5-v9-latest
ENV MAX_JOBS=1

# AReaL-only deps not in veRL image
RUN pip install aiofiles tensorboardX math_verify

# Upgrade megatron-core to 0.17.0 (Apr 16, 2026) — bundles native GDN BSHD context-parallel
# (PR #2614/#2642). Required for Qwen3.5-35B-A3B 16K/32K long-context training.
# mbridge bumped to a 2026-04-24 main commit ("adapt to new mcore for qwen35 mtp")
# which includes Qwen3.5 + new mcore compat fixes.
RUN pip install --upgrade --no-deps \
    megatron-core==0.17.0 \
    && pip install --upgrade --no-deps \
    "mbridge @ git+https://github.com/ISEEKYAN/mbridge.git@310e8fb35ccf4fcd4419d32973e563a6d43ee5fb"
