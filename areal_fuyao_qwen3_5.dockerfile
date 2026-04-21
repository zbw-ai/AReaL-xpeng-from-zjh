# Match veRL's proven environment for Qwen3.5:
# - SGLang 0.5.9 + original transformers (from base image, NOT upgraded)
# - mbridge main branch for Megatron Qwen3.5 support
# - megatron-core 0.16.0 (same as veRL)
# DO NOT upgrade transformers — 5.x breaks SGLang's config parsing for VLM.
FROM infra-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/data-infra/fuyao:zhangjh37-260325-0644
ENV MAX_JOBS=1

# mbridge only — no transformers upgrade
RUN pip install --no-deps --target /AReaL/.venv/lib/python3.12/site-packages \
    git+https://github.com/ISEEKYAN/mbridge.git

# megatron-core 0.16.0 (veRL uses same version)
RUN pip install --no-deps --target /tmp/megatron-scratch \
        megatron-core==0.16.0 \
 && cp -r /tmp/megatron-scratch/. /AReaL/.venv/lib/python3.12/site-packages/ \
 && rm -rf /tmp/megatron-scratch
