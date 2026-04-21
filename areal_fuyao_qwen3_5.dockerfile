# Qwen3.5 Megatron + SGLang image
FROM infra-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/data-infra/fuyao:zhangjh37-260325-0644
ENV MAX_JOBS=1

# transformers: delete old version, install with deps (resolves huggingface_hub etc.)
# Pin torch to prevent pip from upgrading it.
RUN rm -rf /AReaL/.venv/lib/python3.12/site-packages/transformers* && \
    pip install --target /AReaL/.venv/lib/python3.12/site-packages \
    "transformers @ git+https://github.com/huggingface/transformers.git@d64a6d67d8c004a25570db4df5689e06caea6af7" \
    "torch==2.9.1"

# mbridge: no-deps is fine (minimal deps, already satisfied)
RUN pip install --no-deps --target /AReaL/.venv/lib/python3.12/site-packages \
    git+https://github.com/ISEEKYAN/mbridge.git

# megatron-core 0.16.0
RUN pip install --no-deps --target /tmp/megatron-scratch \
        megatron-core==0.16.0 \
 && cp -r /tmp/megatron-scratch/. /AReaL/.venv/lib/python3.12/site-packages/ \
 && rm -rf /tmp/megatron-scratch
