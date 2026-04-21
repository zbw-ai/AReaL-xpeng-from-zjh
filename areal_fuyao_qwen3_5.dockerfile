# Qwen3.5 Megatron + SGLang image
# Need transformers that: 1) registers qwen3_5_moe 2) doesn't break SGLang VLM config
FROM infra-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/data-infra/fuyao:zhangjh37-260325-0644
ENV MAX_JOBS=1

# transformers: specific commit that handles Qwen3.5 VLM properly
RUN pip install --no-deps --target /AReaL/.venv/lib/python3.12/site-packages \
    "transformers @ git+https://github.com/huggingface/transformers.git@d64a6d67d8c004a25570db4df5689e06caea6af7"

# mbridge for Megatron Qwen3.5 support
RUN pip install --no-deps --target /AReaL/.venv/lib/python3.12/site-packages \
    git+https://github.com/ISEEKYAN/mbridge.git

# megatron-core 0.16.0 (same as veRL)
RUN pip install --no-deps --target /tmp/megatron-scratch \
        megatron-core==0.16.0 \
 && cp -r /tmp/megatron-scratch/. /AReaL/.venv/lib/python3.12/site-packages/ \
 && rm -rf /tmp/megatron-scratch
