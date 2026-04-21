# Base: has AReaL deps (psutil, uvloop, swanlab, orjson, httpx, etc.)
# Upgrade key packages to veRL's validated versions for Qwen3.5
FROM infra-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/data-infra/fuyao:zhangjh37-260325-0644
ENV MAX_JOBS=1

# 1. transformers: pin to veRL's exact commit, delete old first
RUN rm -rf /AReaL/.venv/lib/python3.12/site-packages/transformers* \
           /AReaL/.venv/lib/python3.12/site-packages/huggingface_hub* \
           /AReaL/.venv/lib/python3.12/site-packages/tokenizers* \
           /AReaL/.venv/lib/python3.12/site-packages/hf_xet* && \
    pip install --upgrade --target /AReaL/.venv/lib/python3.12/site-packages \
    "torch==2.9.1" \
    "transformers @ git+https://github.com/huggingface/transformers.git@d64a6d67d8c004a25570db4df5689e06caea6af7"

# 2. mbridge: pin to veRL's exact commit (has qwen3_5_moe registration)
RUN rm -rf /AReaL/.venv/lib/python3.12/site-packages/mbridge* && \
    pip install -U --target /AReaL/.venv/lib/python3.12/site-packages \
    "mbridge @ git+https://github.com/ISEEKYAN/mbridge.git@dc1321b95dfdb50b9aa680f785b92936127a51fa"

# 3. megatron-core 0.16.0 (namespace merge)
RUN pip install --no-deps --target /tmp/megatron-scratch \
        megatron-core==0.16.0 \
 && cp -r /tmp/megatron-scratch/. /AReaL/.venv/lib/python3.12/site-packages/ \
 && rm -rf /tmp/megatron-scratch

# 4. vLLM 0.17.0 (veRL uses this for Qwen3.5 inference, not SGLang)
#    Install WITH deps (vllm has many: msgspec, outlines_core, etc.)
#    Pin torch to prevent upgrade.
RUN pip install --target /AReaL/.venv/lib/python3.12/site-packages \
    vllm==0.17.0 "torch==2.9.1"

# 5. flash-linear-attention for Qwen3.5 GDN layers
RUN pip install --no-deps --target /AReaL/.venv/lib/python3.12/site-packages \
    flash-linear-attention==0.4.2

# 5. CuDNN 9.16 for SGLang CuDNN check (if SGLang is used)
RUN mkdir -p /tmp/cudnn-fix && \
    pip download --no-deps -d /tmp/cudnn-fix nvidia-cudnn-cu12==9.16.0.29 && \
    rm -rf /AReaL/.venv/lib/python3.12/site-packages/nvidia/cudnn \
           /AReaL/.venv/lib/python3.12/site-packages/nvidia_cudnn_cu12-*.dist-info && \
    python3 -m zipfile -e /tmp/cudnn-fix/nvidia_cudnn_cu12-9.16.0.29-*.whl /AReaL/.venv/lib/python3.12/site-packages && \
    rm -rf /tmp/cudnn-fix

ENV LD_LIBRARY_PATH=/AReaL/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}
