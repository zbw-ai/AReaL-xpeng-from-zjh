# Qwen3.5 Megatron + SGLang image
FROM infra-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/data-infra/fuyao:zhangjh37-260325-0644
ENV MAX_JOBS=1

# Align the HF stack with the Qwen3.5 path validated in the debug pod.
# Keep torch pinned so pip resolves transformers deps without upgrading torch itself.
RUN rm -rf /AReaL/.venv/lib/python3.12/site-packages/transformers* \
           /AReaL/.venv/lib/python3.12/site-packages/huggingface_hub* \
           /AReaL/.venv/lib/python3.12/site-packages/tokenizers* \
           /AReaL/.venv/lib/python3.12/site-packages/hf_xet* && \
    pip install --upgrade --target /AReaL/.venv/lib/python3.12/site-packages \
    "torch==2.9.1" \
    "transformers @ git+https://github.com/huggingface/transformers.git@cc7ab9be"

# mbridge: install WITH deps (--no-deps misses sub-modules that register qwen3_5_moe bridge).
# Also rm old mbridge to avoid --target overlay issues.
RUN rm -rf /AReaL/.venv/lib/python3.12/site-packages/mbridge* && \
    pip install --upgrade --target /AReaL/.venv/lib/python3.12/site-packages \
    git+https://github.com/ISEEKYAN/mbridge.git

# SGLang on torch 2.9.1 requires CuDNN >= 9.15 at runtime. Unpack the wheel payload
# directly into the venv so libcudnn is guaranteed to be available under site-packages.
RUN mkdir -p /tmp/cudnn-fix && \
    pip download --no-deps -d /tmp/cudnn-fix nvidia-cudnn-cu12==9.16.0.29 && \
    rm -rf /AReaL/.venv/lib/python3.12/site-packages/nvidia/cudnn \
           /AReaL/.venv/lib/python3.12/site-packages/nvidia_cudnn_cu12-*.dist-info && \
    python3 -m zipfile -e /tmp/cudnn-fix/nvidia_cudnn_cu12-9.16.0.29-*.whl /AReaL/.venv/lib/python3.12/site-packages && \
    rm -rf /tmp/cudnn-fix

ENV LD_LIBRARY_PATH=/AReaL/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}

# megatron-core 0.16.0
RUN pip install --no-deps --target /tmp/megatron-scratch \
        megatron-core==0.16.0 \
 && cp -r /tmp/megatron-scratch/. /AReaL/.venv/lib/python3.12/site-packages/ \
 && rm -rf /tmp/megatron-scratch
