# Base: v1 image has transformers>=5.3 + sglang 0.5.9 + torch 2.9.1+cu129
# Add: vLLM, sglang 0.5.10, megatron-bridge, trackio
FROM infra-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/data-infra/fuyao:areal-qwen3_5-megatron-v1-260417-0107
ENV MAX_JOBS=1
RUN pip install --target /AReaL/.venv/lib/python3.12/site-packages \
        "vllm>=0.18.0" && \
    pip install --upgrade --no-deps --target /AReaL/.venv/lib/python3.12/site-packages \
        "sglang>=0.5.10" \
        megatron-bridge==0.3.0 \
        trackio==0.2.2
