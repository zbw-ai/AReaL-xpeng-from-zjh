# Base: v1 image already has transformers>=5.3 + mbridge main + sglang 0.5.9.
# Add: sglang 0.5.10 (Qwen3.5 VLM fix) + vllm (alternative inference backend).
FROM infra-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/data-infra/fuyao:areal-qwen3_5-megatron-v1-260417-0107
ENV MAX_JOBS=1
RUN pip install --upgrade --no-deps --target /AReaL/.venv/lib/python3.12/site-packages \
    "sglang>=0.5.10" && \
    pip install --no-deps --target /AReaL/.venv/lib/python3.12/site-packages \
    "vllm>=0.8.5"
