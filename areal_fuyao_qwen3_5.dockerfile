# Base: v1 image already has transformers>=5.3 + mbridge main.
# Only upgrade sglang (--no-deps to avoid breaking Ray/torch etc).
FROM infra-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/data-infra/fuyao:areal-qwen3_5-megatron-v1-260417-0107
ENV MAX_JOBS=1
RUN pip install --upgrade --no-deps --target /AReaL/.venv/lib/python3.12/site-packages \
    "sglang>=0.5.10"
