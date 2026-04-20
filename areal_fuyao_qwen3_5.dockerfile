FROM infra-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/data-infra/fuyao:zhangjh37-260325-0644
ENV MAX_JOBS=1

# Qwen3.5-35B-A3B (VLM MoE) requires:
#   1) sglang>=0.5.10   — 0.5.9 crashes on Qwen3.5 VLM text_config
#   2) transformers>=5.3 — qwen3_5_moe model type registration
#   3) mbridge main      — Qwen3.5 Megatron bridge (PR#83)
RUN pip install --upgrade --target /AReaL/.venv/lib/python3.12/site-packages \
    "sglang>=0.5.10" \
    transformers tokenizers \
    git+https://github.com/ISEEKYAN/mbridge.git
