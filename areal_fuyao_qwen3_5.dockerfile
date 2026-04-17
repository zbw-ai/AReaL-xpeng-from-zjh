FROM infra-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/data-infra/fuyao:zhangjh37-260325-0644
ENV MAX_JOBS=1

# Qwen3.5 Megatron support:
# 1) mbridge main branch contains Qwen3.5 bridge (PR#83)
# 2) transformers>=5.3.0 has qwen3_5_moe model type
RUN pip install --upgrade --target /AReaL/.venv/lib/python3.12/site-packages \
    transformers tokenizers \
    git+https://github.com/ISEEKYAN/mbridge.git
