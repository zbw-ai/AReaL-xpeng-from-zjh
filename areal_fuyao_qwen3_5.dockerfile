# veRL image: has torch 2.10, vllm 0.17, mbridge@dc1321b, transformers@d64a6d6
# Add AReaL-specific deps that veRL image doesn't have
FROM infra-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/data-infra/fuyao:verl-qwen3_5-v9-latest
ENV MAX_JOBS=1

# AReaL deps missing from veRL image
RUN pip install --target /AReaL/.venv/lib/python3.12/site-packages \
    math_verify \
    uvloop \
    aiofiles \
    colorlog \
    swanlab
