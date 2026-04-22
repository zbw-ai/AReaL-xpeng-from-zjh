# veRL image: has torch 2.10, vllm 0.17, mbridge@dc1321b, transformers@d64a6d6
# Add AReaL-specific deps that veRL image doesn't have
FROM infra-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/data-infra/fuyao:verl-qwen3_5-v9-latest
ENV MAX_JOBS=1

# Only 3 packages AReaL needs that veRL doesn't have
# (uvloop, colorlog, swanlab, psutil, orjson etc. already in veRL)
RUN pip install aiofiles tensorboardX math_verify
