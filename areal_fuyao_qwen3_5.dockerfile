# Use veRL's proven Qwen3.5 image — all packages pre-validated:
# torch 2.10.0, vllm 0.17.0, mbridge@dc1321b, transformers@d64a6d6,
# flash-linear-attention 0.4.2, megatron-core 0.16.0, etc.
FROM infra-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/data-infra/fuyao:verl-qwen3_5-v9-latest
ENV MAX_JOBS=1
