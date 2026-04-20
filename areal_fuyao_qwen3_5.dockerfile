# Base: v1 image has transformers>=5.3 + mbridge main + sglang 0.5.9
#   torch 2.9.1+cu129, Python 3.12
# Add vLLM as inference backend for Qwen3.5 VLM.
# Use venv pip (not --no-deps) so pip resolves torch/CUDA compatibility.
FROM infra-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/data-infra/fuyao:areal-qwen3_5-megatron-v1-260417-0107
ENV MAX_JOBS=1
RUN /AReaL/.venv/bin/pip install "vllm>=0.18.0"
