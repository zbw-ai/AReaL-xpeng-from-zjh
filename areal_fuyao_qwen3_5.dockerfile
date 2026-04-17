FROM infra-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/data-infra/fuyao:areal-qwen3_5-v2-260416-2334
ENV MAX_JOBS=1

# Megatron Qwen3.5: mbridge main branch contains Qwen3.5 bridge (PR#83)
# transformers already upgraded in base image
ARG MBRIDGE_REF=main
RUN pip install --upgrade --target /AReaL/.venv/lib/python3.12/site-packages \
    git+https://github.com/ISEEKYAN/mbridge.git@${MBRIDGE_REF}

# Patch AReaL source for Qwen3.5 VLM weight loading and Megatron parameter conversion.
# These files extend the base image with Qwen3.5-specific logic:
#   hf_load.py    — _get_hf_config_attr handles nested VLM configs (no num_key_value_heads)
#   megatron.py   — convert_qwen3_5_to_hf + _resolve_conversion_fn registry
#   registry.py   — registers Qwen3_5MoeForConditionalGeneration with mbridge path
COPY areal/models/mcore/hf_load.py /code/areal/models/mcore/hf_load.py
COPY areal/models/mcore/registry.py /code/areal/models/mcore/registry.py
COPY areal/engine/megatron_utils/megatron.py /code/areal/engine/megatron_utils/megatron.py
