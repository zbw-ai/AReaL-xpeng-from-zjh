FROM infra-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/data-infra/fuyao:areal-qwen3_5-megatron-v1-260417-0107
ENV MAX_JOBS=1

# Patch AReaL source for Qwen3.5 VLM weight loading and Megatron parameter conversion.
# Base image already has mbridge with Qwen3.5 support (PR#83).
# These COPY commands patch the code that was missing from the base:
#   hf_load.py    — _get_hf_config_attr: handles nested VLM config (Qwen3_5MoeConfig
#                   has no top-level num_key_value_heads; must read from text_config)
#   megatron.py   — convert_qwen3_5_to_hf + _resolve_conversion_fn registry
#   registry.py   — registers Qwen3_5MoeForConditionalGeneration → mbridge path
COPY areal/models/mcore/hf_load.py /code/areal/models/mcore/hf_load.py
COPY areal/models/mcore/registry.py /code/areal/models/mcore/registry.py
COPY areal/engine/megatron_utils/megatron.py /code/areal/engine/megatron_utils/megatron.py
