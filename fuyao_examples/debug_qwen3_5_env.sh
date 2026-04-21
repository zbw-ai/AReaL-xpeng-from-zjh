#!/bin/bash
# Qwen3.5 VLM 环境调试脚本
# 在单 pod 上验证 transformers + SGLang + mbridge 是否兼容
#
# 用法: 在 pod 上执行
#   bash /code/fuyao_examples/debug_qwen3_5_env.sh
#
# 需要: GPU pod，模型路径 /dataset_rc_b1/models/Qwen3.5-0.8B 或 Qwen3.5-35B-A3B

set -euo pipefail

MODEL_PATH="${1:-/dataset_rc_b1/models/Qwen3.5-0.8B}"
echo "========================================="
echo " Qwen3.5 VLM 环境调试"
echo " Model: ${MODEL_PATH}"
echo "========================================="

echo ""
echo "===== Step 1: 检查包版本 ====="
python3 -c "
import transformers; print(f'transformers: {transformers.__version__}')
import sglang; print(f'sglang: {sglang.__version__}')
import torch; print(f'torch: {torch.__version__}')
import ray; print(f'ray: {ray.__version__}')
try:
    import mbridge; print(f'mbridge: OK')
except: print('mbridge: NOT FOUND')
try:
    import megatron.core as mc; print(f'megatron-core: {mc.__version__}')
except: print('megatron-core: NOT FOUND')
"

echo ""
echo "===== Step 2: 检查 transformers 是否注册 qwen3_5_moe ====="
python3 -c "
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
if 'qwen3_5_moe' in CONFIG_MAPPING:
    print('✅ qwen3_5_moe 已注册')
else:
    print('❌ qwen3_5_moe 未注册 — 需要升级 transformers')
    print('   可用的 qwen 类型:', [k for k in CONFIG_MAPPING if 'qwen' in k.lower()])
"

echo ""
echo "===== Step 3: 检查 AutoConfig 加载 ====="
python3 -c "
from transformers import AutoConfig
try:
    config = AutoConfig.from_pretrained('${MODEL_PATH}', trust_remote_code=True)
    print(f'✅ AutoConfig 加载成功: {type(config).__name__}')
    print(f'   architectures: {getattr(config, \"architectures\", \"N/A\")}')
    tc = getattr(config, 'text_config', None)
    vc = getattr(config, 'vision_config', None)
    print(f'   text_config: {type(tc).__name__ if tc else \"None\"}')
    print(f'   vision_config: {type(vc).__name__ if vc else \"None\"}')
    if tc:
        print(f'   text_config.num_attention_heads: {getattr(tc, \"num_attention_heads\", \"MISSING\")}')
        print(f'   text_config.hidden_size: {getattr(tc, \"hidden_size\", \"MISSING\")}')
        print(f'   text_config.num_hidden_layers: {getattr(tc, \"num_hidden_layers\", \"MISSING\")}')
        print(f'   text_config is dict: {isinstance(tc, dict)}')
except Exception as e:
    print(f'❌ AutoConfig 加载失败: {e}')
"

echo ""
echo "===== Step 4: 检查 mbridge 能否加载模型 ====="
python3 -c "
try:
    import mbridge
    bridge = mbridge.AutoBridge.from_pretrained('${MODEL_PATH}', dtype='bfloat16')
    print(f'✅ mbridge 加载成功: {type(bridge).__name__}')
    print(f'   hf_config: {type(bridge.hf_config).__name__}')
except Exception as e:
    print(f'❌ mbridge 加载失败: {e}')
"

echo ""
echo "===== Step 5: 检查 SGLang 能否解析模型 config ====="
python3 -c "
try:
    from sglang.srt.utils.hf_transformers_utils import get_config
    config = get_config('${MODEL_PATH}', trust_remote_code=True)
    print(f'✅ SGLang get_config 成功: {type(config).__name__}')
except Exception as e:
    print(f'❌ SGLang get_config 失败: {e}')

try:
    from sglang.srt.utils.hf_transformers_utils import get_hf_text_config
    from transformers import AutoConfig
    hf_config = AutoConfig.from_pretrained('${MODEL_PATH}', trust_remote_code=True)
    text_config = get_hf_text_config(hf_config)
    print(f'✅ SGLang get_hf_text_config 成功: {type(text_config).__name__}')
    print(f'   num_attention_heads: {getattr(text_config, \"num_attention_heads\", \"MISSING\")}')
except Exception as e:
    print(f'❌ SGLang get_hf_text_config 失败: {e}')
"

echo ""
echo "===== Step 6: 尝试启动 SGLang server (10s 超时) ====="
echo "启动 SGLang server..."
timeout 30 python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --tp 1 \
    --mem-fraction-static 0.5 \
    --context-length 2048 \
    --host 127.0.0.1 \
    --port 30000 \
    --disable-custom-all-reduce \
    2>&1 | head -50 || echo "(超时或退出 — 查看上面的输出判断是否成功)"

echo ""
echo "===== 调试完成 ====="
