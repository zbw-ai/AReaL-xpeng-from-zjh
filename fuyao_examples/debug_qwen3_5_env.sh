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
import importlib.metadata as metadata

def show_version(dist_name, label=None):
    label = label or dist_name
    try:
        print(f'{label}: {metadata.version(dist_name)}')
    except metadata.PackageNotFoundError:
        print(f'{label}: NOT INSTALLED')
    except Exception as e:
        print(f'{label}: ERROR ({type(e).__name__}: {e})')

show_version('transformers')
show_version('huggingface-hub', 'huggingface_hub')
show_version('tokenizers')
show_version('sglang')
show_version('torch')
show_version('ray')

try:
    import transformers
    print(f'transformers import: OK ({transformers.__version__})')
except Exception as e:
    print(f'❌ transformers import failed: {type(e).__name__}: {e}')
    if 'is_offline_mode' in str(e):
        print('   可能原因: transformers 与 huggingface_hub 版本不兼容')
        print('   建议修复 A (Qwen3.5 推荐): uv pip install --upgrade transformers tokenizers')
        print('   建议修复 B (仅兼容 4.x): uv pip install \"huggingface_hub<1.0\"')
    raise SystemExit(2)

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
import socket

try:
    import mbridge
    import torch.distributed as dist
    from megatron.core import parallel_state as mpu

    def find_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('127.0.0.1', 0))
            return s.getsockname()[1]

    if not dist.is_initialized():
        dist.init_process_group(
            backend='gloo',
            init_method=f'tcp://127.0.0.1:{find_free_port()}',
            rank=0,
            world_size=1,
        )
    if not mpu.model_parallel_is_initialized():
        mpu.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            expert_model_parallel_size=1,
            expert_tensor_parallel_size=1,
            use_sharp=False,
            order='tp-cp-ep-dp-pp',
        )

    bridge = mbridge.AutoBridge.from_pretrained(
        '${MODEL_PATH}', trust_remote_code=True, dtype='bfloat16'
    )
    print(f'✅ mbridge 加载成功: {type(bridge).__name__}')
    print(f'   hf_config: {type(bridge.hf_config).__name__}')
except Exception as e:
    print(f'❌ mbridge 加载失败: {e}')
finally:
    try:
        if 'mpu' in globals() and mpu.model_parallel_is_initialized():
            mpu.destroy_model_parallel()
    except Exception:
        pass
    try:
        if 'dist' in globals() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass
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
SGLANG_OUTPUT="$(timeout 30 python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --tp 1 \
    --mem-fraction-static 0.5 \
    --context-length 2048 \
    --host 127.0.0.1 \
    --port 30000 \
    --disable-custom-all-reduce \
    2>&1 || true)"
printf '%s\n' "${SGLANG_OUTPUT}" | head -50
if printf '%s' "${SGLANG_OUTPUT}" | grep -q "SGLANG_DISABLE_CUDNN_CHECK=1"; then
    echo "提示: 当前被 SGLang 的 CuDNN 兼容性检查拦截。"
    echo "  临时跳过检查: export SGLANG_DISABLE_CUDNN_CHECK=1"
    echo "  推荐修复: pip install nvidia-cudnn-cu12==9.16.0.29"
elif [ -z "${SGLANG_OUTPUT}" ]; then
    echo "(超时或退出 — 查看上面的输出判断是否成功)"
fi

echo ""
echo "===== 调试完成 ====="
