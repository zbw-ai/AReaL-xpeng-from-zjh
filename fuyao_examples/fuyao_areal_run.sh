#!/bin/bash
set -euo pipefail

################################################################################
# fuyao_areal_run.sh — AReaL 本地训练统一启动脚本
#
# 训练在当前环境本地执行（本机 GPU），fuyao SDK 仅用于按需部署沙盒服务。
#
# 用法:
#   # 场景1: Math RLVR (无需沙盒)
#   bash fuyao_examples/fuyao_areal_run.sh \
#       --run-type math_rlvr \
#       --config fuyao_examples/math/qwen3_4b_rlvr.yaml
#
#   # 场景2: Search R1 (自动部署 search sandbox)
#   bash fuyao_examples/fuyao_areal_run.sh \
#       --run-type search_r1 \
#       --config fuyao_examples/search_r1/search_r1_qwen3_4b.yaml
#
#   # 场景2: 使用已有检索服务
#   RETRIEVAL_ENDPOINT=http://10.1.x.x:8001/retrieve \
#   bash fuyao_examples/fuyao_areal_run.sh \
#       --run-type search_r1 \
#       --config fuyao_examples/search_r1/search_r1_qwen3_4b.yaml \
#       --skip-deploy
#
#   # 场景3: Code DAPO (默认 local subprocess, 无需沙盒)
#   bash fuyao_examples/fuyao_areal_run.sh \
#       --run-type code_dapo \
#       --config fuyao_examples/code_dapo/code_dapo_qwen3_4b.yaml
################################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_STATE_DIR="${PROJECT_ROOT}/.fuyao_run"
mkdir -p "${RUN_STATE_DIR}"

# ========================== 1. 默认参数 ==========================
RUN_TYPE=""
CONFIG_PATH=""
SKIP_DEPLOY=false
SWANLAB_API_KEY="${SWANLAB_API_KEY:-}"
EXTRA_ARGS=()

# ========================== 2. 解析命令行参数 ==========================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-type)       RUN_TYPE="$2"; shift 2;;
        --config)         CONFIG_PATH="$2"; shift 2;;
        --skip-deploy)    SKIP_DEPLOY=true; shift;;
        --swanlab-api-key) SWANLAB_API_KEY="$2"; shift 2;;
        -h|--help)
            echo "Usage: $0 --run-type <type> --config <yaml> [options]"
            echo ""
            echo "Required:"
            echo "  --run-type        Training type: math_rlvr, search_r1, code_dapo"
            echo "  --config          Path to YAML config file"
            echo ""
            echo "Options:"
            echo "  --skip-deploy     Skip sandbox deployment (use existing endpoints)"
            echo "  --swanlab-api-key SwanLab API key for experiment tracking"
            echo ""
            echo "Environment variables:"
            echo "  RETRIEVAL_ENDPOINT  Pre-deployed search endpoint (for search_r1)"
            echo "  EXECD_ENDPOINT      Pre-deployed code sandbox endpoint (for code_dapo --execd)"
            echo "  SWANLAB_API_KEY     SwanLab API key"
            exit 0
            ;;
        --)               shift; EXTRA_ARGS=("$@"); break;;
        *)                EXTRA_ARGS+=("$1"); shift;;
    esac
done

# ========================== 3. 参数校验 ==========================
VALID_RUN_TYPES=("math_rlvr" "search_r1" "code_dapo")
if [[ -z "$RUN_TYPE" ]]; then
    echo "Error: --run-type is required (${VALID_RUN_TYPES[*]})"
    exit 1
fi
if [[ ! " ${VALID_RUN_TYPES[*]} " =~ " ${RUN_TYPE} " ]]; then
    echo "Error: Invalid --run-type '$RUN_TYPE'. Valid: ${VALID_RUN_TYPES[*]}"
    exit 1
fi
if [[ -z "$CONFIG_PATH" ]]; then
    echo "Error: --config is required"
    exit 1
fi
# Resolve relative config path
if [[ ! "$CONFIG_PATH" = /* ]]; then
    CONFIG_PATH="${PROJECT_ROOT}/${CONFIG_PATH}"
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Error: Config file not found: $CONFIG_PATH"
    exit 1
fi

# ========================== 4. 清理残留进程 ==========================
echo "===== Step 1: Clean up tracked residual processes ====="
for pid_file in "${RUN_STATE_DIR}"/*.pid; do
    [[ -e "$pid_file" ]] || continue
    pid="$(cat "$pid_file" 2>/dev/null || true)"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        echo "[cleanup] Stopping tracked process ${pid} from $(basename "$pid_file")"
        kill "$pid" 2>/dev/null || true
    fi
    rm -f "$pid_file"
done

# ========================== 5. 环境配置 ==========================
echo "===== Step 2: Configure environment ====="

# NCCL 配置
unset NCCL_NET_GDR_LEVEL 2>/dev/null || true
export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_IB_TIMEOUT=22
export NCCL_IB_HCA=mlx5
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_CUMEM_ENABLE=0
export NCCL_MAX_NCHANNELS=16

# CUDA 配置
export CUDA_DEVICE_MAX_CONNECTIONS="1"

# Python 配置
# Include sglang from system install; disable sgl_kernel if A100 (sm80, no sm80 kernel)
EXTRA_PATHS="${PROJECT_ROOT}/areal/_stubs"
[[ -d "/sgl-workspace/sglang/python" ]] && EXTRA_PATHS="${EXTRA_PATHS}:/sgl-workspace/sglang/python"
export PYTHONPATH="${PROJECT_ROOT}:${EXTRA_PATHS}:${PYTHONPATH:-}"
export SGLANG_KERNEL_DISABLE=1
export PYTHONUNBUFFERED=1

# SwanLab
if [[ -n "$SWANLAB_API_KEY" ]]; then
    export SWANLAB_API_KEY
    echo "SwanLab tracking enabled"
fi

# ========================== 6. 按需部署沙盒 ==========================
echo "===== Step 3: Deploy sandboxes if needed ====="

deploy_search_sandbox() {
    if [[ -n "${RETRIEVAL_ENDPOINT:-}" ]]; then
        echo "[sandbox] Using existing RETRIEVAL_ENDPOINT=${RETRIEVAL_ENDPOINT}"
        return
    fi
    if $SKIP_DEPLOY; then
        echo "Error: search_r1 requires RETRIEVAL_ENDPOINT but --skip-deploy is set"
        exit 1
    fi
    echo "[sandbox] Deploying search sandbox via fuyao SDK..."
    local deploy_output
    deploy_output=$(python3 "${SCRIPT_DIR}/deploy_sandboxes.py" --sandbox search --export 2>&1)
    local deploy_rc=$?
    echo "$deploy_output"
    if [[ $deploy_rc -ne 0 ]]; then
        echo "[sandbox] ERROR: Search sandbox deployment failed!"
        exit 1
    fi
    # Parse exported endpoint
    while IFS= read -r line; do
        if [[ "$line" =~ ^export\ RETRIEVAL_ENDPOINT=\"(.+)\"$ ]]; then
            export RETRIEVAL_ENDPOINT="${BASH_REMATCH[1]}"
            echo "[sandbox] Exported RETRIEVAL_ENDPOINT=${RETRIEVAL_ENDPOINT}"
        fi
    done <<< "$deploy_output"
    if [[ -z "${RETRIEVAL_ENDPOINT:-}" ]]; then
        echo "[sandbox] ERROR: RETRIEVAL_ENDPOINT not set after deployment!"
        exit 1
    fi
}

case "$RUN_TYPE" in
    search_r1)
        deploy_search_sandbox
        ;;
    code_dapo)
        # Code DAPO defaults to local subprocess, no sandbox needed
        # If user sets EXECD_ENDPOINT, execd mode will be used
        if [[ -n "${EXECD_ENDPOINT:-}" ]]; then
            echo "[sandbox] Using existing EXECD_ENDPOINT=${EXECD_ENDPOINT}"
        else
            echo "[sandbox] Code DAPO using local subprocess mode (no sandbox needed)"
        fi
        ;;
    math_rlvr)
        echo "[sandbox] Math RLVR does not require sandboxes"
        ;;
esac

# ========================== 7. run-type → Python 入口映射 ==========================
declare -A LAUNCH_SCRIPTS=(
    ["math_rlvr"]="fuyao_examples/math/train_math_rlvr.py"
    ["search_r1"]="fuyao_examples/search_r1/train_search_r1.py"
    ["code_dapo"]="fuyao_examples/code_dapo/train_code_dapo.py"
)
LAUNCH_SCRIPT="${LAUNCH_SCRIPTS[$RUN_TYPE]}"

if [[ ! -f "${PROJECT_ROOT}/${LAUNCH_SCRIPT}" ]]; then
    echo "Error: Launch script not found: ${PROJECT_ROOT}/${LAUNCH_SCRIPT}"
    exit 1
fi

# ========================== 8. 启动训练 ==========================
echo ""
echo "================================================================"
echo " AReaL Fuyao Training"
echo " Type:    ${RUN_TYPE}"
echo " Config:  ${CONFIG_PATH}"
echo " Script:  ${LAUNCH_SCRIPT}"
[[ "$RUN_TYPE" == "search_r1" ]] && echo " Search:  ${RETRIEVAL_ENDPOINT:-not set}"
[[ "$RUN_TYPE" == "code_dapo" ]] && echo " Code:    ${EXECD_ENDPOINT:-local subprocess}"
[[ -n "$SWANLAB_API_KEY" ]] && echo " Logger:  SwanLab enabled"
echo "================================================================"
echo ""

cd "$PROJECT_ROOT"

python3 "${LAUNCH_SCRIPT}" \
    --config "${CONFIG_PATH}" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" &
TRAIN_PID=$!
echo "$TRAIN_PID" > "${RUN_STATE_DIR}/${RUN_TYPE}.pid"

cleanup() {
    if [[ -n "${TRAIN_PID:-}" ]] && kill -0 "${TRAIN_PID}" 2>/dev/null; then
        kill "${TRAIN_PID}" 2>/dev/null || true
        wait "${TRAIN_PID}" 2>/dev/null || true
    fi
    rm -f "${RUN_STATE_DIR}/${RUN_TYPE}.pid"
}

trap cleanup EXIT INT TERM
wait "${TRAIN_PID}"
