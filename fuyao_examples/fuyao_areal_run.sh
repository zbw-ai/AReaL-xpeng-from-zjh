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
            echo "  --run-type        Training type: math_sft, math_rlvr, search_r1, code_dapo"
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
VALID_RUN_TYPES=("math_sft" "math_rlvr" "search_r1" "code_dapo")
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

# ========================== 3.5 Qwen3.5 GDN 运行时依赖 ==========================
# fla + causal-conv1d 是 GDN 核函数依赖（训练侧需要）
# uv 管理的 venv 没有 pip，必须用 uv pip install
if python -c "from fla.ops.gated_delta_rule import chunk_gated_delta_rule" 2>/dev/null; then
    echo "[qwen3.5-deps] flash-linear-attention already installed, skip."
else
    echo "[qwen3.5-deps] Installing flash-linear-attention + causal-conv1d..."
    uv pip install flash-linear-attention 2>&1 | tail -5
    echo "[qwen3.5-deps] Done."
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

# 多节点检测：Fuyao PyTorchJob 注入 Slurm 兼容环境变量
#   NODE_RANK: 节点序号 (0 ~ NNODES-1)
#   MASTER_ADDR: head 节点 IP
#   SLURM_JOB_NUM_NODES: 总节点数
NODE_RANK="${NODE_RANK:-0}"
NUM_NODES="${SLURM_JOB_NUM_NODES:-1}"
HEAD_ADDR="${MASTER_ADDR:-localhost}"
GPUS_PER_NODE="${SLURM_GPUS_PER_NODE:-8}"
RAY_PORT=6379

echo "Node rank: ${NODE_RANK}, Total nodes: ${NUM_NODES}, Head addr: ${HEAD_ADDR}, GPUs/node: ${GPUS_PER_NODE}"

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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Python 配置
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
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
    math_sft)
        echo "[sandbox] Math SFT does not require sandboxes"
        ;;
esac

# ========================== 7. run-type → Python 入口映射 ==========================
case "$RUN_TYPE" in
    math_sft)   LAUNCH_SCRIPT="fuyao_examples/math/train_math_sft.py" ;;
    math_rlvr)  LAUNCH_SCRIPT="fuyao_examples/math/train_math_rlvr.py" ;;
    search_r1)  LAUNCH_SCRIPT="fuyao_examples/search_r1/train_search_r1.py" ;;
    code_dapo)  LAUNCH_SCRIPT="fuyao_examples/code_dapo/train_code_dapo.py" ;;
    *)          echo "Error: No launch script for run-type '$RUN_TYPE'"; exit 1 ;;
esac

if [[ ! -f "${PROJECT_ROOT}/${LAUNCH_SCRIPT}" ]]; then
    echo "Error: Launch script not found: ${PROJECT_ROOT}/${LAUNCH_SCRIPT}"
    exit 1
fi

# ========================== 8. Ray Cluster 启动（多节点时） ==========================

# 判断是否需要 Ray：当 NUM_NODES > 1 时启动 Ray cluster
USE_RAY=false
if [[ "${NUM_NODES}" -gt 1 ]]; then
    USE_RAY=true
fi

if $USE_RAY; then
    echo "===== Step 4: Setting up Ray cluster (${NUM_NODES} nodes) ====="

    # 通过 NFS 信号文件协调各节点退出，避免 worker 先于 head 退出导致 PyTorchJob 判定失败
    # 使用共享存储路径（和 name_resolve/fileroot 同一 NFS），确保所有节点可见
    RAY_SIGNAL_DIR="/dataset_rc_llmrl/zengbw1/areal_experiments/.ray_signals/${BIFROST_JOB_NAME:-default}"
    mkdir -p "${RAY_SIGNAL_DIR}"

    # 先停掉可能残留的 Ray
    ray stop --force 2>/dev/null || true
    sleep 2

    if [[ "${NODE_RANK}" == "0" ]]; then
        # Head 节点：清理旧信号文件
        rm -f "${RAY_SIGNAL_DIR}/job_done"

        echo "[ray] Starting Ray head on node 0..."
        ray start --head --port=${RAY_PORT} --num-gpus=${GPUS_PER_NODE}

        # 等待所有节点加入
        echo "[ray] Waiting for all ${NUM_NODES} nodes to join..."
        MAX_WAIT=300
        ELAPSED=0
        while true; do
            JOINED=$(python3 -c "
import ray
ray.init(address='auto', ignore_reinit_error=True)
nodes = [n for n in ray.nodes() if n.get('Alive', False)]
print(len(nodes))
ray.shutdown()
" 2>/dev/null || echo "0")
            if [[ "${JOINED}" -ge "${NUM_NODES}" ]]; then
                echo "[ray] All ${NUM_NODES} nodes joined the Ray cluster."
                break
            fi
            if [[ "${ELAPSED}" -ge "${MAX_WAIT}" ]]; then
                echo "[ray] ERROR: Timed out waiting for nodes. Joined: ${JOINED}/${NUM_NODES}"
                ray status 2>/dev/null || true
                exit 1
            fi
            echo "[ray] Waiting... ${JOINED}/${NUM_NODES} nodes joined (${ELAPSED}s elapsed)"
            sleep 5
            ELAPSED=$((ELAPSED + 5))
        done
    else
        echo "[ray] Starting Ray worker on node ${NODE_RANK}, connecting to ${HEAD_ADDR}:${RAY_PORT}..."
        sleep 10  # 给 head 节点时间启动
        ray start --address="${HEAD_ADDR}:${RAY_PORT}" --num-gpus=${GPUS_PER_NODE}

        echo "[ray] Worker node ${NODE_RANK} joined. Waiting for head to signal job completion..."
        # 通过 NFS 信号文件等待 head 节点完成训练（比 ray status 更可靠）
        while [[ ! -f "${RAY_SIGNAL_DIR}/job_done" ]]; do
            sleep 10
        done
        echo "[ray] Job completed. Worker node ${NODE_RANK} exiting."
        ray stop --force 2>/dev/null || true
        exit 0
    fi
fi

# ========================== 9. 启动训练 ==========================
echo ""
echo "================================================================"
echo " AReaL Fuyao Training"
echo " Type:    ${RUN_TYPE}"
echo " Config:  ${CONFIG_PATH}"
echo " Script:  ${LAUNCH_SCRIPT}"
echo " Nodes:   ${NUM_NODES} (Ray: ${USE_RAY})"
[[ "$RUN_TYPE" == "search_r1" ]] && echo " Search:  ${RETRIEVAL_ENDPOINT:-not set}"
[[ "$RUN_TYPE" == "code_dapo" ]] && echo " Code:    ${EXECD_ENDPOINT:-local subprocess}"
[[ -n "$SWANLAB_API_KEY" ]] && echo " Logger:  SwanLab enabled"
echo "================================================================"
echo ""

cd "$PROJECT_ROOT"

# 多节点时自动注入 scheduler.type=ray 和 name_resolve.type=ray
RAY_EXTRA_ARGS=()
if $USE_RAY; then
    RAY_EXTRA_ARGS=(
        scheduler.type=ray
        cluster.name_resolve.type=ray
    )
fi

python3 "${LAUNCH_SCRIPT}" \
    --config "${CONFIG_PATH}" \
    "${RAY_EXTRA_ARGS[@]+"${RAY_EXTRA_ARGS[@]}"}" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"; exit_code=$?

# 清理 Ray 并通知 worker 节点退出
if $USE_RAY; then
    echo "[ray] Signaling worker nodes to exit..."
    touch "${RAY_SIGNAL_DIR}/job_done"
    sleep 5  # 给 worker 节点时间读到信号
    echo "[ray] Stopping Ray cluster..."
    ray stop --force 2>/dev/null || true
fi

echo "Training finished with exit code: ${exit_code}"
exit ${exit_code}
