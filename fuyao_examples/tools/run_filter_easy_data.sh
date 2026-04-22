#!/bin/bash
set -euo pipefail

################################################################################
# run_filter_easy_data.sh — POLARIS-style 数据过滤
#
# 在 GPU 节点上：启动 SGLang → 对训练集做 rollout+评分 → 过滤简单题 → 输出新 parquet
#
# 用法（通过 fuyao 提交）:
#   fuyao deploy \
#       --docker-image=infra-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/data-infra/fuyao:zhangjh37-260325-0644 \
#       --project=rc-ai-infra --experiment=zengbw1/llm_rl \
#       --gpu-type a100 --gpus-per-node 4 --node=1 \
#       --label=filter-easy-data \
#       --site=fuyao_b1 --queue=rc-llmrl-a100 \
#       bash fuyao_examples/tools/run_filter_easy_data.sh
################################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# ========================== 配置 ==========================
# 训练后的 checkpoint 路径（step199 = best avg）
CKPT_PATH="${CKPT_PATH:-/dataset_rc_llmrl/zengbw1/areal_experiments/bifrost-2026041721040601-zengbw1-qwen3-4b-polaris-rlvr/files/checkpoints/root/bifrost-2026041721040601-zengbw1-qwen3-4b-polaris-rlvr/trial0/default/epoch0epochstep199globalstep199}"

# 原始训练数据
DATASET_PATH="${DATASET_PATH:-/workspace/zhangjh37@xiaopeng.com/data/dapo_math_17k_processed}"
DATASET_TYPE="${DATASET_TYPE:-dapo_math}"

# 输出路径
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/zengbw1@xiaopeng.com/dataset/dapo_math}"
OUTPUT_FILE="${OUTPUT_DIR}/dapo_math_stage2_filtered.parquet"

# 过滤参数
N_SAMPLES="${N_SAMPLES:-8}"           # 每个 prompt 生成几个 response (与训练一致)
TEMPERATURE="${TEMPERATURE:-1.4}"     # 采样温度
MAX_TOKENS="${MAX_TOKENS:-4096}"      # 最大生成长度 (训练 avg=3.4K, 4K 足够判断难度)
THRESHOLD="${THRESHOLD:-0.9}"         # avg_reward > 此值的题被 drop
BATCH_SIZE="${BATCH_SIZE:-32}"        # 并发请求数

# SGLang 参数 — 自动检测 GPU 数量，启动尽可能多的 TP4 实例
TP="${TP:-4}"
TOTAL_GPUS="${SLURM_GPUS_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
NUM_INSTANCES="${NUM_INSTANCES:-$((TOTAL_GPUS / TP))}"
SGLANG_BASE_PORT=30000

echo "================================================================"
echo " POLARIS-style Easy Data Filter"
echo " Checkpoint: ${CKPT_PATH}"
echo " Dataset:    ${DATASET_PATH} (${DATASET_TYPE})"
echo " Output:     ${OUTPUT_FILE}"
echo " Threshold:  ${THRESHOLD} (drop avg_reward > ${THRESHOLD})"
echo " N_Samples:  ${N_SAMPLES}, Temp: ${TEMPERATURE}"
echo "================================================================"

# ========================== 1. 检查 checkpoint ==========================
if [ ! -d "${CKPT_PATH}" ]; then
    echo "ERROR: Checkpoint not found: ${CKPT_PATH}"
    exit 1
fi
echo "[1/4] Checkpoint exists: $(ls ${CKPT_PATH}/*.safetensors 2>/dev/null | wc -l) safetensors files"

# ========================== 2. 启动 SGLang server ==========================
echo "[2/4] Starting ${NUM_INSTANCES} SGLang server(s) (TP=${TP}, 8 GPU total)..."

SGLANG_PIDS=()
SGLANG_URLS=()

for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    PORT=$((SGLANG_BASE_PORT + i))
    FIRST_GPU=$((i * TP))
    GPU_LIST=$(seq -s, ${FIRST_GPU} $((FIRST_GPU + TP - 1)))

    echo "  Instance ${i}: GPU [${GPU_LIST}], port ${PORT}"
    CUDA_VISIBLE_DEVICES="${GPU_LIST}" python3 -m sglang.launch_server \
        --model-path "${CKPT_PATH}" \
        --tp "${TP}" \
        --port "${PORT}" \
        --dtype bfloat16 \
        --mem-fraction-static 0.85 \
        --disable-custom-all-reduce \
        --context-length 5120 \
        &
    SGLANG_PIDS+=($!)
    SGLANG_URLS+=("http://localhost:${PORT}")
done

# 等待所有 server 就绪
echo "Waiting for SGLang servers to be ready..."
MAX_WAIT=600
ELAPSED=0
ALL_READY=false
while ! $ALL_READY; do
    ALL_READY=true
    for url in "${SGLANG_URLS[@]}"; do
        if ! curl -sf "${url}/health" > /dev/null 2>&1; then
            ALL_READY=false
            break
        fi
    done
    if $ALL_READY; then break; fi

    # Check if any process died
    for pid in "${SGLANG_PIDS[@]}"; do
        if ! kill -0 ${pid} 2>/dev/null; then
            echo "ERROR: SGLang server process ${pid} died"
            for p in "${SGLANG_PIDS[@]}"; do kill ${p} 2>/dev/null || true; done
            exit 1
        fi
    done

    if [ ${ELAPSED} -ge ${MAX_WAIT} ]; then
        echo "ERROR: SGLang servers did not start within ${MAX_WAIT}s"
        for p in "${SGLANG_PIDS[@]}"; do kill ${p} 2>/dev/null || true; done
        exit 1
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    echo "  Waiting... ${ELAPSED}s"
done
echo "All ${NUM_INSTANCES} SGLang servers ready! (took ${ELAPSED}s)"

# 构建逗号分隔的 URL 列表传给 filter 脚本
SGLANG_URL_LIST=$(IFS=,; echo "${SGLANG_URLS[*]}")

# ========================== 3. 运行过滤 ==========================
echo "[3/4] Running data filter..."
mkdir -p "${OUTPUT_DIR}"

python3 fuyao_examples/tools/filter_easy_data.py \
    --dataset-path "${DATASET_PATH}" \
    --dataset-type "${DATASET_TYPE}" \
    --server-url "${SGLANG_URL_LIST}" \
    --n-samples "${N_SAMPLES}" \
    --temperature "${TEMPERATURE}" \
    --max-tokens "${MAX_TOKENS}" \
    --threshold "${THRESHOLD}" \
    --batch-size "${BATCH_SIZE}" \
    --output "${OUTPUT_FILE}"

FILTER_EXIT=$?

# ========================== 4. 清理 ==========================
echo "[4/4] Stopping SGLang servers..."
for pid in "${SGLANG_PIDS[@]}"; do
    kill ${pid} 2>/dev/null || true
    wait ${pid} 2>/dev/null || true
done

if [ ${FILTER_EXIT} -eq 0 ]; then
    echo ""
    echo "================================================================"
    echo " Done! Filtered dataset saved to:"
    echo "   ${OUTPUT_FILE}"
    echo ""
    echo " Stats saved to:"
    echo "   ${OUTPUT_FILE%.parquet}.stats.json"
    echo ""
    echo " Next step: use this in Stage-2 training yaml:"
    echo "   train_dataset:"
    echo "     path: ${OUTPUT_FILE}"
    echo "     type: dapo_math"
    echo "================================================================"
else
    echo "ERROR: Filter script failed with exit code ${FILTER_EXIT}"
    exit ${FILTER_EXIT}
fi
