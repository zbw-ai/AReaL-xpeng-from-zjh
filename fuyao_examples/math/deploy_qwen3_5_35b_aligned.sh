#!/bin/bash
# Deploy Qwen3.5-35B-A3B Math RLVR with 8B-benchmark-aligned hyperparameters.
#
# Usage:
#   bash fuyao_examples/math/deploy_qwen3_5_35b_aligned.sh [LABEL_SUFFIX]
#
# 这是 v17 (qwen3_5_35b_a3b_rlvr_vllm_6node.yaml) 跑通后, 把超参对齐到
# qwen3_8b_rlvr_bench_v070_sync_aligned.yaml 的派生版本。固定 6 节点 PP=4
# 布局, 改 n_samples=8, 采样温度等。详见同目录的
# qwen3_5_35b_a3b_rlvr_vllm_6node_aligned.yaml 顶部注释。
#
# 关键约束 (无法对齐 8B):
#   - weight_update_mode: disk (xccl 与 q||gate 不兼容)
#   - optimizer: adam fp32 (不用 adam_bf16, 精度优先)
#   - enable_offload: true (Actor+Ref colocate 必需)
#   - pad_to_maximum: true (Qwen3.5 GDN 强制 BSHD)
#   - max_new_tokens: 2048 (8B 是 31744; 35B 上 token/step 会爆显存)
#
# Pre-flight:
#   - SWANLAB_API_KEY 必须导出, 否则 stats 无法上传
#   - NFS fileroot (默认 /dataset_rc_llmrl/zengbw1/areal_experiments/) 需有 ≥200 GB 空闲
#   - deepmath_math_rule_20k 数据集需要可读: /workspace/lijl42@xiaopeng.com/datasets/.../deepmath_math_rule_20k.parquet
#   - aime_2024_rule.parquet 需要可读

set -euo pipefail

# ── 默认参数 ──
DOCKER_IMAGE="${DOCKER_IMAGE:-infra-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/data-infra/fuyao:areal-qwen3_5-megatron-v21}"
PROJECT="${PROJECT:-rc-ai-infra}"
EXPERIMENT="${EXPERIMENT:-zengbw1/llm_rl}"
SITE="${SITE:-fuyao_b1}"
QUEUE="${QUEUE:-rc-llmrl-a100}"
GPU_TYPE="${GPU_TYPE:-a100}"
N_NODES="${N_NODES:-6}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
LABEL_SUFFIX="${1:-aligned-v1}"
LABEL="qwen3_5-35b-a3b-${LABEL_SUFFIX}"

# ── 检查 ──
if [ -z "${SWANLAB_API_KEY:-}" ]; then
    echo "[WARN] SWANLAB_API_KEY 未设置, swanlab 上传会被跳过。"
    echo "       export SWANLAB_API_KEY=<your_key> 后再执行可解决。"
fi

CONFIG="fuyao_examples/math/qwen3_5_35b_a3b_rlvr_vllm_6node_aligned.yaml"
if [ ! -f "$CONFIG" ]; then
    echo "[FATAL] config not found: $CONFIG"
    echo "        请确认在 AReaL repo 根目录执行此脚本"
    exit 1
fi

# ── 展示配置, 等待用户确认 ──
cat <<EOF
========================================================================
 Deploy Qwen3.5-35B-A3B Math RLVR (8B-aligned hyperparameters)
========================================================================
  docker image       : $DOCKER_IMAGE
  cluster            : ${N_NODES} × ${GPUS_PER_NODE} GPU (${GPU_TYPE})
  site / queue       : $SITE / $QUEUE
  project / exp      : $PROJECT / $EXPERIMENT
  label              : $LABEL
  config             : $CONFIG
  swanlab            : $([ -n "${SWANLAB_API_KEY:-}" ] && echo "enabled" || echo "DISABLED")

 Aligned to 8B benchmark:
   n_samples=8, batch_size=4 → 32 samples/step (4× v17)
   temperature=0.99, top_p=0.99, top_k=100
   weight_decay=0.1
   ppo_n_minibatches=4 (32/4=8/minibatch ÷ DP=4 = 2 samples/rank)

 35B-mandatory (different from 8B):
   weight_update_mode=disk
   optimizer=adam fp32
   enable_offload=true
   pad_to_maximum=true
   max_new_tokens=2048 (vs 8B's 31744; 显存约束)
========================================================================
EOF

read -r -p "Continue with deploy? [y/N] " confirm
case "$confirm" in
    [yY][eE][sS]|[yY]) ;;
    *)
        echo "Aborted."
        exit 0
        ;;
esac

# ── 提交任务 ──
echo "[INFO] Submitting fuyao job..."
fuyao deploy --disable-fault-tolerance \
    --docker-image="$DOCKER_IMAGE" \
    --project="$PROJECT" \
    --experiment="$EXPERIMENT" \
    --gpu-type "$GPU_TYPE" \
    --gpus-per-node "$GPUS_PER_NODE" \
    --node="$N_NODES" \
    --label="$LABEL" \
    --site="$SITE" \
    --queue="$QUEUE" \
    SWANLAB_API_KEY="${SWANLAB_API_KEY:-}" \
    bash fuyao_examples/fuyao_areal_run.sh \
        --run-type math_rlvr \
        --config "$CONFIG"

echo "[OK] Submitted. 查询任务状态:"
echo "     echo 'N' | fuyao history | grep -A2 '$LABEL'"
echo "     echo 'N' | fuyao log --job-name <bifrost-id> --tail 200 --no-interactive"
