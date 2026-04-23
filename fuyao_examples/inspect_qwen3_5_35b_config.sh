#!/bin/bash
# Pre-flight check for Qwen3.5-35B-A3B before first RL run.
# Reads the model's config.json and prints the key GQA/MoE parameters
# needed to decide actor/vLLM TP.
#
# Usage:
#   fuyao deploy ... bash fuyao_examples/inspect_qwen3_5_35b_config.sh

set -uo pipefail

MODEL_PATH="${1:-/dataset_rc_b1/models/Qwen3.5-35B-A3B}"

echo "================================================================"
echo " Qwen3.5-35B-A3B config check — ${MODEL_PATH}"
echo "================================================================"

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "ERROR: config.json not found at ${MODEL_PATH}"
    exit 1
fi

echo ""
echo "--- Raw text_config fields (relevant for TP/EP sizing):"
python3 <<EOF
import json
with open("${MODEL_PATH}/config.json") as f:
    cfg = json.load(f)
tc = cfg.get("text_config", cfg)
keys = [
    "num_attention_heads",
    "num_key_value_heads",
    "num_hidden_layers",
    "hidden_size",
    "head_dim",
    "intermediate_size",
    "attn_output_gate",
    "num_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
    "vocab_size",
    "max_position_embeddings",
    "model_type",
]
for k in keys:
    v = tc.get(k, cfg.get(k, "<missing>"))
    print(f"  {k:30s} = {v}")
EOF

echo ""
echo "--- TP recommendation based on num_key_value_heads:"
python3 <<EOF
import json
with open("${MODEL_PATH}/config.json") as f:
    cfg = json.load(f)
tc = cfg.get("text_config", cfg)
kv = tc.get("num_key_value_heads", cfg.get("num_key_value_heads"))
qh = tc.get("num_attention_heads", cfg.get("num_attention_heads"))
print(f"  num_q_heads = {qh}, num_kv_heads = {kv}")
print(f"  Actor TP must be <= {kv} (gate 2x bug if TP > num_kv_heads)")
print(f"  vLLM  TP must be <= {kv} (GQA replicate shape bug if TP > num_kv_heads)")
print()
print(f"  Suggested backends:")
if kv >= 8:
    print(f"    actor.backend: megatron:(attn:d2p2t{min(kv, 8)}|ffn:e8t1)")
    print(f"    rollout.backend: vllm:d2t{min(kv, 8)}")
elif kv >= 4:
    print(f"    actor.backend: megatron:(attn:d4p2t{kv}|ffn:e8t1)")
    print(f"    rollout.backend: vllm:d{16 // kv}t{kv}")
elif kv == 2:
    print(f"    actor.backend: megatron:(attn:d4p2t2|ffn:e8t1)   (TP=2 for attn)")
    print(f"    rollout.backend: vllm:d8t2  or  vllm:d16t1 (safer)")
else:
    print(f"    WARNING: num_kv_heads = {kv}, very constrained TP.")
EOF

echo ""
echo "================================================================"
echo " DONE"
echo "================================================================"
