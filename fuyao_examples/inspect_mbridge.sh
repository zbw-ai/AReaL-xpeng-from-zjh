#!/bin/bash
# Inspect cluster's mbridge qwen3_5 source and Megatron _apply_output_gate impl.
# Run via fuyao deploy (no GPUs needed, just prints source code and exits).
#
# Usage:
#   fuyao deploy --docker-image=... --gpu-type a100 --gpus-per-node 1 --node=1 \
#       --label=inspect-mbridge bash fuyao_examples/inspect_mbridge.sh

set -uo pipefail

echo "================================================================"
echo " 1. mbridge version / install info"
echo "================================================================"
pip show mbridge 2>&1 | head -10
echo ""
echo "--- mbridge installed location:"
python3 -c "import mbridge; print(mbridge.__file__)" 2>&1
echo ""

echo "================================================================"
echo " 2. mbridge qwen3_5 folder listing"
echo "================================================================"
MBRIDGE_QWEN3_5="/usr/local/lib/python3.12/dist-packages/mbridge/models/qwen3_5"
if [[ -d "${MBRIDGE_QWEN3_5}" ]]; then
    ls -la "${MBRIDGE_QWEN3_5}"
else
    echo "NOT FOUND. Searching for qwen3_5 folder:"
    find /usr/local/lib/python3.12/dist-packages/mbridge -type d -name "qwen3*" 2>&1
fi
echo ""

echo "================================================================"
echo " 3. mbridge qwen3_5 attention.py FULL SOURCE"
echo "================================================================"
ATT_FILE="${MBRIDGE_QWEN3_5}/attention.py"
if [[ -f "${ATT_FILE}" ]]; then
    echo "--- File: ${ATT_FILE}"
    echo "--- Lines: $(wc -l < ${ATT_FILE})"
    cat -n "${ATT_FILE}"
else
    echo "NOT FOUND: ${ATT_FILE}"
fi
echo ""

echo "================================================================"
echo " 4. Megatron _apply_output_gate implementation"
echo "================================================================"
MEGATRON_ATT="/usr/local/lib/python3.12/dist-packages/megatron/core/transformer/attention.py"
if [[ -f "${MEGATRON_ATT}" ]]; then
    echo "--- File: ${MEGATRON_ATT}"
    # Print 30 lines around _apply_output_gate definition
    grep -n "_apply_output_gate\|def _apply_output_gate\|gate_output\|gated_output\|attention_output_gate\|softmax_scale" "${MEGATRON_ATT}" 2>&1 | head -20
    echo ""
    echo "--- Full _apply_output_gate method:"
    awk '/def _apply_output_gate/,/^    def [a-z]/' "${MEGATRON_ATT}" | head -50
    echo ""
    echo "--- Lines around 1221 (where error occurs):"
    sed -n '1200,1240p' "${MEGATRON_ATT}"
else
    echo "NOT FOUND: ${MEGATRON_ATT}"
fi
echo ""

echo "================================================================"
echo " 5. Megatron version"
echo "================================================================"
pip show megatron-core 2>&1 | head -5
python3 -c "import megatron.core; print('megatron.core.__version__:', getattr(megatron.core, '__version__', 'unknown'))" 2>&1
echo ""

echo "================================================================"
echo " 6. Qwen3.5 model config (if model path accessible)"
echo "================================================================"
MODEL_PATH="/dataset_rc_b1/models/Qwen3.5-0.8B"
if [[ -f "${MODEL_PATH}/config.json" ]]; then
    cat "${MODEL_PATH}/config.json"
else
    echo "Model config not found at ${MODEL_PATH}/config.json"
fi
echo ""

echo "================================================================"
echo " DONE — inspection complete, exiting."
echo "================================================================"
