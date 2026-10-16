#!/usr/bin/env bash
# Localize plan, GPU job 4 (after job 3): Stage B layer scan, then failures on
# unselected prompts for Llama-3 and Qwen2.5-7B (17_pool.py).
set -u
cd "$(dirname "$0")"
PY=~/miniconda3/envs/ood/bin/python
L=ioi_repro/outputs/logs
run() {
  local name=$1; shift
  echo "=== $name $(date +%H:%M)"
  (cd ioi_repro && "$@") > "$L/localize_$name.log" 2>&1 || { echo "FAILED $name"; grep -a -A4 Traceback "$L/localize_$name.log" | tail -8; return 1; }
  tr '\r' '\n' < "$L/localize_$name.log" | grep -a -v -i "warn\|Loading\|nnsight\|^step" | tail -12
}
run stageB_smoke $PY 16_stageB.py --layer 6 --steps 3 --n_eval 32
for layer in 2 4 6 8 10 12; do
  run stageB_L$layer $PY 16_stageB.py --layer $layer
done
# Second model on unselected prompts (the released splits are half Llama-3 failures).
run pool_llama $PY 17_pool.py
export OOD_MODEL=qwen2.5-7b QWEN_REVISION=$(cat ioi_repro/outputs/qwen_revision.txt)
run pool_qwen $PY 17_pool.py
unset OOD_MODEL
echo "=== job 4 finished $(date +%H:%M)"
