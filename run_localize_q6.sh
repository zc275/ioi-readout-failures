#!/usr/bin/env bash
# Localize plan, GPU job 6 (after job 5): Stage B follow-ups.
set -u
cd "$(dirname "$0")"
PY=~/miniconda3/envs/ood/bin/python
L=ioi_repro/outputs/logs
run() {
  local name=$1; shift
  echo "=== $name $(date +%H:%M)"
  (cd ioi_repro && "$@") > "$L/localize_$name.log" 2>&1 || { echo "FAILED $name"; grep -a -A4 Traceback "$L/localize_$name.log" | tail -8; return 1; }
  grep -a "^\[" "$L/localize_$name.log" | cut -c1-400
}
run stageB_L12_chinese $PY 16_stageB.py --layer 12 --train_setting ood_chinese
run stageB_L12_long $PY 16_stageB.py --layer 12 --steps 1000 --lr 3e-3 --tag long
run stageB_L14 $PY 16_stageB.py --layer 14
run stageB_L16 $PY 16_stageB.py --layer 16
run stageB_L12_fold1 $PY 16_stageB.py --layer 12 --fold 1
run stageB_L12_fold2 $PY 16_stageB.py --layer 12 --fold 2
( export OOD_MODEL=qwen2.5-7b QWEN_REVISION=$(cat ioi_repro/outputs/qwen_revision.txt)
  run pool_qwen $PY 17_pool.py )
echo "=== job 6 finished $(date +%H:%M)"
