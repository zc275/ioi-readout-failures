#!/usr/bin/env bash
# Localize plan, GPU job 3 (after job 2): Stage A, then the Qwen2.5-7B replication.
set -u
cd "$(dirname "$0")"
PY=~/miniconda3/envs/ood/bin/python
L=ioi_repro/outputs/logs
run() {  # name, command... (run in ioi_repro)
  local name=$1; shift
  echo "=== $name $(date +%H:%M)"
  (cd ioi_repro && "$@") > "$L/localize_$name.log" 2>&1 || { echo "FAILED $name"; grep -a -A4 Traceback "$L/localize_$name.log" | tail -8; return 1; }
  tr '\r' '\n' < "$L/localize_$name.log" | grep -a -v -i "warn\|Loading\|e.g.\|nnsight\|step [0-9]" | tail -30
}
run stageA $PY 14_stageA.py --score first
export OOD_MODEL=qwen2.5-7b QWEN_REVISION=$(cat ioi_repro/outputs/qwen_revision.txt)
run qwen_relabel $PY 15_relabel.py --folds 0
run qwen_pref11 $PY 11_name_pref.py --fold 0
run qwen_pref12 $PY 12_pref_replicate.py --fold 0
for layer in 8 10 12 14; do
  run qwen_das_L$layer $PY 03_das_train.py --fold 0 --layers $layer
done
echo "=== job 3 finished $(date +%H:%M)"
