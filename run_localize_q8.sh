#!/usr/bin/env bash
# Localize plan, GPU job 8 (after job 7): Qwen site test on rephrased (L-M1).
set -u
cd "$(dirname "$0")"
PY=~/miniconda3/envs/ood/bin/python
L=ioi_repro/outputs/logs
run() {
  local name=$1; shift
  echo "=== $name $(date +%H:%M)"
  (cd ioi_repro && "$@") > "$L/localize_$name.log" 2>&1 || { echo "FAILED $name"; grep -a -A4 Traceback "$L/localize_$name.log" | tail -8; return 1; }
  tr '\r' '\n' < "$L/localize_$name.log" | grep -a "^\[\|AUC\|auc\|iia" | tail -12
}
export OOD_MODEL=qwen2.5-7b QWEN_REVISION=$(cat ioi_repro/outputs/qwen_revision.txt)
run qwen_forced $PY 18_forced_labels.py --fold 0
das=ioi-fold0_layer14_dim256_n10000_seed0
run qwen_cf_truth $PY 04_cf_sim.py --fold 0 --das $das --settings ood_rephrase --splits test --label_source truth
run qwen_cf_output $PY 04_cf_sim.py --fold 0 --das $das --settings ood_rephrase --splits test --label_source output
echo "=== job 8 finished $(date +%H:%M)"
