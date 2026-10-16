#!/usr/bin/env bash
# Localize plan, GPU job 7 (after job 6): exploratory Stage B with a k-dim subspace at L12.
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
run stageB_smoke_dim $PY 16_stageB.py --layer 12 --dim 4 --steps 3 --n_eval 32 --tag smoke
for k in 2 4 8; do
  run stageB_L12_dim$k $PY 16_stageB.py --layer 12 --dim $k
done
echo "=== job 7 finished $(date +%H:%M)"
