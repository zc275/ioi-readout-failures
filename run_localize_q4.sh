#!/usr/bin/env bash
# Localize plan, GPU job 4 (after job 3): Stage B layer scan, then Qwen CF sim
# at the Qwen DAS layer with the best held-out flip IIA (verified examples only).
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
export OOD_MODEL=qwen2.5-7b QWEN_REVISION=$(cat ioi_repro/outputs/qwen_revision.txt)
best=$(cd ioi_repro && $PY -c "
import json, glob
runs = {f: json.load(open(f))['iia_after']['val_correct']['iia_flip'] for f in glob.glob('outputs_qwen2.5-7b/das/*.json')}
print(max(runs, key=runs.get).split('/')[-1][:-5])
for f, v in sorted(runs.items()): print(f, round(v, 3), file=__import__('sys').stderr)
")
echo "Qwen best DAS: $best"
run qwen_cf_output $PY 04_cf_sim.py --fold 0 --das "$best" --splits test --label_source output
run qwen_cf_truth $PY 04_cf_sim.py --fold 0 --das "$best" --splits test --label_source truth
echo "=== job 4 finished $(date +%H:%M)"
