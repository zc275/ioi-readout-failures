#!/usr/bin/env bash
# Follow-ups queued after pipeline_overnight.sh (GPU is single-occupancy).
set -u
cd "$(dirname "$0")"
PY=~/miniconda3/envs/ood/bin/python
LOG=outputs/logs
until grep -q "pipeline finished\|FAILED" "$LOG/pipeline_overnight.log"; do sleep 30; done
echo "=== fold 0 layer 9 ID val rerun (corrected label rule) $(date +%H:%M)"
$PY 04_cf_sim.py --fold 0 --das ioi-fold0_layer9_dim256_n10000_seed0 --settings id --splits val \
  >> "$LOG/cf_ioi-fold0_layer9_dim256_n10000_seed0.log" 2>&1 || echo "FAILED rerun"
grep -a "AUC" "$LOG/cf_ioi-fold0_layer9_dim256_n10000_seed0.log" | tail -1
echo "=== diagnostic: ground-truth labels, joint 9-12, test $(date +%H:%M)"
for fold in 0 1 2; do
  name="ioi-fold${fold}_layer9-10-11-12_dim256_n10000_seed0"
  $PY 04_cf_sim.py --fold "$fold" --das "$name" --splits test --label_source truth \
    >> "$LOG/cf_${name}_truthlabels.log" 2>&1 || echo "FAILED truthlabels fold $fold"
  grep -a "AUC" "$LOG/cf_${name}_truthlabels.log" | tail -3
done
echo "=== post pipeline finished $(date +%H:%M)"
