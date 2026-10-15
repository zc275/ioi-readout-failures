#!/usr/bin/env bash
# Amplification replication: folds 1-2 (paper DAS), and fold-0 single-layer P (L8) as the
# compounding check (PLAN_AMPLIFY.md guess 2).
set -u
cd "$(dirname "$0")"
PY=~/miniconda3/envs/ood/bin/python
for f in 1 2; do
  echo "=== fold $f $(date +%H:%M)"
  $PY 09_amplify.py --fold $f > outputs/logs/amplify_fold$f.log 2>&1 || echo "FAILED fold $f"
done
echo "=== fold 0 single-layer P L8 $(date +%H:%M)"
$PY 09_amplify.py --fold 0 --p_das pt-P-fold0_layer8_dim1_n10000_seed0 > outputs/logs/amplify_fold0_PL8.log 2>&1 || echo "FAILED L8"
echo "=== reps finished $(date +%H:%M)"
