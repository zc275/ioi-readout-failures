#!/usr/bin/env bash
# Localize plan, GPU job 1: dim-1 DAS for V at the final token (Stage A).
set -u
cd "$(dirname "$0")/ioi_repro"
PY=~/miniconda3/envs/ood/bin/python
for layer in 15 17 19 21; do
  name="ioi-fold0_layer${layer}_dim1_n10000_seed0_sitelast"
  [[ -f outputs/das/${name}.pt ]] && continue
  echo "=== DAS ${name} $(date +%H:%M)"
  $PY 03_das_train.py --fold 0 --layers "$layer" --dim 1 --site last > "outputs/logs/das_${name}.log" 2>&1 \
    || { echo "FAILED DAS ${name}"; tail -3 "outputs/logs/das_${name}.log"; continue; }
  tr '\r' '\n' < "outputs/logs/das_${name}.log" | grep -a "IIA after" | tail -1
done
echo "=== job 1 finished $(date +%H:%M)"
