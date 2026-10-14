#!/usr/bin/env bash
# Downstream-variable experiment: DAS for V at the LAST prompt token, then
# counterfactual simulation with output-derived and ground-truth labels.
# Usage: bash run_downstream.sh FOLD LAYER [LAYER ...]   (skips finished steps)
set -u
cd "$(dirname "$0")"
PY=~/miniconda3/envs/ood/bin/python
LOG=outputs/logs
fold=$1; shift
for layer in "$@"; do
  name="ioi-fold${fold}_layer${layer}_dim256_n10000_seed0_sitelast"
  if [[ ! -f outputs/das/${name}.pt ]]; then
    echo "=== DAS ${name} $(date +%H:%M)"
    $PY 03_das_train.py --fold "$fold" --layers "$layer" --site last > "$LOG/das_${name}.log" 2>&1 \
      || { echo "FAILED DAS ${name}"; continue; }
    tr '\r' '\n' < "$LOG/das_${name}.log" | grep -a "IIA after"
  fi
  for labels in output truth; do
    suffix=$([[ $labels == truth ]] && echo _truthlabels || echo "")
    [[ -f outputs/cf_sim/${name}_ood_rephrase_test${suffix}.json ]] && continue
    echo "=== CF sim ${name} labels=${labels} $(date +%H:%M)"
    $PY 04_cf_sim.py --fold "$fold" --das "$name" --splits test --label_source "$labels" \
      >> "$LOG/cf_${name}${suffix}.log" 2>&1 || { echo "FAILED CF ${name} ${labels}"; continue; }
    grep -a "AUC" "$LOG/cf_${name}${suffix}.log" | tail -3
  done
done
echo "=== downstream finished $(date +%H:%M)"
