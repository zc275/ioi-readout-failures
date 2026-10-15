#!/usr/bin/env bash
# Follow-up 2 (fold 0):
#   a. Variable identity: do the P / Q subspaces carry the check or just q?
#   b. Downstream site for typo: P at the final prompt token, layers 14 and 18,
#      to see where clear-cut failures enter (cf. IOI layer 15).
set -u
cd "$(dirname "$0")"
PY=~/miniconda3/envs/ood/bin/python
for run in pt-P-fold0_layer8-9-10_dim1_n10000_seed0 pt-Q-fold0_layer11-12_dim1_n10000_seed0; do
  [[ -f outputs/results/variable_identity_${run}.json ]] && continue
  echo "=== identity ${run} $(date +%H:%M)"
  $PY 07_variable_identity.py --das $run > outputs/logs/identity_${run}.log 2>&1 || echo "FAILED identity ${run}"
  grep -a follows outputs/logs/identity_${run}.log
done
for layer in 14 18; do
  name=pt-P-fold0_layer${layer}_dim1_n10000_seed0_sitelast
  echo "=== DAS ${name} $(date +%H:%M)"
  [[ -f outputs/das/${name}.pt ]] || $PY 02_das_train.py --fold 0 --var P --layers $layer --site last \
    > outputs/logs/das_${name}.log 2>&1 || { echo "FAILED DAS ${name}"; continue; }
  tr '\r' '\n' < outputs/logs/das_${name}.log | grep -a "IIA after"
  $PY 03_cf_sim.py --fold 0 --das $name --settings ood_typo id > outputs/logs/cf_${name}.log 2>&1 \
    || { echo "FAILED CF ${name}"; continue; }
  grep -a AUC outputs/logs/cf_${name}.log
done
echo "=== followup2 finished $(date +%H:%M)"
