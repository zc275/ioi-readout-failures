#!/usr/bin/env bash
# Follow-up 3: the final-token P site (layer 14) on folds 1 and 2, typo + ID.
set -u
cd "$(dirname "$0")"
PY=~/miniconda3/envs/ood/bin/python
for fold in 1 2; do
  name=pt-P-fold${fold}_layer14_dim1_n10000_seed0_sitelast
  echo "=== DAS ${name} $(date +%H:%M)"
  [[ -f outputs/das/${name}.pt ]] || $PY 02_das_train.py --fold $fold --var P --layers 14 --site last \
    > outputs/logs/das_${name}.log 2>&1 || { echo "FAILED DAS ${name}"; continue; }
  tr '\r' '\n' < outputs/logs/das_${name}.log | grep -a "IIA after"
  $PY 03_cf_sim.py --fold $fold --das $name --settings ood_typo id > outputs/logs/cf_${name}.log 2>&1 \
    || { echo "FAILED CF ${name}"; continue; }
  grep -a AUC outputs/logs/cf_${name}.log
done
echo "=== followup3 finished $(date +%H:%M)"
