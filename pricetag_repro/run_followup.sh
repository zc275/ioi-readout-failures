#!/usr/bin/env bash
# Follow-up control (queued after run_overnight.sh): P at the paper's layers
# with dimension 4, fold 0. If close-call "site-wrong" on correct prompts is
# 1-d subspace imprecision, it should drop at dim 4.
set -u
cd "$(dirname "$0")"
PY=~/miniconda3/envs/ood/bin/python
while pgrep -f "bash run_overnight.sh" > /dev/null; do sleep 60; done
name=pt-P-fold0_layer8-9-10_dim4_n10000_seed0
[[ -f outputs/das/${name}.pt ]] || $PY 02_das_train.py --fold 0 --var P --dim 4 > outputs/logs/das_${name}.log 2>&1
tr '\r' '\n' < outputs/logs/das_${name}.log | grep -a "IIA after"
$PY 03_cf_sim.py --fold 0 --das $name > outputs/logs/cf_${name}.log 2>&1
grep -a AUC outputs/logs/cf_${name}.log
echo "=== followup finished $(date +%H:%M)"
