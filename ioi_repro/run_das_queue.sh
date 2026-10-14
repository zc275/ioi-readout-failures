#!/usr/bin/env bash
# Train DAS for each "fold layers" job in order, one at a time (only one
# model fits on the GPU). Jobs whose checkpoint already exists are skipped, so
# the queue can simply be restarted after a crash.
#
# Usage: bash run_das_queue.sh "0 10" "0 11" "1 9-10-11-12" ...
set -u
cd "$(dirname "$0")"
PY=~/miniconda3/envs/ood/bin/python
mkdir -p outputs/logs outputs/das
for job in "$@"; do
  read -r fold layers <<< "$job"
  name="ioi-fold${fold}_layer${layers}_dim256_n10000_seed0"
  if [[ -f "outputs/das/${name}.pt" ]]; then
    echo "skip ${name} (done)"; continue
  fi
  echo "start ${name} $(date +%H:%M)"
  $PY 03_das_train.py --fold "$fold" --layers ${layers//-/ } > "outputs/logs/das_${name}.log" 2>&1
  status=$?
  after=$(tr '\r' '\n' < "outputs/logs/das_${name}.log" | grep -a "IIA after" | head -1)
  echo "end ${name} $(date +%H:%M) exit=${status} ${after}"
  if [[ $status -ne 0 ]]; then
    echo "FAILED ${name}; see outputs/logs/das_${name}.log"
  fi
done
echo "queue finished $(date +%H:%M)"
