#!/usr/bin/env bash
# Overnight queue (one model on the GPU at a time). Steps whose output exists
# are skipped, so the queue can be restarted after a crash.
#   1. PriceTag DAS (P, Q; paper config) + CF sim, folds 0, 1, 2, then summary.
#   2. PriceTag input-fidelity probes (fold 0).
#   3. IOI validity: DAS dimension sweep at fold 0, layer 12 (dims 1, 4, 16, 64,
#      plus truth labels for the existing 256), CF sim with both label rules.
#   4. PriceTag P layer sweep (fold 0, single layers 4, 8, 12, 16): where does
#      the currency failure enter?
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=~/miniconda3/envs/ood/bin/python
PT=$HERE; IOI=$HERE/../ioi_repro
mkdir -p "$PT/outputs/logs"
while pgrep -f "02_das_train.py" > /dev/null; do sleep 30; done

pt_job() {  # fold var layers(space-separated)
  local fold=$1 var=$2; shift 2
  local layers="$*"
  local name="pt-${var}-fold${fold}_layer${layers// /-}_dim1_n10000_seed0"
  cd "$PT"
  if [[ ! -f outputs/das/${name}.pt ]]; then
    echo "=== DAS ${name} $(date +%H:%M)"
    $PY 02_das_train.py --fold "$fold" --var "$var" --layers $layers > "outputs/logs/das_${name}.log" 2>&1 \
      || { echo "FAILED DAS ${name}"; return; }
    tr '\r' '\n' < "outputs/logs/das_${name}.log" | grep -a "IIA after"
  fi
  if [[ ! -f outputs/cf_sim/${name}_id_test.json ]]; then
    echo "=== CF sim ${name} $(date +%H:%M)"
    $PY 03_cf_sim.py --fold "$fold" --das "$name" > "outputs/logs/cf_${name}.log" 2>&1 \
      || { echo "FAILED CF ${name}"; return; }
    grep -a "AUC" "outputs/logs/cf_${name}.log"
  fi
}

# 1. PriceTag, paper config.
for fold in 0 1 2; do
  pt_job $fold P 8 9 10
  pt_job $fold Q 11 12
  (cd "$PT" && $PY 04_summarize.py > outputs/logs/summary.log 2>&1; cat outputs/logs/summary.log)
done

# 2. Input-fidelity probes.
cd "$PT"
if [[ ! -f outputs/results/probe_inputs_fold0.json ]]; then
  echo "=== probes $(date +%H:%M)"
  $PY 05_probe_inputs.py --fold 0 > outputs/logs/probe_inputs_fold0.log 2>&1 || echo "FAILED probes"
fi

# 3. IOI dimension sweep.
cd "$IOI"
for dim in 1 4 16 64 256; do
  name="ioi-fold0_layer12_dim${dim}_n10000_seed0"
  if [[ ! -f outputs/das/${name}.pt ]]; then
    echo "=== IOI DAS ${name} $(date +%H:%M)"
    $PY 03_das_train.py --fold 0 --layers 12 --dim "$dim" > "outputs/logs/das_${name}.log" 2>&1 \
      || { echo "FAILED DAS ${name}"; continue; }
    tr '\r' '\n' < "outputs/logs/das_${name}.log" | grep -a "IIA after"
  fi
  for labels in output truth; do
    suffix=$([[ $labels == truth ]] && echo _truthlabels || echo "")
    [[ -f outputs/cf_sim/${name}_ood_rephrase_test${suffix}.json ]] && continue
    echo "=== IOI CF ${name} ${labels} $(date +%H:%M)"
    $PY 04_cf_sim.py --fold 0 --das "$name" --splits test --label_source "$labels" \
      >> "outputs/logs/cf_${name}${suffix}.log" 2>&1 || { echo "FAILED CF ${name} ${labels}"; continue; }
    grep -a "AUC" "outputs/logs/cf_${name}${suffix}.log" | tail -3
  done
done

# 4. PriceTag P layer sweep.
for layer in 4 8 12 16; do
  pt_job 0 P $layer
done
echo "=== overnight queue finished $(date +%H:%M)"
