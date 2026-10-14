#!/usr/bin/env bash
# Everything after the single-layer DAS queue, in GPU order. Each stage is
# idempotent (skips finished outputs where possible); the script stops at the
# first failing GPU stage.
set -u
cd "$(dirname "$0")"
PY=~/miniconda3/envs/ood/bin/python
LOG=outputs/logs
stage() { echo "=== $1 $(date +%H:%M)"; }
fail() { echo "FAILED: $1 $(date +%H:%M)"; exit 1; }
run_name() { echo "ioi-fold$1_layer$2_dim256_n10000_seed0"; }
cf() {  # fold, layers, splits..., settings via CF_SETTINGS
  local fold=$1 layers=$2; shift 2
  local name; name=$(run_name "$fold" "$layers")
  $PY 04_cf_sim.py --fold "$fold" --das "$name" --settings ${CF_SETTINGS} --splits "$@" \
    >> "$LOG/cf_${name}.log" 2>&1 || fail "cf_sim $name $*"
  grep -a "AUC" "$LOG/cf_${name}.log" | tail -n 3
}

stage "waiting for single-layer DAS queue"
until grep -q "queue finished" "$LOG/queue_single_layer.log" 2>/dev/null; do sleep 30; done
grep -q FAILED "$LOG/queue_single_layer.log" && echo "note: some DAS jobs failed (see queue log)"

stage "cf_sim smoke test: fold 0 layer 9, ID val"
CF_SETTINGS="id" cf 0 9 val

stage "cache activations"
$PY 05_cache_activations.py > "$LOG/05_cache.log" 2>&1 || fail "cache activations"

stage "probes on CPU (background)"
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=16 $PY 06_probes.py > "$LOG/06_probes.log" 2>&1 &

stage "cf_sim ID val: all single-layer subspaces"
for fold in 0 1 2; do for layer in 9 10 11 12; do
  [[ "$fold $layer" == "0 9" ]] && continue
  [[ -f outputs/das/$(run_name $fold $layer).pt ]] || continue
  CF_SETTINGS="id" cf "$fold" "$layer" val
done; done

stage "joint layer 9-12 DAS"
bash run_das_queue.sh "0 9-10-11-12" "1 9-10-11-12" "2 9-10-11-12" > "$LOG/queue_joint.log" 2>&1
cat "$LOG/queue_joint.log"

stage "cf_sim ID val: joint subspaces"
for fold in 0 1 2; do
  [[ -f outputs/das/$(run_name $fold 9-10-11-12).pt ]] && CF_SETTINGS="id" cf "$fold" 9-10-11-12 val
done

stage "cf_sim test: val-selected subspace + joint 9-12, all settings"
for fold in 0 1 2; do
  best=$($PY - "$fold" <<'EOF'
import glob, json, sys
fold = sys.argv[1]
runs = [json.load(open(p)) for p in glob.glob(f'outputs/cf_sim/ioi-fold{fold}_*_id_val.json')]
best = max(runs, key=lambda r: r['auc_output_mapping'])
print(best['das'].split('_layer')[1].split('_')[0])
EOF
)
  echo "fold $fold: best layer config on ID val = $best"
  for layers in $(echo "$best 9-10-11-12" | tr ' ' '\n' | sort -u); do
    CF_SETTINGS="id ood_chinese ood_rephrase" cf "$fold" "$layers" test
  done
done

stage "waiting for probes"
wait
tail -n 20 "$LOG/06_probes.log" | grep -a "method"
stage "pipeline finished"
