#!/usr/bin/env bash
# Localize plan, GPU job 2 (after job 1): Stage 0 and the PriceTag check.
set -u
cd "$(dirname "$0")"
PY=~/miniconda3/envs/ood/bin/python
L=ioi_repro/outputs/logs
run() {  # name, dir, command...
  local name=$1 dir=$2; shift 2
  echo "=== $name $(date +%H:%M)"
  (cd "$dir" && $PY "$@") > "$L/localize_$name.log" 2>&1 || { echo "FAILED $name"; grep -a -A3 Traceback "$L/localize_$name.log" | tail -6; return 1; }
  grep -a -v -i "warn\|Loading\|e.g." "$L/localize_$name.log" | tail -40
}
run smoke_grid ioi_repro 13_pref_grid.py --limit_pairs 8 --score full
run smoke_bias pricetag_repro 10_bias.py --limit 64
run pref_full ioi_repro 12_pref_replicate.py --fold 0 --score full
run pref_first ioi_repro 12_pref_replicate.py --fold 0 --score first
run grid_first ioi_repro 13_pref_grid.py --score first
run grid_full ioi_repro 13_pref_grid.py --score full
run bias_pt pricetag_repro 10_bias.py --fold 0
echo "=== job 2 finished $(date +%H:%M)"
