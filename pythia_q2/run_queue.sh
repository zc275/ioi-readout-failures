#!/usr/bin/env bash
# GPU queue (2026-10-15): IOI amplification (smoke, then full), then Q2 stage 0.
set -u
PY=~/miniconda3/envs/ood/bin/python
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/ioi_repro"
mkdir -p outputs/logs
echo "=== IOI amplify smoke $(date +%H:%M)"
$PY 10_amplify.py --limit 64 > outputs/logs/amplify_smoke.log 2>&1 || { echo "FAILED IOI smoke"; tail -5 outputs/logs/amplify_smoke.log; }
echo "=== IOI amplify full $(date +%H:%M)"
$PY 10_amplify.py > outputs/logs/amplify_fold0.log 2>&1 || { echo "FAILED IOI full"; tail -5 outputs/logs/amplify_fold0.log; }
cd "$ROOT/pythia_q2"
mkdir -p outputs
echo "=== Q2 stage 0 $(date +%H:%M)"
$PY 01_behavior.py --models pythia-1.4b pythia-2.8b > outputs/stage0.log 2>&1 || { echo "FAILED stage 0"; tail -5 outputs/stage0.log; }
grep -a "step" outputs/stage0.log | grep -v -i warn
echo "=== queue finished $(date +%H:%M)"
