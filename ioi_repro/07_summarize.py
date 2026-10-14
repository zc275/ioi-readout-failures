"""Step 7: collect all results into one table (mean ± std over folds). CPU only.

Usage: python 07_summarize.py
Prints a markdown table and writes outputs/results/summary.json.
"""

import glob
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

from common import OUT_DIR
from ioi_data import FOLDS, SETTINGS

# Paper Tables 2-3 (IOI columns): ID, OOD Chinese names, OOD rephrased template.
PAPER = {
  'Confidence, first N tokens, T=1': (0.935, 0.767, 0.650),
  'Confidence, first N tokens, T=2': (0.900, 0.752, 0.558),
  'Confidence, answer token, T=1': (0.928, 0.709, 0.596),
  'Counterfactual sim, first N tokens': (0.999, 0.996, 0.816),
  'Counterfactual sim, output mapping': (0.999, 0.997, 0.815),
  'Value probing': (0.804, 0.611, 0.805),
  'Correctness probing, causal variable': (0.914, 0.806, 0.665),
  'Correctness probing, prompt last token': (0.822, 0.607, 0.444),
}


def fmt(xs):
  xs = [x for x in xs if x is not None]
  if not xs:
    return '—'
  return f'{np.mean(xs):.3f} ± {np.std(xs):.3f}' + ('' if len(xs) == len(FOLDS) else f' (n={len(xs)})')


def load_json(path):
  with open(path) as f:
    return json.load(f)


def selected_config(fold):
  """Layer config with the best ID-val AUC (output mapping) for this fold."""
  runs = [load_json(p) for p in glob.glob(os.path.join(OUT_DIR, 'cf_sim', f'ioi-fold{fold}_*_id_val.json'))]
  return max(runs, key=lambda r: r['auc_output_mapping'])['das'] if runs else None


def cf_rows():
  """Counterfactual simulation: the paper's joint layer 9-12 subspace, and the
  subspace selected per fold on ID validation."""
  rows = {}
  selected = {f: selected_config(f) for f in FOLDS}
  for path in glob.glob(os.path.join(OUT_DIR, 'cf_sim', '*_test.json')):
    r = load_json(path)
    groups = []
    if '_layer9-10-11-12_' in r['das']:
      groups.append('layers 9-12 jointly')
    if r['das'] == selected[r['fold']]:
      groups.append('layer selected on ID val')
    for group in groups:
      for score, key, field in (('first N tokens', 'auc_first_n', 'first_n'),
                                ('output mapping', 'auc_output_mapping', 'output_mapping')):
        name = f'Counterfactual sim, {score} ({group})'
        rows.setdefault(name, {s: {} for s in SETTINGS})[r['setting']][r['fold']] = r[key]
        # Diagnostic: restrict to prompts where our model's verdict matches the
        # released label (near-tie prompts flip between setups).
        sel = [x for x in r['scores'] if (x['our_label'] == 'correct') == bool(x['label'])]
        cname = f'*{name}, label-consistent prompts*'
        rows.setdefault(cname, {s: {} for s in SETTINGS})[r['setting']][r['fold']] = roc_auc_score(
          [x['label'] for x in sel], [x[field] for x in sel])
  print('selected on ID val:', {f: (d or '').split('_layer')[-1].split('_')[0] for f, d in selected.items()})
  return dict(sorted(rows.items(), key=lambda kv: ('selected' in kv[0], kv[0].startswith('*'))))


def main():
  table = {}
  conf_path = os.path.join(OUT_DIR, 'results', 'confidence.json')
  if os.path.exists(conf_path):
    conf = load_json(conf_path)
    for name, t, key in (('Confidence, first N tokens, T=1', '1.0', 'first_n'),
                         ('Confidence, first N tokens, T=2', '2.0', 'first_n'),
                         ('Confidence, answer token, T=1', '1.0', 'answer'),
                         ('*Confidence, first N tokens, T=1, N chosen on ID val*', '1.0', 'first_n_val_N')):
      table[name] = {s: {r['fold']: r[key] for r in conf if r['setting'] == s and r['T'] == t}
                     for s in SETTINGS}
  table.update(cf_rows())
  probes_path = os.path.join(OUT_DIR, 'results', 'probes.json')
  if os.path.exists(probes_path):
    for r in load_json(probes_path):
      if r['method'] == 'value_probe':
        if r['indexing'] != 'correct_index':
          continue  # the demo's off-by-one indexing is reported in the README
        name = f'Value probing (DAS layer {r["das_layer"]})'
      elif r['method'] == 'correctness_probe_site':
        name = 'Correctness probing, causal variable (layer chosen on ID val)'
      elif r['method'] == 'correctness_probe_last_finallayer':
        name = 'Correctness probing, prompt last token (last layer)'
      else:
        name = '*Correctness probing, prompt last token, layer chosen on ID val*'
      for s in SETTINGS:
        table.setdefault(name, {x: {} for x in SETTINGS})[s][r['fold']] = r[s]

  print('| Method | ID (paper) | ID (ours) | Chinese names (paper) | (ours) | Rephrased (paper) | (ours) |')
  print('|---|---|---|---|---|---|---|')
  for name, by_setting in table.items():
    base = name.strip('*').split(' (')[0].split(', label-')[0].split(', N chosen')[0]
    paper = PAPER.get(base) if not name.startswith('*') else None
    cells = []
    for i, s in enumerate(SETTINGS):
      cells += [f'{paper[i]:.3f}' if paper else '—', fmt([by_setting[s].get(f) for f in FOLDS])]
    print(f'| {name} | ' + ' | '.join(cells) + ' |')
  os.makedirs(os.path.join(OUT_DIR, 'results'), exist_ok=True)
  with open(os.path.join(OUT_DIR, 'results', 'summary.json'), 'w') as f:
    json.dump({k: {s: v for s, v in d.items()} for k, d in table.items()}, f, indent=1)


if __name__ == '__main__':
  main()
