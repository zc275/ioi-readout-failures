"""PriceTag step 4: AUC tables from 03_cf_sim.py outputs (mean ± std over folds).

For each setting and label rule:
  - P, Q: AUC of each variable's output-mapping score, all test prompts.
  - P&Q: the two checks combined (min of the two scores, a prompt passes only
    if both variables check out).
  - Stratified by failure type, against all correct prompts:
      left  = wrong with P False (q < lb, said Yes)
      right = wrong with Q False (q > ub, said Yes)

Usage: python 04_summarize.py [--layers_P 8-9-10 --layers_Q 11-12 --run _dim1_n10000_seed0]
Writes outputs/results/cf_sim_summary.json
"""

import argparse
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

import pt_common as C


def auc(y, s):
  y = np.asarray(y)
  return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float('nan')


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--layers_P', default='8-9-10')
  parser.add_argument('--layers_Q', default='11-12')
  parser.add_argument('--run', default='_dim1_n10000_seed0', help='rest of the DAS run name')
  parser.add_argument('--split', default='test')
  args = parser.parse_args()
  table = {}
  for setting in ('ood_currency', 'ood_typo', 'id'):
    per_fold = {}
    for fold in (0, 1, 2):
      runs = {}
      for var in ('P', 'Q'):
        layers = getattr(args, f'layers_{var}')
        path = os.path.join(C.OUT_DIR, 'cf_sim',
                            f'pt-{var}-fold{fold}_layer{layers}{args.run}_{setting}_{args.split}.json')
        if os.path.exists(path):
          with open(path) as f:
            runs[var] = json.load(f)['scores']
      if len(runs) < 2:
        continue
      prompts = [r['prompt'] for r in runs['P']]
      assert prompts == [r['prompt'] for r in runs['Q']]
      rows = runs['P']
      y = np.array([r['label'] for r in rows])
      left = np.array([r['label'] == 0 and not r['P'] for r in rows])
      right = np.array([r['label'] == 0 and r['P'] and not r['Q'] for r in rows])
      out = {}
      for rule in ('output', 'truth'):
        s = {v: np.array([r[f'mapping_{rule}'] for r in runs[v]]) for v in ('P', 'Q')}
        s['P&Q'] = np.minimum(s['P'], s['Q'])
        for name, score in s.items():
          out[f'{rule}/{name}/all'] = auc(y, score)
          for stratum, mask in (('left', left), ('right', right)):
            keep = (y == 1) | mask
            out[f'{rule}/{name}/{stratum}'] = auc(y[keep], score[keep])
      out['n_left'], out['n_right'], out['n_wrong'] = int(left.sum()), int(right.sum()), int((y == 0).sum())
      per_fold[fold] = out
    if not per_fold:
      continue
    keys = next(iter(per_fold.values())).keys()
    table[setting] = {k: (float(np.nanmean([o[k] for o in per_fold.values()])),
                          float(np.nanstd([o[k] for o in per_fold.values()])))
                      for k in keys}
    table[setting]['folds'] = sorted(per_fold)
    print(f'\n== {setting} (folds {sorted(per_fold)}) wrong: {table[setting]["n_wrong"][0]:.0f}, '
          f'left {table[setting]["n_left"][0]:.0f}, right {table[setting]["n_right"][0]:.0f}')
    print(f'{"score":6s} {"rule":7s} {"all":>13s} {"left fails":>13s} {"right fails":>13s}')
    for name in ('P', 'Q', 'P&Q'):
      for rule in ('output', 'truth'):
        cells = [table[setting][f'{rule}/{name}/{st}'] for st in ('all', 'left', 'right')]
        print(f'{name:6s} {rule:7s} ' + ' '.join(f'{m:6.3f}±{s:5.3f}' for m, s in cells))
  os.makedirs(os.path.join(C.OUT_DIR, 'results'), exist_ok=True)
  with open(os.path.join(C.OUT_DIR, 'results', 'cf_sim_summary.json'), 'w') as f:
    json.dump(table, f, indent=1)


if __name__ == '__main__':
  main()
