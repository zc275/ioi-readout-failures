"""PriceTag step 6 (exploratory, added after seeing step 4): is the variable
wrong at the site because the comparison is hard, or because of the shift?

For each boundary check X (P: q vs lb; Q: q vs ub) we take prompts where X is
False (q < lb for P; lb < q > ub for Q), and bin them by the true margin
(lb - q, or q - ub). In each bin we compare correct prompts (said No) with
failures (said Yes) on the *site-wrong rate*: the fraction of informative
bases that answer Yes after the source's X subspace is patched in, i.e. how
often the site carries X = True.

Also reports how well the margin alone separates failures from correct
prompts (AUC), as a difficulty baseline.

Usage: python 06_margin.py      (reads outputs/cf_sim, paper-config runs)
Writes outputs/results/margin_analysis.json
"""

import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

import pt_common as C

RUNS = {'P': 'layer8-9-10_dim1_n10000_seed0', 'Q': 'layer11-12_dim1_n10000_seed0'}
BINS = ((0, 0.5), (0.5, 1), (1, 2), (2, np.inf))


def main():
  result = {}
  for var, run in RUNS.items():
    for setting in ('ood_currency', 'ood_typo', 'id'):
      rows = []
      for fold in (0, 1, 2):
        path = os.path.join(C.OUT_DIR, 'cf_sim', f'pt-{var}-fold{fold}_{run}_{setting}_test.json')
        with open(path) as f:
          for x in json.load(f)['scores']:
            v = C.parse_prompt(x['prompt'])
            if var == 'P' and not v['P']:
              margin = v['lb'] - v['q']
            elif var == 'Q' and v['P'] and not v['Q']:
              margin = v['q'] - v['ub']
            else:
              continue
            rows.append((x['label'], margin, x['patched_yes_rate']))
      a = np.array(rows)
      y, margin, wrong_at_site = a[:, 0], a[:, 1], a[:, 2]
      cell = {'n_correct': int((y == 1).sum()), 'n_wrong': int((y == 0).sum()),
              'auc_margin_predicts_failure': float(roc_auc_score(1 - y, -margin)),
              'bins': []}
      for lo, hi in BINS:
        k = (margin >= lo) & (margin < hi)
        b = {'lo': lo, 'hi': None if np.isinf(hi) else hi}
        for name, lab in (('correct', 1), ('wrong', 0)):
          sel = k & (y == lab)
          b[f'n_{name}'] = int(sel.sum())
          b[f'site_wrong_{name}'] = float(wrong_at_site[sel].mean()) if sel.any() else None
        cell['bins'].append(b)
      result[f'{var}/{setting}'] = cell
      fmt = lambda v: '  -  ' if v is None else f'{v:.2f}'  # noqa: E731
      print(f'{var} {setting:12s} margin-AUC {cell["auc_margin_predicts_failure"]:.2f} | ' + '  '.join(
        f'[{b["lo"]}-{b["hi"] or "inf"}] corr {fmt(b["site_wrong_correct"])} ({b["n_correct"]}) '
        f'wrong {fmt(b["site_wrong_wrong"])} ({b["n_wrong"]})' for b in cell['bins']))
  os.makedirs(os.path.join(C.OUT_DIR, 'results'), exist_ok=True)
  with open(os.path.join(C.OUT_DIR, 'results', 'margin_analysis.json'), 'w') as f:
    json.dump(result, f, indent=1)


if __name__ == '__main__':
  main()
