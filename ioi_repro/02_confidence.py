"""Step 2: confidence-score baseline (paper Sec. 4.1, Eq. 1). CPU only.

Score = mean log-probability of a subset S of the generated tokens:
  - First N tokens: S = first N tokens. The paper sets N to the max token
    length of the expected correct answer (Appendix A.2). For IOI that is 3
    (English names); the paper's numbers match N=3 in all three settings, not
    a per-setting N (Chinese names need up to 6). We report N=3 plus, as a
    sensitivity check, N chosen on the ID validation split.
  - Answer tokens: S = the tokens spelling out the name the model produced.
AUC-ROC is computed on the balanced test split against the released labels.

Usage: python 02_confidence.py   (after 01_behavior.py)
"""

import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

from common import OUT_DIR
from ioi_data import FOLDS, SETTINGS

REPORT_TEMPS = ('1.0', '2.0')
PAPER_N = 3
MAX_N = 8


def load(setting, fold):
  with open(os.path.join(OUT_DIR, 'behavior', f'{setting}_fold{fold}.json')) as f:
    return json.load(f)['results']


def first_n_auc(results, split, temp, n):
  rows = [r for r in results.values() if r['split'] == split]
  labels = [r['released_label'] == 'correct' for r in rows]
  return roc_auc_score(labels, [np.mean(r['logprobs'][temp][:n]) for r in rows])


def main():
  data = {(s, f): load(s, f) for s in SETTINGS for f in FOLDS}
  temps = list(next(iter(data[('id', 0)].values()))['logprobs'])
  # Sensitivity check: N selected on the ID validation split (per temperature).
  val_n = {t: max(range(1, MAX_N + 1), key=lambda n: np.mean(
    [first_n_auc(data[('id', f)], 'val', t, n) for f in FOLDS])) for t in temps}
  rows = []
  for (setting, fold), results in data.items():
    test = [r for r in results.values() if r['split'] == 'test']
    labels = np.array([r['released_label'] == 'correct' for r in test])
    for t in temps:
      answer = [np.mean(r['logprobs'][t][:r['n_answer_tokens'] or PAPER_N]) for r in test]
      rows.append({'setting': setting, 'fold': fold, 'T': t, 'val_N': val_n[t],
                   'first_n': first_n_auc(results, 'test', t, PAPER_N),
                   'first_n_val_N': first_n_auc(results, 'test', t, val_n[t]),
                   'answer': roc_auc_score(labels, answer)})

  def fmt(xs):
    return f'{np.mean(xs):.3f} ± {np.std(xs):.3f}'
  print(f'{"setting":13s} {"T":>4s}  {"first 3 tokens":>14s}  {"first N (val-sel.)":>22s}  {"answer tokens":>14s}')
  for setting in SETTINGS:
    for t in temps:
      sel = [r for r in rows if r['setting'] == setting and r['T'] == t]
      mark = '  <- reported in paper' if t in REPORT_TEMPS else ''
      print(f'{setting:13s} {t:>4s}  {fmt([r["first_n"] for r in sel])}  '
            f'N={sel[0]["val_N"]}: {fmt([r["first_n_val_N"] for r in sel])}  '
            f'{fmt([r["answer"] for r in sel])}{mark}')
  os.makedirs(os.path.join(OUT_DIR, 'results'), exist_ok=True)
  with open(os.path.join(OUT_DIR, 'results', 'confidence.json'), 'w') as f:
    json.dump(rows, f, indent=1)


if __name__ == '__main__':
  main()
