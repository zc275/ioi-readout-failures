"""Step 8 (extension): kill test for geometry-based correctness prediction. CPU only.

Question: does the *geometry* of a representation (where it sits relative to
the verified set) predict correctness, and does that depend on where in the
causal chain we measure it?

Predictions, written down before running these measures:
  P1. Upstream (repeated-name site, hidden states 10-13, DAS subspace or full
      residual): all measures ~0.5 within each position stratum, because the
      position variable is computed correctly on wrong examples too
      (README, observation 1).
  P2. Downstream (last prompt token, hidden states 16-22): clearly above 0.5
      within strata, because that is where the errors appear (observation 2).
If P2 fails, the sharpened Step-2 question has no signal to work with.

Feature spaces (hs i = residual stream after i decoder layers, i.e.
hidden_states[i]; a DAS subspace trained at block_output of layer L lives
at hs L+1):
  site hs{L+1} DAS-256   residual at the site projected onto that layer's DAS subspace
  site hs{L+1} full      full residual at the site
  last hs{h} full        full residual at the last prompt token
  last hs{h} PCA-256     the same, reduced to 256 dims (PCA fit on the verified set)

Measures: all fit only on the verified set (ID train/correct), the same
supervision value probing gets. Higher score = predicted correct.
  knn     -mean cosine distance to the k nearest verified examples (off-manifold;
          Sun et al. 2022 use kNN on normalised features)
  margin  |d_first - d_second|, d_c = mean cosine kNN distance to verified
          examples with V = c (between-classes; small = ambiguous)
  mahal   -min over classes of the Mahalanobis distance to the class mean, with
          a shared Ledoit-Wolf covariance (Lee et al. 2018)
  probe   the paper's linear value probe: |p - 0.5| of a logistic regression for V

Metrics: AUC on each balanced test set, overall and *stratified*, i.e. the
mean of the AUCs within V = first and within V = second. Stratification
removes the position confound (position alone reaches 0.81 overall on the
rephrased template). Score directions are fixed in advance, never flipped
using test labels.

Usage: CUDA_VISIBLE_DEVICES= python 08_kill_test.py
Writes outputs/results/kill_test.json
"""

import glob
import importlib
import json
import os
import warnings

import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from common import OUT_DIR
from ioi_data import FOLDS, SETTINGS

warnings.filterwarnings('ignore')
probes = importlib.import_module('06_probes')

K = 10
DAS_LAYERS = (9, 10, 11, 12)
LAST_HS = (16, 18, 20, 22)
MEASURES = ('knn', 'margin', 'mahal', 'probe')


def l2n(x):
  return x / np.linalg.norm(x, axis=1, keepdims=True)


def knn_dist(train, test, k=K):
  d = 1 - l2n(test) @ l2n(train).T
  return np.sort(d, axis=1)[:, :k].mean(1)


class Measures:
  """All four measures, fit on verified features X with V labels y (bool: first)."""

  def __init__(self, X, y):
    self.X, self.y = X, y
    self.means = {c: X[y == c].mean(0) for c in (True, False)}
    centred = X - np.where(y[:, None], self.means[True], self.means[False])
    self.precision = LedoitWolf().fit(centred).precision_
    self.clf = LogisticRegression(max_iter=1000, tol=0.01, C=0.05).fit(X, y)

  def score(self, Z):
    d_c = {c: knn_dist(self.X[self.y == c], Z) for c in (True, False)}
    maha = {c: np.einsum('ij,jk,ik->i', Z - m, self.precision, Z - m) for c, m in self.means.items()}
    return {
      'knn': -knn_dist(self.X, Z),
      'margin': np.abs(d_c[True] - d_c[False]),
      'mahal': -np.minimum(maha[True], maha[False]),
      'probe': np.abs(self.clf.predict_proba(Z)[:, 1] - 0.5),
    }


def aucs(label, score, first):
  """Overall AUC and mean within-V-stratum AUC."""
  strat = np.mean([roc_auc_score(label[first == v], score[first == v]) for v in (True, False)])
  return float(roc_auc_score(label, score)), float(strat)


def feature_spaces(fold):
  """Yields (name, featurize(acts, idx) -> (n, d) float32 array, needs_pca)."""
  for L in DAS_LAYERS:
    path = os.path.join(OUT_DIR, 'das', f'ioi-fold{fold}_layer{L}_dim256_n10000_seed0.pt')
    W = next(iter(torch.load(path, map_location='cpu').values())).float()
    yield (f'site hs{L + 1} DAS-256',
           lambda a, i, L=L, W=W: (a['site'][i, L + 1].float() @ W.T).numpy(), False)
    yield f'site hs{L + 1} full', lambda a, i, L=L: a['site'][i, L + 1].float().numpy(), False
  for h in LAST_HS:
    yield f'last hs{h} full', lambda a, i, h=h: a['last'][i, h].float().numpy(), False
    yield f'last hs{h} PCA-256', lambda a, i, h=h: a['last'][i, h].float().numpy(), True


def baseline_scores(fold, setting, prompts):
  """Per-prompt scores of the baselines, aligned with `prompts`."""
  with open(os.path.join(OUT_DIR, 'behavior', f'{setting}_fold{fold}.json')) as f:
    beh = json.load(f)['results']
  out = {'confidence N=5': np.array([np.mean(beh[p]['logprobs']['1.0'][:5]) for p in prompts])}
  cf = glob.glob(os.path.join(OUT_DIR, 'cf_sim', f'ioi-fold{fold}_layer9-10-11-12_*_{setting}_test.json'))
  if cf:
    with open(cf[0]) as f:
      by_prompt = {x['prompt']: x['output_mapping'] for x in json.load(f)['scores']}
    out['cf sim (9-12)'] = np.array([by_prompt[p] for p in prompts])
  return out


def main():
  rows = []
  for fold in FOLDS:
    id_acts = probes.load_acts('id', fold)
    tests = {s: (id_acts if s == 'id' else probes.load_acts(s, fold)) for s in SETTINGS}
    ver = probes.select(id_acts, 'train', 'correct')
    y_ver = np.array([id_acts['position'][i] == 'first' for i in ver])
    test_idx = {s: probes.select(a, 'test') for s, a in tests.items()}
    meta = {s: (np.array([a['label'][i] == 'correct' for i in test_idx[s]]),
                np.array([a['position'][i] == 'first' for i in test_idx[s]]),
                [a['prompts'][i] for i in test_idx[s]]) for s, a in tests.items()}

    for setting, (label, first, prompts) in meta.items():
      rows.append({'fold': fold, 'setting': setting, 'space': 'position only (oracle sign)', 'measure': '-',
                   'auc': max(roc_auc_score(label, first), 1 - roc_auc_score(label, first)), 'strat': 0.5})
      for name, score in baseline_scores(fold, setting, prompts).items():
        auc, strat = aucs(label, score, first)
        rows.append({'fold': fold, 'setting': setting, 'space': name, 'measure': '-', 'auc': auc, 'strat': strat})

    for name, featurize, needs_pca in feature_spaces(fold):
      X = featurize(id_acts, ver)
      pca = PCA(256, random_state=0).fit(X) if needs_pca else None
      m = Measures(pca.transform(X) if pca else X, y_ver)
      for setting, acts in tests.items():
        Z = featurize(acts, test_idx[setting])
        scores = m.score(pca.transform(Z) if pca else Z)
        label, first, _ = meta[setting]
        for measure, score in scores.items():
          auc, strat = aucs(label, score, first)
          rows.append({'fold': fold, 'setting': setting, 'space': name, 'measure': measure,
                       'auc': auc, 'strat': strat})
      print(f'fold {fold}: {name} done', flush=True)

  os.makedirs(os.path.join(OUT_DIR, 'results'), exist_ok=True)
  with open(os.path.join(OUT_DIR, 'results', 'kill_test.json'), 'w') as f:
    json.dump(rows, f, indent=1)

  # Mean over folds. Cells: overall AUC / stratified AUC.
  def cell(sel):
    return f'{np.mean([r["auc"] for r in sel]):.2f} / {np.mean([r["strat"] for r in sel]):.2f}'
  baselines = list(dict.fromkeys(r['space'] for r in rows if r['measure'] == '-'))
  spaces = list(dict.fromkeys(r['space'] for r in rows if r['measure'] != '-'))
  print(f'\nCells: overall AUC / within-position-stratum AUC, mean of {len(FOLDS)} folds.')
  print('\n| baseline | ' + ' | '.join(SETTINGS) + ' |\n|---|' + '---|' * len(SETTINGS))
  for b in baselines:
    print(f'| {b} | ' + ' | '.join(
      cell([r for r in rows if r['space'] == b and r['setting'] == s]) for s in SETTINGS) + ' |')
  for setting in SETTINGS:
    print(f'\n### {setting}\n| feature space | ' + ' | '.join(MEASURES) + ' |\n|---|' + '---|' * len(MEASURES))
    for space in spaces:
      print(f'| {space} | ' + ' | '.join(
        cell([r for r in rows if r['setting'] == setting and r['space'] == space and r['measure'] == m])
        for m in MEASURES) + ' |')


if __name__ == '__main__':
  main()
