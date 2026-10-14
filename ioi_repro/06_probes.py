"""Step 6: value probing and correctness probing (paper Secs. 4.2-4.3). CPU only.

Both use the authors' compute_auc_roc_for_all_locations (logistic regression,
C=0.05) on activations cached by 05_cache_activations.py.

Value probing (Eqs. 7-8): project the residual at the DAS site onto the DAS
subspace (W x), train a classifier for the value of V (first/second) on
verified examples (ID train/correct; validation = ID val/correct), and score
a test input by the classifier's confidence. For two classes the authors'
code uses |p - 0.5|, which ranks like max_i p_i.

Correctness probing (Eq. 3): a classifier for correct/wrong on the full
residual at a location (the DAS site = "causal variable", or the last prompt
token), trained on ID train (correct + wrong). Its layer is chosen by AUC on
the ID validation split.

Layer indexing: hidden_states[L + 1] is the output of decoder layer L, i.e.
where a DAS subspace trained at block_output of layer L lives. The authors'
MMLU demo projects hidden_states[L] instead (one layer earlier); we report
both for value probing.

Usage: CUDA_VISIBLE_DEVICES= python 06_probes.py --das_layers 9 10 11 12
Writes outputs/results/probes.json
"""

import argparse
import contextlib
import io
import json
import os

import numpy as np
import torch

from common import OUT_DIR
from ioi_data import FOLDS, SETTINGS
from probe import compute_auc_roc_for_all_locations


def load_acts(setting, fold):
  return torch.load(os.path.join(OUT_DIR, 'activations', f'{setting}_fold{fold}.pt'))


def select(acts, split, label=None):
  return [i for i, (s, y) in enumerate(zip(acts['split'], acts['label']))
          if s == split and (label is None or y == label)]


def quiet(fn, *args):
  """The authors' probe code prints accuracies for every layer."""
  with contextlib.redirect_stdout(io.StringIO()):
    return fn(*args)


def value_probe(id_acts, test_acts, W, hs_index):
  """AUC of value probing for one DAS subspace W (dim x 4096) at one hidden-state index."""
  def feats(acts, idx):
    x = acts['site'][idx, hs_index].float() @ W.T.float()
    return x[:, None, :].numpy()                        # (n, 1 layer, dim)
  tr, va = select(id_acts, 'train', 'correct'), select(id_acts, 'val', 'correct')
  te = select(test_acts, 'test')
  hidden = {'train': {'v': feats(id_acts, tr)}, 'val': {'v': feats(id_acts, va)},
            'test': {'v': feats(test_acts, te)}}
  labels = {'train': [id_acts['position'][i] for i in tr],
            'val': [id_acts['position'][i] for i in va],
            'test': [test_acts['label'][i] for i in te]}
  return quiet(compute_auc_roc_for_all_locations, hidden, labels, 'lr')[0]['v']['auc_roc']


def correctness_probe(id_acts, eval_acts, eval_split, loc, layers=slice(None)):
  """Per-layer AUC of a correctness probe at `loc` ('site' or 'last')."""
  tr, va, ev = select(id_acts, 'train'), select(id_acts, 'val'), select(eval_acts, eval_split)
  hidden = {'train': {loc: id_acts[loc][tr][:, layers].float().numpy()},
            'val': {loc: id_acts[loc][va][:, layers].float().numpy()},
            'test': {loc: eval_acts[loc][ev][:, layers].float().numpy()}}
  labels = {'train': [id_acts['label'][i] for i in tr],
            'val': [id_acts['label'][i] for i in va],
            'test': [eval_acts['label'][i] for i in ev]}
  per_layer = quiet(compute_auc_roc_for_all_locations, hidden, labels, 'lr')
  return {layer: m[loc]['auc_roc'] for layer, m in per_layer.items()}


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--das_layers', nargs='+', type=int, default=[9, 10, 11, 12])
  parser.add_argument('--skip_correctness', action='store_true')
  args = parser.parse_args()
  results = []
  for fold in FOLDS:
    id_acts = load_acts('id', fold)
    test_acts = {s: (id_acts if s == 'id' else load_acts(s, fold)) for s in SETTINGS}

    for layer in args.das_layers:
      run = f'ioi-fold{fold}_layer{layer}_dim256_n10000_seed0'
      path = os.path.join(OUT_DIR, 'das', f'{run}.pt')
      if not os.path.exists(path):
        print('missing', run)
        continue
      W = next(iter(torch.load(path, map_location='cpu').values()))
      for name, hs_index in (('correct_index', layer + 1), ('demo_index', layer)):
        row = {'method': 'value_probe', 'fold': fold, 'das_layer': layer, 'indexing': name}
        for setting, acts in test_acts.items():
          row[setting] = value_probe(id_acts, acts, W, hs_index)
        print(row, flush=True)
        results.append(row)

    if not args.skip_correctness:
      for loc in ('site', 'last'):
        val_auc = correctness_probe(id_acts, id_acts, 'val', loc)
        best = max(val_auc, key=val_auc.get)
        row = {'method': f'correctness_probe_{loc}', 'fold': fold, 'layer': best,
               'val_auc': val_auc[best]}
        for setting, acts in test_acts.items():
          row[setting] = correctness_probe(id_acts, acts, 'test', loc, slice(best, best + 1))[0]
        print(row, flush=True)
        results.append(row)
      # The last-layer, last-token probe used in prior work (guess 9 in README).
      row = {'method': 'correctness_probe_last_finallayer', 'fold': fold, 'layer': 32}
      for setting, acts in test_acts.items():
        row[setting] = correctness_probe(id_acts, acts, 'test', 'last', slice(32, 33))[0]
      print(row, flush=True)
      results.append(row)

  os.makedirs(os.path.join(OUT_DIR, 'results'), exist_ok=True)
  with open(os.path.join(OUT_DIR, 'results', 'probes.json'), 'w') as f:
    json.dump(results, f, indent=1)


if __name__ == '__main__':
  main()
