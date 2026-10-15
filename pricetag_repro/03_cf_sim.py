"""PriceTag step 3: counterfactual simulation for P or Q (paper Sec. 4.3, Fig. 6).

For each test prompt (source) we patch the DAS subspace at the query integer
token into K informative verified bases (other variable True, "Answer:" suffix),
and record the patched bases' [no, yes] family masses. Because the bases are
informative, the counterfactual label is just the source's value of the
variable (Yes iff X = True), under two rules:

  truth:  X from the prompt's numbers. Needs ground truth (not deployable).
  output: X inferred from the source's output (the released label):
          correct -> X = truth; wrong and said Yes -> X = True (Yes needs both
          checks to pass); wrong and said No -> X = False (assume the other
          check was right; these are rare, < 3% of wrong examples).

Both rules agree on correct examples, so they differ only on failures.
Writes per-prompt scores for both rules; 04_summarize.py computes AUCs.

Usage: python 03_cf_sim.py --fold 0 --das pt-P-fold0_layer8-9-10_dim1_n10000_seed0
"""

import argparse
import importlib
import json
import os
import random

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import pt_common as C
import pyvene as pv
from causal_interventions import (LowRankRotatedSpaceIntervention, PretrainedFeaturizer,
                                  get_intervention_config)

das = importlib.import_module('02_das_train')


def load_das(model, path):
  """As ioi_repro/04_cf_sim.load_das."""
  weights = torch.load(path, map_location='cuda')
  layers = [int(k.split('.')[1]) for k in weights]
  config = get_intervention_config(
    type(model), 'block_output', layers, LowRankRotatedSpaceIntervention,
    intervention_units=['pos'] * len(layers), intervention_dimension=0)
  intervenable = pv.IntervenableModel(config, model)
  intervenable.set_device('cuda')
  intervenable.disable_model_gradients()
  assert list(intervenable.interventions) == list(weights)
  for k, w in weights.items():
    featurizer = PretrainedFeaturizer(w).eval()
    featurizer.weight = featurizer.weight.to('cuda')
    intervenable.interventions[k][0].rotate_layer = featurizer
    intervenable.interventions[k][0].set_interchange_dim(interchange_dim=featurizer.weight.shape[0])
  intervenable.model.eval()
  return intervenable


def output_value(v, var, correct):
  if correct:
    return v[var]
  return not v['answer']  # wrong: said Yes (truth No) -> True; said No -> False


@torch.no_grad()
def both_masses(intervenable, tokenizer, examples, ids, batch_size):
  res = []
  for inputs in das.loader(tokenizer, examples, batch_size, shuffle=False):
    m, _ = das.step(intervenable, tokenizer, inputs, ids)
    res += m.cpu().tolist()
  return np.array(res)  # (n, 2): [no, yes]


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--fold', type=int, required=True)
  parser.add_argument('--das', required=True)
  parser.add_argument('--settings', nargs='+', default=['ood_currency', 'ood_typo', 'id'])
  parser.add_argument('--splits', nargs='+', default=['test'])
  parser.add_argument('--k', type=int, default=10)
  parser.add_argument('--pool', type=int, default=16)
  parser.add_argument('--batch_size', type=int, default=64)
  parser.add_argument('--seed', type=int, default=0)
  args = parser.parse_args()
  var = args.das.split('-')[1]
  assert f'fold{args.fold}_' in args.das and var in ('P', 'Q')

  C.set_seed(args.seed)
  model, tokenizer = C.load_model_and_tokenizer()
  ids = C.family_ids(tokenizer)
  intervenable = load_das(model, os.path.join(C.OUT_DIR, 'das', f'{args.das}.pt'))
  verified = C.verified_bases(model, tokenizer, args.fold)
  pool = [p for p in verified if C.parse_prompt(p)[C.OTHER[var]]][:args.pool]
  os.makedirs(os.path.join(C.OUT_DIR, 'cf_sim'), exist_ok=True)

  for setting in args.settings:
    data = C.load_split(setting, args.fold)
    behavior = C.load_behavior(setting)
    for split in args.splits:
      rng = random.Random(args.seed)
      sources = [(p, 1) for p in data[split]['correct']] + [(p, 0) for p in data[split]['wrong']]
      site = {}
      for p in pool:
        site[C.with_answer(p)] = C.padded_site(tokenizer, C.with_answer(p))
      examples, owner = [], []
      for j, (src, _) in enumerate(sources):
        site[src] = C.padded_site(tokenizer, src)
        for base in rng.sample(pool, args.k):
          examples.append(das.make_example(tokenizer, base, src, var, site))
          owner.append(j)
      m = both_masses(intervenable, tokenizer, examples, ids, args.batch_size)
      owner = np.array(owner)
      said_yes = m[:, 1] > m[:, 0]
      rows = []
      for j, (src, y) in enumerate(sources):
        v = C.parse_prompt(src)
        sel = owner == j
        row = {'prompt': src, 'label': y, 'our_label': behavior[src]['our_label'],
               'P': v['P'], 'Q': v['Q'], 'answer': v['answer'],
               'patched_yes_rate': float(said_yes[sel].mean()),
               'patched_yes_mass': float(m[sel, 1].mean()),
               'patched_no_mass': float(m[sel, 0].mean())}
        for rule, x in (('truth', v[var]), ('output', output_value(v, var, y == 1))):
          row[f'cf_{rule}'] = bool(x)
          row[f'mapping_{rule}'] = float((said_yes[sel] == x).mean())
          row[f'first_n_{rule}'] = float(m[sel, int(x)].mean())
        rows.append(row)
      labels = np.array([r['label'] for r in rows])
      result = {'das': args.das, 'var': var, 'setting': setting, 'split': split, 'fold': args.fold,
                'k': args.k, 'pool': pool, 'scores': rows}
      for rule in ('output', 'truth'):
        s = np.array([r[f'mapping_{rule}'] for r in rows])
        result[f'auc_mapping_{rule}'] = roc_auc_score(labels, s)
        result[f'auc_first_n_{rule}'] = roc_auc_score(labels, [r[f'first_n_{rule}'] for r in rows])
      print(f'[{args.das} | {setting} {split}] AUC output-mapping: output-labels '
            f'{result["auc_mapping_output"]:.3f}, truth-labels {result["auc_mapping_truth"]:.3f}', flush=True)
      with open(os.path.join(C.OUT_DIR, 'cf_sim', f'{args.das}_{setting}_{split}.json'), 'w') as f:
        json.dump(result, f)


if __name__ == '__main__':
  main()
