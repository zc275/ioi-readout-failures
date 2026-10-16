"""IOI step 12 (PLAN_PREF.md, P4-P5): does a name pair's preference replicate
in fresh contexts, and does it predict which prompts fail?

For each test prompt (names a, b in intro order; X the answer, Y the repeated
name), the same two names go into K other prompts of the same set (their place,
object and template), in both versions (Y repeated, X repeated). The pair's
preference for X in context k is the mean of logit(X) - logit(Y) over the two
versions, which cancels the IOI signal. Done with the intro order kept (a, b)
and swapped (b, a).

Reported per set: AUC of the fresh-context preference for predicting the
original prompt's failure (P4), correlation of fresh vs original preference,
and the same-order vs swapped-order correlation (P5).

Usage: python 12_pref_replicate.py [--fold 0] [--k 4] [--score first|full]
  first: first-token logits (default); full: full-name log-probs (PLAN_LOCALIZE 0a)
Writes outputs/results/pref_replicate_fold{fold}[_full].json, and per-prompt
arrays to outputs/pref/pref_replicate_fold{fold}_{score}.json (for Stage A)
"""

import argparse
import json
import os
import random

import numpy as np
from sklearn.metrics import roc_auc_score

from common import OUT_DIR, load_model_and_tokenizer, set_seed
from ioi_data import _IOI_PATTERN, load_split, parse_ioi_prompt

nm = __import__('11_name_pref')


def fill(donor, a, b, c):
  """The donor prompt with its names replaced (intro a, b; repeated c)."""
  m = _IOI_PATTERN.search(donor)
  out = donor
  for group, name in sorted((('a', a), ('b', b), ('c', c)), key=lambda x: -m.start(x[0])):
    out = out[:m.start(group)] + name + out[m.end(group):]
  return out


def preference(model, tokenizer, pairs, donors, batch_size, score=nm.logit_diff):
  """pairs: [(a, b, X, Y)]; donors: [[prompt] * k] per pair. Mean over contexts
  of the preference for X: ((lX - lY | Y repeated) + (lX - lY | X repeated)) / 2."""
  v1, v2 = [], []
  for (a, b, x, y), ds in zip(pairs, donors):
    v1 += [fill(d, a, b, y) for d in ds]
    v2 += [fill(d, a, b, x) for d in ds]
  d1 = score(model, tokenizer, v1, batch_size)    # answer X: lX - lY
  d2 = score(model, tokenizer, v2, batch_size)    # answer Y: lY - lX
  k = len(donors[0])
  return ((d1 - d2) / 2).reshape(-1, k).mean(1)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--fold', type=int, default=0)
  parser.add_argument('--k', type=int, default=4)
  parser.add_argument('--batch_size', type=int, default=32)
  parser.add_argument('--score', choices=('first', 'full'), default='first')
  args = parser.parse_args()
  score = nm.score_fn(args.score)
  set_seed(0)
  rng = random.Random(0)
  model, tokenizer = load_model_and_tokenizer()
  res, per_prompt = {}, {}
  for s in nm.SETS:
    split = load_split(s, args.fold)['test']
    prompts = [p for p in split['correct'] + split['wrong'] if len(set(nm.first_tokens(tokenizer, p))) == 2]
    d = score(model, tokenizer, prompts, args.batch_size)
    d_m = score(model, tokenizer, [nm.mirror(p) for p in prompts], args.batch_size)
    pref_orig = (d - d_m) / 2
    fail = d <= 0
    pairs = []
    for p in prompts:
      v = parse_ioi_prompt(p)
      x = v['answer']
      pairs.append((v['name_a'], v['name_b'], x, v['name_c']))
    donors = [rng.sample([q for q in prompts if q != p], args.k) for p in prompts]
    same = preference(model, tokenizer, pairs, donors, args.batch_size, score)
    swapped = preference(model, tokenizer, [(b, a, x, y) for a, b, x, y in pairs], donors, args.batch_size, score)
    per_prompt[s] = {'prompts': prompts, 'd': d.tolist(), 'd_mirror': d_m.tolist(),
                     'fresh_pref_same': same.tolist(), 'fresh_pref_swapped': swapped.tolist()}
    cell = {
      'n': len(d), 'n_fail': int(fail.sum()),
      'auc_fail_from_fresh_pref_same_order': float(roc_auc_score(fail, -same)),
      'auc_fail_from_fresh_pref_swapped_order': float(roc_auc_score(fail, -swapped)),
      'auc_fail_from_orig_pref_selection_confounded': float(roc_auc_score(fail, -pref_orig)),
      'corr_fresh_vs_orig': float(np.corrcoef(same, pref_orig)[0, 1]),
      'corr_same_vs_swapped_order': float(np.corrcoef(same, swapped)[0, 1]),
      'fail_sign_agree_fresh_orig': float((np.sign(same[fail]) == np.sign(pref_orig[fail])).mean()),
      'fail_mean_fresh_pref': float(same[fail].mean()), 'succ_mean_fresh_pref': float(same[~fail].mean()),
    }
    res[s] = cell
    print(f'[{s}] e.g. fresh: {fill(donors[0][0], *pairs[0][:2], pairs[0][3])[-140:]!r}')
    print('  ' + '\n  '.join(f'{k} {v:.3f}' if isinstance(v, float) else f'{k} {v}' for k, v in cell.items()),
          flush=True)
  suffix = '' if args.score == 'first' else f'_{args.score}'
  path = os.path.join(OUT_DIR, 'results', f'pref_replicate_fold{args.fold}{suffix}.json')
  with open(path, 'w') as f:
    json.dump(res, f, indent=1)
  os.makedirs(os.path.join(OUT_DIR, 'pref'), exist_ok=True)
  with open(os.path.join(OUT_DIR, 'pref', f'pref_replicate_fold{args.fold}_{args.score}.json'), 'w') as f:
    json.dump(per_prompt, f)


if __name__ == '__main__':
  main()
