"""IOI step 13 (PLAN_LOCALIZE.md, Stage 0b): what is the bias b made of?

40 names per set, all ordered pairs (first-mentioned X, second Y), each in K
random contexts from the set's fold-0 test prompts, in both mirror versions.
b(X, Y) is the lean toward the first-mentioned name X with the IOI signal
cancelled (as in 12_pref_replicate.py).

Exact split per unordered pair {X, Y}:
  name part      n = (b(X, Y) - b(Y, X)) / 2   flips with the names
  position part  p = (b(X, Y) + b(Y, X)) / 2   lean toward the first slot
One fixed decomposition, fitted on 80% of unordered pairs, scored on the rest:
  b(X, Y) = u(X) - u(Y) + beta
Shares are of E[b^2] (relative to no bias, b = 0), not of the variance around
the mean, so a constant position lean counts as explained by beta.

Usage: python 13_pref_grid.py [--n_names 40] [--k 4] [--score first|full]
Writes outputs/results/pref_grid_{score}.json and outputs/pref/pref_grid_{score}.json
"""

import argparse
import itertools
import json
import os
import random

import numpy as np

from common import OUT_DIR, load_model_and_tokenizer, set_seed
from ioi_data import FOLDS, answer_prefix, load_split, parse_ioi_prompt

nm = __import__('11_name_pref')
rep = __import__('12_pref_replicate')

# Rephrased uses the ID names in the rephrased template.
NAME_SOURCE = {'id': 'id', 'ood_chinese': 'ood_chinese', 'ood_rephrase': 'id'}


def all_names(setting):
  names = set()
  for fold in FOLDS:
    for split in load_split(setting, fold).values():
      for prompts in split.values():
        for p in prompts:
          v = parse_ioi_prompt(p)
          names.update((v['name_a'], v['name_b']))
  return sorted(names)


def pick_names(tokenizer, setting, donor, n, rng):
  """n names with pairwise-distinct first tokens (after the answer prefix)."""
  pre = answer_prefix(donor)
  pool = all_names(NAME_SOURCE[setting])
  rng.shuffle(pool)
  out, seen = [], set()
  for name in pool:
    t = tokenizer(pre + name, add_special_tokens=False).input_ids[0]
    if t not in seen:
      seen.add(t)
      out.append(name)
    if len(out) == n:
      return out
  raise ValueError(f'only {len(out)} names with distinct first tokens in {setting}')


def explained(b, pred):
  return float(1 - np.mean((b - pred) ** 2) / np.mean(b ** 2))


def fit(names, b, rng, test_frac=0.2):
  """b: {(X, Y): lean toward first-mentioned X}. Fit u, beta on train unordered pairs."""
  idx = {x: i for i, x in enumerate(names)}
  pairs = sorted({tuple(sorted(k)) for k in b})
  rng.shuffle(pairs)
  n_test = int(len(pairs) * test_frac)
  test, train = set(pairs[:n_test]), pairs[n_test:]

  def design(keys):
    A = np.zeros((len(keys), len(names) + 1))
    for r, (x, y) in enumerate(keys):
      A[r, idx[x]], A[r, idx[y]], A[r, -1] = 1, -1, 1
    return A

  tr = [k for k in b if tuple(sorted(k)) not in test]
  te = [k for k in b if tuple(sorted(k)) in test]
  coef = np.linalg.lstsq(design(tr), np.array([b[k] for k in tr]), rcond=None)[0]
  u, beta = coef[:-1] - coef[:-1].mean(), coef[-1]
  bt = np.array([b[k] for k in te])
  pred = design(te) @ np.concatenate([u, [beta]])
  # Exact name / position split per unordered pair.
  n_part = np.array([(b[(x, y)] - b[(y, x)]) / 2 for x, y in pairs])
  p_part = np.array([(b[(x, y)] + b[(y, x)]) / 2 for x, y in pairs])
  n_te = np.array([(b[(x, y)] - b[(y, x)]) / 2 for x, y in pairs[:n_test]])
  n_pred = np.array([u[idx[x]] - u[idx[y]] for x, y in pairs[:n_test]])
  tot = np.mean(n_part ** 2) + np.mean(p_part ** 2)
  return {
    'n_pairs_ordered': len(b), 'n_test_unordered': n_test,
    'explained_u_beta_heldout': explained(bt, pred),            # L0-2
    'explained_beta_only_heldout': explained(bt, np.full_like(bt, beta)),   # L0-3
    'explained_u_only_heldout': explained(bt, pred - beta),
    'name_part_explained_by_u_heldout': explained(n_te, n_pred),
    'share_name_part': float(np.mean(n_part ** 2) / tot),
    'share_position_part': float(np.mean(p_part ** 2) / tot),
    'position_part_pair_specific_share': float(np.var(p_part) / np.mean(p_part ** 2)),
    'beta': float(beta), 'mean_abs_b': float(np.mean(np.abs(list(b.values())))),
  }, {x: float(u[idx[x]]) for x in names}


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--n_names', type=int, default=40)
  parser.add_argument('--k', type=int, default=4)
  parser.add_argument('--batch_size', type=int, default=32)
  parser.add_argument('--score', choices=('first', 'full'), default='first')
  parser.add_argument('--limit_pairs', type=int, default=None, help='smoke test')
  args = parser.parse_args()
  set_seed(0)
  rng = random.Random(0)
  score = nm.score_fn(args.score)
  model, tokenizer = load_model_and_tokenizer()
  fam_path = os.path.join(OUT_DIR, 'results', 'name_familiarity.json')
  fam = None
  if os.path.exists(fam_path):
    with open(fam_path) as f:
      raw = json.load(f)
    per_context = raw.get('per_context', raw)
    if isinstance(per_context, dict) and per_context and isinstance(next(iter(per_context.values())), dict):
      ctxs = list(per_context.values())
      fam = {n: float(np.mean([c[n] for c in ctxs if n in c])) for n in set().union(*ctxs)}
  res, raw_out = {}, {}
  for s in nm.SETS:
    split = load_split(s, 0)['test']
    donors_pool = split['correct'] + split['wrong']
    names = pick_names(tokenizer, s, donors_pool[0], args.n_names, rng)
    ordered = list(itertools.permutations(names, 2))
    if args.limit_pairs:
      ordered = ordered[:args.limit_pairs]
    # rep.preference takes (a, b, X, Y): intro order a, b; X the first answer.
    # Lean toward the first-mentioned name X = preference for X with a = X.
    quads = [(x, y, x, y) for x, y in ordered]
    donors = [rng.sample(donors_pool, args.k) for _ in ordered]
    lean = rep.preference(model, tokenizer, quads, donors, args.batch_size, score)
    b = dict(zip(ordered, lean.tolist()))
    print(f'[{s}] names e.g. {names[:4]}, {len(ordered)} ordered pairs, mean |b| {np.mean(np.abs(lean)):.2f}')
    if args.limit_pairs:
      continue
    cell, u = fit(names, b, random.Random(1))
    ntok = {x: len(tokenizer(answer_prefix(donors_pool[0]) + x, add_special_tokens=False).input_ids) for x in names}
    xs = np.array([u[x] for x in names])
    cell['corr_u_ntokens'] = float(np.corrcoef(xs, [ntok[x] for x in names])[0, 1])
    if fam and all(x in fam for x in names):
      cell['corr_u_familiarity'] = float(np.corrcoef(xs, [fam[x] for x in names])[0, 1])
    res[s] = cell
    raw_out[s] = {'names': names, 'u': u, 'b': [[x, y, v] for (x, y), v in b.items()]}
    print('  ' + '\n  '.join(f'{k} {v:.3f}' if isinstance(v, float) else f'{k} {v}' for k, v in cell.items()),
          flush=True)
  if args.limit_pairs:
    return
  with open(os.path.join(OUT_DIR, 'results', f'pref_grid_{args.score}.json'), 'w') as f:
    json.dump(res, f, indent=1)
  os.makedirs(os.path.join(OUT_DIR, 'pref'), exist_ok=True)
  with open(os.path.join(OUT_DIR, 'pref', f'pref_grid_{args.score}.json'), 'w') as f:
    json.dump(raw_out, f)


if __name__ == '__main__':
  main()
