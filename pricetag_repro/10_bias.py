"""PriceTag step 10 (PLAN_LOCALIZE.md, generalization check G): does a
truth-independent Yes/No lean predict failures, as the name/position lean does
in IOI?

L = log P(yes family) - log P(no family) after "Answer:". For a prompt and a
mirror whose query price q is moved so the true answer flips, the lean toward
Yes that doesn't depend on the truth is b = (L + L_mirror) / 2 (exact split, as
in IOI). The mirror reflects q across the nearer boundary (same distance on the
other side), keeping the number format.

Fresh-context b (to avoid the selection confound in PLAN_PREF.md): the same
interval [lb, ub] with K new inside/outside query pairs at random distances;
b(interval) is their mean lean. Score for failure: b oriented against the truth.

Usage: python 10_bias.py [--fold 0] [--k 2]
Writes outputs/results/bias_fold{fold}.json
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from pt_common import (OUT_DIR, family_ids, family_mass, load_model_and_tokenizer, load_split,
                       parse_prompt, set_seed, with_answer)
from pricetag_data import _NUMBER, _USER_TEXT

SETS = ('id', 'ood_currency', 'ood_typo')


def fmt(x, template):
  """x formatted like the number string `template` (decimals, separator)."""
  sep = ',' if ',' in template else '.'
  dec = len(template.split(sep)[1]) if sep in template else 0
  s = f'{x:.{dec}f}'
  return s.replace('.', ',') if sep == ',' else s


def with_query(prompt, q):
  """The prompt with its query price replaced by q (same format)."""
  m = _USER_TEXT.search(prompt)
  start = m.start('text')
  num = list(_NUMBER.finditer(m.group('text')))[2]
  text = fmt(q, num.group())
  if float(text.replace(',', '.')) <= 0:
    return None
  return prompt[:start + num.start()] + text + prompt[start + num.end():]


def mirror(prompt):
  v = parse_prompt(prompt)
  lb, ub, q = v['lb'], v['ub'], v['q']
  edge = lb if abs(q - lb) <= abs(q - ub) else ub
  out = with_query(prompt, 2 * edge - q)
  if out is None or parse_prompt(out)['answer'] == v['answer']:
    return None
  return out


def fresh(prompt, k, rng):
  """k (inside, outside) prompts for the same interval, or None."""
  v = parse_prompt(prompt)
  lb, ub = v['lb'], v['ub']
  w = ub - lb
  out = []
  for _ in range(20 * k):
    q_in = rng.uniform(lb, ub)
    side = rng.choice((-1, 1)) if lb > 0.2 * w else 1
    q_out = lb - rng.uniform(0, w) if side < 0 else ub + rng.uniform(0, w)
    a, b = with_query(prompt, q_in), with_query(prompt, q_out)
    if a is None or b is None or not parse_prompt(a)['answer'] or parse_prompt(b)['answer']:
      continue
    out.append((a, b))
    if len(out) == k:
      return out
  return None


@torch.no_grad()
def yes_lean(model, tokenizer, prompts, batch_size):
  ids = family_ids(tokenizer)
  out = []
  for i in range(0, len(prompts), batch_size):
    enc = tokenizer([with_answer(p) for p in prompts[i:i + batch_size]], return_tensors='pt',
                    padding=True).to(model.device)
    m = family_mass(model(**enc).logits[:, -1], ids).clamp_min(1e-12)
    out += (m[:, 1].log() - m[:, 0].log()).cpu().tolist()
  return np.array(out)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--fold', type=int, default=0)
  parser.add_argument('--k', type=int, default=2)
  parser.add_argument('--batch_size', type=int, default=64)
  parser.add_argument('--limit', type=int, default=None, help='smoke test')
  args = parser.parse_args()
  set_seed(0)
  rng = random.Random(0)
  model, tokenizer = load_model_and_tokenizer()
  res, per_prompt = {}, {}
  for s in SETS:
    split = load_split(s, args.fold)['test']
    items = []
    for p in split['correct'] + split['wrong']:
      m, f = mirror(p), fresh(p, args.k, rng)
      if m is not None and f is not None:
        items.append((p, m, f))
    if args.limit:
      items = items[:args.limit]
    prompts = [p for p, _, _ in items]
    truth = np.array([parse_prompt(p)['answer'] for p in prompts])
    L = yes_lean(model, tokenizer, prompts, args.batch_size)
    L_m = yes_lean(model, tokenizer, [m for _, m, _ in items], args.batch_size)
    flat = [x for _, _, f in items for pair in f for x in pair]
    L_f = yes_lean(model, tokenizer, flat, args.batch_size).reshape(len(items), args.k, 2)
    d = np.where(truth, L, -L)
    fail = d <= 0
    b_orig = (L + L_m) / 2
    b_fresh = L_f.mean(-1).mean(-1)
    against = lambda b: np.where(truth, -b, b)
    s_sig = np.where(truth, L - L_m, L_m - L) / 2
    cell = {'n': len(items), 'acc': float((~fail).mean()), 'n_fail': int(fail.sum()),
            'auc_fail_from_fresh_b': float(roc_auc_score(fail, against(b_fresh))) if 0 < fail.sum() < len(fail) else None,
            'auc_fail_from_orig_b_selection_confounded':
              float(roc_auc_score(fail, against(b_orig))) if 0 < fail.sum() < len(fail) else None,
            'corr_fresh_vs_orig_b': float(np.corrcoef(b_fresh, b_orig)[0, 1]),
            'mean_b_fresh': float(b_fresh.mean()),
            'fail_mean_signal': float(s_sig[fail].mean()) if fail.any() else None,
            'succ_mean_signal': float(s_sig[~fail].mean()),
            'fail_mean_b_against': float(against(b_fresh)[fail].mean()) if fail.any() else None,
            'succ_mean_b_against': float(against(b_fresh)[~fail].mean()),
            # A constant Yes-lean predicts failures from the truth alone; the
            # within-truth AUCs ask whether the interval's own lean adds to that.
            'auc_fail_from_truth_no': float(roc_auc_score(fail, ~truth)) if 0 < fail.sum() < len(fail) else None,
            'fail_rate_truth_yes': float(fail[truth].mean()), 'fail_rate_truth_no': float(fail[~truth].mean())}
    for name, sel in (('truth_yes', truth), ('truth_no', ~truth)):
      f_ = fail[sel]
      cell[f'auc_within_{name}'] = float(roc_auc_score(f_, against(b_fresh)[sel])) if 0 < f_.sum() < len(f_) else None
    per_prompt[s] = {'prompts': prompts, 'truth': truth.tolist(), 'L': L.tolist(), 'L_mirror': L_m.tolist(),
                     'L_fresh': L_f.tolist()}
    res[s] = cell
    print(f'[{s}] e.g. mirror {items[0][1][items[0][1].find("Does"):][:90]!r}')
    print('  ' + '\n  '.join(f'{k} {v:.3f}' if isinstance(v, float) else f'{k} {v}' for k, v in cell.items()),
          flush=True)
  if args.limit:
    return
  with open(os.path.join(OUT_DIR, 'results', f'bias_fold{args.fold}.json'), 'w') as f:
    json.dump(res, f, indent=1)
  os.makedirs(os.path.join(OUT_DIR, 'bias'), exist_ok=True)
  with open(os.path.join(OUT_DIR, 'bias', f'bias_fold{args.fold}.json'), 'w') as f:
    json.dump(per_prompt, f)


if __name__ == '__main__':
  main()
