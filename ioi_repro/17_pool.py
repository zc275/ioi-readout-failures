"""IOI step 17 (PLAN_LOCALIZE.md, second model): failures on *unselected* prompts.

The released splits are half Llama-3 failures by construction, so they can't
show another model's failures (Qwen2.5-7B gets 99% of them right). Here: random
prompts from the released templates, places and objects (a donor prompt) with
a random name pair and a random repeated name, scored by forced choice (first
tokens, correct minus other). Reports the natural failure rate, then the
bias test of 12_pref_replicate.py on up to N failures and N random successes:
does the pair's fresh-context bias predict which prompts fail?

Usage: [OOD_MODEL=qwen2.5-7b] python 17_pool.py [--n_pool 20000] [--n_test 500]
Writes {OUT_DIR}/results/pool_fold0.json
"""

import argparse
import json
import os
import random

import numpy as np
from sklearn.metrics import roc_auc_score

from common import MODEL_TAG, OUT_DIR, load_model_and_tokenizer, set_seed
from ioi_data import load_split, parse_ioi_prompt

nm = __import__('11_name_pref')
rep = __import__('12_pref_replicate')
grid = __import__('13_pref_grid')


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--n_pool', type=int, default=20000)
  parser.add_argument('--n_test', type=int, default=500)
  parser.add_argument('--k', type=int, default=4)
  parser.add_argument('--batch_size', type=int, default=64)
  args = parser.parse_args()
  set_seed(0)
  rng = random.Random(0)
  model, tokenizer = load_model_and_tokenizer()
  res = {'model': MODEL_TAG}
  for s in nm.SETS:
    split = load_split(s, 0)
    donors = [p for sp in split.values() for k in ('correct', 'wrong') for p in sp[k]]
    names = grid.all_names(grid.NAME_SOURCE[s])
    pool = []
    while len(pool) < args.n_pool:
      a, b = rng.sample(names, 2)
      rep_name = rng.choice((a, b))
      p = rep.fill(rng.choice(donors), a, b, rep_name)
      if len(set(nm.first_tokens(tokenizer, p))) == 2:
        pool.append(p)
    d = nm.logit_diff(model, tokenizer, pool, args.batch_size)
    fail = d <= 0
    V_first = np.array([parse_ioi_prompt(p)['position'] == 'first' for p in pool])
    cell = {'n_pool': len(pool), 'fail_rate': float(fail.mean()),
            'fail_rate_answer_first': float(fail[V_first].mean()),
            'fail_rate_answer_second': float(fail[~V_first].mean())}
    fi = [i for i in range(len(pool)) if fail[i]]
    si = [i for i in range(len(pool)) if not fail[i]]
    pick = rng.sample(fi, min(args.n_test, len(fi))) + rng.sample(si, min(args.n_test, len(si)))
    if len(fi) >= 20:
      sub = [pool[i] for i in pick]
      pairs = []
      for p in sub:
        v = parse_ioi_prompt(p)
        pairs.append((v['name_a'], v['name_b'], v['answer'], v['name_c']))
      ctx = [rng.sample(donors, args.k) for _ in sub]
      same = rep.preference(model, tokenizer, pairs, ctx, 32)
      swapped = rep.preference(model, tokenizer, [(b, a, x, y) for a, b, x, y in pairs], ctx, 32)
      f = fail[pick]
      cell.update({
        'n_test_fail': int(f.sum()), 'n_test_succ': int((~f).sum()),
        'auc_fail_from_fresh_pref_same_order': float(roc_auc_score(f, -same)),
        'auc_fail_from_fresh_pref_swapped_order': float(roc_auc_score(f, -swapped)),
        'corr_same_vs_swapped_order': float(np.corrcoef(same, swapped)[0, 1]),
        'fail_mean_fresh_pref': float(same[f].mean()), 'succ_mean_fresh_pref': float(same[~f].mean())})
    res[s] = cell
    print(f'[{MODEL_TAG} {s}] ' + '  '.join(f'{k} {v:.3f}' if isinstance(v, float) else f'{k} {v}'
                                            for k, v in cell.items()), flush=True)
  os.makedirs(os.path.join(OUT_DIR, 'results'), exist_ok=True)
  with open(os.path.join(OUT_DIR, 'results', 'pool_fold0.json'), 'w') as f:
    json.dump(res, f, indent=1)


if __name__ == '__main__':
  main()
