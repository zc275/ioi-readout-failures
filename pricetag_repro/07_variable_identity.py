"""PriceTag step 7 (control): does the P subspace carry P, or just the query q?

If the patched subspace only carried the query's value, a patched base would
compare the *source's* q with the *base's* own lower bound: its answer would
follow [lb_base < q_src]. If it carries P, the answer follows
[lb_src < q_src]. We take pairs where the two disagree and count which one
the patched base follows. Same for Q with [q_src < ub_base] vs [q_src < ub_src].

Sources: ID validation prompts our model answers correctly. Bases:
informative verified bases, as in 02/03.

Usage: python 07_variable_identity.py --das pt-P-fold0_layer8-9-10_dim1_n10000_seed0
Writes outputs/results/variable_identity_{das}.json
"""

import argparse
import importlib
import json
import os
import random

import numpy as np

import pt_common as C

das = importlib.import_module('02_das_train')
cf = importlib.import_module('03_cf_sim')


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--das', required=True)
  parser.add_argument('--n_pairs', type=int, default=4000)
  parser.add_argument('--seed', type=int, default=0)
  args = parser.parse_args()
  var = args.das.split('-')[1]
  fold = int(args.das.split('fold')[1].split('_')[0])
  where = 'last' if args.das.endswith('_sitelast') else 'query'

  C.set_seed(args.seed)
  model, tokenizer = C.load_model_and_tokenizer()
  ids = C.family_ids(tokenizer)
  intervenable = cf.load_das(model, os.path.join(C.OUT_DIR, 'das', f'{args.das}.pt'))
  bases = [p for p in C.verified_bases(model, tokenizer, fold) if C.parse_prompt(p)[C.OTHER[var]]]
  behavior = C.load_behavior('id')
  val = C.load_split('id', fold)['val']
  sources = [p for k in ('correct', 'wrong') for p in val[k] if behavior[p]['our_label'] == 'correct']

  def alt(b, s):  # what a "q only" subspace would produce
    return b['lb'] < s['q'] if var == 'P' else s['q'] < b['ub']

  rng = random.Random(args.seed)
  site, examples = {}, []
  while len(examples) < args.n_pairs:
    b, s = rng.choice(bases), rng.choice(sources)
    bv, sv = C.parse_prompt(b), C.parse_prompt(s)
    if alt(bv, sv) == sv[var]:
      continue
    for p in (C.with_answer(b), s):
      site.setdefault(p, C.padded_site(tokenizer, p, where))
    examples.append(das.make_example(tokenizer, b, s, var, site))
  m = cf.both_masses(intervenable, tokenizer, examples, ids, 64)
  said_yes = m[:, 1] > m[:, 0]
  follows_var = np.array([sy == (e['inv_label'] == ' Yes') for sy, e in zip(said_yes, examples)])
  result = {'das': args.das, 'n_pairs': len(examples),
            'follows_variable': float(follows_var.mean()),
            'follows_query_only': float(1 - follows_var.mean())}
  print(json.dumps(result), flush=True)
  os.makedirs(os.path.join(C.OUT_DIR, 'results'), exist_ok=True)
  with open(os.path.join(C.OUT_DIR, 'results', f'variable_identity_{args.das}.json'), 'w') as f:
    json.dump(result, f, indent=1)


if __name__ == '__main__':
  main()
