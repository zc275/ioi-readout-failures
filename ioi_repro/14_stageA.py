"""IOI step 14 (PLAN_LOCALIZE.md, Stage A): where does the bias b join the
readout?

At the final prompt token, layers 15/17/19/21, the 1-d DAS direction for V
(trained with --site last --dim 1). Per test prompt: the coordinate c, the
answer position V, and the fresh-context bias from 12_pref_replicate.py turned
into a lean toward the first-mentioned name, b_first.

Correlational: c ~ 1 + V + b_first (standardized b coefficient, bootstrap CI).
Causal: for pairs of prompts with the same V, set the base's c to the source's
c (a 1-d interchange) and measure the change in the base's logit difference
(first-mentioned name minus second-mentioned, first tokens). If c carries b,
the change tracks b_first(source) - b_first(base). Reported: the slope of the
observed change on that predicted change, over pairs with opposite-sign b
(|b| >= 1 each) plus control pairs with similar b. Slope 1: the coordinate
carries all of b; 0: none of it.

Usage: python 14_stageA.py [--layers 15 17 19 21] [--score first]
Writes outputs/results/stageA_fold0_{score}.json
"""

import argparse
import json
import os
import random

import numpy as np
import torch

from amplify import load_directions
from common import OUT_DIR, load_model_and_tokenizer, set_seed
from ioi_data import parse_ioi_prompt

nm = __import__('11_name_pref')


class Coord:
  """Record or set the 1-d coordinate along w at the last position of a layer."""

  def __init__(self, model, layer, w):
    self.w, self.target, self.record = w, None, []
    self.handle = model.model.layers[layer].register_forward_hook(self.hook)

  def hook(self, module, inputs, output):
    h = output[0]
    x = h[:, -1].float()
    c = x @ self.w
    self.record.append(c.cpu())
    if self.target is None:
      return output
    x = x + (self.target - c)[:, None] * self.w
    h = h.clone()
    h[:, -1] = x.to(h.dtype)
    return (h,) + tuple(output[1:])

  def remove(self):
    self.handle.remove()


@torch.no_grad()
def run(model, tokenizer, coord, prompts, targets, batch_size):
  """(coordinates, first-token logit diff correct - other) with optional targets."""
  coord.record, out = [], []
  for i in range(0, len(prompts), batch_size):
    chunk = prompts[i:i + batch_size]
    coord.target = None if targets is None else torch.tensor(targets[i:i + batch_size], device='cuda')
    enc = tokenizer(chunk, return_tensors='pt', padding=True).to('cuda')
    toks = torch.tensor([nm.first_tokens(tokenizer, p) for p in chunk], device='cuda')
    logits = model(**enc).logits[:, -1].float()
    out += (logits.gather(1, toks[:, :1]) - logits.gather(1, toks[:, 1:])).squeeze(1).cpu().tolist()
  coord.target = None
  return torch.cat(coord.record).numpy(), np.array(out)


def ols_b(c, V, b, rng, n_boot=1000):
  """Standardized coefficient of b in c ~ 1 + V + b, with a bootstrap 95% CI."""
  def coef(idx):
    X = np.column_stack([np.ones(len(idx)), V[idx], b[idx] / b.std()])
    return np.linalg.lstsq(X, c[idx] / c.std(), rcond=None)[0][2]
  n = len(c)
  boots = [coef(rng.integers(0, n, n)) for _ in range(n_boot)]
  return float(coef(np.arange(n))), [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def make_pairs(V, b, n_pairs, rng):
  """(base, source) index pairs with the same V: half opposite-sign strong b,
  half controls with |b_src - b_base| < 0.5."""
  idx = np.arange(len(V))
  opp, ctl = [], []
  for i in rng.permutation(idx):
    same = idx[(V == V[i]) & (idx != i)]
    if abs(b[i]) >= 1:
      cand = same[(np.sign(b[same]) == -np.sign(b[i])) & (np.abs(b[same]) >= 1)]
      if len(cand) and len(opp) < n_pairs // 2:
        opp.append((i, rng.choice(cand)))
    cand = same[np.abs(b[same] - b[i]) < 0.5]
    if len(cand) and len(ctl) < n_pairs // 2:
      ctl.append((i, rng.choice(cand)))
  return opp, ctl


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--layers', type=int, nargs='+', default=[15, 17, 19, 21])
  parser.add_argument('--score', default='first')
  parser.add_argument('--n_pairs', type=int, default=600)
  parser.add_argument('--batch_size', type=int, default=32)
  args = parser.parse_args()
  set_seed(0)
  model, tokenizer = load_model_and_tokenizer()
  with open(os.path.join(OUT_DIR, 'pref', f'pref_replicate_fold0_{args.score}.json')) as f:
    pref = json.load(f)
  res = {}
  for layer in args.layers:
    name = f'ioi-fold0_layer{layer}_dim1_n10000_seed0_sitelast'
    path = os.path.join(OUT_DIR, 'das', f'{name}.pt')
    if not os.path.exists(path):
      print(f'missing {name}, skipped')
      continue
    w = load_directions(path)[layer]
    coord = Coord(model, layer, w)
    res[layer] = {}
    for s, d in pref.items():
      rng = np.random.default_rng(0)
      prompts = d['prompts']
      V = np.array([parse_ioi_prompt(p)['position'] == 'first' for p in prompts], float)
      sign = np.where(V == 1, 1.0, -1.0)            # +1: answer is the first-mentioned name
      b_first = np.array(d['fresh_pref_same']) * sign
      c, ld = run(model, tokenizer, coord, prompts, None, args.batch_size)
      ld_first = ld * sign
      beta, ci = ols_b(c, V, b_first, rng)
      opp, ctl = make_pairs(V, b_first, args.n_pairs, rng)
      pairs = opp + ctl
      base = [prompts[i] for i, _ in pairs]
      _, ld_int = run(model, tokenizer, coord, base, [float(c[j]) for _, j in pairs], args.batch_size)
      bi, sj = np.array([i for i, _ in pairs]), np.array([j for _, j in pairs])
      obs = ld_int * sign[bi] - ld_first[bi]
      pred = b_first[sj] - b_first[bi]
      slope = float(np.polyfit(pred, obs, 1)[0])
      n_opp = len(opp)
      cell = {'n': len(prompts), 'b_coef_std': beta, 'b_coef_ci': ci,
              'corr_c_V': float(np.corrcoef(c, V)[0, 1]),
              'swap_slope_obs_on_pred': slope,
              'opp_mean_obs': float(obs[:n_opp].mean()) if n_opp else None,
              'opp_mean_pred': float(pred[:n_opp].mean()) if n_opp else None,
              'opp_frac_same_sign': float((np.sign(obs[:n_opp]) == np.sign(pred[:n_opp])).mean()) if n_opp else None,
              'ctl_mean_abs_obs': float(np.abs(obs[n_opp:]).mean()) if len(ctl) else None,
              'n_opp': n_opp, 'n_ctl': len(ctl)}
      res[layer][s] = cell
      print(f'[L{layer} {s}] ' + '  '.join(f'{k} {v:.3f}' if isinstance(v, float) else f'{k} {v}'
                                          for k, v in cell.items()), flush=True)
    coord.remove()
  with open(os.path.join(OUT_DIR, 'results', f'stageA_fold0_{args.score}.json'), 'w') as f:
    json.dump(res, f, indent=1)


if __name__ == '__main__':
  main()
