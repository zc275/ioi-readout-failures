"""IOI step 16 (PLAN_LOCALIZE.md, Stage B): is a name's "default-ness" stored on
its own tokens?

Write A, B for the first- and second-mentioned names and D = logit(A) - logit(B)
(first tokens). For a prompt with B repeated (answer A) and its mirror with A
repeated (answer B): D1 = s + b and D2 = -s + b, where b is the lean toward A
and s the IOI signal (exact split). If b = u(A) - u(B) + beta and u is stored
on each name's tokens, exchanging u between the two names gives b' = -b + 2 beta
with s unchanged, so both D's should move by -2 (b - beta).

We train a 1-d direction w at layer L, at the last token of each intro name
(a and b), with that objective: swap the coordinate along w between the two
positions, in both versions, and regress D1', D2' onto the targets above
(MSE). beta is the mean b over training prompts. Train on ID train prompts
(fold 0); evaluate on ID and Chinese test prompts, with a random-direction
control.

Reported (L-B1, L-B2): among test prompts with |b - beta| >= 1, the fraction
where the swap flips the sign of b - beta; the median relative change of s; on
test failures, the fraction the swap fixes; on successes, the fraction it
breaks.

Usage: python 16_stageB.py --layer 6 [--steps 300] [--train_setting id] [--fold 0] [--tag x]
Writes outputs/results/stageB_L{layer}{suffix}.json and outputs/stageB/w_L{layer}{suffix}.pt
(suffix empty for the defaults, else _{train_setting}_fold{fold}[_{tag}])
"""

import argparse
import json
import os
import random

import numpy as np
import torch

from common import OUT_DIR, load_model_and_tokenizer, set_seed
from ioi_data import _IOI_PATTERN, load_split, parse_ioi_prompt

nm = __import__('11_name_pref')


def intro_positions(tokenizer, prompt):
  """Indices (unpadded, BOS included) of the last token of intro names a and b."""
  m = _IOI_PATTERN.search(prompt)
  enc = tokenizer(prompt, return_offsets_mapping=True)
  out = []
  for g in ('a', 'b'):
    end = m.end(g)
    hits = [i for i, (s, e) in enumerate(enc['offset_mapping']) if s < end <= e]
    out.append(hits[-1])
  return out


def first_vs_second_tokens(tokenizer, prompt):
  """First tokens of (A, B): first- and second-mentioned names."""
  v = parse_ioi_prompt(prompt)
  pre = nm.answer_prefix(prompt)
  return [tokenizer(pre + n, add_special_tokens=False).input_ids[0] for n in (v['name_a'], v['name_b'])]


class Swap:
  """Swap the coordinate along w between two positions per row at one layer."""

  def __init__(self, model, layer):
    self.w, self.pos = None, None
    self.handle = model.model.layers[layer].register_forward_hook(self.hook)

  def hook(self, module, inputs, output):
    if self.w is None:
      return output
    h = output[0]
    # w: (d,) or (d, k); swap the coordinates in its (orthonormalized) span.
    Q = (self.w / self.w.norm())[:, None] if self.w.dim() == 1 else torch.linalg.qr(self.w)[0]
    rows = torch.arange(h.shape[0], device=h.device)
    pa, pb = self.pos[:, 0], self.pos[:, 1]
    xa, xb = h[rows, pa].float(), h[rows, pb].float()
    ca, cb = xa @ Q, xb @ Q
    h = h.clone()
    h[rows, pa] = (xa + (cb - ca) @ Q.T).to(h.dtype)
    h[rows, pb] = (xb + (ca - cb) @ Q.T).to(h.dtype)
    return (h,) + tuple(output[1:])


def encode(tokenizer, prompts):
  enc = tokenizer(prompts, return_tensors='pt', padding=True).to('cuda')
  n = enc.input_ids.shape[1]
  pos = torch.tensor([[n - len(tokenizer(p).input_ids) + i for i in intro_positions(tokenizer, p)]
                      for p in prompts], device='cuda')
  toks = torch.tensor([first_vs_second_tokens(tokenizer, p) for p in prompts], device='cuda')
  return enc, pos, toks


def D(model, enc, toks):
  logits = model(**enc).logits[:, -1].float()
  return (logits.gather(1, toks[:, :1]) - logits.gather(1, toks[:, 1:])).squeeze(1)


def usable(tokenizer, prompts):
  return [p for p in prompts if len(set(first_vs_second_tokens(tokenizer, p))) == 2]


@torch.no_grad()
def measure(model, tokenizer, swap, prompts, w, batch_size):
  """(D1, D2) per prompt (original, mirror), with swap along w (None = no swap)."""
  out = {1: [], 2: []}
  for i in range(0, len(prompts), batch_size):
    chunk = prompts[i:i + batch_size]
    for k, ps in ((1, chunk), (2, [nm.mirror(p) for p in chunk])):
      enc, pos, toks = encode(tokenizer, ps)
      swap.w, swap.pos = w, pos
      out[k] += D(model, enc, toks).cpu().tolist()
      swap.w = None
  return np.array(out[1]), np.array(out[2])


def evaluate(D1, D2, E1, E2, beta, answer_is_A):
  """Before (D) vs after (E) the swap."""
  s, b = (D1 - D2) / 2, (D1 + D2) / 2
  s2, b2 = (E1 - E2) / 2, (E1 + E2) / 2
  strong = np.abs(b - beta) >= 1
  # Correct on the original prompt: D1 > 0 if the answer is A, < 0 if B.
  d = np.where(answer_is_A, D1, -D1)
  e = np.where(answer_is_A, E1, -E1)
  fail = d <= 0
  return {
    'n': int(len(D1)), 'n_strong': int(strong.sum()),
    'flip_frac_strong': float((np.sign(b2 - beta) == -np.sign(b - beta))[strong].mean()) if strong.any() else None,
    'median_rel_change_s': float(np.median(np.abs(s2 - s) / np.maximum(np.abs(s), 1e-3))),
    'mean_b_minus_beta_before_strong': float((b - beta)[strong].mean()) if strong.any() else None,
    'mean_b_minus_beta_after_strong': float((b2 - beta)[strong].mean()) if strong.any() else None,
    'n_fail': int(fail.sum()),
    'fix_frac_failures': float((e[fail] > 0).mean()) if fail.any() else None,
    'break_frac_successes': float((e[~fail] <= 0).mean()) if (~fail).any() else None,
  }


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--layer', type=int, required=True)
  parser.add_argument('--steps', type=int, default=300)
  parser.add_argument('--batch_size', type=int, default=16)
  parser.add_argument('--lr', type=float, default=1e-2)
  parser.add_argument('--n_eval', type=int, default=600)
  parser.add_argument('--train_setting', default='id', choices=('id', 'ood_chinese'))
  parser.add_argument('--fold', type=int, default=0)
  parser.add_argument('--tag', default='')
  parser.add_argument('--dim', type=int, default=1, help='subspace dimension (exploratory, > 1)')
  args = parser.parse_args()
  default = args.train_setting == 'id' and args.fold == 0 and not args.tag and args.dim == 1
  suffix = '' if default else (f'_{args.train_setting}_fold{args.fold}' + (f'_dim{args.dim}' if args.dim > 1 else '')
                               + (f'_{args.tag}' if args.tag else ''))
  set_seed(0)
  rng = random.Random(0)
  model, tokenizer = load_model_and_tokenizer()
  swap = Swap(model, args.layer)
  split = load_split(args.train_setting, args.fold)
  train = usable(tokenizer, split['train']['correct'] + split['train']['wrong'])
  rng.shuffle(train)

  # beta: mean lean toward the first-mentioned name over training prompts.
  D1, D2 = measure(model, tokenizer, swap, train[:512], None, 32)
  beta = float(((D1 + D2) / 2).mean())
  print(f'beta ({args.train_setting} train) {beta:.3f}', flush=True)

  shape = (model.config.hidden_size,) if args.dim == 1 else (model.config.hidden_size, args.dim)
  w = torch.randn(*shape, device='cuda')
  w = (w / w.norm(dim=0)).requires_grad_(True)
  opt = torch.optim.Adam([w], lr=args.lr)
  for step in range(args.steps):
    chunk = rng.sample(train, args.batch_size)
    loss = 0.0
    for ps in (chunk, [nm.mirror(p) for p in chunk]):
      enc, pos, toks = encode(tokenizer, ps)
      with torch.no_grad():
        swap.w = None
        base = D(model, enc, toks)
      if ps is chunk:
        base1 = base
      else:
        base2 = base
    b_lean = (base1 + base2) / 2
    target = -2 * (b_lean - beta)
    for ps, base in ((chunk, base1), ([nm.mirror(p) for p in chunk], base2)):
      enc, pos, toks = encode(tokenizer, ps)
      swap.w, swap.pos = w, pos
      loss = loss + ((D(model, enc, toks) - (base + target)) ** 2).mean()
      swap.w = None
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 25 == 0 or step == args.steps - 1:
      print(f'step {step} loss {loss.item() / 2:.3f}', flush=True)

  w_final = (w / w.norm(dim=0)).detach()
  g = torch.Generator(device='cpu').manual_seed(1)
  w_rand = torch.randn(*w_final.shape, generator=g).cuda()
  w_rand = w_rand / w_rand.norm(dim=0)
  res = {'layer': args.layer, 'dim': args.dim, 'train_setting': args.train_setting, 'fold': args.fold, 'steps': args.steps,
         'lr': args.lr, 'beta_train': beta, 'sets': {}}
  for s in ('id', 'ood_chinese'):
    test = usable(tokenizer, load_split(s, args.fold)['test']['correct'] + load_split(s, args.fold)['test']['wrong'])
    test = random.Random(1).sample(test, min(args.n_eval, len(test)))
    answer_is_A = np.array([parse_ioi_prompt(p)['position'] == 'first' for p in test])
    D1, D2 = measure(model, tokenizer, swap, test, None, 32)
    beta_s = float(((D1 + D2) / 2).mean())
    cell = {'beta_test': beta_s}
    for name, ww in (('das', w_final), ('random', w_rand)):
      E1, E2 = measure(model, tokenizer, swap, test, ww, 32)
      cell[name] = evaluate(D1, D2, E1, E2, beta_s, answer_is_A)
    res['sets'][s] = cell
    print(f'[{s}] ' + json.dumps(cell), flush=True)
  os.makedirs(os.path.join(OUT_DIR, 'stageB'), exist_ok=True)
  torch.save(w_final.cpu(), os.path.join(OUT_DIR, 'stageB', f'w_L{args.layer}{suffix}.pt'))
  with open(os.path.join(OUT_DIR, 'results', f'stageB_L{args.layer}{suffix}.json'), 'w') as f:
    json.dump(res, f, indent=1)


if __name__ == '__main__':
  main()
