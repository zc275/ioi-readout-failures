"""Q2 control (PLAN_Q2.md): is a Pythia OOD gap a name preference rather than
an IOI failure? For each prompt, also score its mirror, where the other name is
the repeated one (so the correct answer swaps). A model that just prefers one
name of the pair gets exactly one of the two right.

Per setting: accuracy on originals and mirrors, the fraction of pairs with both
right, both wrong, and exactly one right. With d = log P(correct) - log P(other)
on the original and d' on the mirror, a fixed preference p for one name adds +p
to one and -p to the other, so (d + d')/2 is the preference-free IOI signal and
(d - d')/2 the preference. "acc_corrected" is the fraction of pairs with
d + d' > 0 (also removes the length penalty of full-name scoring).

Usage: python 02_name_prior.py --model pythia-160m --step 143000
Writes outputs/name_prior.json (one entry per model x step x setting)
"""

import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from importlib import import_module

behavior = import_module('01_behavior')


def mirror(setting, items):
  """Same prompt with the other name repeated: swap which name is the answer."""
  out = []
  for prompt, answer, other in items:
    # The repeated name is `other`; it appears once in the intro and once later.
    head, sep, tail = prompt.rpartition(other)
    out.append((head + answer + tail if sep else prompt, other, answer))
  return out


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', default='pythia-160m')
  parser.add_argument('--step', type=int, default=143000)
  parser.add_argument('--n', type=int, default=2000)
  parser.add_argument('--batch_size', type=int, default=128)
  parser.add_argument('--device', default='cuda')
  parser.add_argument('--out', default=os.path.join(behavior.OUT_DIR, 'name_prior.json'))
  args = parser.parse_args()

  tokenizer = AutoTokenizer.from_pretrained(f'EleutherAI/{args.model}')
  tokenizer.pad_token = tokenizer.eos_token
  tokenizer.padding_side = 'left'
  model = AutoModelForCausalLM.from_pretrained(
    f'EleutherAI/{args.model}', revision=f'step{args.step}',
    torch_dtype=torch.float16 if args.device == 'cuda' else torch.float32).to(args.device).eval()
  results = {}
  if os.path.exists(args.out):
    with open(args.out) as f:
      results = json.load(f)
  for s in behavior.TEMPLATES:
    items = behavior.build_prompts(s, args.n, seed=0)
    mirrored = mirror(s, items)
    print(f'{s}: e.g. {items[0][0]!r} -> {items[0][1]!r}\n  mirror {mirrored[0][0]!r} -> {mirrored[0][1]!r}')
    da = np.array(behavior.score(model, tokenizer, items, args.batch_size, args.device))
    db = np.array(behavior.score(model, tokenizer, mirrored, args.batch_size, args.device))
    a, b = da > 0, db > 0
    r = {'n': len(a), 'acc': float(a.mean()), 'acc_mirror': float(b.mean()),
         'both_right': float((a & b).mean()), 'both_wrong': float((~a & ~b).mean()),
         'one_right': float((a ^ b).mean()), 'acc_corrected': float((da + db > 0).mean()),
         'mean_signal': float(((da + db) / 2).mean()), 'mean_abs_pref': float((np.abs(da - db) / 2).mean())}
    results[f'{args.model}/step{args.step}/{s}'] = r
    print('  ' + '  '.join(f'{k} {v:.3f}' for k, v in r.items() if k != 'n'), flush=True)
  os.makedirs(os.path.dirname(args.out), exist_ok=True)
  with open(args.out, 'w') as f:
    json.dump(results, f, indent=1)


if __name__ == '__main__':
  main()
