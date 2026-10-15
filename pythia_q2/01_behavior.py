"""Q2 stage 0 (PLAN_Q2.md): IOI behavior of Pythia across training checkpoints,
in distribution and under two shifts. No interventions.

Prompts are in the classic completion form (Pythia is a base model), built
from the names, places and objects of the released IOI data:
  id:           "Then, A and B were working at the office. B decided to give a bone to" -> " A"
  ood_chinese:  same template, the released Chinese name pairs
  ood_rephrase: "At the office, A and B spent the day working. Later, B handed a bone to" -> " A"
Scoring: log-probability of the full correct name vs the other name (Chinese
names are byte tokens, so first tokens often coincide).

Usage: python 01_behavior.py --models pythia-1.4b pythia-2.8b [--device cuda]
Writes outputs/behavior.json (one entry per model x checkpoint x setting)
"""

import argparse
import json
import os
import random
import re
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'ioi_repro'))
from ioi_data import FOLDS, load_split  # noqa: E402

OUT_DIR = os.path.join(HERE, 'outputs')
CHECKPOINTS = (0, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 100000, 143000)
TEMPLATES = {
  'id': 'Then, {a} and {b} were working at the {place}. {c} decided to give {art} {obj} to',
  'ood_chinese': 'Then, {a} and {b} were working at the {place}. {c} decided to give {art} {obj} to',
  'ood_rephrase': 'At the {place}, {a} and {b} spent the day working. Later, {c} handed {art} {obj} to',
}
SOURCE = {'id': 'id', 'ood_chinese': 'ood_chinese', 'ood_rephrase': 'id'}
_PAT = re.compile(r'(?:Then, |Story: )(?P<a>.+?) and (?P<b>.+?) were working at the (?P<place>.+?)\. '
                  r'(?P<c>.+?) decided to give (?P<art>an?) (?P<obj>.+?) to')


def build_prompts(setting, n, seed):
  """[(prompt, correct name, other name)], deduplicated, n sampled."""
  items = set()
  for fold in FOLDS:
    for split in load_split(SOURCE[setting], fold).values():
      for prompts in split.values():
        for p in prompts:
          m = _PAT.search(p)
          items.add(tuple(m.group(k) for k in ('a', 'b', 'place', 'c', 'art', 'obj')))
  out = []
  for a, b, place, c, art, obj in sorted(items):
    answer, other = (a, b) if c == b else (b, a)
    out.append((TEMPLATES[setting].format(a=a, b=b, place=place, c=c, art=art, obj=obj), answer, other))
  random.Random(seed).shuffle(out)
  return out[:n]


@torch.no_grad()
def continuation_logprob(model, tokenizer, prompts, conts, batch_size, device):
  """Sum of log-probs of each continuation's tokens after its prompt (ids
  concatenated, so tokenization doesn't merge across the boundary)."""
  out = []
  for i in range(0, len(prompts), batch_size):
    seqs, lens = [], []
    for p, c in zip(prompts[i:i + batch_size], conts[i:i + batch_size]):
      c_ids = tokenizer(c).input_ids
      seqs.append(tokenizer(p).input_ids + c_ids)
      lens.append(len(c_ids))
    n = max(map(len, seqs))
    ids = torch.full((len(seqs), n), tokenizer.pad_token_id)
    mask = torch.zeros((len(seqs), n), dtype=torch.long)
    for j, s in enumerate(seqs):
      ids[j, n - len(s):] = torch.tensor(s)
      mask[j, n - len(s):] = 1
    logp = torch.log_softmax(model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits.float(), -1)
    for j, k in enumerate(lens):
      tgt = ids[j, n - k:].to(device)
      out.append(float(logp[j, n - k - 1:n - 1].gather(1, tgt[:, None]).sum()))
  return out


def score(model, tokenizer, items, batch_size, device):
  """Per item: log P(correct name) - log P(other name), full names with a leading space."""
  prompts = [p for p, _, _ in items]
  right = continuation_logprob(model, tokenizer, prompts, [' ' + a for _, a, _ in items], batch_size, device)
  wrong = continuation_logprob(model, tokenizer, prompts, [' ' + o for _, _, o in items], batch_size, device)
  return [r - w for r, w in zip(right, wrong)]


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--models', nargs='+', default=['pythia-1.4b', 'pythia-2.8b'])
  parser.add_argument('--checkpoints', type=int, nargs='+', default=list(CHECKPOINTS))
  parser.add_argument('--n', type=int, default=2000)
  parser.add_argument('--batch_size', type=int, default=64)
  parser.add_argument('--device', default='cuda')
  parser.add_argument('--out', default=os.path.join(OUT_DIR, 'behavior.json'))
  args = parser.parse_args()

  prompts = {s: build_prompts(s, args.n, seed=0) for s in TEMPLATES}
  for s, items in prompts.items():
    print(f'{s}: {len(items)} prompts, e.g. {items[0][0]!r} -> {items[0][1]!r}', flush=True)
  results = {}
  if os.path.exists(args.out):
    with open(args.out) as f:
      results = json.load(f)
  for name in args.models:
    tokenizer = AutoTokenizer.from_pretrained(f'EleutherAI/{name}')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    for step in args.checkpoints:
      key = f'{name}/step{step}'
      if all(f'{key}/{s}' in results for s in TEMPLATES):
        continue
      t0 = time.time()
      model = AutoModelForCausalLM.from_pretrained(
        f'EleutherAI/{name}', revision=f'step{step}',
        torch_dtype=torch.float16 if args.device == 'cuda' else torch.float32).to(args.device).eval()
      for s, items in prompts.items():
        d = np.array(score(model, tokenizer, items, args.batch_size, args.device))
        results[f'{key}/{s}'] = {'n': len(d), 'acc': float((d > 0).mean()), 'mean_logprob_diff': float(d.mean()),
                                 'n_fail': int((d <= 0).sum())}
      print(f'{key}: ' + '  '.join(f'{s} {results[f"{key}/{s}"]["acc"]:.3f}' for s in TEMPLATES)
            + f'  ({time.time() - t0:.0f}s)', flush=True)
      del model
      if args.device == 'cuda':
        torch.cuda.empty_cache()
      os.makedirs(os.path.dirname(args.out), exist_ok=True)
      with open(args.out, 'w') as f:
        json.dump(results, f, indent=1)


if __name__ == '__main__':
  main()
