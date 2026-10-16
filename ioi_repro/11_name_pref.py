"""IOI step 11 (PLAN_PREF.md): are failures driven by a preference for one name?

For each test prompt, score it and its mirror (the repeated name's second
mention replaced by the other name, so the answer swaps). With d the logit
difference correct - other on the original and d' on the mirror, a fixed
preference for one name cancels in d + d'. Failures with d + d' > 0 are
"preference-driven": the IOI signal points the right way and the preference wins.

Usage: python 11_name_pref.py [--fold 0]
Writes outputs/results/name_pref_fold{fold}.json
"""

import argparse
import json
import os

import numpy as np
import torch

from common import OUT_DIR, load_model_and_tokenizer, set_seed
from ioi_data import answer_prefix, load_split, parse_ioi_prompt

SETS = ('id', 'ood_chinese', 'ood_rephrase')


def mirror(prompt):
  v = parse_ioi_prompt(prompt)
  start, end = v['name_c_span']
  return prompt[:start] + v['answer'] + prompt[end:]


def first_tokens(tokenizer, prompt):
  """(correct name's first token, other name's first token)."""
  v = parse_ioi_prompt(prompt)
  other = v['name_b'] if v['answer'] == v['name_a'] else v['name_a']
  pre = answer_prefix(prompt)
  return tuple(tokenizer(pre + n, add_special_tokens=False).input_ids[0] for n in (v['answer'], other))


@torch.no_grad()
def logit_diff(model, tokenizer, prompts, batch_size):
  out = []
  for i in range(0, len(prompts), batch_size):
    chunk = prompts[i:i + batch_size]
    enc = tokenizer(chunk, return_tensors='pt', padding=True).to('cuda')
    toks = torch.tensor([first_tokens(tokenizer, p) for p in chunk], device='cuda')
    logits = model(**enc).logits[:, -1].float()
    out += (logits.gather(1, toks[:, :1]) - logits.gather(1, toks[:, 1:])).squeeze(1).cpu().tolist()
  return np.array(out)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--fold', type=int, default=0)
  parser.add_argument('--batch_size', type=int, default=32)
  args = parser.parse_args()
  set_seed(0)
  model, tokenizer = load_model_and_tokenizer()
  res = {}
  for s in SETS:
    split = load_split(s, args.fold)['test']
    prompts = [p for p in split['correct'] + split['wrong'] if len(set(first_tokens(tokenizer, p))) == 2]
    mirrors = [mirror(p) for p in prompts]
    assert all(parse_ioi_prompt(m)['answer'] != parse_ioi_prompt(p)['answer'] for p, m in zip(prompts, mirrors))
    d, d2 = logit_diff(model, tokenizer, prompts, args.batch_size), logit_diff(model, tokenizer, mirrors, args.batch_size)
    fail = d <= 0
    signal, pref = (d + d2) / 2, np.abs(d - d2) / 2
    cell = {'n': len(d), 'acc': float((~fail).mean()), 'acc_mirror': float((d2 > 0).mean()),
            'acc_corrected': float((signal > 0).mean()), 'n_fail': int(fail.sum())}
    for name, sel in (('fail', fail), ('succ', ~fail)):
      cell[f'{name}_pref_driven'] = float((signal[sel] > 0).mean())
      cell[f'{name}_mirror_right'] = float((d2[sel] > 0).mean())
      cell[f'{name}_mean_signal'] = float(signal[sel].mean())
      cell[f'{name}_mean_abs_pref'] = float(pref[sel].mean())
    res[s] = cell
    print(f'[{s}] e.g. {prompts[0][-160:]!r}\n  mirror {mirrors[0][-160:]!r}')
    print('  ' + '  '.join(f'{k} {v:.3f}' if isinstance(v, float) else f'{k} {v}' for k, v in cell.items()), flush=True)
  path = os.path.join(OUT_DIR, 'results', f'name_pref_fold{args.fold}.json')
  with open(path, 'w') as f:
    json.dump(res, f, indent=1)


if __name__ == '__main__':
  main()
