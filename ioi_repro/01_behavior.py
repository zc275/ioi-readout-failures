"""Step 1: run the model on every IOI prompt.

Two purposes:
  1. Sanity check. Re-derive correct/wrong for each prompt and compare with
     the released labels. If they disagree much, our setup (weights, dtype,
     tokenization, padding) does not match the authors', and nothing
     downstream can be trusted.
  2. Confidence-score baseline. Save the log-probability of every generated
     token at several temperatures (paper Eq. 1).

Usage: python 01_behavior.py [--settings id ood_chinese ood_rephrase] [--folds 0 1 2]
Writes outputs/behavior/{setting}_fold{fold}.json
"""

import argparse
import json
import os
import re

import torch
from tqdm import tqdm

from common import OUT_DIR, load_model_and_tokenizer, set_seed
from ioi_data import FOLDS, SETTINGS, answer_prefix, load_split, parse_ioi_prompt

TEMPERATURES = (0.5, 1.0, 1.5, 2.0, 3.0)
MAX_NEW_TOKENS = 8  # longer than any answer (English names <= 3 tokens, Chinese <= 5)


def judge(output_text, variables):
  """Returns (verdict, name the model produced) for a decoded model output.

  verdict is 'correct' only for an exact match of the non-repeated name;
  'wrong' for the repeated name; 'other' for anything else. In the ID data
  'other' is almost always a spelling 'correction' of the right name
  (Harrisson -> Harrison), which the released labels count as wrong.
  """
  text = output_text.lstrip(' "\'')
  io, s = variables['answer'], variables['name_c']
  # Prefer the longer name when one is a prefix of the other (e.g. Ash / Ashley).
  for name, verdict in sorted([(io, 'correct'), (s, 'wrong')], key=lambda x: -len(x[0])):
    if text.startswith(name):
      return verdict, name
  produced = re.match(r'[^"\'.,\n]*', text).group(0).strip()
  return 'other', produced or None


def num_answer_tokens(tokenizer, token_ids, name):
  """How many leading generated tokens it takes to spell out `name`."""
  for k in range(1, len(token_ids) + 1):
    if len(tokenizer.decode(token_ids[:k]).lstrip(' "\'')) >= len(name):
      return k
  return len(token_ids)


@torch.no_grad()
def run_prompts(model, tokenizer, prompts, batch_size):
  # Batch prompts of similar length together to minimise padding.
  order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
  results = {}
  for b in tqdm(range(0, len(order), batch_size), desc='generate'):
    batch = [prompts[i] for i in order[b:b + batch_size]]
    enc = tokenizer(batch, return_tensors='pt', padding=True).to(model.device)
    out = model.generate(
      **enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, temperature=None, top_p=None,
      pad_token_id=tokenizer.pad_token_id, output_logits=True, return_dict_in_generate=True)
    gen = out.sequences[:, enc.input_ids.shape[1]:]          # (B, steps)
    logits = torch.stack(out.logits, dim=1).float()          # (B, steps, vocab)
    per_temp = {}
    for t in TEMPERATURES:
      logp = torch.log_softmax(logits / t, dim=-1)
      per_temp[t] = logp.gather(-1, gen.unsqueeze(-1)).squeeze(-1).cpu()
    for row, prompt in enumerate(batch):
      ids = gen[row].tolist()
      # Keep tokens up to and including the first end-of-turn token.
      if tokenizer.eos_token_id in ids:
        ids = ids[:ids.index(tokenizer.eos_token_id) + 1]
      results[prompt] = {
        'output_ids': ids,
        'output_text': tokenizer.decode(ids, skip_special_tokens=True),
        'logprobs': {str(t): per_temp[t][row, :len(ids)].tolist() for t in TEMPERATURES},
      }
  return results


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--settings', nargs='+', default=list(SETTINGS))
  parser.add_argument('--folds', nargs='+', type=int, default=list(FOLDS))
  parser.add_argument('--batch_size', type=int, default=64)
  args = parser.parse_args()

  set_seed(0)
  model, tokenizer = load_model_and_tokenizer()
  os.makedirs(os.path.join(OUT_DIR, 'behavior'), exist_ok=True)

  for setting in args.settings:
    for fold in args.folds:
      data = load_split(setting, fold)
      prompts = [p for s in data for k in ('correct', 'wrong') for p in data[s][k]]
      released = {p: k for s in data for k in ('correct', 'wrong') for p in data[s][k]}
      split_of = {p: s for s in data for k in ('correct', 'wrong') for p in data[s][k]}
      results = run_prompts(model, tokenizer, prompts, args.batch_size)

      counts = {'agree': 0, 'disagree': 0, 'other': 0}
      for p, r in results.items():
        v = parse_ioi_prompt(p)
        verdict, said = judge(r['output_text'], v)
        r['released_label'] = released[p]
        r['our_label'] = verdict
        r['split'] = split_of[p]
        r['answer_prefix'] = answer_prefix(p)
        r['n_answer_tokens'] = num_answer_tokens(tokenizer, r['output_ids'], said) if said else None
        if verdict == 'other':
          counts['other'] += 1
        # Released labels count every non-exact answer as wrong.
        effective = 'wrong' if verdict == 'other' else verdict
        counts['agree' if effective == released[p] else 'disagree'] += 1
      n = len(results)
      print(f'[{setting} fold {fold}] n={n}  agree={counts["agree"]/n:.4f}  '
            f'disagree={counts["disagree"]/n:.4f}  (other={counts["other"]/n:.4f})', flush=True)
      with open(os.path.join(OUT_DIR, 'behavior', f'{setting}_fold{fold}.json'), 'w') as f:
        json.dump({'counts': counts, 'results': results}, f)


if __name__ == '__main__':
  main()
