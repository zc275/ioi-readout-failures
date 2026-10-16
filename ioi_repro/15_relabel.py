"""IOI step 15 (PLAN_LOCALIZE.md, second model): build a second model's own splits.

Same train/val/test membership as the released Llama-3 splits, prompts moved
into the second model's chat template (user text and the assistant's answer
prefix unchanged), and correct/wrong judged from its own greedy outputs with
01_behavior.judge ('other' counts as wrong, as in the released labels).

Usage: OOD_MODEL=qwen2.5-7b python 15_relabel.py [--folds 0 1 2]
Writes outputs_{tag}/splits/{setting}_fold{fold}.json and a summary
"""

import argparse
import json
import os
import re

from common import MODEL_TAG, OUT_DIR, load_model_and_tokenizer, set_seed
from ioi_data import FOLDS, SETTINGS, load_split, parse_ioi_prompt

beh = __import__('01_behavior')

_LLAMA = re.compile(r'<\|start_header_id\|>user<\|end_header_id\|>\n\n(?P<user>.*?)<\|eot_id\|>'
                    r'<\|start_header_id\|>assistant<\|end_header_id\|>\n\n(?P<prefix>.*)$', re.S)


def retemplate(tokenizer, prompt):
  m = _LLAMA.search(prompt)
  assert m, prompt
  chat = tokenizer.apply_chat_template([{'role': 'user', 'content': m.group('user')}],
                                       tokenize=False, add_generation_prompt=True)
  return chat + m.group('prefix')


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--folds', nargs='+', type=int, default=list(FOLDS))
  parser.add_argument('--batch_size', type=int, default=64)
  args = parser.parse_args()
  assert MODEL_TAG != 'llama3', 'set OOD_MODEL'
  set_seed(0)
  model, tokenizer = load_model_and_tokenizer()
  os.makedirs(os.path.join(OUT_DIR, 'splits'), exist_ok=True)
  summary = {}
  for setting in SETTINGS:
    for fold in args.folds:
      data = load_split(setting, fold, released=True)
      new = {sp: {'correct': [], 'wrong': []} for sp in data}
      prompts = {retemplate(tokenizer, p): sp for sp in data for k in ('correct', 'wrong') for p in data[sp][k]}
      results = beh.run_prompts(model, tokenizer, list(prompts), args.batch_size)
      counts = {'correct': 0, 'wrong': 0, 'other': 0}
      for p, r in results.items():
        verdict, _ = beh.judge(r['output_text'], parse_ioi_prompt(p))
        counts[verdict] += 1
        new[prompts[p]]['correct' if verdict == 'correct' else 'wrong'].append(p)
      n = len(results)
      summary[f'{setting}_fold{fold}'] = {k: v / n for k, v in counts.items()} | {'n': n}
      print(f'[{setting} fold {fold}] n={n} ' + ' '.join(f'{k}={v / n:.3f}' for k, v in counts.items())
            + f'  e.g. {next(iter(results.values()))["output_text"]!r}', flush=True)
      with open(os.path.join(OUT_DIR, 'splits', f'{setting}_fold{fold}.json'), 'w') as f:
        json.dump(new, f)
  with open(os.path.join(OUT_DIR, 'splits', 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=1)


if __name__ == '__main__':
  main()
