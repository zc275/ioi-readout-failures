"""IOI step 18 (PLAN_LOCALIZE.md, second model): forced-choice labels for CF sim.

Qwen2.5-7B starts explaining instead of naming someone on the rephrased
template, so greedy generation gives no label. Here every test prompt is
labelled by forced choice (first tokens, correct minus other), and the chosen
name is saved as the "output" that 04_cf_sim.py reads. The generation-based
splits are kept in splits_generation/. Train and val keep their generation
labels (DAS bases come from ID train/correct).

Usage: OOD_MODEL=qwen2.5-7b python 18_forced_labels.py [--fold 0]
Rewrites {OUT_DIR}/splits/{setting}_fold{fold}.json (test only) and writes
{OUT_DIR}/behavior/{setting}_fold{fold}.json
"""

import argparse
import json
import os
import shutil

from common import MODEL_TAG, OUT_DIR, load_model_and_tokenizer, set_seed
from ioi_data import SETTINGS, answer_prefix, load_split, parse_ioi_prompt

nm = __import__('11_name_pref')


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--fold', type=int, default=0)
  parser.add_argument('--batch_size', type=int, default=64)
  args = parser.parse_args()
  assert MODEL_TAG != 'llama3', 'set OOD_MODEL'
  set_seed(0)
  model, tokenizer = load_model_and_tokenizer()
  gen_dir = os.path.join(OUT_DIR, 'splits_generation')
  os.makedirs(gen_dir, exist_ok=True)
  os.makedirs(os.path.join(OUT_DIR, 'behavior'), exist_ok=True)
  for s in SETTINGS:
    path = os.path.join(OUT_DIR, 'splits', f'{s}_fold{args.fold}.json')
    backup = os.path.join(gen_dir, f'{s}_fold{args.fold}.json')
    if not os.path.exists(backup):
      shutil.copy(path, backup)
    with open(backup) as f:
      data = json.load(f)
    prompts = data['test']['correct'] + data['test']['wrong']
    prompts = [p for p in prompts if len(set(nm.first_tokens(tokenizer, p))) == 2]
    d = nm.logit_diff(model, tokenizer, prompts, args.batch_size)
    data['test'] = {'correct': [p for p, x in zip(prompts, d) if x > 0],
                    'wrong': [p for p, x in zip(prompts, d) if x <= 0]}
    behavior = {}
    for p, x in zip(prompts, d):
      v = parse_ioi_prompt(p)
      other = v['name_b'] if v['answer'] == v['name_a'] else v['name_a']
      behavior[p] = {'output_text': answer_prefix(p) + (v['answer'] if x > 0 else other), 'forced_choice_diff': float(x)}
    with open(path, 'w') as f:
      json.dump(data, f)
    with open(os.path.join(OUT_DIR, 'behavior', f'{s}_fold{args.fold}.json'), 'w') as f:
      json.dump({'results': behavior, 'label_source': 'forced_choice'}, f)
    print(f'[{s}] test {len(prompts)}: correct {len(data["test"]["correct"])}, wrong {len(data["test"]["wrong"])}',
          flush=True)


if __name__ == '__main__':
  main()
