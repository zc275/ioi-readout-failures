"""PriceTag step 1: run the model on every prompt (free-form, greedy).

Same two purposes as ioi_repro/01_behavior.py:
  1. Sanity check: re-judge each output and compare with the released
     correct/wrong labels. Large disagreement means our setup differs from the
     authors' and nothing downstream can be trusted.
  2. Confidence baseline: log-probability of each generated token at T = 1, 2.

Differences from IOI:
  - Answers are free-form ("No, $7.59 is less than $9.32..."), so we generate up
    to MAX_NEW_TOKENS and judge by the first standalone yes/no (pricetag_data.judge).
  - Prompts are generated once per setting (the three folds overlap heavily)
    and the per-fold split and released label are stored alongside.
  - Log-probs are recorded during generation by a logits processor instead of
    keeping every step's full logits, which would not fit in 24 GB. Under
    greedy decoding the chosen token is the argmax of the raw logits, so its
    log-prob at temperature T is log_softmax(logits / T)[argmax].

Usage: python 01_behavior.py [--settings ...] [--limit N]   (--limit: pilot run)
Writes outputs/behavior/{setting}.json
"""

import argparse
import json
import os
import sys

import torch
from tqdm import tqdm
from transformers import LogitsProcessor, LogitsProcessorList

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ioi_repro'))
from common import load_model_and_tokenizer, set_seed  # noqa: E402
from pricetag_data import FOLDS, SETTINGS, judge, load_split, parse_prompt  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
TEMPERATURES = (1.0, 2.0)
MAX_NEW_TOKENS = 160  # the paper selects N up to 98 tokens for PriceTag


class GreedyLogprobRecorder(LogitsProcessor):
  """Records, per step, the log-prob of the greedy token at each temperature."""

  def __init__(self):
    self.steps = []  # list of (B, len(TEMPERATURES)) tensors

  def __call__(self, input_ids, scores):
    s = scores.float()
    top = s.argmax(-1, keepdim=True)
    self.steps.append(torch.stack(
      [torch.log_softmax(s / t, -1).gather(-1, top).squeeze(-1) for t in TEMPERATURES], -1).cpu())
    return scores


@torch.no_grad()
def run_prompts(model, tokenizer, prompts, batch_size):
  stop_ids = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<|eot_id|>'),
              tokenizer.convert_tokens_to_ids('<|end_of_text|>')}
  order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
  results = {}
  for b in tqdm(range(0, len(order), batch_size), desc='generate'):
    batch = [prompts[i] for i in order[b:b + batch_size]]
    # The released prompts start at the user header; the tokenizer adds BOS,
    # as in the IOI pipeline (which matches the released labels).
    enc = tokenizer(batch, return_tensors='pt', padding=True).to(model.device)
    rec = GreedyLogprobRecorder()
    out = model.generate(
      **enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, temperature=None, top_p=None,
      pad_token_id=tokenizer.pad_token_id, logits_processor=LogitsProcessorList([rec]))
    gen = out[:, enc.input_ids.shape[1]:].tolist()
    logps = torch.stack(rec.steps, 1)  # (B, steps, temps)
    for row, prompt in enumerate(batch):
      ids = gen[row]
      cut = next((i + 1 for i, t in enumerate(ids) if t in stop_ids), len(ids))
      ids = ids[:cut]
      results[prompt] = {
        'output_ids': ids,
        'output_text': tokenizer.decode(ids, skip_special_tokens=True),
        'truncated': cut == MAX_NEW_TOKENS and ids[-1] not in stop_ids,
        'logprobs': {str(t): [round(x, 5) for x in logps[row, :len(ids), j].tolist()]
                     for j, t in enumerate(TEMPERATURES)},
      }
  return results


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--settings', nargs='+', default=list(SETTINGS))
  parser.add_argument('--batch_size', type=int, default=64)
  parser.add_argument('--limit', type=int, default=None, help='pilot: first N prompts per setting')
  args = parser.parse_args()

  set_seed(0)
  model, tokenizer = load_model_and_tokenizer()
  os.makedirs(os.path.join(OUT_DIR, 'behavior'), exist_ok=True)

  for setting in args.settings:
    folds = {f: load_split(setting, f) for f in FOLDS}
    # prompt -> {fold: (split, released label)}
    membership = {}
    for f, data in folds.items():
      for split, by_label in data.items():
        for label, ps in by_label.items():
          for p in ps:
            membership.setdefault(p, {})[f] = (split, label)
    prompts = sorted(membership)
    if args.limit:
      prompts = prompts[::max(1, len(prompts) // args.limit)][:args.limit]
    results = run_prompts(model, tokenizer, prompts, args.batch_size)

    counts = {'agree': 0, 'disagree': 0, 'undecided': 0, 'truncated': 0}
    for p, r in results.items():
      verdict, said = judge(r['output_text'], parse_prompt(p))
      r['our_label'], r['said_yes'] = verdict, said
      r['folds'] = {str(f): list(v) for f, v in membership[p].items()}
      released = {lab for _, lab in membership[p].values()}
      counts['truncated'] += r['truncated']
      if verdict == 'undecided':
        counts['undecided'] += 1
      else:
        counts['agree' if released == {verdict} else 'disagree'] += 1
    n = len(results)
    print(f'[{setting}] n={n}  ' + '  '.join(f'{k}={v / n:.4f}' for k, v in counts.items()), flush=True)
    name = f'{setting}_pilot.json' if args.limit else f'{setting}.json'
    with open(os.path.join(OUT_DIR, 'behavior', name), 'w') as f:
      json.dump({'counts': counts, 'results': results}, f)


if __name__ == '__main__':
  main()
