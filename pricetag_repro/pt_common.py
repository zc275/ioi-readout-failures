"""Shared PriceTag pieces for DAS and counterfactual simulation.

- Intervention site: the last token of the integer part of the query price
  (paper Table 5, "the integer part of the item price").
- Answers: base prompts get the "Answer:" suffix (PLAN.md, agreed), so the
  model answers Yes/No as the first token. A verdict is the family (all case
  variants of yes / no, as in the paper's answer tokens) with more probability.
- Verified bases: released ID train/correct, our free-form verdict correct,
  and the first token after "Answer:" correct. Cached per fold.
"""

import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'ioi_repro'))
from common import load_model_and_tokenizer, set_seed  # noqa: E402,F401
from pricetag_data import _NUMBER, _USER_TEXT, load_split, parse_prompt  # noqa: E402

OUT_DIR = os.path.join(HERE, 'outputs')
INPUT_MAX_LEN = 40   # longest prompt is 36 tokens, with "Answer:" 35 (ID)
ANSWER_SUFFIX = 'Answer:'
# Paper Table 5: P at layers 8-10, Q at 11-12, intervention dimension 1.
PAPER_LAYERS = {'P': [8, 9, 10], 'Q': [11, 12]}
OTHER = {'P': 'Q', 'Q': 'P'}


def with_answer(prompt):
  return prompt if prompt.endswith(ANSWER_SUFFIX) else prompt + ANSWER_SUFFIX


def family_ids(tokenizer):
  """Token ids of all case variants of yes / no, with and without a leading space."""
  ids = {}
  for name in ('yes', 'no'):
    found = set()
    for w in (name, name.capitalize(), name.upper()):
      for s in (w, ' ' + w):
        t = tokenizer(s, add_special_tokens=False).input_ids
        if len(t) == 1:
          found.add(t[0])
    ids[name] = sorted(found)
  return ids


def family_mass(logits, ids):
  """(B, V) logits -> (B, 2) probabilities of [no, yes] families."""
  p = torch.softmax(logits.float(), -1)
  return torch.stack([p[:, ids['no']].sum(-1), p[:, ids['yes']].sum(-1)], -1)


def query_token_index(tokenizer, prompt):
  """Index (in the tokenized prompt, BOS included) of the last token of the
  query price's integer part."""
  m = _USER_TEXT.search(prompt)
  numbers = list(_NUMBER.finditer(m.group('text')))
  assert len(numbers) == 3, prompt
  q = numbers[2]
  int_len = len(q.group(0).replace(',', '.').split('.')[0])
  last_char = m.start('text') + q.start() + int_len - 1
  enc = tokenizer(prompt, return_offsets_mapping=True)
  for i, (a, b) in enumerate(enc.offset_mapping):
    if a <= last_char < b:
      return i
  raise ValueError(prompt)


def padded_site(tokenizer, prompt, site='query'):
  """Intervention position in the left-padded sequence. site='query': the
  query's integer token (paper); 'last': the final prompt token."""
  n = len(tokenizer(prompt).input_ids)
  assert n <= INPUT_MAX_LEN, (n, prompt)
  if site == 'last':
    return INPUT_MAX_LEN - 1
  return INPUT_MAX_LEN - n + query_token_index(tokenizer, prompt)


def locations():
  return {f'pos{i}': {'max_input_length': INPUT_MAX_LEN, 'inv_position': [i]}
          for i in range(INPUT_MAX_LEN)}


def load_behavior(setting):
  with open(os.path.join(OUT_DIR, 'behavior', f'{setting}.json')) as f:
    return json.load(f)['results']


@torch.no_grad()
def answer_first_token(model, tokenizer, prompts, batch_size=64):
  """Greedy family verdict (True = yes) and family masses after "Answer:"."""
  ids = family_ids(tokenizer)
  said, masses = [], []
  for i in range(0, len(prompts), batch_size):
    enc = tokenizer([with_answer(p) for p in prompts[i:i + batch_size]],
                    return_tensors='pt', padding=True).to(model.device)
    logits = model(**enc).logits[:, -1]
    m = family_mass(logits, ids)
    said += (m[:, 1] > m[:, 0]).cpu().tolist()
    masses += m.cpu().tolist()
  return said, masses


def verified_bases(model, tokenizer, fold):
  """Released ID train/correct prompts our model also gets right, free-form
  and after "Answer:". Cached in outputs/verified/fold{fold}.json."""
  path = os.path.join(OUT_DIR, 'verified', f'fold{fold}.json')
  if os.path.exists(path):
    with open(path) as f:
      return json.load(f)['prompts']
  behavior = load_behavior('id')
  candidates = [p for p in load_split('id', fold)['train']['correct']
                if behavior[p]['our_label'] == 'correct']
  said, masses = answer_first_token(model, tokenizer, candidates)
  keep = [p for p, s in zip(candidates, said) if s == parse_prompt(p)['answer']]
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, 'w') as f:
    json.dump({'fold': fold, 'n_released_correct': len(load_split('id', fold)['train']['correct']),
               'n_ours_correct': len(candidates), 'n_answer_correct': len(keep),
               'prompts': keep}, f)
  print(f'verified bases fold {fold}: {len(candidates)} free-form correct -> '
        f'{len(keep)} also correct after "Answer:"', flush=True)
  return keep
