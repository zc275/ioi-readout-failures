"""Step 3: localize the IOI position variable with DAS (paper Sec. 2.1, App. B.2).

Learns an orthogonal projection Q (dim x 4096) at the residual stream
(block_output) of the given layer(s), at the last token of the repeated
name's second occurrence. Training pairs are (base, source) drawn from the
verified set (train/correct). The target is the counterfactual label: the
base's name at the source's position. If patching Q's subspace from the
source into the base makes the base output that name, Q encodes the position
variable.

Training follows the authors' MMLU demo exactly: batch 16, AdamW (lr 1e-4,
no weight decay), their linear schedule over 10x the steps (so ~constant over
our one epoch), cross-entropy on the first N label tokens.

Usage: python 03_das_train.py --fold 0 --layers 9 [--dim 256]
Writes outputs/das/{run_name}.pt (loadable by the authors' load_intervenable)
and outputs/das/{run_name}.json (config + IIA before/after training).
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
from datasets import Dataset
from torch.nn import CrossEntropyLoss
from transformers import get_linear_schedule_with_warmup

from common import OUT_DIR, load_model_and_tokenizer, set_seed
from ioi_data import (answer_prefix, counterfactual_label, intervention_token_index,
                      load_split, parse_ioi_prompt)
# Authors' code (../src).
import pyvene as pv
from causal_data_utils import FEATURE_TYPES, get_dataloader
from causal_interventions import (LowRankRotatedSpaceIntervention, get_intervention_config,
                                  train_intervention_step)

INPUT_MAX_LEN = 64      # longest IOI prompt is 55 tokens
N_LABEL_TOKENS = 3      # longest English name is 3 tokens


def compute_cross_entropy_loss(logits, labels, pad_token_id, next_n_tokens=1):
  """Copied from the authors' demo: cross-entropy over the last n tokens."""
  vocab_size = logits.shape[-1]
  labels = labels.clone()
  shift_logits = logits[..., -next_n_tokens - 1: -1, :].contiguous()
  shift_labels = labels[..., -next_n_tokens:].contiguous()
  shift_logits = shift_logits.view(-1, vocab_size)
  shift_labels = shift_labels.view(-1).to(shift_logits.device)
  shift_labels[shift_labels == pad_token_id] = -100
  return CrossEntropyLoss()(shift_logits, shift_labels)


def exact_match(logits, labels, pad_token_id):
  """Per-example: does the argmax match every (non-padding) label token?

  Label padding is the pad id (counterfactual labels) or -100, as in the
  authors' compute_metrics.
  """
  pred = logits[:, -N_LABEL_TOKENS - 1:-1].argmax(-1)
  labels = labels[:, :N_LABEL_TOKENS].to(pred.device)
  ok = (pred == labels) | (labels < 0) | (labels == pad_token_id)
  return ok.all(-1).cpu().tolist()


def make_example(tokenizer, base, source, site):
  """One (base, source) pair in the authors' dataset format.

  'split' / 'source_split' name the intervention position in the left-padded
  sequence; split_to_inv_locations maps each name to that position.
  """
  bv, sv = parse_ioi_prompt(base), parse_ioi_prompt(source)
  prefix = answer_prefix(base)
  return {
    'input': base,
    'label': prefix + bv['answer'],
    'source_input': source,
    'source_label': answer_prefix(source) + sv['answer'],
    'inv_label': prefix + counterfactual_label(bv, sv),
    'split': f'pos{site[base]}',
    'source_split': f'pos{site[source]}',
  }


def padded_site(tokenizer, prompt, site='name'):
  """Intervention position in the left-padded sequence.

  site='name': last token of the repeated name's second occurrence (Table 5).
  site='last': the final prompt token, where the answer is read out.
  """
  n_tokens = len(tokenizer(prompt).input_ids)
  assert n_tokens <= INPUT_MAX_LEN
  if site == 'last':
    return INPUT_MAX_LEN - 1
  return INPUT_MAX_LEN - n_tokens + intervention_token_index(tokenizer, prompt)


@torch.no_grad()
def evaluate(intervenable, tokenizer, examples, split_to_inv_locations, batch_size):
  loader = get_dataloader(
    Dataset.from_list(examples, features=FEATURE_TYPES), tokenizer=tokenizer,
    batch_size=batch_size, prompt_max_length=INPUT_MAX_LEN,
    output_max_length=N_LABEL_TOKENS + 1, first_n=N_LABEL_TOKENS, shuffle=False)
  matches = []
  for inputs in loader:
    inputs = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    out = train_intervention_step(intervenable, inputs, split_to_inv_locations,
                                  pad_token_id=tokenizer.pad_token_id)
    matches += exact_match(out.logits, inputs['labels'], tokenizer.pad_token_id)
  return matches


def iia_report(intervenable, tokenizer, eval_sets, split_to_inv_locations, batch_size):
  """IIA overall, and on 'flip' pairs where base and source positions differ
  (there the counterfactual label differs from the base's own answer, so a
  no-op intervention scores 0)."""
  report = {}
  for name, examples in eval_sets.items():
    m = np.array(evaluate(intervenable, tokenizer, examples, split_to_inv_locations, batch_size))
    flip = np.array([e['inv_label'] != e['label'] for e in examples])
    report[name] = {'iia': float(m.mean()), 'iia_flip': float(m[flip].mean()),
                    'iia_same': float(m[~flip].mean()), 'n': len(m)}
  return report


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--fold', type=int, required=True)
  parser.add_argument('--layers', type=int, nargs='+', required=True)
  parser.add_argument('--dim', type=int, default=256)
  parser.add_argument('--n_pairs', type=int, default=10000)
  parser.add_argument('--batch_size', type=int, default=16)
  parser.add_argument('--lr', type=float, default=1e-4)
  parser.add_argument('--n_eval', type=int, default=512)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--site', choices=('name', 'last'), default='name')
  args = parser.parse_args()
  run_name = (f'ioi-fold{args.fold}_layer{"-".join(map(str, args.layers))}_dim{args.dim}'
              f'_n{args.n_pairs}_seed{args.seed}' + ('_sitelast' if args.site == 'last' else ''))
  print(run_name, flush=True)

  set_seed(args.seed)
  model, tokenizer = load_model_and_tokenizer()
  data = load_split('id', args.fold)
  verified = data['train']['correct']
  site = {p: padded_site(tokenizer, p, args.site) for s in data for k in data[s] for p in data[s][k]}
  split_to_inv_locations = {
    f'pos{i}': {'max_input_length': INPUT_MAX_LEN, 'inv_position': [i]}
    for i in range(INPUT_MAX_LEN)}

  # Training pairs: uniform sample (without replacement) from verified x verified.
  rng = random.Random(args.seed)
  pair_ids = rng.sample(range(len(verified) ** 2), args.n_pairs)
  train = [make_example(tokenizer, verified[i // len(verified)], verified[i % len(verified)], site)
           for i in pair_ids]
  # Held-out IIA: verified bases, sources from the validation split.
  eval_sets = {
    f'val_{k}': [make_example(tokenizer, rng.choice(verified), src, site)
                 for src in rng.sample(data['val'][k], args.n_eval)]
    for k in ('correct', 'wrong')}

  intervenable_config = get_intervention_config(
    type(model), 'block_output', args.layers, LowRankRotatedSpaceIntervention,
    intervention_units=['pos'] * len(args.layers), intervention_dimension=args.dim)
  intervenable = pv.IntervenableModel(intervenable_config, model)
  intervenable.set_device('cuda')
  intervenable.disable_model_gradients()
  print('trainable parameters:', intervenable.count_parameters(), flush=True)

  before = iia_report(intervenable, tokenizer, eval_sets, split_to_inv_locations, args.batch_size)
  print('IIA before training:', json.dumps(before), flush=True)

  train_loader = get_dataloader(
    Dataset.from_list(train, features=FEATURE_TYPES), tokenizer=tokenizer,
    batch_size=args.batch_size, prompt_max_length=INPUT_MAX_LEN,
    output_max_length=N_LABEL_TOKENS + 1, first_n=N_LABEL_TOKENS)
  params = [p for v in intervenable.interventions.values() for p in v[0].rotate_layer.parameters()]
  optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0)
  scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=10 * len(train_loader))

  log, losses, accs, t0 = [], [], [], time.time()
  for step, inputs in enumerate(train_loader):
    inputs = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    if step == 0:
      # Show what is being patched, as the authors' demo does.
      for i in range(3):
        b, s = int(inputs['split'][i][3:]), int(inputs['source_split'][i][3:])
        print(f'  base token {tokenizer.decode(inputs["input_ids"][i][b])!r} <- '
              f'source token {tokenizer.decode(inputs["source_input_ids"][i][s])!r} | '
              f'base label {inputs["label"][i]!r} -> counterfactual label {inputs["inv_label"][i]!r}')
    out = train_intervention_step(intervenable, inputs, split_to_inv_locations,
                                  pad_token_id=tokenizer.pad_token_id)
    loss = compute_cross_entropy_loss(out.logits, inputs['labels'][:, :N_LABEL_TOKENS],
                                      pad_token_id=tokenizer.pad_token_id,
                                      next_n_tokens=N_LABEL_TOKENS)
    loss.backward()
    optimizer.step()
    scheduler.step()
    intervenable.set_zero_grad()
    losses.append(loss.item())
    accs += exact_match(out.logits.detach(), inputs['labels'], tokenizer.pad_token_id)
    if (step + 1) % 25 == 0 or step + 1 == len(train_loader):
      entry = {'step': step + 1, 'loss': float(np.mean(losses[-25:])),
               'train_iia': float(np.mean(accs[-25 * args.batch_size:])),
               'minutes': (time.time() - t0) / 60}
      log.append(entry)
      print(f'step {entry["step"]}/{len(train_loader)}  loss {entry["loss"]:.3f}  '
            f'train IIA {entry["train_iia"]:.3f}  {entry["minutes"]:.1f} min', flush=True)

  after = iia_report(intervenable, tokenizer, eval_sets, split_to_inv_locations, args.batch_size)
  print('IIA after training:', json.dumps(after), flush=True)

  os.makedirs(os.path.join(OUT_DIR, 'das'), exist_ok=True)
  torch.save({k: v[0].rotate_layer.weight.detach().cpu() for k, v in intervenable.interventions.items()},
             os.path.join(OUT_DIR, 'das', f'{run_name}.pt'))
  with open(os.path.join(OUT_DIR, 'das', f'{run_name}.json'), 'w') as f:
    json.dump({'args': vars(args), 'iia_before': before, 'iia_after': after, 'log': log}, f, indent=1)
  print('saved', run_name, flush=True)


if __name__ == '__main__':
  main()
