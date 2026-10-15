"""PriceTag step 2: localize P (left-boundary check) or Q (right-boundary
check) with DAS at the query price's integer token (paper Table 5).

Same training recipe as ioi_repro/03_das_train.py (batch 16, AdamW lr 1e-4,
one epoch over 10,000 pairs), with three PriceTag-specific choices:

- Bases carry the "Answer:" suffix and are scored on the first answer token
  (PLAN.md). Sources are the released prompts, unchanged.
- Bases are *informative* for the variable: for P, bases with Q = True; for Q,
  bases with P = True. Otherwise the output is No whatever the patched value
  (Q = False forces No; P = False forces No via the shortcut in the paper's
  Fig. 6), and the pair carries no signal. The counterfactual label is then
  simply the source's value of the variable: Yes iff X_src is True.
- Loss: -log of the counterfactual family's probability (all case variants of
  yes / no), instead of cross-entropy on one spelling.

Usage: python 02_das_train.py --fold 0 --var P [--layers 8 9 10] [--dim 1]
Writes outputs/das/{run}.pt and {run}.json
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
from datasets import Dataset
from transformers import get_linear_schedule_with_warmup

import pt_common as C
import pyvene as pv
from causal_data_utils import FEATURE_TYPES, get_dataloader
from causal_interventions import (LowRankRotatedSpaceIntervention, get_intervention_config,
                                  train_intervention_step)


def make_example(tokenizer, base, source, var, site):
  bv, sv = C.parse_prompt(base), C.parse_prompt(source)
  assert bv[C.OTHER[var]]
  return {
    'input': C.with_answer(base),
    'label': ' Yes' if bv['answer'] else ' No',
    'source_input': source,
    'source_label': ' Yes' if sv['answer'] else ' No',
    'inv_label': ' Yes' if sv[var] else ' No',
    'split': f'pos{site[C.with_answer(base)]}',
    'source_split': f'pos{site[source]}',
    'flip': bv[var] != sv[var],
  }


def loader(tokenizer, examples, batch_size, shuffle):
  ds = Dataset.from_list([{k: v for k, v in e.items() if k != 'flip'} for e in examples],
                         features=FEATURE_TYPES)
  return get_dataloader(ds, tokenizer=tokenizer, batch_size=batch_size,
                        prompt_max_length=C.INPUT_MAX_LEN, output_max_length=2,
                        first_n=1, shuffle=shuffle)


def step(intervenable, tokenizer, inputs, ids):
  """Family masses [no, yes] at the answer position, after the intervention."""
  inputs = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
  out = train_intervention_step(intervenable, inputs, C.locations(),
                                pad_token_id=tokenizer.pad_token_id)
  return C.family_mass(out.logits[:, -2], ids), inputs


def target(inputs):
  return torch.tensor([s.strip() == 'Yes' for s in inputs['inv_label']], device='cuda').long()


@torch.no_grad()
def iia(intervenable, tokenizer, examples, ids, batch_size):
  hits = []
  for inputs in loader(tokenizer, examples, batch_size, shuffle=False):
    m, inputs = step(intervenable, tokenizer, inputs, ids)
    hits += (m.argmax(-1) == target(inputs)).cpu().tolist()
  hits, flip = np.array(hits), np.array([e['flip'] for e in examples])
  return {'iia': float(hits.mean()), 'iia_flip': float(hits[flip].mean()),
          'iia_same': float(hits[~flip].mean()), 'n': len(hits), 'n_flip': int(flip.sum())}


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--fold', type=int, required=True)
  parser.add_argument('--var', choices=('P', 'Q'), required=True)
  parser.add_argument('--layers', type=int, nargs='+')
  parser.add_argument('--dim', type=int, default=1)
  parser.add_argument('--n_pairs', type=int, default=10000)
  parser.add_argument('--batch_size', type=int, default=16)
  parser.add_argument('--lr', type=float, default=1e-4)
  parser.add_argument('--n_eval', type=int, default=512)
  parser.add_argument('--seed', type=int, default=0)
  args = parser.parse_args()
  args.layers = args.layers or C.PAPER_LAYERS[args.var]
  run = (f'pt-{args.var}-fold{args.fold}_layer{"-".join(map(str, args.layers))}'
         f'_dim{args.dim}_n{args.n_pairs}_seed{args.seed}')
  print(run, flush=True)

  C.set_seed(args.seed)
  model, tokenizer = C.load_model_and_tokenizer()
  ids = C.family_ids(tokenizer)
  print('answer tokens:', {k: [tokenizer.decode(i) for i in v] for k, v in ids.items()}, flush=True)
  verified = C.verified_bases(model, tokenizer, args.fold)
  bases = [p for p in verified if C.parse_prompt(p)[C.OTHER[args.var]]]
  behavior = C.load_behavior('id')
  val = C.load_split('id', args.fold)['val']
  # Held-out sources: ID validation prompts, split by our own verdict (the
  # released ID labels don't reproduce, see PLAN.md).
  val_sources = {k: [p for s in ('correct', 'wrong') for p in val[s]
                     if behavior[p]['our_label'] == k] for k in ('correct', 'wrong')}
  all_prompts = set(verified) | {p for v in val_sources.values() for p in v}
  site = {}
  for p in all_prompts:
    site[p] = C.padded_site(tokenizer, p)
    site[C.with_answer(p)] = C.padded_site(tokenizer, C.with_answer(p))
  print(f'{len(verified)} verified, {len(bases)} informative bases for {args.var}; '
        f'val sources: { {k: len(v) for k, v in val_sources.items()} }', flush=True)

  rng = random.Random(args.seed)
  train = [make_example(tokenizer, rng.choice(bases), rng.choice(verified), args.var, site)
           for _ in range(args.n_pairs)]
  print(f'train pairs: {np.mean([e["flip"] for e in train]):.2f} flip, '
        f'{np.mean([e["inv_label"] == " Yes" for e in train]):.2f} counterfactual Yes', flush=True)
  eval_sets = {f'val_{k}': [make_example(tokenizer, rng.choice(bases), s, args.var, site)
                            for s in rng.sample(v, min(args.n_eval, len(v)))]
               for k, v in val_sources.items()}

  config = get_intervention_config(
    type(model), 'block_output', args.layers, LowRankRotatedSpaceIntervention,
    intervention_units=['pos'] * len(args.layers), intervention_dimension=args.dim)
  intervenable = pv.IntervenableModel(config, model)
  intervenable.set_device('cuda')
  intervenable.disable_model_gradients()

  before = {k: iia(intervenable, tokenizer, v, ids, 32) for k, v in eval_sets.items()}
  print('IIA before:', json.dumps(before), flush=True)

  train_loader = loader(tokenizer, train, args.batch_size, shuffle=True)
  params = [p for v in intervenable.interventions.values() for p in v[0].rotate_layer.parameters()]
  optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0)
  scheduler = get_linear_schedule_with_warmup(optimizer, 0, 10 * len(train_loader))
  log, losses, hits, t0 = [], [], [], time.time()
  for i, inputs in enumerate(train_loader):
    if i == 0:
      for j in range(3):
        b, s = int(inputs['split'][j][3:]), int(inputs['source_split'][j][3:])
        print(f'  base token {tokenizer.decode(inputs["input_ids"][j][b])!r} <- source token '
              f'{tokenizer.decode(inputs["source_input_ids"][j][s])!r} | '
              f'{inputs["label"][j]!r} -> {inputs["inv_label"][j]!r}', flush=True)
    m, inputs = step(intervenable, tokenizer, inputs, ids)
    y = target(inputs)
    loss = -torch.log(m.gather(1, y[:, None]).squeeze(1).clamp(min=1e-9)).mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    intervenable.set_zero_grad()
    losses.append(loss.item())
    hits += (m.argmax(-1) == y).cpu().tolist()
    if (i + 1) % 25 == 0 or i + 1 == len(train_loader):
      e = {'step': i + 1, 'loss': float(np.mean(losses[-25:])),
           'train_iia': float(np.mean(hits[-25 * args.batch_size:])), 'minutes': (time.time() - t0) / 60}
      log.append(e)
      print(f'step {e["step"]}/{len(train_loader)} loss {e["loss"]:.3f} '
            f'train IIA {e["train_iia"]:.3f} {e["minutes"]:.1f} min', flush=True)

  after = {k: iia(intervenable, tokenizer, v, ids, 32) for k, v in eval_sets.items()}
  print('IIA after:', json.dumps(after), flush=True)
  os.makedirs(os.path.join(C.OUT_DIR, 'das'), exist_ok=True)
  torch.save({k: v[0].rotate_layer.weight.detach().cpu() for k, v in intervenable.interventions.items()},
             os.path.join(C.OUT_DIR, 'das', f'{run}.pt'))
  with open(os.path.join(C.OUT_DIR, 'das', f'{run}.json'), 'w') as f:
    json.dump({'args': vars(args), 'iia_before': before, 'iia_after': after, 'log': log}, f, indent=1)
  print('saved', run, flush=True)


if __name__ == '__main__':
  main()
