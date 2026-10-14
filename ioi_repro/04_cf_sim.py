"""Step 4: counterfactual simulation (paper Sec. 4.3, Eqs. 4-6).

For each test input (the source) we sample K verified ID examples as bases
and patch the DAS subspace from the source into each base. The counterfactual
label is the base's name at the position the source's *output* names
(y_cf = 1(tau(M(x_src))), Sec. 4.3). If the source's output came from the
mechanism identified on verified data, the patched bases should produce y_cf.

Scores per test input, averaged over the K bases (the marginalisation in Eq. 4):
  - first_n:        exp(-mean token cross-entropy of y_cf). This is the
                    authors' demo's `loss_exp`, a monotone variant of Eq. 5.
  - output_mapping: fraction of bases whose (greedy) output is exactly y_cf
                    (Eq. 6). Teacher-forced argmax matching every label token
                    is equivalent to greedy decoding producing y_cf.
AUC-ROC is computed against the released correct/wrong labels.

Usage: python 04_cf_sim.py --fold 0 --das ioi-fold0_layer9_dim256_n10000_seed0
                           [--settings id ood_chinese ood_rephrase] [--splits test]
Writes outputs/cf_sim/{das}_{setting}_{split}.json
"""

import argparse
import importlib
import json
import os
import random
import re

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import roc_auc_score

from common import OUT_DIR, load_model_and_tokenizer, set_seed
from ioi_data import (SETTINGS, answer_prefix, counterfactual_label, load_split,
                      output_position, parse_ioi_prompt)
import pyvene as pv
from causal_data_utils import FEATURE_TYPES, get_dataloader
from causal_interventions import (LowRankRotatedSpaceIntervention, PretrainedFeaturizer,
                                  get_intervention_config, train_intervention_step)

das = importlib.import_module('03_das_train')
N = das.N_LABEL_TOKENS


def load_das(model, path):
  """The authors' load_intervenable, generalised to several layers.

  (Their helper builds a config with a single unit list, which fails for the
  joint layer 9-12 subspaces, and leaves CPU-saved weights on the CPU.)
  """
  weights = torch.load(path, map_location='cuda')
  layers = [int(k.split('.')[1]) for k in weights]
  config = get_intervention_config(
    type(model), 'block_output', layers, LowRankRotatedSpaceIntervention,
    intervention_units=['pos'] * len(layers), intervention_dimension=0)
  intervenable = pv.IntervenableModel(config, model)
  intervenable.set_device('cuda')
  intervenable.disable_model_gradients()
  assert list(intervenable.interventions) == list(weights), (list(intervenable.interventions), list(weights))
  for k, w in weights.items():
    featurizer = PretrainedFeaturizer(w).eval()
    featurizer.weight = featurizer.weight.to('cuda')
    intervenable.interventions[k][0].rotate_layer = featurizer
    intervenable.interventions[k][0].set_interchange_dim(interchange_dim=featurizer.weight.shape[0])
  intervenable.model.eval()
  return intervenable


@torch.no_grad()
def score_pairs(intervenable, tokenizer, examples, locations, batch_size):
  """Per pair: (exp(-mean CE of y_cf), exact match of y_cf)."""
  loader = get_dataloader(
    Dataset.from_list(examples, features=FEATURE_TYPES), tokenizer=tokenizer,
    batch_size=batch_size, prompt_max_length=das.INPUT_MAX_LEN,
    output_max_length=N + 1, first_n=N, shuffle=False)
  pad = tokenizer.pad_token_id
  probs, matches = [], []
  for inputs in loader:
    inputs = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    logits = train_intervention_step(intervenable, inputs, locations, pad_token_id=pad).logits
    logits = logits[:, -N - 1:-1].float()
    labels = inputs['labels'][:, :N]
    valid = (labels >= 0) & (labels != pad)
    ce = torch.nn.functional.cross_entropy(
      logits.transpose(1, 2), labels.clamp(min=0), reduction='none')
    mean_ce = (ce * valid).sum(-1) / valid.sum(-1)
    probs += torch.exp(-mean_ce).cpu().tolist()
    matches += ((logits.argmax(-1) == labels) | ~valid).all(-1).cpu().tolist()
  return probs, matches


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--fold', type=int, required=True)
  parser.add_argument('--das', required=True, help='run name of a 03_das_train.py checkpoint')
  parser.add_argument('--settings', nargs='+', default=list(SETTINGS))
  parser.add_argument('--splits', nargs='+', default=['test'])
  parser.add_argument('--k', type=int, default=10, help='bases per test input')
  parser.add_argument('--pool', type=int, default=16, help='verified examples bases are drawn from')
  parser.add_argument('--batch_size', type=int, default=32)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--label_source', choices=('output', 'truth'), default='output',
                      help="diagnostic: 'truth' takes V from the source's ground truth instead of its output")
  args = parser.parse_args()
  suffix = '' if args.label_source == 'output' else '_truthlabels'
  assert f'fold{args.fold}_' in args.das

  set_seed(args.seed)
  model, tokenizer = load_model_and_tokenizer()
  intervenable = load_das(model, os.path.join(OUT_DIR, 'das', f'{args.das}.pt'))
  locations = {f'pos{i}': {'max_input_length': das.INPUT_MAX_LEN, 'inv_position': [i]}
               for i in range(das.INPUT_MAX_LEN)}
  # As in the authors' demo: a small pool of verified examples, K bases each.
  pool = load_split('id', args.fold)['train']['correct'][:args.pool]
  os.makedirs(os.path.join(OUT_DIR, 'cf_sim'), exist_ok=True)

  for setting in args.settings:
    data = load_split(setting, args.fold)
    with open(os.path.join(OUT_DIR, 'behavior', f'{setting}_fold{args.fold}.json')) as f:
      behavior = json.load(f)['results']
    for split in args.splits:
      rng = random.Random(args.seed)
      sources = [(p, 1) for p in data[split]['correct']] + [(p, 0) for p in data[split]['wrong']]
      where = 'last' if args.das.endswith('_sitelast') else 'name'
      site = {p: das.padded_site(tokenizer, p, where) for p in pool + [s for s, _ in sources]}
      examples, owner = [], []
      for j, (src, _) in enumerate(sources):
        sv = parse_ioi_prompt(src)
        output = behavior[src]['output_text']
        position, exact = output_position(output, sv)
        produced = re.match(r'[^"\'.,\n]*', output.lstrip(' "\'')).group(0).strip() or output.strip()
        for base in rng.sample(pool, args.k):
          bv = parse_ioi_prompt(base)
          # tau maps an output naming name a / name b to V; any other output
          # (e.g. a misspelled name) is not a value of V, so, as in the
          # authors' demo, the label falls back to the source's raw output,
          # which a patched base cannot produce.
          cf = counterfactual_label(bv, {'position': position}) if exact else produced
          if args.label_source == 'truth':
            cf = counterfactual_label(bv, sv)
          examples.append({
            'input': base, 'label': answer_prefix(base) + bv['answer'],
            'source_input': src, 'source_label': output,
            'inv_label': answer_prefix(base) + cf,
            'split': f'pos{site[base]}', 'source_split': f'pos{site[src]}'})
          owner.append(j)
      probs, matches = score_pairs(intervenable, tokenizer, examples, locations, args.batch_size)
      owner = np.array(owner)
      first_n = np.array([np.mean(np.array(probs)[owner == j]) for j in range(len(sources))])
      mapping = np.array([np.mean(np.array(matches)[owner == j]) for j in range(len(sources))])
      labels = np.array([y for _, y in sources])
      result = {
        'das': args.das, 'setting': setting, 'split': split, 'fold': args.fold, 'k': args.k,
        'label_source': args.label_source,
        'auc_first_n': roc_auc_score(labels, first_n),
        'auc_output_mapping': roc_auc_score(labels, mapping),
        'iia_correct': float(mapping[labels == 1].mean()),
        'iia_wrong': float(mapping[labels == 0].mean()),
        # our_label: our model's own verdict (01_behavior.py). It differs from the
        # released label on near-tie prompts, which caps achievable AUC.
        'scores': [{'prompt': s, 'label': int(y), 'our_label': behavior[s]['our_label'],
                    'first_n': float(a), 'output_mapping': float(b)}
                   for (s, y), a, b in zip(sources, first_n, mapping)],
      }
      print(f'[{args.das}{suffix} | {setting} {split}] AUC first-N {result["auc_first_n"]:.3f}  '
            f'output-mapping {result["auc_output_mapping"]:.3f}  '
            f'IIA correct {result["iia_correct"]:.3f} wrong {result["iia_wrong"]:.3f}', flush=True)
      with open(os.path.join(OUT_DIR, 'cf_sim', f'{args.das}_{setting}_{split}{suffix}.json'), 'w') as f:
        json.dump(result, f)


if __name__ == '__main__':
  main()
