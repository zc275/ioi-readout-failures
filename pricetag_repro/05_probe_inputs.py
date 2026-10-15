"""PriceTag step 5: input-fidelity probes (PLAN.md, C2).

Does the model *read* the prices correctly? Linear (ridge) probes trained on
ID prompts decode the lower bound lb, upper bound ub and query q from the
residual stream; we then ask whether decoding is worse, or decoded relations
are wrong, on OOD failures. Needs no answer and no ground-truth label, only
the input text, so it is a deployable check.

Positions (all in the unpadded prompt):
  q_int   last token of the query's integer part (the paper's P/Q site)
  q_end   last token of the query number (after any decimal part)
  last    final prompt token
Layers: every second layer 2..20, plus 24, 28.

Matched comparison for the left-boundary failure (said Yes with q < lb): we
compare those failures with *correct* prompts that also have q < lb (said No).
Both have the same true relation, so a difference in decoded (q - lb) is about
reading, not about which prompts happen to fail.

Usage: python 05_probe_inputs.py [--fold 0]
Writes outputs/results/probe_inputs_fold{f}.json
"""

import argparse
import json
import os

import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score

import pt_common as C

LAYERS = list(range(2, 21, 2)) + [24, 28]
POSITIONS = ('q_int', 'q_end', 'last')


def positions(tokenizer, prompt):
  m = C._USER_TEXT.search(prompt)
  q = list(C._NUMBER.finditer(m.group('text')))[2]
  end_char = m.start('text') + q.end() - 1
  enc = tokenizer(prompt, return_offsets_mapping=True)
  q_end = next(i for i, (a, b) in enumerate(enc.offset_mapping) if a <= end_char < b)
  return {'q_int': C.query_token_index(tokenizer, prompt), 'q_end': q_end,
          'last': len(enc.input_ids) - 1}


@torch.no_grad()
def cache(model, tokenizer, prompts, batch_size=32):
  """(n, len(POSITIONS), len(LAYERS), d) float16 residual stream after each layer."""
  feats = []
  for i in range(0, len(prompts), batch_size):
    batch = prompts[i:i + batch_size]
    enc = tokenizer(batch, return_tensors='pt', padding=True).to(model.device)
    hs = model(**enc, output_hidden_states=True).hidden_states  # [emb, layer0 out, ...]
    width = enc.input_ids.shape[1]
    for j, p in enumerate(batch):
      pos = positions(tokenizer, p)
      pad = width - int(enc.attention_mask[j].sum())
      feats.append(torch.stack([
        torch.stack([hs[layer + 1][j, pad + pos[name]] for layer in LAYERS])
        for name in POSITIONS]).half().cpu())
  return torch.stack(feats).float().numpy()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--fold', type=int, default=0)
  args = parser.parse_args()
  C.set_seed(0)
  model, tokenizer = C.load_model_and_tokenizer()

  id_data = C.load_split('id', args.fold)
  train = [p for s in ('train',) for k in ('correct', 'wrong') for p in id_data[s][k]]
  test = {'id': [p for k in ('correct', 'wrong') for p in id_data['test'][k]]}
  labels = {'id': {p: 1 for p in id_data['test']['correct']} | {p: 0 for p in id_data['test']['wrong']}}
  for setting in ('ood_currency', 'ood_typo'):
    d = C.load_split(setting, args.fold)['test']
    test[setting] = d['correct'] + d['wrong']
    labels[setting] = {p: 1 for p in d['correct']} | {p: 0 for p in d['wrong']}

  def targets(prompts):
    v = [C.parse_prompt(p) for p in prompts]
    return {k: np.array([x[k] for x in v]) for k in ('lb', 'ub', 'q')}

  print(f'caching {len(train)} ID train prompts', flush=True)
  X_train, y_train = cache(model, tokenizer, train), targets(train)
  X_test = {}
  for s, ps in test.items():
    print(f'caching {len(ps)} {s} test prompts', flush=True)
    X_test[s] = cache(model, tokenizer, ps)

  results = {'layers': LAYERS, 'positions': POSITIONS, 'fold': args.fold, 'cells': []}
  for pi, pos in enumerate(POSITIONS):
    for li, layer in enumerate(LAYERS):
      cell = {'position': pos, 'layer': layer}
      preds = {}
      for var in ('lb', 'ub', 'q'):
        probe = RidgeCV(alphas=np.logspace(-1, 4, 11)).fit(X_train[:, pi, li], y_train[var])
        for s in test:
          preds.setdefault(s, {})[var] = probe.predict(X_test[s][:, pi, li])
      for s, ps in test.items():
        t = targets(ps)
        y = np.array([labels[s][p] for p in ps])
        err_q = np.abs(preds[s]['q'] - t['q'])
        rel = preds[s]['q'] - preds[s]['lb']   # decoded (q - lb): > 0 means "P True"
        left = t['q'] < t['lb']
        out = {
          'mae': {v: float(np.abs(preds[s][v] - t[v]).mean()) for v in ('lb', 'ub', 'q')},
          'r2_q': float(1 - ((preds[s]['q'] - t['q']) ** 2).sum() / ((t['q'] - t['q'].mean()) ** 2).sum()),
          'mae_q_correct': float(err_q[y == 1].mean()), 'mae_q_wrong': float(err_q[y == 0].mean()),
          # AUC of -|q error| for correct vs wrong: > 0.5 means failures are decoded worse.
          'auc_q_err': float(roc_auc_score(y, -err_q)),
          # Among prompts with q < lb: does decoded (q - lb) separate failures (said Yes)
          # from successes (said No)? > 0.5 means failures look more like "q > lb".
          'n_left_correct': int((left & (y == 1)).sum()), 'n_left_wrong': int((left & (y == 0)).sum()),
          'auc_left_rel': (float(roc_auc_score(1 - y[left], rel[left]))
                           if 0 < y[left].sum() < left.sum() else None),
          'decoded_P_wrong_rate_left_correct': float((rel[left & (y == 1)] > 0).mean()) if (left & (y == 1)).any() else None,
          'decoded_P_wrong_rate_left_wrong': float((rel[left & (y == 0)] > 0).mean()) if (left & (y == 0)).any() else None,
        }
        cell[s] = out
      results['cells'].append(cell)
      print(f'{pos:5s} L{layer:2d} ' + '  '.join(
        f'{s}: R2q {cell[s]["r2_q"]:.2f} errAUC {cell[s]["auc_q_err"]:.2f} '
        f'leftAUC {cell[s]["auc_left_rel"] if cell[s]["auc_left_rel"] is None else round(cell[s]["auc_left_rel"], 2)}'
        for s in test), flush=True)

  os.makedirs(os.path.join(C.OUT_DIR, 'results'), exist_ok=True)
  with open(os.path.join(C.OUT_DIR, 'results', f'probe_inputs_fold{args.fold}.json'), 'w') as f:
    json.dump(results, f, indent=1)


if __name__ == '__main__':
  main()
