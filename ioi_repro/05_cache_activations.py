"""Step 5: cache residual-stream activations once, for all probing methods.

For each prompt we save hidden states at every layer (HF `hidden_states`:
index 0 = embeddings, index L+1 = output of decoder layer L = pyvene's
'block_output' of layer L) at two token positions:
  - site:  last token of the repeated name's second occurrence (DAS site)
  - last:  the final prompt token
Value probing, correctness probing and the planned geometric measures all
run on these caches, on the CPU.

Prompts cached per fold: all ID splits (probes train on ID train/val), plus
the test split of each OOD setting.

Usage: python 05_cache_activations.py [--folds 0 1 2]
Writes outputs/activations/{setting}_fold{fold}.pt with
  {'prompts': [...], 'split': [...], 'label': [...], 'position': [...],
   'site': fp16 (n, 33, 4096), 'last': fp16 (n, 33, 4096)}
"""

import argparse
import os

import torch
from tqdm import tqdm

from common import OUT_DIR, load_model_and_tokenizer
from ioi_data import FOLDS, SETTINGS, intervention_token_index, load_split, parse_ioi_prompt


@torch.no_grad()
def extract(model, tokenizer, prompts, batch_size):
  sites, lasts = [], []
  for b in tqdm(range(0, len(prompts), batch_size), desc='forward'):
    batch = prompts[b:b + batch_size]
    enc = tokenizer(batch, return_tensors='pt', padding=True).to(model.device)
    # Left padding: position ids must count from the first real token.
    position_ids = (enc.attention_mask.cumsum(-1) - 1).clamp(min=0)
    out = model(**enc, position_ids=position_ids, output_hidden_states=True)
    hs = torch.stack(out.hidden_states, dim=1)  # (B, 33, T, 4096)
    T = enc.input_ids.shape[1]
    site_idx = torch.tensor(
      [T - len(tokenizer(p).input_ids) + intervention_token_index(tokenizer, p) for p in batch],
      device=hs.device)
    rows = torch.arange(len(batch), device=hs.device)
    sites.append(hs[rows, :, site_idx].half().cpu())
    lasts.append(hs[:, :, -1].half().cpu())
  return torch.cat(sites), torch.cat(lasts)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--folds', nargs='+', type=int, default=list(FOLDS))
  parser.add_argument('--batch_size', type=int, default=32)
  args = parser.parse_args()
  model, tokenizer = load_model_and_tokenizer()
  os.makedirs(os.path.join(OUT_DIR, 'activations'), exist_ok=True)
  for fold in args.folds:
    for setting in SETTINGS:
      path = os.path.join(OUT_DIR, 'activations', f'{setting}_fold{fold}.pt')
      if os.path.exists(path):
        print('skip', path)
        continue
      data = load_split(setting, fold)
      splits = list(data) if setting == 'id' else ['test']
      rows = [(p, s, k) for s in splits for k in ('correct', 'wrong') for p in data[s][k]]
      prompts = [p for p, _, _ in rows]
      site, last = extract(model, tokenizer, prompts, args.batch_size)
      torch.save({'prompts': prompts, 'split': [s for _, s, _ in rows], 'label': [k for _, _, k in rows],
                  'position': [parse_ioi_prompt(p)['position'] for p in prompts],
                  'site': site, 'last': last}, path)
      print(f'saved {path}: {tuple(site.shape)}', flush=True)


if __name__ == '__main__':
  main()
