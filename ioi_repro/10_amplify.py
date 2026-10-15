"""IOI step 10 (PLAN_AMPLIFY.md, lead 1): amplify the model's own position
variable at the repeated name's token and see which failures it fixes.

DAS: fold 0, layer 12, dim 1. alpha in {2, 4, 8} (1 = baseline), along the DAS
direction or 3 magnitude-matched random directions. Scoring: the first token
of the correct name vs the other name (with the answer prefix); prompts whose
two names share a first token are dropped. Sets: ID validation (to choose
alpha) and ID, Chinese and rephrased test.

Usage: python 10_amplify.py [--das ioi-fold0_layer12_dim1_n10000_seed0]
Writes outputs/amplify/{das}.json and _summary.json
"""

import argparse
import json
import os

import numpy as np
import torch

from amplify import Amplifier, load_directions, midpoint, random_directions
from common import OUT_DIR, load_model_and_tokenizer, set_seed
from ioi_data import answer_prefix, intervention_token_index, load_split, parse_ioi_prompt

ALPHAS = (2.0, 4.0, 8.0)
N_RANDOM = 3
SETS = ('id_val', 'id', 'ood_chinese', 'ood_rephrase')
MAX_BREAK = 0.05


def first_tokens(tokenizer, prompt):
  """(correct name's first token, other name's first token)."""
  v = parse_ioi_prompt(prompt)
  other = v['name_b'] if v['answer'] == v['name_a'] else v['name_a']
  pre = answer_prefix(prompt)
  return tuple(tokenizer(pre + n, add_special_tokens=False).input_ids[0] for n in (v['answer'], other))


def batches(tokenizer, prompts, batch_size):
  out = []
  for i in range(0, len(prompts), batch_size):
    chunk = prompts[i:i + batch_size]
    enc = tokenizer(chunk, return_tensors='pt', padding=True)
    n = enc.input_ids.shape[1]
    pos = [n - len(tokenizer(p).input_ids) + intervention_token_index(tokenizer, p) for p in chunk]
    toks = torch.tensor([first_tokens(tokenizer, p) for p in chunk], device='cuda')
    out.append((enc.to('cuda'), torch.tensor(pos, device='cuda'), toks))
  return out


@torch.no_grad()
def correct(model, amp, cached):
  out = []
  for enc, pos, toks in cached:
    amp.positions = pos
    logits = model(**enc).logits[:, -1].float()
    out += (logits.gather(1, toks[:, :1]) > logits.gather(1, toks[:, 1:])).squeeze(1).cpu().tolist()
  amp.positions = None
  return np.array(out)


def fit_centers(model, amp, cached, classes):
  amp.alpha = 1.0
  amp.start_recording()
  correct(model, amp, cached)
  rec = amp.stop_recording()
  cls = np.asarray(classes, bool)
  m = {layer: midpoint(c[:, 0], cls) for layer, c in rec.items()}
  m_r = {layer: midpoint(c[:, 1], cls) for layer, c in rec.items()} if amp.r is not None else None
  stats = {layer: {'mean_first': float(c[cls, 0].mean()), 'mean_second': float(c[~cls, 0].mean()),
                   'sd': float(c[:, 0].std())} for layer, c in rec.items()}
  return m, m_r, stats


def summarize(res, das):
  out = {}
  d = res['id_val']
  ok0 = np.array(d['baseline'])
  chosen = {}
  for direction in ['das'] + [f'rand{i}' for i in range(N_RANDOM)]:
    best = 1.0
    for alpha in ALPHAS:
      if (~np.array(d['runs'][f'{direction}|{alpha}'])[ok0]).mean() <= MAX_BREAK:
        best = alpha
    chosen[direction] = best
  out['chosen_alpha'] = chosen
  print('alpha chosen on ID val (break <= 5%):', chosen)
  for s in SETS:
    d = res[s]
    ok0 = np.array(d['baseline'])
    sw = {}
    path = os.path.join(OUT_DIR, 'cf_sim', f'{das}_{s}_test_truthlabels.json')
    if os.path.exists(path):
      with open(path) as f:
        sw = {x['prompt']: x['output_mapping'] < 0.5 for x in json.load(f)['scores']}
    site_wrong = np.array([sw.get(p, np.nan) for p in d['prompts']], dtype=float)
    cell = {'n': len(ok0), 'baseline_acc': float(ok0.mean()), 'n_fail': int((~ok0).sum()), 'runs': {}}
    print(f'\n[{s}] n {len(ok0)}, baseline acc {ok0.mean():.3f}, failures {(~ok0).sum()} '
          f'(site-wrong {int(np.nansum(site_wrong[~ok0]))})')
    for alpha in ALPHAS:
      for direction in ['das'] + [f'rand{i}' for i in range(N_RANDOM)]:
        v = np.array(d['runs'][f'{direction}|{alpha}'])
        fixed, broke = ~ok0 & v, ok0 & ~v
        r = {'fix_all': float(fixed.sum() / max(1, (~ok0).sum())),
             'break': float(broke.sum() / max(1, ok0.sum())), 'net_acc': float(v.mean() - ok0.mean())}
        for name, sel in (('siteright', ~ok0 & (site_wrong == 0)), ('sitewrong', ~ok0 & (site_wrong == 1))):
          r[f'n_{name}'] = int(sel.sum())
          r[f'fix_{name}'] = float(fixed[sel].mean()) if sel.any() else None
        cell['runs'][f'{direction}|{alpha}'] = r
      das_r = cell['runs'][f'das|{alpha}']
      rnd = np.mean([cell['runs'][f'rand{i}|{alpha}']['fix_all'] for i in range(N_RANDOM)])
      rnd_b = np.mean([cell['runs'][f'rand{i}|{alpha}']['break'] for i in range(N_RANDOM)])
      mark = '*' if alpha == chosen['das'] else ' '
      print(f' {mark}a={alpha:<3g} fix {das_r["fix_all"]:.2f} break {das_r["break"]:.2f} '
            f'net {das_r["net_acc"]:+.3f} | random fix {rnd:.2f} break {rnd_b:.2f}')
    out[s] = cell
  return out


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--das', default='ioi-fold0_layer12_dim1_n10000_seed0')
  parser.add_argument('--batch_size', type=int, default=32)
  parser.add_argument('--limit', type=int, default=None, help='smoke test: n prompts per set')
  parser.add_argument('--summarize_only', action='store_true')
  args = parser.parse_args()
  fold = int(args.das.split('fold')[1].split('_')[0])
  path = os.path.join(OUT_DIR, 'amplify', f'{args.das}{"_smoke" if args.limit else ""}.json')
  if args.summarize_only:
    with open(path) as f:
      res = json.load(f)['sets']
    with open(path.replace('.json', '_summary.json'), 'w') as f:
      json.dump(summarize(res, args.das), f, indent=1)
    return

  set_seed(0)
  model, tokenizer = load_model_and_tokenizer()
  amp = Amplifier(model, load_directions(os.path.join(OUT_DIR, 'das', f'{args.das}.pt')))
  data = {s: load_split(s, fold) for s in ('id', 'ood_chinese', 'ood_rephrase')}

  def usable(prompts):
    return [p for p in prompts if len(set(first_tokens(tokenizer, p))) == 2]

  verified = usable(data['id']['train']['correct'])
  ver_cached = batches(tokenizer, verified, args.batch_size)
  classes = [parse_ioi_prompt(p)['position'] == 'first' for p in verified]
  amp.m, _, stats = fit_centers(model, amp, ver_cached, classes)
  das_m = dict(amp.m)
  print('DAS coordinate stats on verified ID prompts:', json.dumps(stats), flush=True)
  print('sanity: verified correct with hooks at alpha 1:',
        float(correct(model, amp, ver_cached).mean()), flush=True)
  rand = []
  for seed in range(N_RANDOM):
    r = random_directions(amp.w, seed)
    amp.set_random(r, None)
    _, m_r, _ = fit_centers(model, amp, ver_cached, classes)
    rand.append((r, m_r))
    amp.set_random(None, None)
  amp.m = das_m

  sets = {'id_val': data['id']['val']['correct'] + data['id']['val']['wrong']}
  for s in SETS[1:]:
    sets[s] = data[s]['test']['correct'] + data[s]['test']['wrong']
  sets = {s: usable(p) for s, p in sets.items()}
  if args.limit:
    sets = {s: p[:args.limit // 2] + p[-(args.limit // 2):] for s, p in sets.items()}
  res = {}
  for s, prompts in sets.items():
    cached = batches(tokenizer, prompts, args.batch_size)
    amp.alpha = 1.0
    d = {'prompts': prompts, 'baseline': correct(model, amp, cached).tolist(), 'runs': {}}
    for direction in ['das'] + [f'rand{i}' for i in range(N_RANDOM)]:
      if direction == 'das':
        amp.set_random(None, None)
      else:
        amp.set_random(*rand[int(direction[4:])])
      for alpha in ALPHAS:
        amp.alpha = alpha
        d['runs'][f'{direction}|{alpha}'] = correct(model, amp, cached).tolist()
      amp.alpha = 1.0
      amp.set_random(None, None)
    print(f'{s}: {len(prompts)} prompts, baseline acc {np.mean(d["baseline"]):.3f}', flush=True)
    res[s] = d

  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, 'w') as f:
    json.dump({'args': vars(args), 'stats': stats, 'sets': res}, f)
  with open(path.replace('.json', '_summary.json'), 'w') as f:
    json.dump(summarize(res, args.das), f, indent=1)


if __name__ == '__main__':
  main()
