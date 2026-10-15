"""PriceTag step 9 (PLAN_AMPLIFY.md, lead 1): amplify the model's own P / Q
value at the query integer token and see which failures it fixes.

Conditions: P only, Q only, P + Q; DAS direction or 3 magnitude-matched random
directions; alpha in {2, 4, 8} (alpha = 1 is the baseline). Prompts carry the
"Answer:" suffix and are scored on the first token's yes/no family, as for
the DAS bases. Sets: ID validation (to choose alpha), and ID, currency and
typo test, fold 0 by default.

Post hoc (added after the smoke run, not in the plan's predictions): '-F'
conditions amplify only rows whose coordinate reads False. Amplifying a True
value overrides the other check (P True, Q False prompts flip to Yes), so for
an AND the False side is the one to strengthen.

Usage: python 09_amplify.py [--fold 0] [--p_das ...] [--q_das ...]
Writes outputs/amplify/{tag}_fold{f}.json (per-prompt verdicts) and prints
the summary used in PLAN_AMPLIFY.md.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

import pt_common as C

sys.path.insert(0, os.path.join(os.path.dirname(C.HERE), 'ioi_repro'))
from amplify import Amplifier, load_directions, midpoint, random_directions  # noqa: E402

ALPHAS = (2.0, 4.0, 8.0)
CONDS = ('P', 'Q', 'PQ', 'P-F', 'Q-F', 'PQ-F')
N_RANDOM = 3
SETS = ('id_val', 'id', 'ood_currency', 'ood_typo')
MAX_BREAK = 0.05


def batches(tokenizer, prompts, batch_size):
  """Cached (encoding, site positions) for left-padded batches of prompts + "Answer:"."""
  out = []
  for i in range(0, len(prompts), batch_size):
    chunk = [C.with_answer(p) for p in prompts[i:i + batch_size]]
    enc = tokenizer(chunk, return_tensors='pt', padding=True)
    n = enc.input_ids.shape[1]
    pos = [n - len(tokenizer(p).input_ids) + C.query_token_index(tokenizer, p) for p in chunk]
    out.append((enc.to('cuda'), torch.tensor(pos, device='cuda')))
  return out


@torch.no_grad()
def said_yes(model, amps, cached, ids):
  out = []
  for enc, pos in cached:
    for a in amps:
      a.positions = pos
    m = C.family_mass(model(**enc).logits[:, -1], ids)
    out += (m[:, 1] > m[:, 0]).cpu().tolist()
  for a in amps:
    a.positions = None
  return np.array(out)


def fit_centers(model, amp, cached, classes, ids):
  amp.alpha = 1.0
  amp.start_recording()
  said_yes(model, [amp], cached, ids)
  rec = amp.stop_recording()
  amp.m = {layer: midpoint(c[:, 0], classes) for layer, c in rec.items()}
  cls = np.asarray(classes, bool)
  amp.gate_sign = {layer: float(np.sign(c[~cls, 0].mean() - c[cls, 0].mean())) for layer, c in rec.items()}
  if amp.r is not None:
    amp.m_r = {layer: midpoint(c[:, 1], classes) for layer, c in rec.items()}
  return {layer: {'mean_true': float(c[cls, 0].mean()), 'mean_false': float(c[~cls, 0].mean()),
                  'sd': float(c[:, 0].std())} for layer, c in rec.items()}


def site_wrong(fold, run, setting):
  """{prompt: variable wrong at the site} from the existing CF-sim scores."""
  path = os.path.join(C.OUT_DIR, 'cf_sim', f'{run}_{setting}_test.json')
  if not os.path.exists(path):
    return {}
  with open(path) as f:
    scores = json.load(f)['scores']
  var = run.split('-')[1]
  return {x['prompt']: (x['patched_yes_rate'] >= 0.5) != bool(x[var]) for x in scores}


def summarize(res, args):
  """Fix / break rates and the PLAN_AMPLIFY.md predictions."""
  out = {}
  sw = {var: {s: site_wrong(args.fold, run, s) for s in SETS[1:]}
        for var, run in (('P', args.p_das), ('Q', args.q_das))}
  # Choose alpha per (condition, direction) on ID val.
  chosen = {}
  for cond in CONDS:
    for direction in ['das'] + [f'rand{i}' for i in range(N_RANDOM)]:
      d = res['id_val']
      truth = np.array(d['answer'])
      ok0 = np.array(d['baseline']) == truth
      best = 1.0
      for alpha in ALPHAS:
        v = np.array(d['runs'][f'{cond}|{direction}|{alpha}'])
        if (v[ok0] != truth[ok0]).mean() <= MAX_BREAK:
          best = alpha
      chosen[cond, direction] = best
  das_alpha = {cond: chosen[cond, 'das'] for cond in CONDS}
  out['chosen_alpha'] = {f'{c}|{d}': a for (c, d), a in chosen.items()}
  print('alpha chosen on ID val (break <= 5%):', das_alpha)

  for s in SETS:
    d = res[s]
    prompts, truth = d['prompts'], np.array(d['answer'])
    base = np.array(d['baseline'])
    ok0 = base == truth
    v_ = [C.parse_prompt(p) for p in prompts]
    left = np.array([not v['P'] for v in v_]) & base         # P False, said Yes
    right = np.array([v['P'] and not v['Q'] for v in v_]) & base
    margin_l = np.array([v['lb'] - v['q'] for v in v_])
    sw_p = np.array([sw['P'].get(s, {}).get(p, np.nan) for p in prompts], dtype=float)
    sw_q = np.array([sw['Q'].get(s, {}).get(p, np.nan) for p in prompts], dtype=float)
    cell = {'n': len(prompts), 'baseline_acc': float(ok0.mean()), 'n_fail': int((~ok0).sum()),
            'n_left_fail': int(left.sum()), 'n_right_fail': int(right.sum()), 'runs': {}}
    print(f'\n[{s}] baseline acc {ok0.mean():.3f}; failures {(~ok0).sum()} '
          f'(left {left.sum()}, right {right.sum()})')
    for cond in CONDS:
      for alpha in ALPHAS:
        for direction in ['das'] + [f'rand{i}' for i in range(N_RANDOM)]:
          v = np.array(d['runs'][f'{cond}|{direction}|{alpha}'])
          fixed, broke = ~ok0 & (v == truth), ok0 & (v != truth)
          r = {'fix_all': float(fixed.sum() / max(1, (~ok0).sum())),
               'break': float(broke.sum() / max(1, ok0.sum())),
               'net_acc': float((v == truth).mean() - ok0.mean())}
          strata = {
            'left': left, 'right': right,
            'left_clear_siteright': left & (margin_l >= 1) & (sw_p == 0),
            'left_siteright': left & (sw_p == 0), 'left_sitewrong': left & (sw_p == 1),
            'right_siteright': right & (sw_q == 0), 'right_sitewrong': right & (sw_q == 1)}
          for name, sel in strata.items():
            r[f'n_{name}'] = int(sel.sum())
            r[f'fix_{name}'] = float(fixed[sel].mean()) if sel.any() else None
          cell['runs'][f'{cond}|{direction}|{alpha}'] = r
        das = cell['runs'][f'{cond}|das|{alpha}']
        rnd = np.mean([cell['runs'][f'{cond}|rand{i}|{alpha}']['fix_all'] for i in range(N_RANDOM)])
        f = lambda x: ' -- ' if x is None else f'{x:.2f}'  # noqa: E731
        mark = '*' if alpha == das_alpha[cond] else ' '
        print(f' {mark}{cond:4s} a={alpha:<3g} fix {das["fix_all"]:.2f} break {das["break"]:.2f} '
              f'net {das["net_acc"]:+.3f} | rand fix {rnd:.2f} | left clear+right-site '
              f'{f(das["fix_left_clear_siteright"])} ({das["n_left_clear_siteright"]}) '
              f'left site-right {f(das["fix_left_siteright"])} ({das["n_left_siteright"]}) '
              f'site-wrong {f(das["fix_left_sitewrong"])} ({das["n_left_sitewrong"]}) | right '
              f'site-right {f(das["fix_right_siteright"])} ({das["n_right_siteright"]}) '
              f'site-wrong {f(das["fix_right_sitewrong"])} ({das["n_right_sitewrong"]})')
    out[s] = cell
  return out


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--fold', type=int, default=0)
  parser.add_argument('--p_das', default=None)
  parser.add_argument('--q_das', default=None)
  parser.add_argument('--batch_size', type=int, default=64)
  parser.add_argument('--summarize_only', action='store_true')
  parser.add_argument('--limit', type=int, default=None, help='smoke test: first n prompts per set')
  args = parser.parse_args()
  args.p_das = args.p_das or f'pt-P-fold{args.fold}_layer8-9-10_dim1_n10000_seed0'
  args.q_das = args.q_das or f'pt-Q-fold{args.fold}_layer11-12_dim1_n10000_seed0'
  tag = f'{args.p_das}__{args.q_das.split("_", 1)[1]}'
  path = os.path.join(C.OUT_DIR, 'amplify', f'{tag}.json')
  if args.summarize_only:
    with open(path) as f:
      res = json.load(f)
    json.dump(summarize(res['sets'], args), open(path.replace('.json', '_summary.json'), 'w'), indent=1)
    return

  C.set_seed(0)
  model, tokenizer = C.load_model_and_tokenizer()
  ids = C.family_ids(tokenizer)
  amps = {var: Amplifier(model, load_directions(os.path.join(C.OUT_DIR, 'das', f'{run}.pt')))
          for var, run in (('P', args.p_das), ('Q', args.q_das))}
  verified = C.verified_bases(model, tokenizer, args.fold)
  ver_cached = batches(tokenizer, verified, args.batch_size)
  classes = {var: [C.parse_prompt(p)[var] for p in verified] for var in amps}
  rand = {var: [random_directions(a.w, seed) for seed in range(N_RANDOM)] for var, a in amps.items()}

  # Centers: DAS (from a pass with no random direction) and per random direction.
  centers = {var: fit_centers(model, a, ver_cached, classes[var], ids) for var, a in amps.items()}
  print('DAS coordinate stats on verified ID prompts:', json.dumps(centers), flush=True)
  das_m = {var: dict(a.m) for var, a in amps.items()}
  base_ver = said_yes(model, list(amps.values()), ver_cached, ids)
  print('sanity: verified bases correct with hooks at alpha 1:',
        float(np.mean(base_ver == np.array([C.parse_prompt(p)['answer'] for p in verified]))), flush=True)
  rand_m = {}
  for var, a in amps.items():
    for i, r in enumerate(rand[var]):
      a.set_random(r, None)
      fit_centers(model, a, ver_cached, classes[var], ids)
      rand_m[var, i] = dict(a.m_r)
      a.set_random(None, None)
      a.m = dict(das_m[var])  # fit_centers overwrote m with the same DAS midpoints

  splits = C.load_split('id', args.fold)
  sets = {'id_val': splits['val']['correct'] + splits['val']['wrong']}
  for s in SETS[1:]:
    t = C.load_split(s, args.fold)['test']
    sets[s] = t['correct'] + t['wrong']
  if args.limit:
    sets = {s: p[:args.limit // 2] + p[-(args.limit // 2):] for s, p in sets.items()}
    path = path.replace('.json', '_smoke.json')
  res = {}
  for s, prompts in sets.items():
    cached = batches(tokenizer, prompts, args.batch_size)
    for a in amps.values():
      a.alpha = 1.0
    d = {'prompts': prompts, 'answer': [C.parse_prompt(p)['answer'] for p in prompts],
         'baseline': said_yes(model, list(amps.values()), cached, ids).tolist(), 'runs': {}}
    for cond in CONDS:
      vars_ = cond.split('-')[0]
      active = [amps[v] for v in vars_]
      for a in active:
        a.gate = cond.endswith('-F')
      for direction in ['das'] + [f'rand{i}' for i in range(N_RANDOM)]:
        for var in vars_:
          if direction == 'das':
            amps[var].set_random(None, None)
          else:
            i = int(direction[4:])
            amps[var].set_random(rand[var][i], rand_m[var, i])
        for alpha in ALPHAS:
          for a in active:
            a.alpha = alpha
          d['runs'][f'{cond}|{direction}|{alpha}'] = said_yes(model, active, cached, ids).tolist()
          for a in active:
            a.alpha = 1.0
        for a in active:
          a.set_random(None, None)
      for a in active:
        a.gate = False
    acc = np.mean(np.array(d['baseline']) == np.array(d['answer']))
    print(f'{s}: {len(prompts)} prompts, baseline acc {acc:.3f}', flush=True)
    res[s] = d

  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, 'w') as f:
    json.dump({'args': vars(args), 'centers': centers, 'sets': res}, f)
  summary = summarize(res, args)
  with open(path.replace('.json', '_summary.json'), 'w') as f:
    json.dump(summary, f, indent=1)


if __name__ == '__main__':
  main()
