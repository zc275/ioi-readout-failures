"""PriceTag step 8 (PLAN_AMPLIFY.md, lead 5): at matched true margin, does the
shift change whether the variable is wrong at the site?

Same quantities as 06_margin.py (X-False prompts, margin = distance to the
bound, site-wrong = patched-yes rate), with three changes:
- outcome is our model's own verdict (our_label), not the released label;
- finer margin bins, and bootstrap CIs for the shift-minus-ID gaps;
- a margin-matched gap: per-bin gaps averaged with weights n_shift + n_id,
  and a linear fit site_wrong ~ bin + setting, separately for failures and
  successes.

Usage: python 08_matched_difficulty.py
Writes outputs/results/matched_difficulty.json
"""

import json
import os

import numpy as np

import pt_common as C

RUNS = {'P': 'layer8-9-10_dim1_n10000_seed0', 'Q': 'layer11-12_dim1_n10000_seed0'}
SETTINGS = ('id', 'ood_currency', 'ood_typo')
EDGES = (0, 0.25, 0.5, 0.75, 1, 1.5, 2, np.inf)
N_BOOT = 2000


def load_rows(var):
  """(setting, bin, failed, site_wrong) for X-False prompts, pooled over folds."""
  rows = []
  for s_i, setting in enumerate(SETTINGS):
    behavior = C.load_behavior(setting)
    for fold in (0, 1, 2):
      path = os.path.join(C.OUT_DIR, 'cf_sim', f'pt-{var}-fold{fold}_{RUNS[var]}_{setting}_test.json')
      with open(path) as f:
        scores = json.load(f)['scores']
      for x in scores:
        v = C.parse_prompt(x['prompt'])
        if var == 'P' and not v['P']:
          margin = v['lb'] - v['q']
        elif var == 'Q' and v['P'] and not v['Q']:
          margin = v['q'] - v['ub']
        else:
          continue
        label = behavior[x['prompt']]['our_label']
        if label not in ('correct', 'wrong'):
          continue
        b = int(np.searchsorted(EDGES, margin, side='right') - 1)
        rows.append((s_i, b, label == 'wrong', x['patched_yes_rate']))
  return np.array(rows, dtype=float)


def cell_means(a):
  """(setting, bin, failed) -> (mean site-wrong, n)."""
  out = {}
  for s in range(len(SETTINGS)):
    for b in range(len(EDGES) - 1):
      for fl in (0, 1):
        sel = (a[:, 0] == s) & (a[:, 1] == b) & (a[:, 2] == fl)
        out[s, b, fl] = (a[sel, 3].mean() if sel.any() else np.nan, int(sel.sum()))
  return out


def matched_gap(a, s, failed, bins):
  """Weighted mean over bins of [site-wrong(shift s) - site-wrong(ID)]."""
  m = cell_means(a)
  num = den = 0.0
  for b in bins:
    (x, nx), (y, ny) = m[s, b, failed], m[0, b, failed]
    if nx and ny:
      num += (nx + ny) * (x - y)
      den += nx + ny
  return num / den if den else np.nan


def fit_setting_effects(a, failed):
  """OLS site_wrong ~ bin dummies + setting dummies (ID baseline)."""
  a = a[a[:, 2] == failed]
  bins = np.unique(a[:, 1])
  X = [np.ones(len(a))] + [a[:, 1] == b for b in bins[1:]] + [a[:, 0] == s for s in (1, 2)]
  coef = np.linalg.lstsq(np.array(X, dtype=float).T, a[:, 3], rcond=None)[0]
  return coef[-2:]


def boot(a, fn, rng):
  vals = [fn(a[rng.integers(0, len(a), len(a))]) for _ in range(N_BOOT)]
  return np.nanpercentile(vals, [2.5, 97.5], axis=0).tolist()


def main():
  rng = np.random.default_rng(0)
  result = {'edges': [e if np.isfinite(e) else None for e in EDGES], 'settings': SETTINGS}
  for var in ('P', 'Q'):
    a = load_rows(var)
    m = cell_means(a)
    res = {'cells': [], 'gaps': {}}
    print(f'\n{var}: site-wrong rate by margin bin, failures / successes (n)')
    for b in range(len(EDGES) - 1):
      line = f'  [{EDGES[b]:.2f}, {EDGES[b + 1]:.2f})'
      for s, setting in enumerate(SETTINGS):
        (w, nw), (c, nc) = m[s, b, 1], m[s, b, 0]
        res['cells'].append({'setting': setting, 'bin': b, 'fail': w, 'n_fail': nw, 'succ': c, 'n_succ': nc})
        f = lambda v: ' -- ' if np.isnan(v) else f'{v:.2f}'  # noqa: E731
        line += f' | {setting[-6:]:>6s} {f(w)} ({nw:3d}) / {f(c)} ({nc:3d})'
      print(line)
    low = [0, 1, 2, 3]  # margin 0-1
    for s in (1, 2):
      for failed, name in ((1, 'fail'), (0, 'succ')):
        key = f'{SETTINGS[s]}_minus_id_{name}_margin0-1'
        est = matched_gap(a, s, failed, low)
        ci = boot(a, lambda x: matched_gap(x, s, failed, low), rng)
        res['gaps'][key] = {'est': est, 'ci95': ci}
        print(f'  {key:40s} {est:+.3f}  CI [{ci[0]:+.3f}, {ci[1]:+.3f}]')
    for failed, name in ((1, 'fail'), (0, 'succ')):
      est = fit_setting_effects(a, failed)
      ci = boot(a, lambda x: fit_setting_effects(x, failed), rng)
      res['gaps'][f'ols_setting_effect_{name}'] = {
        'currency': est[0], 'typo': est[1], 'ci95_currency': [ci[0][0], ci[1][0]],
        'ci95_typo': [ci[0][1], ci[1][1]]}
      print(f'  OLS (all margins) {name}: currency {est[0]:+.3f} [{ci[0][0]:+.3f}, {ci[1][0]:+.3f}]  '
            f'typo {est[1]:+.3f} [{ci[0][1]:+.3f}, {ci[1][1]:+.3f}]')
    result[var] = res
  with open(os.path.join(C.OUT_DIR, 'results', 'matched_difficulty.json'), 'w') as f:
    json.dump(result, f, indent=1, default=float)


if __name__ == '__main__':
  main()
