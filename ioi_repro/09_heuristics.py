"""Step 9 (extension): level 0 of the competing-heuristic hypothesis.

Hypothesis: the readout combines the computed position variable V with a cheap
heuristic H, and OOD failures are cases where H wins. Level 0 asks only the
correlational question: do input features that would make a heuristic point
at the repeated name S predict failure, *within* each V stratum?

Why "which inputs fail" and not "which wrong name": the prompts contain only
two names, so a wrong output is either S itself or a spelling "correction" of
the right name (README, guess 5). The wrong name carries no information; the
question is which inputs fail.

Candidate heuristics and features (computed from the prompt alone, plus the
model's unconditional log-probability of each name; IO = right answer,
S = repeated name):
  H1 slot bias     prefer the later name in "{a} and {b}". Points at S exactly
                   when V = first. Feature: 1[V = first]. Overall AUC only,
                   since it *is* the stratum variable.
  H2 familiarity   prefer the name the model finds more familiar.
                   Feature: fam(S) - fam(IO), with fam(x) = log p(" x." | c)
                   averaged over three neutral contexts c (FAM_CONTEXTS).
                   Higher -> predicted wrong.
  H3 IO rarity     an unfamiliar right answer is "normalised" or dropped.
                   Feature: -fam(IO). Higher -> predicted wrong. Aimed at the
                   spelling-correction failures.
  H4 similarity    similar names interfere. Feature: difflib ratio(IO, S).
                   Higher -> predicted wrong.
  H5 token length  long (multi-token) right answers are harder.
                   Feature: ntok(IO) - ntok(S). Higher -> predicted wrong.

Predictions, written down before computing any of these features:
  P1 (H1). AUC of 1[V = first] for "wrong" > 0.5 in all three settings.
      Already observed on the rephrased template (about 80% of wrong examples
      have V = first, README obs. 3), so only ID and Chinese test it.
  P2 (H2). Within-stratum AUC >= 0.60 in at least two settings. This is the
      main candidate for a heuristic that varies across inputs.
  P3 (H3). Within-stratum AUC > 0.5 overall, and higher on the
      spelling-correction failures than on S-failures.
  P4 (H4, H5). Weak expectation: 0.5-0.6. Included so that a null is
      reported, not to look for a winner.
Pass rule for level 0: mean within-stratum AUC >= 0.60, in the predicted
direction, in >= 2 of 3 settings, with every fold above 0.5. Directions are
fixed above and never flipped using test labels. Every candidate is reported.

Outcomes:
  primary    released label (wrong vs correct), test split, balanced.
  secondary  S-failures only: released-wrong prompts whose output (our run)
             is S, against released-correct prompts. This drops the spelling
             corrections and our ~2% label disagreements.
Joint check: a logistic regression on all five features (standardised), fit
on each fold's train split and scored on its test split (within-stratum AUC).
It asks whether the features add up beyond V. It uses OOD labels for fitting,
so it is an analysis, not a predictor that competes with the paper's methods.

Familiarity is defined by us, so it is checked before use: the script reports
the Spearman correlation of fam between the three contexts. If it is low
(< 0.7), the measure depends on the arbitrary choice of context and H2/H3
results should not be interpreted.

Caveat: the same names recur across prompts, so prompts are not independent
and the fold spread understates the uncertainty. If anything passes, the next
step is a bootstrap over names.

Usage:
  python 09_heuristics.py --compute_familiarity   # once, ~1-2 min GPU
  CUDA_VISIBLE_DEVICES= python 09_heuristics.py   # analysis, CPU
Writes outputs/results/name_familiarity.json and outputs/results/heuristics.json
"""

import argparse
import difflib
import importlib
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from common import OUT_DIR
from ioi_data import FOLDS, SETTINGS, answer_prefix, parse_ioi_prompt

behavior = importlib.import_module('01_behavior')

FAM_PATH = os.path.join(OUT_DIR, 'results', 'name_familiarity.json')
FAM_CONTEXTS = ("The following is a person's first name:", 'Name:', 'My friend is called')
FEATURES = ('slot', 'fam_gap', 'io_rarity', 'similarity', 'tok_gap')


def load_behavior(setting, fold):
  with open(os.path.join(OUT_DIR, 'behavior', f'{setting}_fold{fold}.json')) as f:
    return json.load(f)['results']


def all_names():
  names = set()
  for setting in SETTINGS:
    for fold in FOLDS:
      for prompt in load_behavior(setting, fold):
        v = parse_ioi_prompt(prompt)
        names.update((v['name_a'], v['name_b']))
  return sorted(names)


def compute_familiarity(batch_size=64):
  """fam_c(x) = sum of log p(token) over " x." after context c (with BOS), per context.

  The trailing period scores the name as complete, so "Ash" is not credited
  with the probability of "Ashley".
  """
  from common import load_model_and_tokenizer
  model, tok = load_model_and_tokenizer()
  names = all_names()
  per_context = {c: familiarity_in_context(model, tok, names, c, batch_size) for c in FAM_CONTEXTS}
  os.makedirs(os.path.dirname(FAM_PATH), exist_ok=True)
  with open(FAM_PATH, 'w') as f:
    json.dump(per_context, f, indent=1)
  print(f'Wrote familiarity for {len(names)} names x {len(FAM_CONTEXTS)} contexts to {FAM_PATH}')


def familiarity_in_context(model, tok, names, context, batch_size):
  import torch
  ctx = tok(context)['input_ids']  # includes BOS
  fam = {}
  for i in range(0, len(names), batch_size):
    batch = names[i:i + batch_size]
    conts = [tok(f' {n}.', add_special_tokens=False)['input_ids'] for n in batch]
    seqs = [ctx + c for c in conts]
    width = max(map(len, seqs))
    # Right-pad by hand; we only read positions inside each real sequence.
    ids = torch.full((len(seqs), width), tok.pad_token_id)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for j, s in enumerate(seqs):
      ids[j, :len(s)] = torch.tensor(s)
      mask[j, :len(s)] = 1
    with torch.no_grad():
      logp = model(input_ids=ids.cuda(), attention_mask=mask.cuda()).logits.float().log_softmax(-1).cpu()
    for j, (name, c) in enumerate(zip(batch, conts)):
      # Token at position t is predicted by the logits at t - 1.
      pos = torch.arange(len(ctx) - 1, len(ctx) - 1 + len(c))
      fam[name] = float(logp[j, pos, torch.tensor(c)].sum())
  return fam


def load_familiarity():
  """Mean familiarity over contexts, after printing how well the contexts agree."""
  from scipy.stats import spearmanr
  with open(FAM_PATH) as f:
    per_context = json.load(f)
  names = sorted(next(iter(per_context.values())))
  M = np.array([[per_context[c][n] for n in names] for c in FAM_CONTEXTS])
  print('Spearman correlation of familiarity between contexts (rows/cols in FAM_CONTEXTS order):')
  for i in range(len(FAM_CONTEXTS)):
    print('  ' + '  '.join(f'{spearmanr(M[i], M[j])[0]:.2f}' for j in range(len(FAM_CONTEXTS))))
  return dict(zip(names, M.mean(0)))


def featurize(prompts, fam, tokenizer):
  """(n, len(FEATURES)) array, each column oriented so that higher = predicted wrong."""
  rows = []
  for p in prompts:
    v = parse_ioi_prompt(p)
    io, s = v['answer'], v['name_c']
    pre = answer_prefix(p)
    ntok = lambda n: len(tokenizer(pre + n, add_special_tokens=False)['input_ids'])
    rows.append([
      float(v['position'] == 'first'),
      fam[s] - fam[io],
      -fam[io],
      difflib.SequenceMatcher(None, io, s).ratio(),
      ntok(io) - ntok(s),
    ])
  return np.array(rows)


def strat_auc(label, score, first):
  """Mean of the AUCs within V = first and V = second (NaN if a stratum is one-class)."""
  out = []
  for v in (True, False):
    sel = first == v
    if len(set(label[sel])) < 2:
      return float('nan')
    out.append(roc_auc_score(label[sel], score[sel]))
  return float(np.mean(out))


def split_data(beh, split, fam, tokenizer):
  prompts = [p for p, r in beh.items() if r['split'] == split]
  wrong = np.array([beh[p]['released_label'] == 'wrong' for p in prompts])
  verdict = np.array([behavior.judge(beh[p]['output_text'], parse_ioi_prompt(p))[0] for p in prompts])
  return prompts, featurize(prompts, fam, tokenizer), wrong, verdict


def analyse(fam, tokenizer):
  rows = []
  for setting in SETTINGS:
    for fold in FOLDS:
      beh = load_behavior(setting, fold)
      _, X, wrong, verdict = split_data(beh, 'test', fam, tokenizer)
      first = X[:, 0] == 1
      row = {'setting': setting, 'fold': fold,
             'failure_types': {k: int(((verdict == k) & wrong).sum()) for k in ('wrong', 'other', 'correct')}}
      # Secondary outcome: S-failures vs released-correct.
      s_sel = ~wrong | (verdict == 'wrong')
      # Spelling-correction failures vs released-correct (for P3).
      o_sel = ~wrong | (verdict == 'other')
      for j, name in enumerate(FEATURES):
        row[name] = {
          'overall': float(roc_auc_score(wrong, X[:, j])),
          'strat': strat_auc(wrong, X[:, j], first),
          'strat_s_failures': strat_auc(wrong[s_sel], X[s_sel, j], first[s_sel]),
          'strat_other_failures': strat_auc(wrong[o_sel], X[o_sel, j], first[o_sel]),
        }
      # Joint check: fit on train, score test.
      _, Xtr, ytr, _ = split_data(beh, 'train', fam, tokenizer)
      scaler = StandardScaler().fit(Xtr)
      clf = LogisticRegression(max_iter=1000).fit(scaler.transform(Xtr), ytr)
      score = clf.decision_function(scaler.transform(X))
      row['joint'] = {'strat': strat_auc(wrong, score, first),
                      'coef': dict(zip(FEATURES, map(float, clf.coef_[0])))}
      rows.append(row)
  return rows


def report(rows):
  def cell(setting, get):
    vals = np.array([get(r) for r in rows if r['setting'] == setting])
    return f'{np.nanmean(vals):.2f} ± {np.nanstd(vals):.2f}', vals
  print('\nFailure types among released-wrong test prompts (summed over folds):')
  for setting in SETTINGS:
    tot = {k: sum(r['failure_types'][k] for r in rows if r['setting'] == setting)
           for k in ('wrong', 'other', 'correct')}
    print(f'  {setting:13s} outputs S: {tot["wrong"]}  spelling/other: {tot["other"]}  '
          f'correct in our run: {tot["correct"]}')
  for key, title in (('strat', 'within-stratum AUC, primary outcome'),
                     ('strat_s_failures', 'within-stratum AUC, S-failures only'),
                     ('strat_other_failures', 'within-stratum AUC, spelling failures only'),
                     ('overall', 'overall AUC (not stratified)')):
    print(f'\n{title} (mean ± std over folds; higher = feature predicts wrong as predicted):')
    print(f'  {"feature":12s}' + ''.join(f'{s:>18s}' for s in SETTINGS))
    for name in FEATURES:
      if name == 'slot' and key != 'overall':
        continue
      print(f'  {name:12s}' + ''.join(f'{cell(s, lambda r: r[name][key])[0]:>18s}' for s in SETTINGS))
  print('\nJoint logistic regression (fit on train, within-stratum AUC on test):')
  print('  ' + ''.join(f'{s:>18s}' for s in SETTINGS))
  print('  ' + ''.join(f'{cell(s, lambda r: r["joint"]["strat"])[0]:>18s}' for s in SETTINGS))
  print('\nPass rule: within-stratum AUC >= 0.60 in >= 2 settings, every fold > 0.5:')
  for name in FEATURES[1:]:
    passed = []
    for s in SETTINGS:
      _, vals = cell(s, lambda r: r[name]['strat'])
      passed.append(np.nanmean(vals) >= 0.60 and np.all(vals > 0.5))
    print(f'  {name:12s} {"PASS" if sum(passed) >= 2 else "fail"}  ({", ".join(s for s, p in zip(SETTINGS, passed) if p) or "none"})')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--compute_familiarity', action='store_true')
  args = ap.parse_args()
  if args.compute_familiarity:
    compute_familiarity()
    return
  from transformers import AutoTokenizer
  from common import MODEL_ID, MODEL_REVISION
  tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
  fam = load_familiarity()
  rows = analyse(fam, tokenizer)
  with open(os.path.join(OUT_DIR, 'results', 'heuristics.json'), 'w') as f:
    json.dump(rows, f, indent=1)
  report(rows)


if __name__ == '__main__':
  main()
