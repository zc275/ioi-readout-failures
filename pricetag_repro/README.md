# PriceTag: where do OOD failures enter?

Second task for the IOI follow-up (`../ioi_repro`). The plan and predictions, committed before any DAS or counterfactual-simulation run, are in `PLAN.md`. The questions this feeds into are in `../RESEARCH_QUESTIONS.md`.

## Pipeline

| Step | Script | Output |
|---|---|---|
| 1. Free-form behavior, agreement with released labels | `01_behavior.py` | `outputs/behavior/{setting}.json` |
| 2. DAS for P (layers 8–10) or Q (layers 11–12), dim 1, query integer token | `02_das_train.py` | `outputs/das/` |
| 3. CF sim, output-label and truth-label rules in one pass | `03_cf_sim.py` | `outputs/cf_sim/` |
| 4. AUC tables, stratified by failure type | `04_summarize.py` | `outputs/results/cf_sim_summary.json` |
| 5. Input-fidelity probes (lb, ub, q) | `05_probe_inputs.py` | `outputs/results/probe_inputs_fold{f}.json` |

`run_overnight.sh` runs steps 2–5 for all folds, plus the IOI dimension sweep and a P layer sweep.

## Behavior check (step 1)

| Setting | Agree, released-correct | Agree, released-wrong | No yes/no in answer (correct / wrong) |
|---|---|---|---|
| Typo | 0.99–1.00 | 0.96–0.97 | 0% |
| Currency | 0.81–0.85 | 0.91 | 11–16% / 1% |
| ID | 0.74–0.76 | 0.40–0.45 | 24% / 20% |

The ID labels don't reproduce (question for the authors, see `PLAN.md`). So bases come from prompts that are released-correct *and* correct for our model, and we don't report ID AUCs.

## Guesses (choices the paper doesn't pin down)

1. **"Answer:" on bases.** Bases get the suffix, so the first token is the verdict (agreed in `PLAN.md`). 739 of 759 fold-0 candidates still answer correctly with it, and only those are kept.
2. **"Layers 8–10" means a joint intervention.** We read it as one dim-1 subspace in each of layers 8, 9 and 10, swapped together, as for IOI "9–12".
3. **Site.** The last token of the query's integer part. Llama-3 tokenizes integers up to 999 as one token, so this is the whole integer. The decimal part comes *after* the site, so a misread decimal (e.g. "₺2,6") can't affect the representation there.
4. **Informative bases.** For P we use only bases with Q = True; for Q, only bases with P = True. Otherwise the answer is No whatever is patched (Q = False forces No, and P = False forces No via the paper's shortcut), and the pair carries no signal. The counterfactual label is then just the source's value of the variable.
5. **Scoring.** We use the probability of each answer family (all case variants of yes / no, the paper's answer tokens) at the first answer position. The verdict is the family with more mass, and the loss is −log of the target family's mass.
6. **Output-label rule.** A source's variable value comes from its released label. If correct, it's the true value. If wrong and the model said Yes, it's True (Yes needs both checks to pass). If wrong and it said No, it's False (we assume the other check was right; under 3% of failures).
7. **CF-sim pool.** The first 16 informative verified bases, with K = 10 per test prompt, as in the authors' demo.
8. **DAS held-out sources.** ID validation prompts, split by *our* verdict, since the released ID labels don't reproduce.

## Results: CF sim at the paper's sites (3 folds, test split)

Within-setting AUC of the output-mapping score against the released labels, mean over folds (`04_summarize.py`, `outputs/results/cf_sim_summary.json`). "Left" failures are q < lb with the model saying Yes; "right" failures are q > ub with the model saying Yes. Each stratum is compared with all correct prompts.

| Setting | Score | Output labels, all | Truth labels, all | Output labels, left | Truth labels, left | Output labels, right | Truth labels, right |
|---|---|---|---|---|---|---|---|
| Currency | P&Q | 0.93 | 0.67 | 0.95 | 0.64 | 0.77 | 0.82 |
| Typo | P&Q | 0.96 | 0.59 | 0.96 | 0.52 | 0.95 | 0.67 |

DAS held-out flip IIA (our-correct ID sources): P 0.95–0.99, Q 0.77–0.82. Q localizes less cleanly at dimension 1.

**Against the plan's predictions:**
- **C1 fails as written.** Currency truth-label AUC is 0.65 for P (needed ≥ 0.70). Output-label AUC is higher on currency than typo, not lower.
- **T1 holds.** Typo truth-label 0.59 (≤ 0.60) and output-label 0.96 (≥ 0.80).
- **C2 is inconclusive.** At the last token of the query, the ID-trained probe decodes q equally well on currency failures and successes (error AUC 0.39–0.51 across layers). At the final prompt token, failures decode worse (0.65–0.77), but failures have systematically different q values, so this is confounded. The decoded relation (q − lb) doesn't separate left failures from matched correct prompts (AUC 0.45–0.62, true-margin baseline 0.57). So there's no sign that the model misreads "₺2,6" as larger than the lower bound.

## Exploratory finding: where the failure enters depends on how close the call is

Added after seeing the results above (`06_margin.py`, `outputs/results/margin_analysis.json`). The site-wrong rate is the share of patched bases that answer Yes, i.e. the site carries the variable as True when it is False. Pooled over 3 folds:

| Check / setting | Margin < 0.5: correct | Margin < 0.5: wrong | 0.5–1: correct | 0.5–1: wrong | ≥ 1: correct | ≥ 1: wrong | Failures with margin ≥ 1 |
|---|---|---|---|---|---|---|---|
| P currency | 0.65 | 0.79 | 0.18 | 0.47 | ≤ 0.06 | ≤ 0.05 | 1099 / 1298 |
| P typo | 0.45 | 0.61 | 0.16 | 0.24 | ≤ 0.01 | 0.00 | 585 / 801 |
| Q currency | 0.58 | 0.83 | 0.09 | 0.77 | ≤ 0.02 | ≤ 0.07 | 71 / 231 |
| Q typo | 0.50 | 0.68 | 0.02 | 0.36 | ≤ 0.01 | ≤ 0.08 | 524 / 731 |
| P ID (released labels) | 0.62 | 0.83 | 0.21 | 0.73 | 0.00 | 0.00 | 84 / 423 |

Reading:
1. **Clear-cut failures (margin ≥ 1) are readout failures.** The site carries the right value whether the model ends up right or wrong. Most shift-induced failures are of this kind: 85% of currency P failures and 73% of typo P failures. Output-label CF sim catches them.
2. **Close-call failures carry the wrong value at the site.** At 0.5–1 this separates failures from successes (e.g. Q currency 0.77 vs 0.09). Below 0.5 the site is fuzzy for correct prompts too (≈ 0.6), so part of that signal is imprecision of the 1-d subspace or of the representation itself. These are the failures output-label CF sim is blind to, and truth-label CF sim catches.
3. **ID failures are mostly close calls** (median distance below lb 0.13 vs 3.36 for correct prompts; margin alone predicts failure with AUC 0.85). **Currency and typo add clear-cut failures**, where the margin is useless (AUC 0.55–0.56 for P).

So the split isn't "surface shift → misreading". It's closer to: **difficulty → the variable is computed wrong; distribution shift → the variable is computed right and then not used.**

**Control: is close-call fuzziness just a too-narrow subspace?** No. P at the paper's layers with a 4-dimensional subspace (fold 0, `run_followup.sh`) gives the same picture as dimension 1. The site-wrong rate on *correct* prompts with margin < 0.5 is 0.61 vs 0.49 (currency) and 0.49 vs 0.47 (ID), and the AUCs are within 0.01. So the fuzziness near the boundary looks like a property of the model's own representation (a graded comparison), not of the DAS subspace size. The fold-0 close-call cells are small (3–16 correct prompts), so treat this as indicative.

**Caveat on typo:** the close-call effect is strong for ID and for Q under currency, but weak for typo (pooled P 0.5–1: 0.16 vs 0.24; it reverses in fold 0 on 5 vs 30 prompts).

**P layer sweep (fold 0, single layers, dim 1).** Val flip IIA: layer 4 0.65, layer 8 0.96, layer 12 0.95, layer 16 0.27. So P lives at the query token in layers 8–12 and has moved elsewhere by 16. Currency truth-label AUC drops from 0.70 (layer 8) to 0.61 (layer 12), and typo stays at about 0.50.
