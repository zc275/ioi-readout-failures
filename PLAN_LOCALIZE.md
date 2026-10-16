# Plan: localize the bias that outweighs a correct V (DRAFT for your review)

Written 2026-10-16, before running anything. The user approved executing the
plan on 2026-10-16 ("let's execute the plan"). Background: `PLAN_PREF.md` (the bias result), `ioi_repro/README.md`
("where does the failure enter?"), `RESEARCH_QUESTIONS.md`.

## Headline this plan serves

In causal-abstraction terms: **shift failures show the high-level model is
incomplete. V is computed correctly; the readout combines V with a bias B that
the high-level model leaves out, and the model fails when B outweighs V.** The
goal is to make B a localized, causally verified variable, not just a number
computed from outputs.

## Why this doesn't turn into an endless list of heuristics

There is one variable, B, and it is defined by an exact split, not by guessing
heuristics. For a prompt and its mirror (the other name repeated), write
L1 and L2 for logit(X) − logit(Y) in the two versions (X is the answer in the
first). Then exactly:

- signal s = (L1 − L2) / 2: the part that flips when the repeated name flips
  (what V should drive);
- bias b = (L1 + L2) / 2: the part that doesn't depend on which name repeats;
- the original prompt fails exactly when s + b ≤ 0.

This is an identity for a two-name choice, so any influence that doesn't
depend on which name repeats is inside b. **That is bookkeeping, not
explanation** (point raised by the user, 2026-10-16). It shows failures are
fully accounted for by s versus b, with no third category of failure. It does
not show that b is one mechanism: b could be the sum of many mechanisms
spread across the network.

Whether b is one thing is therefore an empirical question, and the plan tests it
directly instead of listing mechanisms:
1. **Is b simple as a function?** (Stage 0b.) If one score per name plus a
   constant for position explains most of b on held-out pairs, then whatever
   produces b acts through one number per name: low-dimensional, even if many
   components contribute.
2. **Is that number stored in one place?** (Stage B.) This is the
   causal-abstraction criterion: a high-level variable is localized if
   intervening on one low-dimensional subspace changes behavior as the
   high-level model predicts (interchange intervention accuracy). If a 1-d
   direction passes, it doesn't matter how many microscopic mechanisms feed it:
   at this level of description b is one variable, the way temperature is one
   variable over many molecules. If no direction passes, b is distributed, and
   we report that rather than look for an eleventh mechanism.

What the 0.99 correlation does and doesn't show: b measured in 4 unrelated
sentences correlates 0.99 with b in the original (fold 0, ID and Chinese). So b
depends only on **which two names, in which order**; place, object and template
add at most about 2%. That rules out context as an ingredient. It says nothing
about how many mechanisms compute b from the names.

## Stage 0: what is b made of? (half a day; behavioral, no training)

**0a. Full-name scoring.** Rerun `12_pref_replicate.py` scoring the log-prob of
each full name after the answer prefix, not the first token, on fold 0.

- L0-1. Fresh-context AUC for failures stays ≥ 0.85 in ID and Chinese. If it
  drops below 0.75, the bias was mostly a first-token artifact. We report that
  and stop Stage 0.

**0b. One decomposition, fitted once.** Choose 40 names per set (ID and
Chinese). Score all ordered pairs (1,560 per set) in 4 random contexts, in both
mirror versions. Fit by least squares on a random 80% of the ordered pairs, and test
on the held-out 20% (every name appears in many pairs, so each u is estimated
from training pairs only):

  b(first = X, second = Y) = u(X) − u(Y) + β + residual

where u is one "default-ness" score per name and β is a constant preference for
the first-mentioned slot. Rephrased uses the ID names in the rephrased template.

- L0-2. ID and Chinese: u and β together explain ≥ 60% of b's variance on
  held-out pairs. That means b is mostly a per-name score, which makes Stage B
  well-posed.
- L0-3. Rephrased: β alone explains ≥ 50% of b's variance (position dominates).
- L0-4. No prediction: whether u relates to token count, the familiarity score
  from `09_heuristics`, or gender. Reported as correlations only, with no
  further hunting.

**Stopping rule.** Whatever the share explained by u + β, Stage 0 ends there.
If it is under 30%, b is mostly pair-specific: we skip Stage B and do Stage A
only.

## Stage A: where does b join the readout? (about 1 day)

Train a 1-d DAS direction for V at the final token, layers 15, 17, 19 and 21
(`03_das_train.py --site last`, dim 1; the existing final-token runs are dim
256). Fold 0, verified ID examples, as before.

- **Correlational.** At each layer, regress the final-token DAS coordinate on V
  and on b (fresh-context, so it isn't tied to the prompt's own noise).
- **Causal.** For pairs of prompts with the same V and opposite-sign b (from
  Stage 0's measurements), swap the 1-d coordinate from one into the other at
  layer L. The high-level model predicts the logit difference moves by about the
  difference in b.

Predictions:
- L-A1. Rephrased: b already loads on the final-token coordinate at layer 15
  (regression coefficient on b significant, and the swap moves the logit
  difference by at least half the predicted amount).
- L-A2. ID: b loads on the coordinate only from layer 17 or later. That's where
  ID failures enter, per the README.
- L-A3. No prediction for Chinese.

## Stage B: where does b come from? (2–3 days; only if L0-2 holds)

If b ≈ u(X) − u(Y), u is a property of each name and may be stored on the
name's own tokens. Test: train a 1-d DAS direction at the two intro-name
positions (layers 2–12 scanned) with this objective: **swapping the coordinate
between the two names in the same prompt exchanges their u**. Then the model's
bias b should flip sign while s is unchanged. Training uses mirror-pair logit
differences on ID training prompts; testing uses held-out names.

- L-B1. Some layer gives a direction where the within-prompt swap flips the
  sign of b on ≥ 70% of held-out pairs with |b| ≥ 1, while s changes by less
  than 20%.
- L-B2. Applying that swap to Chinese and ID test **failures** fixes ≥ 50% of
  them (they become correct), and breaks successes only where b was helping.
- If L-B1 fails at every layer: u isn't stored linearly at the name tokens. We
  report that, and the result falls back to Stage A ("b joins the readout at
  layer L"), which is still a causal-abstraction refinement.

Position bias (rephrased) doesn't need Stage B. Its b is mostly β (L0-3), a
property of the slots rather than the names, and Stage A localizes where it
enters.

## Generalization check G: PriceTag (behavioral, about 15 min GPU)

Added 2026-10-16 after the user's question: will the ingredients carry over? We
can't know in advance; the claim that might generalize is the *shape* (a term
that ignores the task variable outweighs it), not names or position.
`pricetag_repro/10_bias.py`: L = log P(yes) − log P(no) after "Answer:". The
mirror moves the query price across the nearer boundary by the same distance
(truth flips), so b = (L + L_mirror)/2 is the Yes-lean that ignores the truth.
Fresh-context b: the same interval with 2 new inside/outside query pairs at
random distances. Failure score: fresh b oriented against the truth. Fold 0,
test prompts whose mirror and fresh versions are constructible (about 75–85%).

- G1. Fresh-interval b predicts failures with AUC ≥ 0.70 in at least one of
  currency and typo.
- G2. No prediction for ID, or for whether the lean is toward Yes or No.
- If G1 fails, the failures there are "weak signal" (small s), not "bias wins",
  which fits lead 1. That still says the bias story is IOI-specific, and we
  report it that way.

## Second model (background, GPU only)

Llama-3.1-8B-Instruct or Qwen2.5-7B-Instruct (whichever runs the released
templates cleanly).
1. Regenerate the correct/wrong splits from that model's own outputs. The
   released splits were built from Llama-3's answers.
2. Run the bias test (`11`, `12`) and DAS at the repeated name with truth-label
   CF sim (is V right on failures?).

- L-M1. On the second model, shift failures are mostly site-right
  (truth-label CF sim ≤ 0.65) and fresh-context b predicts failures with
  AUC ≥ 0.80 in at least two of three settings.

## Order and timeline

| Step | When | GPU |
|---|---|---|
| Stage 0 (0a, 0b) | 10/16–10/17 | about 2 h |
| Second model: splits + bias test | in background from 10/17 | about 4 h |
| Stage A | 10/18–10/19 | DAS training about 2–4 h |
| Second model: DAS + CF sim | in background | about 1 day |
| Stage B | 10/20–10/23 | DAS scan about 1 day |
| Start writing | about 10/25 | |

One big GPU job at a time (two at once stalled the GPU on 10/15).

## Guesses (choices without a paper to follow)

1. 40 names per set, chosen at random (fixed seed) from the released data's
   names, with all 40 first tokens distinct so every pair is scorable.
2. Held-out evaluation is by *pairs*: u can only be estimated for names the fit
   has seen, so we test whether u fitted on some pairs predicts b on unseen
   pairs of the same names.
3. Stage A uses fresh-context b, never the prompt's own b, so b isn't
   contaminated by that prompt's noise (the selection issue in `PLAN_PREF.md`).
4. Stage B's objective uses only the logit difference between the two names in
   the prompt, as in all IOI work here.

## Results

### Stage 0 (2026-10-16, fold 0)

**0a, full-name scoring (L0-1: holds).** Fresh-context AUC for failures with
full-name log-probs: ID 0.94, Chinese 0.96, rephrased 0.89 (first tokens:
0.95 / 0.96 / 0.89). Not a first-token artifact.

**0b, one decomposition (40 names, 1,560 ordered pairs per set, 4 contexts).**
Shares of E[b²] on held-out unordered pairs, first-token scoring (full-name
scoring gives the same to within 0.02):

| | ID | Chinese | rephrased |
|---|---|---|---|
| u + β | **0.83** | **0.75** | **0.88** |
| β only | 0.18 | 0.10 | **0.77** |
| name part explained by u | 0.91 | 0.85 | 0.71 |
| share of b that is name / position | 0.70 / 0.30 | 0.75 / 0.25 | 0.18 / 0.82 |
| β (lean toward the first-mentioned slot) | −0.98 | +0.64 | −2.14 |
| corr(u, familiarity) | −0.26 | −0.02 | 0.15 |
| corr(u, token count) | 0.28 | 0.14 | −0.07 |

- **L0-2 holds** (≥ 0.60): ID and Chinese b is mostly one "default-ness" score
  per name plus a constant slot lean. Stage B is well-posed.
- **L0-3 holds** (≥ 0.50): rephrased b is mostly the constant slot lean toward
  the second-mentioned name.
- L0-4: u is not familiarity and barely relates to token count, which is why
  `09_heuristics` found nothing. What u is stays open (stopping rule: not
  hunted further). The position part has a pair-specific share (0.36 ID,
  0.65 Chinese), so β is a simplification there.
- Unexplained: 17% (ID), 25% (Chinese), 12% (rephrased) of b.

### Generalization check G, PriceTag (fold 0, `pricetag_repro/10_bias.py`)

| | ID | currency | typo |
|---|---|---|---|
| AUC: fresh-interval b → failure | 0.56 | **0.78** | **0.86** |
| mean Yes-lean b | −0.21 | **+2.59** | **+2.41** |
| signal s, failures / successes | 1.28 / 4.86 | 1.95 / 3.49 | 1.54 / 2.05 |
| corr fresh vs original b | 0.30 | 0.41 | 0.45 |

- **G1 holds** (≥ 0.70 in at least one shift: both). Both shifts add a strong
  Yes-lean that ignores the checks; failures are where it runs against the
  truth. ID failures are weak-signal close calls instead (no lean). That
  matches lead 5 (difficulty and shift are separate piles) and lead 1 (gated
  amplification of a False site beats the Yes-lean).
- **Caveat being checked:** b here is mostly a per-setting constant (fresh vs
  original only 0.30–0.45), so the AUC may come from "the truth is No" alone.
  Rerun queued with a truth-only baseline and within-truth AUCs.
- Same shape as IOI, different ingredients: names and slot in IOI, a Yes/No
  lean in PriceTag.

### Stage A (2026-10-16, fold 0, `ioi_repro/14_stageA.py`)

1-d DAS for V at the final token (held-out flip IIA on verified prompts: L15
0.58, L17 0.25, so this direction captures V only partly). Fresh-context b
turned into a lean toward the first-mentioned name.

| | b coefficient (std, 95% CI), L15 | swap slope, L15 | swap right direction, L15 | L17–21 |
|---|---|---|---|---|
| ID | 0.34 [0.31, 0.37] | 0.21 | 0.86 | coef 0.41–0.47, slope 0.09–0.13 |
| Chinese | 0.45 [0.41, 0.49] | 0.24 | 0.88 | coef 0.41–0.49, slope 0.14–0.21 |
| rephrased | 0.12 [0.06, 0.17] | 0.03 | 0.53 | coef 0.16–0.20, slope 0.04–0.07 |

- **L-A1 fails:** the rephrased position bias does not load on the final-token
  V coordinate at any layer, and swapping it does nothing. The position bias
  reaches the output by another path. That fits amplification not helping
  rephrased failures; lead 2 (name movers) is the candidate path.
- **L-A2 fails in the other direction:** ID b already loads at L15.
- ID and Chinese: b has partly merged into the V readout by L15. The swap moves
  the output the predicted way 86–91% of the time but by only about a fifth of
  the predicted amount. So most of b's effect bypasses this 1-d coordinate,
  or the coordinate is a weak handle (flip IIA 0.58). We can't tell these apart
  here.

### Second model, first look (Qwen2.5-7B-Instruct, revision a09a354, fold 0)

- **The released prompts can't test Qwen.** Every released split is half
  Llama-3 failures by construction (fold 0: 2,048 of 4,096 per setting). Qwen's
  greedy outputs on them: ID 99.0% correct, Chinese 98.3%, 0% repeated-name
  answers. Rephrased: 100% "other", because Qwen starts explaining ("Based on
  the partial story provided, …") instead of naming someone within 8 tokens.
  So Llama's failures are Llama-specific, which fits b being a model-specific
  per-name score.
- **Forced choice on the released test prompts (`11_name_pref.py`):** ID 0.998,
  Chinese 1.000 (signal +13 to +20 logits). Rephrased 0.24 (bias-corrected
  0.06), with failures' signal **−7.7**: Qwen computes the *opposite* answer
  (the repeated name, the giver), and the bias explains almost none of it (3%).
  That's a computing failure, not a using one. The s/b split tells the two
  apart across models.
- So L-M1 can't be tested on these prompts. `17_pool.py` (unselected random
  prompts, both models) is queued in job 4. It gives each model's natural
  failure rate and the bias test on its own failures.

### Stage B (2026-10-16, fold 0, `ioi_repro/16_stageB.py`, trained on ID train names)

Swap a learned 1-d coordinate between the two intro-name tokens at layer L
(300 steps, batch 16, lr 1e-2). Test: 600 held-out test prompts per set.
Random-direction control fixes 0.00–0.02 of failures everywhere.

| layer | 2 | 4 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|---|
| ID: b − β flips sign (|b − β| ≥ 1) | 0.04 | 0.13 | 0.37 | 0.45 | 0.46 | **0.51** |
| ID: failures fixed | 0.37 | 0.59 | 0.73 | 0.80 | 0.80 | **0.82** |
| ID: successes broken | 0.02 | 0.01 | 0.03 | 0.03 | 0.04 | 0.04 |
| ID: median relative change of s | 0.04 | 0.05 | 0.06 | 0.07 | 0.07 | 0.07 |
| Chinese: failures fixed (transfer) | 0.04 | 0.08 | 0.15 | 0.17 | 0.19 | 0.21 |

- **L-B1 fails** (best flip 0.51 at L12; predicted ≥ 0.70), though the swap
  leaves s nearly unchanged (7%), as the high-level model requires.
- **L-B2 holds for ID, fails for Chinese transfer.** Swapping one direction on
  the two name tokens fixes 82% of ID test failures and breaks 4% of
  successes. A direction trained on English names barely moves Chinese
  failures (21%). The follow-up trains on Chinese names directly.
- Reading: a 1-d per-name "default-ness" coordinate on the name tokens carries
  enough of b to fix most ID failures causally. It carries only about half the
  sign of b, so either b is partly stored elsewhere (position part, the second
  mention, other dimensions) or training hasn't converged (loss was noisy).
  Job 6 tests convergence (1,000 steps), layers 14 and 16, Chinese-trained
  directions, and folds 1–2.

### Unselected prompts, Llama-3 (`ioi_repro/17_pool.py`, 20,000 random prompts per setting)

Random name pair, random repeated name, template/place/object from a random
released prompt; forced choice (first tokens). Bias test on 500 failures + 500
successes.

| | ID | Chinese | rephrased |
|---|---|---|---|
| natural failure rate | **3.5%** | **9.0%** | **44.9%** |
| failure rate, answer first / second mentioned | 5.2% / 1.9% | 4.5% / 13.7% | 76.1% / 13.5% |
| AUC: fresh-context b → failure | **0.96** | **0.97** | **0.87** |
| AUC with intro order swapped | 0.82 | 0.80 | 0.25 |
| corr b same vs swapped order | 0.61 | 0.66 | −0.57 |

- The shift effect is real at natural rates: 3.5% → 9.0% (Chinese) → 44.9%
  (rephrased). The released splits (50% failures) hid the base rates.
- **The bias result replicates on unselected prompts** (AUC 0.96 / 0.97 / 0.87),
  so it isn't produced by how the released splits were chosen.
- The slot pattern matches β: rephrased and ID fail more when the answer is the
  first-mentioned name, Chinese when it is the second (β = +0.64).

### PriceTag G, rerun with baselines (resolves the caveat above)

| | ID | currency | typo |
|---|---|---|---|
| AUC fresh-interval b | 0.56 | 0.78 | 0.86 |
| AUC from "truth is No" alone | 0.68 | 0.67 | 0.83 |
| failure rate, truth Yes / No | 0.02 / 0.51 | 0.00 / 0.33 | 0.01 / 0.74 |
| AUC within truth-No prompts | 0.62 | 0.67 | 0.60 |

**Qualified.** PriceTag failures are almost all on truth-No prompts (saying
Yes). Most of G1's AUC comes from a constant, setting-wide Yes-lean (+2.4 to
+2.6 under the shifts), not from an interval-specific bias. Within truth-No
prompts, the interval's own lean adds modestly (0.60–0.67). So the *shape*
generalizes (a lean that ignores the checks outweighs them), but in PriceTag it
is mostly one constant per setting, unlike IOI's per-name scores.

### Stage B follow-ups (job 6)

| run | ID flip | ID fixed / broken | Chinese fixed / broken | s change |
|---|---|---|---|---|
| L12, fold 0 (from the scan) | 0.51 | 0.82 / 0.04 | 0.21 / 0.04 | 0.07 |
| L12, 1,000 steps, lr 3e-3 | 0.46 | 0.80 / 0.04 | 0.20 / 0.03 | 0.07 |
| L14 | 0.40 | 0.74 / 0.05 | 0.11 / 0.04 | 0.05 |
| L16 | 0.41 | 0.75 / 0.08 | 0.13 / 0.03 | 0.06 |
| L12, fold 1 | 0.57 | 0.82 / 0.09 | 0.20 / 0.02 | 0.09 |
| L12, fold 2 | 0.48 | 0.71 / 0.07 | 0.29 / 0.01 | 0.07 |
| L12, trained on Chinese names | 0.12 | 0.46 / 0.05 | **0.47 / 0.11** (flip 0.18) | 0.07 |

Random directions fix 0.00–0.02 everywhere.

- **The ID result replicates across folds:** one direction at L12 fixes
  71–82% of ID test failures, breaking 4–9% of successes, with s nearly
  unchanged.
- **Not a convergence problem:** 1,000 steps gives the same as 300. L12 is the
  peak; it's the same layer where V is localized at the repeated name.
- **Chinese names are stored less cleanly:** a Chinese-trained direction fixes
  47% of Chinese failures but breaks 11%, and flips b only 18% of the time.
  Chinese names are byte tokens, and only each name's last token is swapped.
- L-B1 stays failed (flip 0.46–0.57 vs ≥ 0.70). Job 7 (exploratory, post hoc)
  tries 2-, 4- and 8-dim subspaces to see whether one direction is the cap.
