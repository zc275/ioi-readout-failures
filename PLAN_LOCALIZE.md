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
