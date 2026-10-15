# Plan: amplification fix (lead 1) and matched difficulty (lead 5)

Written 2026-10-15, before running either. Exploratory: a missed prediction
redirects the work, it doesn't end it. Background in `RESEARCH_QUESTIONS.md`.

## Lead 5: matched difficulty (CPU, existing results)

**Question.** At the same true margin, does the shift change whether failures
have the variable wrong at the site? The reformulation ("difficulty → computed
wrong, shift → computed right then not used") says the shift adds failures
whose site is *right*, so at matched margin shifted failures should be site-wrong
less often than ID failures, while successes look the same across settings.

**Already seen (so not a prediction).** `06_margin.py` with the *released*
labels: at margin 0.5–1, P failures are site-wrong 0.47 (currency), 0.24 (typo),
0.73 (ID); Q failures 0.77, 0.36, 0.68. Q under currency looks like ID.

**What's new.** (a) Use our model's own verdict (`our_label`), because the
released ID labels don't reproduce (0.74 / 0.40 agreement), so the ID "failures"
above are contaminated. (b) Finer bins and bootstrap CIs. (c) A regression
among X-False prompts: site-wrong ~ margin bin × outcome + setting.

**Predictions.**
- M1. With our labels, in the 0–1 margin range, P failures under currency and
  typo are site-wrong less often than ID failures (gap ≥ 0.15, CI excludes 0).
- M2. Successes at matched margin are site-wrong at similar rates across
  settings (all pairwise gaps ≤ 0.10).
- M3. Q under currency is the exception seen above. No prediction, just report.

## Lead 1: amplify the model's own value (GPU, no training)

**Question.** On "using" failures, the variable is correct at the site, and the
answer still comes out wrong downstream. If the downstream reading is just too
weak, strengthening the model's *own* value at the site should fix the failure
with no ground truth. On failures where the site value is wrong, the same
operation should strengthen the error and not fix it.

**Operation.** At the DAS site, for each layer in the DAS run, take the 1-d
coordinate c = h·w and replace it with m + α(c − m), where m is the midpoint
of the two class means of c on ID verified training prompts. Nothing about the
test prompt's label is used. α ∈ {1, 2, 4, 8}; α = 1 is the baseline.

**Runs.**
- PriceTag, fold 0, paper DAS (P at layers 8–10, Q at 11–12, query integer
  token). Conditions: P only, Q only, P + Q (the deployable version, since we
  don't know which check failed). Settings: currency, typo, ID test (1,024 each).
- IOI, fold 0, layer 12, dim 1, repeated-name token. Settings: ID, Chinese,
  rephrased test (1,024 each).
- Control: for each, 3 random unit directions orthogonal to w, with a per-prompt
  displacement of the same norm as the DAS one, signed by the prompt's own
  random coordinate (c_r − m_r).
- Then, if fold 0 is interesting: folds 1–2.

**Choosing α without test labels.** The largest α whose break rate on ID
*validation* correct prompts is ≤ 5%. Results for every α are reported too.

**Scoring.** PriceTag: first-token yes/no family after "Answer:". IOI: first
token of the correct name vs the other name. A failure means wrong at α = 1
in this same format (our own baseline, not the released label).

**Predictions** (at the α chosen on ID val):
- A1. PriceTag currency and typo, P + Q: fix ≥ 30% of clear-cut (margin ≥ 1)
  site-right left failures, break ≤ 5% of baseline-correct, net accuracy ≥ +5
  points.
- A2. PriceTag, P only, left failures (P False, said Yes): fix rate on
  site-wrong failures ≤ 10% and at least 20 points below site-right failures.
  Site-wrong / site-right comes from the existing fold-0 CF-sim scores
  (patched-yes rate ≥ 0.5 for a P-False prompt).
- A3. Random-direction control: fix rate ≤ 10% and ≤ 1/3 of the DAS fix rate
  in every setting.
- A4. IOI Chinese and rephrased: fix ≥ 30% of failures with ≤ 5% breaks. No
  prediction for ID versus the shifts. ID failures enter after layer 15 and
  rephrased ones between 12 and 15, and I don't know which helps.

**How to read the outcomes.**
- A1 and A2 hold: the diagnosis tells you the fix (Q3), and the two failure
  types respond differently. That's the strongest SOP result we could get from
  this week.
- A1 fails but A3 holds: the readout failure isn't a weak signal. Something
  downstream routes around a correct value, which points to lead 2 (the IOI
  name-mover heads).
- A3 fails: amplifying anything at that site helps, so the result isn't about
  the variable.

**Guesses** (choices made without a paper to follow):
1. Center m from ID training class means. This uses ID labels only, as DAS
   training does.
2. Amplify at every layer of the multi-layer runs, computing c on the current
   (already amplified upstream) state, so effective strength compounds over
   layers 8–10. Fold-0 single-layer P runs (L8, L12) as a check.
3. Only the site token is changed.
4. IOI prompts whose two names share a first token are dropped.
5. The site-wrong threshold of 0.5 on the patched-yes rate.

## Results

### Lead 5 (2026-10-15, `pricetag_repro/08_matched_difficulty.py`)

Our own labels, 3 folds pooled. Margin-matched gap = shift minus ID in
site-wrong rate over margins 0–1, with 95% bootstrap CIs.

| | P failures | P successes | Q failures | Q successes |
|---|---|---|---|---|
| currency − ID | −0.02 [−0.12, +0.08] | +0.01 [−0.14, +0.21] | +0.02 [−0.04, +0.07] | −0.15 [−0.25, −0.06] |
| typo − ID | −0.20 [−0.30, −0.10] | −0.14 [−0.33, +0.07] | −0.25 [−0.31, −0.19] | −0.22 [−0.54, −0.07] |

- **M1: failed for currency, held for typo.** At matched margin, currency
  close-call failures are wrong at the site exactly as often as ID ones. Typo
  close-call failures are wrong at the site less often.
- **M2: mixed.** Holds for P under currency. The typo success cells are tiny
  (about 20 prompts below margin 1), and Q successes under currency are *less*
  often site-wrong than ID.
- **What the shift does under currency:** it adds failures far from the bound,
  where the site is right. 870 of 1,256 P currency failures have margin ≥ 2,
  against 33 of 249 ID failures. Close calls behave as in ID. So difficulty and
  shift look *additive* under currency (separate failure piles that don't
  interact), not the "shift turns close calls site-right" version M1 tested.
  Typo also adds site-right failures among close calls.
- Correction to the earlier 06_margin numbers: with the released labels, ID
  failures at 0.5–1 looked 0.73 site-wrong. With our labels they are 0.39–0.47,
  on only 20 prompts. The released ID labels were inflating that contrast.

### Lead 1, PriceTag fold 0 (2026-10-15, `pricetag_repro/09_amplify.py`)

Two-sided amplification (as registered) breaks correct answers: amplifying a
*correct* True value of one check makes the model say Yes even when the other
check is False (e.g. P True, Q False → Yes). The readout behaves like adding up
evidence, not a strict AND. So the ID-val rule picks α = 1 (no fix) for P, Q
and P + Q.

- **A1: failed.** At every α, P + Q breaks 18–36% of correct ID-val answers.
- **A2: holds at every α ≥ 2, but vacuous at the chosen α = 1, and partly true by
  construction** (amplifying a site that reads True can't produce No). P only,
  left failures: site-right fixed 0.91–0.92 (currency, n = 86) and 0.62–0.64
  (typo, n = 297); site-wrong fixed 0.00 (n = 34, 14).
- **A3: failed on the 10% bar in currency.** Random directions fix up to 0.22
  of currency failures (two-sided) and 0.19 (gated). Typo random ≤ 0.10. As a
  ratio, random is ≤ 1/3 of DAS in most cells but not all (two-sided P + Q
  currency 0.22 vs 0.37; Q-F 0.09 vs 0.25).

**Post hoc, designed after the smoke run: amplify only a site that reads False
("-F").** For an AND, False is the evidence that should win. The gate uses ID
class means only, with no test labels. α = 8 is chosen on ID val for all three
-F conditions (0% breaks).

| P + Q, -F, α = 8 | fixed | broke | accuracy change | clear-cut site-right left failures | site-right right failures | site-wrong (left / right) |
|---|---|---|---|---|---|---|
| currency | 0.68 | 0.00 | +13.8 pts | 1.00 (61) | 1.00 (50) | 0.21 / 0.05 |
| typo | 0.67 | 0.02 | +31.7 pts | 0.65 (239) | 0.92 (143) | 0.07 / 0.00 |
| ID | 0.21 | 0.00 | +7.3 pts | n/a | 0.90 (30) | 0.19 / 0.07 |
| random -F, same α | 0.03–0.19 | | | | | |

How to read this:
- The non-trivial part is that **site-right failures get fixed by strengthening
  the model's own value.** The downstream readout does respond: these failures
  are a weak-signal problem, not a broken path. Currency clear-cut left failures:
  100%. Typo: 65%; the remaining 35% may be the ones that go wrong later (final
  token at layer 14, see the README).
- Site-wrong failures not being fixed is mostly by construction under the gate.
  The earlier evidence (CF sim) is what shows they're a separate type.
- Gated amplification is close to "say No when a site reads False". A probe on
  the site would do about as well. The interest is mechanistic (the readout can
  use the value when it's stronger), not a better classifier.
- Random pushes fix up to 19% of currency failures, so currency failures are
  fragile to any perturbation at that token.
- Fold 0 only. Folds 1–2 and the single-layer (L8) check are queued.

Next, cheap (CPU-only once coordinates are logged): are site-right failures'
coordinates *weaker* (|c − m| smaller) than successes' at matched margin? That
would show the "weak signal" directly.
