# Plan: do name preferences drive Llama's IOI failures? (small control)

Written 2026-10-15 (night), before running. Exploratory; prompted by the Pythia
Stage 0 correction (`PLAN_Q2.md`), where a name preference explained the
transient raw gap.

## Question

On Llama-3-8B-Instruct, every IOI test failure is "using" (V right at layer
12). One candidate for *why* the readout fails: a preference for one name of the
pair, independent of which name repeats, outweighs a correct but modest IOI
signal. Test: score each prompt and its mirror (the repeated name replaced by
the other name, so the answer swaps). With d = logit(correct) − logit(other)
(first tokens, as in `10_amplify.py`) on the original and d' on the mirror, a
fixed preference cancels in d + d'. A failure is "preference-driven" if d + d' > 0
(the IOI signal points the right way and the preference wins).

Fold 0 test sets (the released balanced correct/wrong split), names sharing a
first token dropped. Failures and successes by our own first-token score.

## Predictions

- P1. Chinese names: ≥ 70% of failures are preference-driven. Chinese names are
  rarer and byte-tokenized, so their unconditional preferences should be large.
- P2. Rephrased template: ≤ 40% of failures are preference-driven. Amplification
  didn't fix them, and they enter between layers 12 and 15, so I expect them to
  be a broken path rather than a preference winning.
- P3. ID: no prediction.

## How to read it

P1 holding gives a concrete answer to "why does the readout fail" for the
Chinese shift (a name prior outweighs the IOI signal), and fits the
amplification result there (40% fixed at α = 4: strengthening the signal beats
the preference). An earlier name-familiarity heuristic (`09_heuristics.py`)
failed, but it used familiarity proxies rather than the model's own preference.

## Guesses

1. The mirror replaces only the second mention (character span from
   `parse_ioi_prompt`), so both names still appear in the intro in the same order.
2. First-token logits, as in lead 1, since Llama's Chinese names that share a
   first token are already dropped.

## First result (fold 0) and why it isn't enough

| | acc | failures preference-driven | failures' mirror right | mean signal (fail / succ) | mean preference size (fail / succ) |
|---|---|---|---|---|---|
| ID | 0.54 | 0.98 | 1.00 | 3.2 / 4.3 | 4.2 / 1.7 |
| Chinese | 0.51 | 1.00 | 1.00 | 2.9 / 3.9 | 4.4 / 1.9 |
| rephrased | 0.50 | 0.43 | 0.70 | −0.5 / 1.3 | 2.3 / 2.2 |

P1 holds on its face, P2 narrowly fails (0.43 vs ≤ 0.40). **But the measure is
confounded by selection:** if d and d' were independent noise around a positive
signal, picking prompts with d ≤ 0 would also make d + d' > 0 and |d − d'|
large. So "preference-driven" here doesn't yet show a *stable* preference.

## Added test (written before running): does the preference replicate?

For each test prompt, put the same two names into K = 4 other prompts from the
same set (their place, object, template), in both mirror versions, keeping the
intro order. Then again with the intro order swapped. The pair's preference for
the original answer name X in context k is (d_k − d'_k)/2.

- P4. Out-of-sample prediction: the mean fresh-context preference (same intro
  order) predicts which original prompts fail, AUC ≥ 0.75 for ID and Chinese.
  For rephrased, AUC ≤ 0.65.
- P5. No prediction: whether the preference survives swapping the intro order
  (a property of the names) or flips with it (a property of the slot, i.e. a
  position bias).

## Replication result (fold 0, `ioi_repro/12_pref_replicate.py`)

| | AUC: fresh preference (same order) → failure | AUC, intro order swapped | corr fresh vs original | corr same vs swapped order |
|---|---|---|---|---|
| ID | **0.95** | 0.77 | 0.99 | 0.54 |
| Chinese | **0.96** | 0.82 | 0.99 | 0.69 |
| rephrased | **0.89** | 0.23 (reversed) | 0.91 | −0.56 |

Under the selection-only null, the fresh-context AUC would be 0.5 (fresh
contexts share only the two names and their order with the original prompt).

- **P4: holds for ID and Chinese, fails for rephrased** (0.89, predicted ≤ 0.65).
  A pair's preference, measured in 4 unrelated sentences with the IOI signal
  cancelled, predicts which prompts fail almost perfectly. Failures' mean fresh
  preference is −4.1 (ID) and −4.3 (Chinese) logits against the answer, vs
  +0.1 / +0.4 for successes.
- **P5: ID and Chinese, mostly the names; rephrased, the slot.** ID/Chinese
  preferences partly survive swapping the intro order (corr 0.54–0.69). Rephrased
  preferences flip with it (corr −0.56): a position bias.
- Position check (each pair in both answer positions): accuracy when the answer
  is the first- vs second-mentioned name: ID 0.62 / 0.89, Chinese 0.85 / 0.61,
  rephrased **0.22 / 0.83**. Rephrased failures are mostly the model picking the
  second-mentioned name. (These sets are the balanced correct/wrong split, so the
  absolute levels are lower than on random prompts.)

**What it means.** Llama's IOI failures are "using" failures in which V is right
at layer 12 and a stable bias outweighs it at the readout: a bias toward one
*name* of the pair (ID, Chinese) or toward one *position* (rephrased). That is a
concrete answer to "why does the readout fail", and it fits lead 1: under
Chinese names, strengthening V beats the bias in 40% of failures. Rephrased
failures didn't respond to strengthening, which may mean the position bias
enters downstream of the site (lead 2: name movers attending by position).

**Tension to resolve:** `09_heuristics.py` (dead ends) found slot-bias and
name-familiarity heuristics didn't predict failures within strata. Those were
proxies computed from the prompt; this uses the model's own measured preference.
Check how the two differ before claiming more. Prior work
on prior-vs-context biases needs a literature search before any novelty claim.

**Folds 1–2 replicate** (AUC same order / swapped order / corr same vs swapped):
ID 0.93 / 0.76 / 0.52 and 0.93 / 0.77 / 0.56; Chinese 0.96 / 0.83 / 0.70 and
0.96 / 0.82 / 0.70; rephrased 0.88 / 0.23 / −0.60 and 0.89 / 0.24 / −0.55.

## Related work (from a quick search 2026-10-16; search summaries only, not read yet)

- "gpt 2 sometimes fails at ioi" (LessWrong): GPT-2 Small fails on structured
  subsets of name pairs (e.g. one specific pair in a large share of cases). Same
  phenomenon as ours, informal, small model. Our addition: measured preference
  predicts failures out of sample, at 8B, plus the name-vs-position split and the
  link to V being right at the site.
- "Detection Without Suppression: A Sign Flip in IOI Circuit Formation" (ICML
  2026): IOI accuracy drops below chance early in training while name
  duplication is decodable at 99% at the second-name position. That matches
  Pythia's corrected accuracy ≈ 0 at steps 512–2000 (`PLAN_Q2.md`), and is close
  to our "variable right, readout wrong" framing over training. **Read before
  continuing Q2.**
- "Reading Task Failure Off the Activations: A Sparse-Feature Audit of GPT-2
  Small on IOI" (arXiv 2605.22719): predicting IOI failures from activations.
  Read to see whether they find a name/position bias.
- Knowledge-action gap on clinical triage (arXiv 2603.18353): probes recover what
  the output misses; steering fixed about a fifth of misses but broke many
  correct ones. Similar to our IOI amplification result.
