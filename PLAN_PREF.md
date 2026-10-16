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
