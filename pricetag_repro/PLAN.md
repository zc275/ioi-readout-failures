# PriceTag blind-spot test: plan and predictions (DRAFT, not committed)

Review this before any counterfactual-simulation run. Once agreed, commit it unchanged so the predictions are timestamped before the results.

## What we're testing

On IOI, failures enter *after* the paper's site: V is correct at the repeated name, and the readout breaks later. So output-label counterfactual simulation (CF sim) catches them, while truth-label CF sim scores at chance there. The claim to test on PriceTag: when a shift corrupts a variable *at or before* the site, the pattern flips. Output-label CF sim misses those failures (the blind spot), and truth-label CF sim catches them.

## Correction to what I said earlier

I said the **typo** setting would garble the inputs. It doesn't. The typos are in the **template** ("b/t", "&&", "?!!"); the numbers are untouched. The shift that could make the model misread the numbers is the **currency** setting: Turkish lira with a decimal comma, e.g. "₺5,44". So the predictions below swap the roles. Currency is the candidate for a reading failure, and typos are the candidate for a readout failure, like IOI.

## Already seen (disclosed, so it isn't a prediction)

From the released labels alone, parsed before any modelling:

| | Wrong answers that are a failed left check (q < lb, said "Yes") | Wrong answers that are a failed right check (q > ub, said "Yes") |
|---|---|---|
| ID | 1799 | 4081 |
| Currency | **5357** | 758 |
| Typo | 3487 | 2643 |

(The rows sum over the three folds and all splits, so prompts that appear in several folds are counted more than once.) Under the currency shift, failures move to the **left boundary**. That fits "₺2,2 vs ₺3,13 is misread", but there are other explanations too.

## Predictions (fill in thresholds together, then commit)

- **C1, currency: reading failure.** At the paper's site for P (query integer token, layers 8–10), truth-label CF sim on P is clearly above chance on currency failures (≥ 0.70). Output-label CF sim at the same site scores lower on currency than on typo.
- **C2, currency: the input-fidelity check works.** A probe for the query's numeric value at the query token, trained on ID, decodes q worse on currency failures than on currency successes. This is the deployable check: it compares against the input text, so it needs no answer.
- **T1, typo: readout failure (like IOI).** At the same site, truth-label CF sim on P and Q is near chance (≤ 0.60), and output-label CF sim is high (≥ 0.80).
- **Kill condition.** If currency looks like typo (truth-label near chance at the site), the blind spot doesn't show up in practice here. The paper then keeps the IOI analysis and the clarification of what CF sim detects, and drops the "silent failures" headline.

## Open design question (needs your call)

**PriceTag answers are free-form.** In ID and currency the model writes "Let's check! The item costs $19.9, which is greater than $12…" before saying yes or no. Typo prompts end with "Answer:", so there it answers immediately.

This matters in two places:
1. **DAS training and CF-sim bases** need a short, scoreable answer: did the patched example say Yes or No? **Proposal:** add "Answer:" to the *base* prompts only, the verified ID examples we patch into. The *source* (test) prompts stay exactly as released. Their representation at the query token can't see anything after it (causal attention), so the suffix doesn't affect it. That keeps the upstream test faithful to the released data. I agree, let's use the Answer: format 
2. **A downstream site** at the last prompt token works for typo (it answers right there). For ID and currency, the decision may be formed *during* the reasoning text, not at the prompt's last token. **Proposal:** run the upstream site × both label rules first, since it answers C1 and T1. Treat a downstream site for currency as a follow-up. I agree 

## Reproduction issue for Jing

The released **ID** PriceTag labels don't match our model's greedy outputs: only 57% agreement on a 128-prompt pilot. Our model answers many released-"wrong" prompts correctly, after reasoning. We tested the authors' exact `generate_batched` settings (57.8%) and the eot-masking variant (49%). Adding an "Answer:" prefix gets 73–77%, so it's not that either. The OOD settings do agree (currency 91%, typo 99%). Question for the authors: how were the ID labels generated?

**Workaround:** the verified set comes from prompts that are released-correct *and* that our model answers correctly. OOD AUCs use the released labels, as for IOI. We don't report ID AUCs for PriceTag.
