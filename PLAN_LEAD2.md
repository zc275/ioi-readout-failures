# Plan: where IOI rephrased failures get lost (lead 2), DRAFT for your review

Written 2026-10-15 (night), before running anything. Mark predictions "I agree"
or change them, as with the other plans. Background: `PLAN_AMPLIFY.md` (lead 1)
and `ioi_repro/README.md` ("where does the failure enter?").

## Question

On rephrased-template failures, the position variable V is right at the repeated
name (layer 12, and strengthening it there doesn't help) but already wrong at the
final token by layer 15. Something between those layers loses it. In GPT-2 IOI
(Wang et al. 2022), the final token learns the answer through attention heads:
**S-inhibition heads** carry "which name is repeated" from the repeated name to
the final token, and **name-mover heads** at the final token attend to the
non-repeated name and copy it into the output. So: **on rephrased failures, which
of these steps goes wrong?**

- (a) **Moving:** V never reaches the final token (the heads that read it from
  the repeated name's position attend elsewhere).
- (b) **Copying:** V arrives, but the name movers attend to the wrong name.
- (c) **Outvoted:** the name movers do the right thing, and other heads or MLPs
  push the other name harder.

## Stage A: find the heads, compare failures to successes (about 1 day)

Llama-3-8B-Instruct, fold 0. Prompts: ID test and rephrased test, both
successes and failures (as scored in `10_amplify.py`: first token of the correct
name vs the other name).

1. **Direct logit attribution** at the final token, per attention head in layers
   12–31: each head's output projected onto the (correct − other) name-token
   unembedding difference, with the final RMSNorm scale frozen at its actual
   value. Name-mover candidates: the heads with the largest mean attribution on
   ID successes.
2. **Attention** of each candidate from the final token to the three name
   positions (non-repeated name, first mention of the repeated name, second
   mention), on successes vs failures in each setting.
3. **Moving test:** for heads in layers 12–15 at the final token, ablate each one
   (replace its output with its mean over ID prompts) while doing the L12 DAS
   swap. Heads whose ablation removes most of the swap's effect on the output are
   the ones that carry V from the repeated name to the final token.

## Predictions (draft)

- L2-1. Name movers exist: ≤ 10 heads in layers 13–31 account for ≥ 50% of the
  total head attribution on ID successes, and attend mostly (≥ 0.5 of their
  name-attention) to the non-repeated name.
- L2-2. Rephrased failures are mostly (b) copying: on failures, the top name
  movers give more attention to the repeated name than to the non-repeated one,
  and on rephrased successes they don't. Measure: the fraction of name-attention
  on the correct name, failures < 0.5 < successes.
- L2-3. ID failures (which enter after layer 15) look like (c) outvoted instead:
  name-mover attention on ID failures is close to ID successes, and the
  attribution difference comes from other components.
- No prediction for which heads carry V in step 3. I don't know whether Llama
  routes it the way GPT-2 does.

**How to read the outcomes.** L2-2 holding gives a concrete story: the
rephrasing changes *where the name movers look*, not what the model knows. That
connects to Q1 (a surface change breaks a specific readout step). If (a) holds
instead, the failure is in moving the variable, between layers 12 and 15, and
step 3 names the heads.

## Guesses (choices without a paper to follow)

1. Attribution through the final RMSNorm uses the actual per-prompt scale
   (standard approximation; the norm makes attribution not exactly additive).
2. Llama-3 uses grouped-query attention (32 query heads share 8 key/value
   heads). Heads are the 32 query heads, each with its slice of the output
   projection.
3. Name positions are each name's first token. The rephrased template is
   "Story: A and B were working at the place. B decided to give a X to...
   Question: Which of the two characters ... receive the X in the end?". The
   question mentions no names, so there are three name positions as in ID.
4. Mean ablation over ID prompts, not zero ablation.

## Alternative model: Pythia-160M

Tigges et al. map Pythia-160M's IOI circuit head by head across training, and
160M has a large Chinese-name gap at the end of training (0.975 vs 0.554). The
same Stage A on 160M is much cheaper, comes with a published circuit to check
against, and connects to Q2 (does the circuit form before it becomes robust to
names?). I'd run Llama first, since that's where the lead-1 result is, and Pythia
after Q2 Stage 1.

## Timeline

Stage A by 10/18 (padded 10/20). It shares the GPU with Q2 Stage 1, so I'll
run them one at a time (running two at once stalled the GPU on 10/15).
