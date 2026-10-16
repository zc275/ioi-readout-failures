# Plan: Q2 across training (lead 4), DRAFT for your review

Written 2026-10-15. Stage 0 has no predictions (it only asks whether the setup
works), so it can run now. Stages 1–2 have draft predictions; mark them "I agree"
or change them before they run, as with `pricetag_repro/PLAN.md`.

## Question

From the evidence so far (2 tasks, 5 shifts): on shifted inputs, the variable is
usually right at the site, and the error enters on the way to the output.
Sharpened Q2: **over training, does the variable become robust to the shift
before the readout path does?** If yes, the OOD gap late in training is a
readout problem that training is still fixing, and the gap early in training
is a different (variable) problem. This is also the SOP pitch, and it connects
to Tigges, Hanna, Yu & Biderman (NeurIPS 2024), who track the IOI circuit
across Pythia training from 70M to 2.8B.

## Setup choices (guesses, tell me if you disagree)

1. **Model: Pythia** (EleutherAI), 1.4B and 2.8B: the sizes Tigges et al.
   cover, and both fit on the 4090. 154 checkpoints each (step 0 to 143,000).
   Backup if Pythia doesn't show an OOD gap: OLMo-2 1B (checkpoints every 1,000
   steps).
2. **Task: IOI only.** Pythia is a base model, and PriceTag needs instruction
   following and number comparison, which it likely can't do. IOI has no
   difficulty axis, so lead 5's "two piles" question can't be tracked here.
3. **Prompt format:** the classic completion form, since base models don't
   follow "Output the non-repeated name": "Then, A and B were working at the
   office. B decided to give a bone to" → " A". Names, places and objects come
   from the released data.
4. **Shifts:** (a) Chinese names: the released name pairs, same template.
   (b) A rephrased narrative template written by us, since the released one is
   a question-answer format a base model may not handle. Example: "At the office,
   A and B spent the day working. Later, B handed a bone to" → " A".
5. **Scoring:** the log-probability of the full correct name vs the other name.
   First tokens don't work, because Chinese names are byte tokens and often
   share their first token.

## Stage 0: does Pythia have an OOD gap worth studying? (about half a day, no DAS)

Behavior at about 12 log-spaced checkpoints (step 0, 512, 1k, 2k, 4k, 8k,
16k, 32k, 64k, 100k, 143k), 2,000 prompts per setting.

**Go criteria at the final checkpoint:** ID accuracy ≥ 80%, an OOD drop of
≥ 10 points in at least one shift, and ≥ 200 OOD failures. If neither size
passes: OLMo-2 1B, or a different shift.

## Stage 1: does the "using" pattern hold in Pythia at the end of training? (1–2 days)

DAS on the final checkpoint, at the repeated name's token, with a layer sweep
(Pythia's layers don't map to Llama's layer 12), then CF simulation with output
and truth labels, as in `ioi_repro`.

- Q2-1 (draft). As in Llama: OOD failures are mostly site-right (truth-label
  AUC ≤ 0.65, output-label AUC ≥ 0.85). If this fails, Stage 2 still runs, but
  the question becomes "when do the failures become 'using' failures?" I agree

## Stage 2: across training (3–4 days)

At each checkpoint from the point IOI works on ID (≥ 70% ID accuracy):
retrain DAS (the subspace can move over training, per Tigges et al.), then
measure:
- (i) ID IIA (is the variable there?);
- (ii) the site-right fraction of OOD failures (truth-label CF sim);
- (iii) OOD accuracy;
- (iv) if lead 1 works, the amplification fix rate as a readout-strength measure.

- Q2-2 (draft). The variable becomes shift-robust first: the training step at
  which (ii) crosses 0.8 is earlier than the step at which OOD accuracy reaches
  90% of its final value.
- Q2-3 (draft). Earlier OOD failures are more often site-wrong than later ones
  ((ii) rises over training).
   I agree
## Timeline (aggressive, then padded 1.5×)

Stage 0 by 10/16, Stage 1 by 10/18, Stage 2 preliminary curve by 10/24.
Padded: about 10/31. That still leaves November for the SOP, ahead of the
December deadlines.

## What leads 1 and 5 change here

- Lead 1: if amplification fixes "using" failures, it becomes measure (iv). If
  not, Stage 2 keeps measures (i)–(iii).
- Lead 5: difficulty and shift failures look like separate piles. IOI has no
  difficulty axis, so this needs a second task later; out of scope for now.

## Stage 0 results (2026-10-15 night)

`pythia_q2/01_behavior.py` (raw) and `pythia_q2/02_name_prior.py` (corrected).
Pythia-2.8B is excluded: its hub checkpoints are corrupted (byte-identical
weight files across steps 0, 1000 and 32000, and across 64000 and 143000).

**The raw score was contaminated by a name preference.** Full-name log-prob
scoring favors whichever name the model finds likelier on its own (for Chinese
names, mostly the shorter one). Control: score each prompt and its mirror (the
other name repeated, so the answer swaps). With d and d' the two
log-prob differences, a fixed preference cancels in d + d'. "Corrected" = the
fraction of pairs with d + d' > 0.

Corrected accuracy, ID / Chinese / rephrased:

| step | 160M | 410M | 1.4B |
|---|---|---|---|
| 0 | 0.52 / 0.49 / 0.52 | 0.47 / 0.56 / 0.48 | 0.49 / 0.54 / 0.49 |
| 512–1000 | ≈ 0 / ≈ 0 / ≈ 0 | ≈ 0 / ≈ 0 / ≈ 0 | ≈ 0 / ≈ 0 / ≈ 0 |
| 2000 | 0.02 / 0.00 / 0.01 | 0.10 / 0.00 / 0.14 | **0.74 / 0.01** / 0.93 |
| 4000 | 0.81 / 0.00 / 0.74 | **0.95 / 0.15** / 1.00 | 0.97 / 0.96 / 1.00 |
| 8000 | 1.00 / 0.34 / 1.00 | 1.00 / 0.95 / 1.00 | 1.00 / 1.00 / 1.00 |
| 32000 | 1.00 / 0.44 / 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| 143000 | 1.00 / 0.60 / 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |

What this shows:
- **The transient raw gaps in 410M and 1.4B (steps 4k–32k) were the name
  preference,** not IOI failing. Corrected, those checkpoints do IOI on Chinese
  names as well as on English ones.
- **Early in training (steps 512–2000) corrected accuracy is ≈ 0:** the model
  systematically picks the *repeated* name (copying the more frequent name),
  before IOI forms.
- **The real shift effect is a lag:** IOI on Chinese names forms later than on
  English names. About 2k steps in 1.4B (step 2000: 0.74 vs 0.01), about 4k in
  410M (step 4000: 0.95 vs 0.15). 160M never fully gets there (0.60 at the end,
  with the mean preference-free signal 0.19 vs 3.8 on ID).
- The rephrased template causes no gap in any model.

**Go criteria at the final checkpoint: no model passes once corrected**
(160M's gap is real but its Chinese signal is near zero, which looks like
"computing", not "using"). Options for the user:
1. Study the lag window densely (Pythia has checkpoints every 1,000 steps):
   410M steps 4k–8k or 1.4B steps 2k–4k. In those windows, are the Chinese
   failures "computing" (V wrong at the site) or "using"? A sharper Q2: when a
   variable forms later for shifted inputs, does it form at the site first and
   reach the output later?
2. 160M at the end of training as a "computing" failure case (useful contrast
   with Llama, where failures are "using").
3. OLMo-2 1B, or a different shift.
4. Put Q2 behind lead 2.

**Consequence for Llama:** Llama's Chinese-name failures could also be driven by
a name preference that outweighs a correct V at the readout. That would be a
concrete answer to "why does the readout fail" for that shift. Next: the
mirrored-pair test on Llama, with a prediction written first.
