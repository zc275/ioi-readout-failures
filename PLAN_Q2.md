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
