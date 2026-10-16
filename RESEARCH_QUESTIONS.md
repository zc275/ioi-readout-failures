# Research questions (exploratory)

The working idea is that an out-of-distribution failure enters at one of two points:
- **Reading:** the model misreads the input, so the variable is already wrong at the site.
- **Using:** the model reads the input correctly and then fails to use it, so the variable is right at the site and the output goes wrong later.

Counterfactual simulation with output labels catches "using" failures. With a "reading" failure, a clean but wrong value at the site that the output follows consistently gets predicted as correct (the blind spot).

This is exploratory. We write predictions down before each run so we can't fool ourselves, but a missed prediction redirects the work and doesn't end it.

## The base claim everything rests on

**Both failure types happen.** All three questions below assume this.

| Evidence | Strength |
|---|---|
| IOI, all 3 settings: V is correct at the upstream site (layers 9–12) on failures. Truth-label CF sim ≈ 0.50, output-label ≈ 0.97. These are "using" failures. | Solid for one task and one model. Holds at every subspace dimension from 1 to 256 (2026-10-15 sweep). Shuffled-label control passes (DAS can't fit random targets). Random-init-model control still open. |
| IOI at L15: ID failures enter after L15. Rephrased-template failures enter between layers 12 and 15 (truth-label 0.80, output-label 0.42). | Shows *where* in the "using" stage the failure enters, and that this moves with the shift. Still all "using". |
| PriceTag, 3 folds (`pricetag_repro/README.md`): clear-cut failures (true margin ≥ 1) have P/Q correct at the paper's site (site-wrong ≤ 0.08). These are 73–85% of currency and typo P failures. | Same "using" pattern as IOI, on a second task and two more shifts. |
| PriceTag close calls (margin 0.5–1): the variable is *wrong at the site* on failures far more than on successes (e.g. Q currency 0.77 vs 0.09). Below 0.5 the site is fuzzy even for correct prompts. | **First evidence of the other failure type**, but it's "computed wrong on hard instances", not "misread the input". Mostly in ID and on Q under currency. |
| Misreading the input text | Not observed. Probes read q equally well on currency failures and successes. The "₺2,6" misreading idea has no support so far. |

## Q1. Which shifts break reading, and which break using?

The hypothesis: surface shifts (typos, a different script or number format) break reading, and context shifts (rephrasing, distractors) break using. If that holds, it's a general statement about how LLMs fail OOD, not a note about one method.

- **Evidence so far: against the hypothesis as stated.** Every shift tested (Chinese names, rephrased template, currency format, typos) mainly adds "using" failures. On PriceTag, whether the variable is wrong at the site depends on *instance difficulty* (how close q is to the bound), not on the shift. Prediction C1 failed and T1 held (`pricetag_repro/PLAN.md`).
- **Matched difficulty (2026-10-15, lead 5).** With our own labels, currency
  close-call failures are wrong at the site as often as ID ones. The shift adds
  a separate pile of clear-cut, site-right failures. Typo also adds site-right
  failures among close calls. Difficulty and shift look like two failure
  sources that don't interact.
- **Candidate reformulation:** difficulty → the variable is computed wrong; distribution shift → the variable is computed right and then not used. To test it: does it hold at matched difficulty, is the close-call effect more than 1-d DAS imprecision, and does a third task show the same thing?
- **Timeline:** IOI + PriceTag by January is feasible. A third task would be a stretch.

## Q2. When during training does the readout become robust?

**Sharpened (2026-10-15, `PLAN_Q2.md`, predictions agreed):** over training,
does the variable become robust to the shift before the readout path does?
Pythia 1.4B and 2.8B, IOI in completion form. Stage 0 (behavior across
checkpoints) is running.


On Pythia checkpoints, do early-training OOD failures come from reading and later ones from using? This connects to Tigges, Hanna, Yu & Biderman (NeurIPS 2024, arXiv 2407.10827), which tracks the IOI circuit across training (co-author Qinan Yu, a Potts PhD student), and to Jing Huang's current interest in learning dynamics.

- **Evidence so far:** none. Not started.
- **Role:** the M.S. project to pitch in the SOP, not something to finish by January. It needs a new model family and setup.

## Q3. Does the diagnosis tell you what to fix?

**Update 2026-10-15 (`PLAN_AMPLIFY.md`).** Partly yes. Amplifying the model's own
value at the site (no test labels) fixes most PriceTag shift failures where the
site is right: 68% of currency and 67% of typo failures with the post-hoc
False-side version, against ≤ 19% for random directions. It doesn't fix IOI
rephrased failures (≤ 1%), even though all of them are site-right. So the
"using" type splits into "weak signal" (fixable at the site) and "lost
downstream" (not fixable there). The pre-registered two-sided version failed in
both tasks: strengthening a correct value also breaks correct answers.


If reading failures respond to one intervention (cleaning up the input) and using failures to another (steering at the readout), while a generic fix helps neither, then the diagnosis is useful and not just descriptive.

- **Evidence so far:** none. Repairing IOI at the readout alone would only confirm what we already know. Q3 needs both failure types to compare.
- **Role:** the weakest of the three, but it answers "so what" directly.

## Dead ends (kept so we don't revisit them)

- **Geometry / off-manifold measures** (`ioi_repro/08_kill_test.py`): didn't predict failures.
- **Name-familiarity and slot-bias heuristics** for why the IOI readout fails (`ioi_repro/09_heuristics.py`): failed the preregistered rule. Within-stratum AUCs were 0.43–0.59. So we still don't know *why* the IOI readout fails.

## Update log

- 2026-10-14: file created. Base claim: half observed (using). Q1–Q3: no direct evidence yet.
- 2026-10-15: PriceTag (3 folds). Shift-induced failures are "using" (as in IOI). Close-call failures have the variable wrong at the site, which is the first sign of a second type, but driven by difficulty, not misreading. Q1 needs reformulating (above). Q3 is now testable: two failure types to compare.
- 2026-10-15 (later): IOI result holds at DAS dim 1–256 and passes a shuffled-label control. PriceTag subspaces carry the checks, not just q (P 92%, Q 83%). Clear-cut typo failures are right at the query token (0.00) and about half wrong by the final token at layer 14 (0.44–0.55 vs 0.21–0.26 for correct prompts; 3 folds). That's the IOI pattern on a second task.
- 2026-10-15 (afternoon): lead 5 (matched difficulty) shows difficulty and shift as separate piles, not interacting. Lead 1 (amplification): PriceTag shift failures are mostly fixable at the site (weak signal); IOI rephrased ones are not (lost downstream). Pre-registered two-sided versions failed (A1, A3 in currency, A4). Q2 plan agreed; Stage 0 running.
- 2026-10-15 (night): PriceTag gated amplification replicates in folds 1–2 (60–80% of shift failures fixed, ≤ 4% breaks; site-right fixed 0.88–0.99). Single-layer P (L8) does as well and fixes no site-wrong failures. Pythia Stage 0: raw gaps in 410M and 1.4B were a name preference in the scoring (mirrored-pair control); corrected, Chinese-name IOI just forms 2k–4k steps later than English. 160M has a real end-of-training Chinese gap (corrected 0.60) with near-zero signal, which looks like "computing". No model passes the go criteria; Q2 direction is open (PLAN_Q2.md). 2.8B's hub checkpoints are corrupted.
