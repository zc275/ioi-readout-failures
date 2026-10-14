# IOI reproduction: Huang et al. (ICML 2025)

Independent reproduction of the IOI results in *Internal Causal Mechanisms Robustly Predict Language Model Out-of-Distribution Behaviors* (Huang, Tao, Icard, Yang, Potts), built on the authors' released code (`../src`) and data (`../data/ioi_task`).

**Status:** all four method families reproduced on IOI, over 3 folds, for the ID setting and both OOD settings.

## Summary

- **Confidence scores** reproduce to within 0.006 (first N tokens, T=1). They are very sensitive to N: selecting N on ID validation raises the rephrased-template AUC from 0.65 to 0.86.
- **Counterfactual simulation** is the strongest method, as in the paper. ID and Chinese names are slightly below the paper (0.97–0.98 vs 0.997–0.999), and the rephrased template is far above it (0.98 vs 0.815). On every prompt where our model's verdict matches the released label it is perfect (AUC 1.000). The whole gap to 1.0 comes from near-tie prompts whose outcome differs between setups.
- **Value probing at the Table 5 site does not reproduce** (≈ 0.45 vs 0.804). At that site the position variable is computed correctly even when the model's answer is wrong, so the probe has no signal. A probe at the last prompt token (layers 16–22) does match the paper (0.82–0.85, exploratory, fold 0).
- **Correctness probing** at the causal-variable site roughly reproduces (0.885 / 0.771 / 0.580 vs 0.914 / 0.806 / 0.665).

## Setup

- Model: `meta-llama/Meta-Llama-3-8B-Instruct` (revision `8afb486c`), bf16, single RTX 4090 (24 GB). No quantization.
- Environment: `requirements.txt`. Transformers and datasets are pinned to versions from early 2025 (the authors' code targets transformers 4.x). **pyvene must be 0.1.6 or older.** From 0.1.7 on, `IntervenableModel.interventions[k]` holds the intervention object itself rather than an `(intervention, hook)` tuple, which breaks the authors' `interventions[k][0]` accesses.
- DAS: 10K pairs from verified × verified (ID train/correct), batch 16, AdamW lr 1e-4, one epoch, dim 256, block_output at the last token of the repeated name's second occurrence. Single layers 9, 10, 11 and 12, plus layers 9–12 jointly, for each fold. Held-out flip-pair IIA is 0.94–1.00 for every run.
- Counterfactual simulation: K = 10 bases per test input, drawn from a pool of 16 verified examples (as in the authors' demo).
- Compute (benchmarked with random-weight Llama-3-8B, same architecture):

  | Workload | Peak GPU memory | Speed |
  |---|---|---|
  | Inference, batch 64 × 96 tokens | 17.2 GB | ~95 prompts/s |
  | DAS, 1 layer, dim 256, batch 16 | 20.8 GB | ~12 min per 10K pairs |
  | DAS, layers 9–12 jointly, dim 256, batch 16 | 21.2 GB | ~36 min per 10K pairs |

  Step time is dominated by the orthogonal parametrization of the rotation, not by the model. **Under WSL2, do not exceed ~23 GB:** a DAS batch of 32 (26.5 GB peak) did not raise an out-of-memory error. It silently spilled into system RAM and ran about 70× slower. The full pipeline took about 7 GPU-hours.

## Results (mean ± std over 3 folds; AUC-ROC on balanced test sets, against the released labels)

| Method | ID (paper) | ID (ours) | Chinese names (paper) | (ours) | Rephrased template (paper) | (ours) |
|---|---|---|---|---|---|---|
| Confidence, first N tokens, T=1 | 0.935 ± 0.004 | 0.929 ± 0.008 | 0.767 ± 0.012 | 0.766 ± 0.012 | 0.650 ± 0.002 | 0.646 ± 0.003 |
| Confidence, first N tokens, T=2 | 0.900 ± 0.016 | 0.881 ± 0.018 | 0.752 ± 0.005 | 0.727 ± 0.022 | 0.558 ± 0.009 | 0.601 ± 0.005 |
| Confidence, answer token, T=1 | 0.928 ± 0.006 | 0.926 ± 0.008 | 0.709 ± 0.017 | 0.756 ± 0.005 | 0.596 ± 0.007 | 0.524 ± 0.008 |
| Counterfactual sim, first N tokens (layers 9–12 jointly) | 0.999 ± 0.000 | 0.973 ± 0.012 | 0.996 ± 0.002 | 0.980 ± 0.001 | 0.816 ± 0.014 | 0.983 ± 0.004 |
| Counterfactual sim, output mapping (layers 9–12 jointly) | 0.999 ± 0.000 | 0.971 ± 0.007 | 0.997 ± 0.001 | 0.978 ± 0.003 | 0.815 ± 0.004 | 0.984 ± 0.003 |
| Counterfactual sim, output mapping (layer selected on ID val) | 0.999 ± 0.000 | 0.970 ± 0.007 | 0.997 ± 0.001 | 0.977 ± 0.003 | 0.815 ± 0.004 | 0.984 ± 0.003 |
| Value probing, DAS subspace (layers 9 / 10 / 11 / 12) | 0.804 ± 0.021 | 0.47 / 0.44 / 0.43 / 0.47 | 0.611 ± 0.063 | 0.50 / 0.50 / 0.53 / 0.59 | 0.805 ± 0.010 | 0.70 / 0.46 / 0.59 / 0.49 |
| Correctness probing, causal variable (layer chosen on ID val) | 0.914 ± 0.007 | 0.885 ± 0.011 | 0.806 ± 0.023 | 0.771 ± 0.029 | 0.665 ± 0.032 | 0.580 ± 0.016 |
| Correctness probing, prompt last token (last layer) | 0.822 ± 0.023 | 0.887 ± 0.026 | 0.607 ± 0.034 | 0.650 ± 0.029 | 0.444 ± 0.073 | 0.527 ± 0.012 |
| *Diagnostics (not methods from the paper):* | | | | | | |
| *Counterfactual sim, output mapping, label-consistent prompts only* | — | 1.000 ± 0.000 | — | 1.000 ± 0.000 | — | 1.000 ± 0.000 |
| *Confidence, first N tokens, T=1, N=5 chosen on ID val* | — | 0.938 ± 0.009 | — | 0.818 ± 0.009 | — | 0.863 ± 0.013 |
| *Correctness probing, prompt last token, layer chosen on ID val (~21)* | — | 0.929 ± 0.022 | — | 0.842 ± 0.007 | — | 0.662 ± 0.020 |
| *Position-only predictor (oracle sign)* | — | 0.623 ± 0.021 | — | 0.595 ± 0.007 | — | 0.811 ± 0.005 |

Paper numbers are from Tables 2 and 3. For IOI, "Change Language" is the Chinese-names setting and "Rephrase Templates" is the rephrased template. The layer selected on ID validation was 12, 9–12 jointly and 9 for folds 0, 1 and 2. Regenerate the table with `python 07_summarize.py`.

## Observations

**1. Why counterfactual simulation works here: the site holds an internal answer key.** At the DAS site (layers 9–12, last token of the repeated name), the position variable V is computed correctly even when the model's final answer is wrong. On held-out flip pairs, IIA with the ground-truth V is 0.94–1.00 for wrong sources, about the same as for correct ones. Counterfactual simulation derives the label from the source's *output* (guess 1), so it effectively checks whether the output agrees with the internally computed V, and they disagree exactly when the output is wrong. On prompts where our model's verdict matches the released label, AUC is 1.000 in all settings and folds. The remaining errors are near-tie prompts whose outcome differs between our setup and the authors' (see "Label agreement").

**2. Value probing at the Table 5 site has no signal, for the same reason.** A value probe on the DAS subspace classifies V with accuracy 1.000 on both correct and wrong examples, with equal confidence (0.41 vs 0.40; fold 0, layer 10). Its AUC is therefore near chance, and below it where the position confound points the other way. A layer scan with a full-residual V-probe (ID fold 0, exploratory) locates where the errors enter:

| Location | Layers | V-probe accuracy on wrong examples | AUC of probe confidence |
|---|---|---|---|
| repeated-name site | 6–32 | 1.00 | ≈ 0.50 |
| last prompt token | 16–22 | 0.85–0.88 | 0.82–0.85 |

The errors arise when the position is read out at the final token, from about layer 16 on, and a value probe there roughly matches the paper's 0.804. Hypothesis: the paper's IOI value probe read the last token, as the MMLU demo's value probe does (`pos_key = 'causal_intermediate'`, the last position). The demo also projects `hidden_states[layer]`, which is the *input* to the DAS layer rather than its output. We report both indexings in `outputs/results/probes.json`; neither changes the conclusion.

**3. Our rephrased-template counterfactual simulation is far above the paper's (0.98 vs 0.815). The cause is unexplained.** Correctness here is strongly confounded with V: about 80% of correct test examples have V = second and about 80% of wrong ones V = first. A predictor that sees only V reaches 0.811, close to the paper's 0.815 and 0.805 (value probing). We tested one candidate explanation, that the paper's labels used the ground-truth V rather than the output. With ground-truth labels, counterfactual simulation (output mapping, layers 9–12 jointly, 3 folds) gives 0.506 ± 0.006 (ID), 0.506 ± 0.007 (Chinese) and 0.363 ± 0.085 (rephrased), not 0.815, so that explanation is ruled out (`04_cf_sim.py --label_source truth`). *(Question for the authors.)*

**4. The confidence baseline is very sensitive to N out of distribution.** Using N=3, which matches the paper, we reproduce its numbers. Selecting N on the ID validation split instead, the standard way to pick a hyperparameter, gives N=5. That N barely changes ID (0.929 → 0.938) but raises the rephrased-template AUC from 0.646 to 0.863, above the paper's counterfactual simulation result in that setting (0.815). Mechanism: under the rephrased template, wrong answers tend to keep generating ("Eloy. The story states that…"), while correct ones end the turn right after the name. Token 4 or 5 is often that end-of-turn decision, so it only enters the score once N > 3.

## Places where we had to guess

1. **Counterfactual label source.** The paper says V is "inferred from either the source input or the test example output" (Sec. 4.3). The paper's definition, y_cf = 1(τ(M(x_src))), and the authors' MMLU demo (test-time labels come from `generate_batched`) both take it from the source's *model output*, and we do the same (`ioi_data.output_position`). Outputs that are not exactly name a or name b (mostly spelling "corrections" such as Harrisson → Harrison) are not values of V. For these, the label falls back to the source's raw output, as in the demo's `inv_label_fn`, and a patched base cannot produce it. Taking V from the ground truth instead gives AUC ≈ 0.5 (observation 3). Our first version mapped misspelled outputs to the most similar name, which scored all 47 spelling corrections in ID fold-0 val as correct (AUC 0.928, versus 0.975 with the fallback).
2. **Layers.** Table 5 lists "layers 9–12" for IOI. We ran single layers 9, 10, 11 and 12 and the four layers jointly. ID validation AUC differs by at most 0.015 across these, and test results for the joint and the validation-selected subspaces agree to within 0.005, so this choice does not matter here.
3. **K, the number of verified bases per test input.** It is not stated for IOI. We use the MMLU demo's setting: 10 bases drawn from a pool of 16 verified examples.
4. **Value-probe location.** Not stated. We use the DAS site and layer (observation 2).
5. **Correctness rule.** We count an output as correct only if it exactly matches the non-repeated name. On ID fold 0, 72 outputs were spelling "corrections" of the right name (Harrisson → Harrison, Issaic → Isaac), and the released labels mark all 72 as wrong, so this appears to be the authors' rule.
6. **N for "first N tokens".** We use N=3 for all IOI settings, the max English-name answer length. A per-setting N (6 for Chinese names) does not reproduce the paper (0.828 vs 0.767). N=3 matches all three settings to within 0.006.
7. **"Answer tokens".** We take the generated tokens that spell out the produced name, including the leading-space token. This matches ID (0.926 vs 0.928) but not OOD (Chinese 0.756 vs 0.709; rephrased 0.524 vs 0.596), so the authors' definition evidently differs. *(Unresolved.)*
8. **Pad token (deviation from the demo, needed for correctness).** The authors' demo sets `pad_token = eos_token`, which for Llama-3-8B-Instruct is `<|eot_id|>`. Their `train_intervention_step` and evaluation code rebuild the attention mask as `input_ids != pad_token_id`. Every IOI prompt contains a real `<|eot_id|>` in its chat template, so that token gets masked out and the position ids after it shift. We measured the effect on 256 ID validation examples per class, with no intervention: under this mask the model outputs the correct name on 53.9% of the released-*wrong* examples, versus 5.5% with the true mask. The correct examples go from 98.8% to 95.7%. We pad with `<|end_of_text|>`, which never occurs in the prompts, so the authors' mask logic recovers the true mask. Check: self-patching (source = base) gives IIA 1.000 on verified examples and 0.047 on released-wrong ones.
9. **Layer for the prompt-last-token correctness probe.** It is not stated. The last layer (0.887 / 0.650 / 0.527) is closer to the paper (0.822 / 0.607 / 0.444) than a layer selected on validation (0.929 / 0.842 / 0.662), so we report the last layer.

## Label agreement with the released data

Greedy decoding on our setup agrees with the released correct/wrong labels on 97.3–98.6% of prompts in every setting and fold (ID 97.5–97.8%, Chinese 98.0–98.3%, rephrased template 98.3–98.6%). Nearly all disagreements are released-*wrong* examples that we get right, and they are near-ties: median first-token probability 0.53, versus 0.89 for prompts where we agree. Re-running the 102 ID fold-0 disagreements one at a time, with no padding, flips 58 of them back into agreement, so the outcome on these knife-edge examples depends on the amount of padding in the batch (bf16 numerics). **All AUCs use the released labels as ground truth.** Our generations are used only for confidence scores and for counterfactual labels.

## Questions for the authors

1. Was a different pad token (or tokenizer revision) used for IOI? With `pad = <|eot_id|>`, the chat template's `<|eot_id|>` is masked out (guess 8).
2. Where was the IOI value probe trained: at the Table 5 site, or at the last prompt token (observation 2)?
3. What explains the rephrased-template counterfactual-simulation result of 0.815, given that output-derived labels give 0.98 for us (observation 3)?
4. How are "answer tokens" defined for multi-token or leading-space answers (guess 7)?

## Reproducing

```
python 01_behavior.py            # model outputs + labels check            (~10 min GPU)
python 02_confidence.py          # confidence baseline                     (CPU)
bash run_das_queue.sh "0 9" ...  # DAS, one job per "fold layers"          (~12 / 36 min each)
python 04_cf_sim.py --fold 0 --das ioi-fold0_layer9-10-11-12_dim256_n10000_seed0 --splits test
python 05_cache_activations.py   # residual stream at site + last token    (~5 min GPU, ~10 GB)
python 06_probes.py              # value + correctness probing             (CPU)
python 07_summarize.py           # the results table
```

`pipeline_overnight.sh` runs everything after the single-layer DAS queue in order.

## Extension: geometry kill test (`08_kill_test.py`)

Do geometric measures of where a representation sits, relative to the verified set, predict correctness? And does the answer depend on where in the causal chain we measure? Predictions were written into the script before running it: (P1) no signal at the repeated-name site; (P2) clear signal at the last prompt token. All measures are fit on verified examples only. Cells are AUCs *within position strata* (removing the position confound), mean ± std over 3 folds, for two feature spaces fixed in advance:

| Feature space | Measure | ID | Chinese names | Rephrased |
|---|---|---|---|---|
| repeated-name site, hs11, DAS-256 | kNN distance (off-manifold) | 0.67 ± 0.02 | 0.63 ± 0.02 | 0.43 ± 0.03 |
| | class margin (between-class) | 0.39 ± 0.02 | 0.45 ± 0.02 | 0.46 ± 0.01 |
| | Mahalanobis | 0.58 ± 0.00 | 0.61 ± 0.02 | 0.48 ± 0.01 |
| | linear value probe | 0.46 ± 0.02 | 0.47 ± 0.01 | 0.46 ± 0.01 |
| last prompt token, hs20, PCA-256 | kNN distance (off-manifold) | 0.56 ± 0.01 | 0.71 ± 0.04 | 0.60 ± 0.03 |
| | class margin (between-class) | 0.81 ± 0.02 | 0.81 ± 0.01 | 0.55 ± 0.05 |
| | Mahalanobis | 0.47 ± 0.03 | 0.38 ± 0.04 | 0.38 ± 0.03 |
| | linear value probe | 0.88 ± 0.01 | 0.86 ± 0.01 | 0.55 ± 0.03 |
| *Baselines* | confidence, N=5 | 0.94 | 0.81 | 0.78 |
| | counterfactual sim (9–12) | 0.97 | 0.98 | 0.97 |

Full tables (16 feature spaces × 4 measures) are in `outputs/logs/08_kill_test.log`.

- **P2 holds for between-class measures, P1 holds for them too.** The linear probe and the kNN class margin are at chance at the paper's site but reach 0.81–0.88 at the last token (ID and Chinese names). The correctness signal lives at the readout stage, not at the localization site.
- **Off-manifold measures do not behave as hoped.** kNN and Mahalanobis distances carry a modest signal at the *upstream* site (0.6–0.67 in ID and Chinese; perhaps unusual names, untested), but it inverts on the rephrased template. At the last token they are near or below chance, and full-dimensional kNN is strongly inverted (0.29–0.35). Of the two failure sources the paper hypothesises in Sec. 4.3, "between classes" is supported and "outside the region the classes cover" is not.
- **Geometry does not beat the linear probe.** The nonlinear margin is at most equal to the linear probe, and no cheap method approaches counterfactual simulation. Confidence with N=5 beats all of them on ID and the rephrased template. None of the methods works well on the rephrased template.
- Caveat: the measure hyperparameters (k = 10, the layers) were not tuned, and picking the best of the 64 cells per setting would select on test data.

## Extension: where does the failure enter? DAS at the readout (`run_downstream.sh`)

**Idea.** The paper localizes the position variable V at the repeated name (upstream). There V is correct even on failures (observation 1). The model must carry V to the final token and read the answer out there. If failures happen during that readout, a *downstream* realization of V, localized with DAS at the final prompt token, should be corrupted on failures. In causal-abstraction terms, the high-level model refines from names → V → output to names → V → V_read → output, and failures are cases where V_read ≠ V.

**Test.** We train DAS for V at the final prompt token (`03_das_train.py --site last`), using the same counterfactual pairs, verified examples only and the same hyperparameters, at layers 15/17/19/21. We then run counterfactual simulation with **ground-truth labels**, i.e. "does this realization of V hold the correct position?", which does not use the model's output. The layer is chosen by localization quality on verified examples only (held-out flip IIA: 0.86 / 0.75 / 0.71 / 0.62 on fold 0, so layer 15), without failure labels, as in the paper's design.

**Result.** Ground-truth-label counterfactual simulation, AUC within position strata, mean ± std over 3 folds:

| Realization of V | Score | ID | Chinese names | Rephrased |
|---|---|---|---|---|
| upstream (repeated name, layers 9–12 jointly) | output mapping | 0.51 ± 0.00 | 0.51 ± 0.00 | 0.48 ± 0.01 |
| | first N tokens | 0.66 ± 0.03 | 0.62 ± 0.04 | 0.49 ± 0.01 |
| **downstream (final token, layer 15)** | output mapping | **0.70 ± 0.03** | **0.78 ± 0.03** | **0.80 ± 0.01** |
| | first N tokens | **0.81 ± 0.01** | **0.81 ± 0.01** | **0.82 ± 0.01** |
| **downstream (final token, layer 17)**, robustness check | output mapping | **0.73 ± 0.01** | **0.78 ± 0.01** | **0.80 ± 0.03** |

On fold 0, every final-token layer tried (15–21) gives 0.71–0.84 (output mapping, within strata).

- **The downstream realization of V carries the failure, and the upstream one does not.** The site where V is localized most cleanly (upstream, IIA 0.99) is the least informative about failures, because localization quality is measured on correct examples only, where both realizations agree.
- **This explains the label-rule results.** With *output-derived* labels, the downstream realization fails on the rephrased template (0.41 within strata, fold 0) while the upstream one succeeds (0.97). Downstream V agrees with the wrong output, since the output is built from it, so comparing the two finds nothing. Upstream V is still correct, so it disagrees with a wrong output.
- Caveats: the first-N score retains a weak upstream signal (0.62–0.66 on ID and Chinese names), perhaps input atypicality (cf. the kill test's kNN result; untested). Downstream localization is less clean (IIA 0.81–0.86 vs 0.99). As a *predictor*, this does not beat the paper's output-label method (0.97–0.98); the contribution is locating the failure, not a better score.
