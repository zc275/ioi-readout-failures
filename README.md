# Where do language-model failures enter? An IOI reproduction of Huang et al. (ICML 2025) and a follow-up

Arthur Cao

This repository reproduces the IOI results of [*Internal Causal Mechanisms Robustly Predict Language Model Out-of-Distribution Behaviors*](https://openreview.net/forum?id=Ofa1cspTrv) (Huang, Tao, Icard, Yang & Potts, ICML 2025) on Llama-3-8B-Instruct. It then adds a follow-up experiment that locates where the model's failures arise.

> **Acknowledgement.** This is a fork of [explanare/ood-prediction](https://github.com/explanare/ood-prediction), the authors' official release. The original code (`src/`), data (`data/`), demo notebook and pretrained subspaces (`models/`) are theirs, under the MIT license in [`LICENSE`](LICENSE). Their README is preserved as [`ORIGINAL_README.md`](ORIGINAL_README.md). Everything in this fork is in [`ioi_repro/`](ioi_repro/). Please cite the original paper (below).

## What this fork adds

1. **An independent reproduction of the IOI results**: in-distribution plus two distribution shifts (Chinese names, rephrased template), 3 folds, all four method families from the paper, with an end-to-end pipeline the original release does not include for IOI.
2. **A follow-up finding**: the causal variable the paper localizes is computed *correctly* even when the model fails. The errors enter later, when the answer is read out at the final token, and DAS can localize that readout.

## Reproduction results (AUC-ROC for predicting whether the model's answer is correct, mean over 3 folds)

| Method | ID: paper / ours | Chinese names: paper / ours | Rephrased: paper / ours |
|---|---|---|---|
| Confidence score (first N tokens, T=1) | 0.935 / 0.929 | 0.767 / 0.766 | 0.650 / 0.646 |
| Counterfactual simulation (output mapping, layers 9–12) | 0.999 / 0.971 | 0.997 / 0.978 | 0.815 / 0.984 |
| Value probing (the paper's localization site, layers 9–12) | 0.804 / 0.43–0.47 | 0.611 / 0.50–0.59 | 0.805 / 0.46–0.70 |
| Correctness probing (causal variable) | 0.914 / 0.885 | 0.806 / 0.771 | 0.665 / 0.580 |

The paper's central claim reproduces: counterfactual simulation is the strongest method and stays robust under distribution shift. Confidence scores match to within 0.006. Our small in-distribution gap to 0.999 comes entirely from near-tie prompts whose answer differs between our setup and the authors' (1.4–2.7% of prompts). On prompts where our model's behaviour matches the released labels, counterfactual simulation scores 1.000. Value probing does not reproduce at the paper's localization site; the follow-up below explains why.

## Follow-up: failures enter at the readout

IOI asks the model to complete *"Then, Issak and Naya were working at the office. Issak decided to give a ring to"* with the non-repeated name (*Naya*). The paper's causal model is **names → position variable V ("the answer is the first/second name") → output**, with V localized by DAS at the repeated name.

- **At the paper's site, V is always right.** Even on examples the model gets wrong, V there holds the correct position (interchange-intervention accuracy 0.94–1.00 for wrong sources). That is why value probing there sits near chance. It is also why counterfactual simulation works: it compares the model's output with this internal "answer key".
- **At the final token, V is corrupted exactly on failures.** We localized the same variable with DAS at the final prompt token, choosing the layer by localization quality on correct examples only, as in the paper's design. Counterfactual simulation with *ground-truth* labels, which tests whether that realization of V holds the correct position and never looks at the model's output, then predicts failures:

| Realization of V (ground-truth labels, AUC within position groups, mean of 3 folds) | ID | Chinese names | Rephrased |
|---|---|---|---|
| Repeated name, layers 9–12 (paper's site) | 0.51 | 0.51 | 0.48 |
| **Final token, layer 15** | **0.70** | **0.78** | **0.80** |
| **Final token, layer 17** (robustness check) | **0.73** | **0.78** | **0.80** |

So the model computes the right answer and then loses it on the way to the output. A refined causal model, **names → V → V_read → output**, captures this: failures are cases where V_read ≠ V. The site where V is localized most cleanly is the least informative about failures, because localization is scored on correct examples, where the two realizations agree.

## Other findings

- **The confidence baseline is sensitive to N.** Choosing the number of scored tokens on validation data (N=5 instead of 3) raises its AUC on the rephrased template from 0.646 to 0.863.
- **A geometry kill test** (`ioi_repro/08_kill_test.py`): between-class geometry at the final token predicts failures (0.81–0.88), while off-manifold distance does not. This bears on the two probe-failure sources the paper hypothesises in Sec. 4.3.
- **Reproduction notes**: we document a pad-token interaction that changes model behaviour on chat-template prompts, a pyvene version constraint (≤ 0.1.6), and every detail we had to infer, in [`ioi_repro/README.md`](ioi_repro/README.md).

## Repository layout and quick start

```
ioi_repro/      this fork's work: pipeline, results, full write-up (README.md)
src/  data/  models/  mmlu_ood_prediction_demo.ipynb   original release (authors)
```

Setup: Python 3.11, `pip install -r ioi_repro/requirements.txt`, access to `meta-llama/Meta-Llama-3-8B-Instruct`, and one 24 GB GPU (no quantization). The full pipeline takes about 10 GPU-hours. Step-by-step commands are in [`ioi_repro/README.md`](ioi_repro/README.md#reproducing).

## Citation

Please cite the original paper:

```bibtex
@inproceedings{huang2025internal,
  title={Internal Causal Mechanisms Robustly Predict Language Model Out-of-Distribution Behaviors},
  author={Jing Huang and Junyi Tao and Thomas Icard and Diyi Yang and Christopher Potts},
  booktitle={Forty-second International Conference on Machine Learning},
  year={2025},
  url={https://openreview.net/forum?id=Ofa1cspTrv}
}
```
