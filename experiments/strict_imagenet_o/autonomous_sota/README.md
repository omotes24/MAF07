# PULSE strict-inductive OOD research

This directory contains the autonomous research record and the frozen PULSE
implementation for `timm/resnet50d.ra2_in1k`.

## Method

PULSE sums four independently ID-standardized OOD components with fixed unit
weights:

```text
ReLU(z_ID(log1p(activation_shift)))
+ ReLU(z_ID(class_center_uniformity))
+ z_ID(-localized_prototype_support)
+ z_ID(-compact_multiview_knn_support)
```

The method uses no target-OOD fit or calibration, shares no information between
test images, changes no classifier prediction, and applies one configuration to
every OOD dataset. Its only method hyperparameters are the activation
perturbation fraction, activation perturbation magnitude, and KNN neighbor
count. The train support is a compact IVF-PQ index rather than a raw feature
bank.

## Reproduced legacy result

The frozen legacy result is in `legacy_results/cycle87_pulse/`:

| Metric | PULSE | RC-MSPS | Difference |
|---|---:|---:|---:|
| Macro AUROC | 0.911426 | 0.887636 | +0.023790 |
| Macro FPR95 | 0.399970 | 0.434417 | -0.034447 |
| Macro AUPR-OUT | 0.626109 | 0.619273 | +0.006836 |

ImageNet-O AUROC is 0.821093. The paired bootstrap and the four-component
ablation are stored beside the summary. Every component removal worsens both
Macro AUROC and Macro FPR95.

## Integrity workflow

`configs/final_benchmark_preregistration.json` and
`final_results/prereg_manifest/` fix the untouched NINCO and SSB-hard suite.
After the code commit is fixed, `configs/final_locked_config.json` records every
equation, hyperparameter, model-state hash, data-manifest hash, artifact hash,
code hash, environment version, and seed.

The final one-shot command is:

```bash
python run_final_locked.py --locked-config configs/final_locked_config.json
```

The launcher accepts no score or data-selection settings. It verifies the lock,
runs four fixed GPU shards, evaluates PULSE and RC-MSPS, performs 2,000 paired
bootstrap draws, and writes `RESULTS.md`, per-dataset metrics, runtime/storage
accounting, and post-lock error analysis.

## Untouched final result

The locked evaluation completed once on NINCO (5,879 images) and SSB-hard
(49,000 images):

| Metric | PULSE | RC-MSPS | Difference |
|---|---:|---:|---:|
| Macro AUROC | **0.723284** | 0.687187 | **+0.036097** |
| Macro FPR95 | **0.775802** | 0.818535 | **-0.042734** |
| Macro AUPR-OUT | **0.486622** | 0.444561 | **+0.042061** |

All three paired-bootstrap 95% confidence intervals exclude zero in the
favorable direction, and PULSE improves AUROC on each final dataset. The full
score-level record, confidence intervals, failure cases, runtime, and storage
accounting are in `final_results/pulse_locked/RESULTS.md`.

## Research record

- `STATE.json`: current checkpoint and final-access state
- `HYPOTHESES.md`: preregistered hypotheses and predictions
- `RESEARCH_LOG.md`: chronological experiment record
- `FAILED_IDEAS.md`: rejected mechanisms and retry conditions
- `ALL_EXPERIMENTS.csv`: tested hypothesis ledger
- `PARETO_FRONT.csv`: same-protocol Pareto frontier
- `SOTA_ELIGIBILITY.md`: comparison eligibility audit
