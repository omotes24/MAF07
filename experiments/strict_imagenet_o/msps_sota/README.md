# RC-MSPS: Strict Fair Inductive OOD Detection

This directory contains the implementation, immutable configuration, tests, and complete results for Reliability-Calibrated Multi-Stage Prototype Support (RC-MSPS). It keeps the existing `timm/resnet50d.ra2_in1k` backbone, classifier, preprocessing, ID/OOD data, and metric implementation fixed.

## Final result

| Method | Macro AUROC | Macro FPR95 | Macro AUPR-OUT |
|---|---:|---:|---:|
| RC-MSPS | **0.887636** | **0.434417** | **0.619273** |
| Original MSPS | 0.881466 | 0.485732 | 0.576734 |
| ViM | 0.836215 | 0.688679 | 0.443039 |
| KNN | 0.833174 | 0.654297 | 0.452506 |

The paired 1,000-replicate bootstrap difference from MSPS is `+0.006171` AUROC (`95% CI [0.005182, 0.007098]`) and `-0.051316` FPR95 (`[-0.057179, -0.046179]`). This aggregate improvement is heterogeneous: RC-MSPS improves iNaturalist and OpenImage-O, but degrades ImageNet-O and Texture. It therefore does not meet all requested success targets and is not claimed as universal OOD SOTA.

## Locked method

The selected method uses:

- normalized mean prototypes from 150 ImageNet train images per class;
- a 50-image-per-class ID-only empirical CDF after locking;
- top-5 deepest-stage candidate classes;
- harmonic fusion of four calibrated stage supports;
- fixed ID-derived weights `[0.015206, 0.028525, 0.099700, 0.856569]`;
- class-CDF weight `0.1` and global-CDF weight `0.9`;
- cross-stage consensus reward `0.5`;
- no residual term and no variance penalty.

The OOD score is the negative of the final ID confidence. Each query is scored independently against frozen compressed ID statistics. No target OOD data, test-batch statistic, test neighbor, adaptation, or OOD-specific setting is used.

The configuration was written before final OOD evaluation and has file SHA-256:

```text
395b6583e842c37847e24e8c1052cc9a71ae38ce6f3b39133a1550a453640b08
```

## Files

- `METHOD.md`: complete definition and inductive-data-flow argument.
- `results/RESULTS_SOTA.md`: audit, all results, bootstrap CIs, failures, and claim limits.
- `configs/search.yaml`: declared ID-only staged search space.
- `configs/locked_config.json`: immutable selected configuration and provenance hashes.
- `run_search.py`: five-fold ImageNet class-holdout selection.
- `lock_config.py`: final ID-statistic fitting and immutable configuration creation.
- `evaluate_locked.py`: one-shot evaluator accepting no hyperparameters.
- `bootstrap.py`: paired stratified image bootstrap.
- `results/`: raw scores, complete CSVs, logs, audit records, and provenance.

## Verification

From the repository root:

```bash
PYTHONPATH=experiments/strict_imagenet_o/msps_sota \
  pytest -q experiments/strict_imagenet_o/msps_sota/tests
```

Nine tests cover OOD-path rejection, independent scoring, score orientation, prototype normalization, cache loading, lock tampering, and bootstrap metric equivalence. The exact Hades commands are in `results/command_log.txt`.

## Important scope

The comparison-set-best statement applies only to methods reproduced with the exact frozen ResNet50d/direct-resize protocol. CARef/CADRef, Dynamic Covariance variants without a protocol-matched implementation, and published numbers with different weights or preprocessing are excluded from the ranking. See `results/baseline_eligibility.csv`.
