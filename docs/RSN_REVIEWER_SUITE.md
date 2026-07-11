# RSN reviewer experiment suite

This suite addresses the reviewer-facing gaps without changing the reported RSN
definition. The entry point on Hades is:

```bash
bash scripts/run_rsn_reviewer_hades.sh
```

The default output root is
`results/hades_results/rsn_reviewer_20260712`. All CSV writers use file locks,
and completed method/fold jobs are skipped on restart.

## 1. Matched factor decomposition

`scripts/run_rsn_reviewer_suite.py` evaluates matched controls on the same fold,
feature cache, test rows, and candidate classes. The core block covers every ID
set, seed, and DINOv2 backbone for `m=2,...,7`.

The controls independently vary:

- raw versus L2-normalized features;
- pooled versus class-wise candidate banks;
- `k=50` versus `k=150` and mean versus kth-neighbor reduction;
- no scaling, global standard deviation, class standard deviation, MAD, or IQR;
- Euclidean, squared, or Huber residual aggregation;
- calibration versus no calibration;
- equalized class-bank sizes.

The sensitivity block changes one RSN setting at a time for `k`, Huber `delta`,
and the minimum scale `tau` at `m=2,4,7`. This avoids attributing a difference
between the old KNN configuration and RSN to class-conditional scaling alone.

## 2. ID-set-size confounds

Every fold row stores ID/OOD counts, prevalence, ID training-bank size, class
counts, and minimum/maximum class-bank sizes. Both ordinary AUPR-OUT and a
sample-weighted 50:50-prevalence AUPR-OUT are written. Per-OOD-species metrics
support an eight-species macro average.

`scripts/run_rsn_candidate_count_counterfactual.py` provides a narrower causal
test. It fixes one anchor ID species, one target OOD species, their test images,
and their prevalence, then adds only distractor candidate banks. This isolates
candidate-count effects from OOD-species composition. It does not claim to
isolate changes to a separately trained classification head.

## 3. Calibration diagnostics

`scripts/analyze_rsn_calibration_assumptions.py` measures raw/calibrated score
rank agreement, rank displacement, score-margin-dependent reversal rates, and a
class-count union bound derived from DKW. The report deliberately treats this as
an empirical rank-equivalence diagnostic: DKW alone does not prove an average
AUROC loss. `rsn_equalbank3000` directly tests the unequal-order-statistic issue.

## 4. Dependence-aware uncertainty

`scripts/analyze_rsn_reviewer_suite.py` first averages seed/backbone repetitions
within each ID set, then resamples ID-set clusters separately inside every
`m` stratum. It also performs cluster sign-flip tests and reports equal-weight
`m=2,...,7` estimates. It never treats the 1,476 reused-image folds as IID.
Species-macro results are emitted separately.

## 5. Strong nearby baseline

`nnguide_k10` follows the released NNGuide scoring rule: training features are
weighted by log-sum-exp confidence, top-k inner-product guidance is averaged,
and the result is multiplied by query confidence. It uses this repository's
shared ridge head, which is recorded as an adaptation rather than described as
the authors' classifier training recipe. `k=50` is included as a sensitivity.

## 6. Dataset and image audits

- `audit_rsn_dataset_leakage.py`: exact/cross-class hashes, DINO near-duplicate
  pairs at a declared threshold, cross-split pairs, and source/license/
  photographer/individual/site metadata coverage. A leakage-safety claim is
  allowed only when all provenance fields are complete and no checked crossing
  exists.
- `build_rsn_human_label_audit.py`: stratified annotation sheets and a template
  for blinded review. It remains explicitly incomplete until a reviewer fills
  the sheet.
- `audit_rsn_visual_shortcuts.py`: low-resolution full, center, border, and
  corner probes. These detect exploitable shortcuts but cannot prove their
  absence.
- `extract_dino_saliency_ablation_features.py`: foreground-only and
  background-only DINO features from CLS-to-patch similarity masks.
- `analyze_rsn_representative_fold.py`: success/failure cases, KNN/RSN nearest
  neighbors, score distributions, species confusion, Huber-suppressed
  dimensions and image extrema, runtime, memory, and bank size.

## 7. Representation robustness

After the DINO experiment, the queued pipeline extracts OpenAI CLIP ViT-B/16,
BioCLIP, and supervised ImageNet ViT-B/16 features from the same cleaned
manifest. The matched RSN/KNN/NNGuide core is run at `m=2,4,7`. These results
test representation-family and pretraining-source robustness; they are kept
separate from the DINO main table.

## Completion

`core_coverage_report.json` and `sensitivity_coverage_report.json` are strict
coverage gates. A valid core run has all combinations of two DINO backbones,
three split seeds, every ID set for `m=2,...,7`, and all registered core methods,
plus one per-OOD-species row for every fold/method result.
