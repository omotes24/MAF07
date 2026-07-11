# Experiment Procedure

This document records the end-to-end procedure used for MAF07 and RSN.

## 1. Dataset layout

The repository expects the image dataset under:

```text
/home/omote/CAT/
  cheetah/
  jaguar/
  leopard/
  lion/
  ocelot/
  puma/
  serval/
  tiger/
```

The images are not stored in Git. On another machine, either reproduce this
layout or edit `configs/dataset.yaml`.

Verify the dataset:

```bash
python -m maf07.cli verify-dataset --strict
```

Create the manifest:

```bash
python -m maf07.cli build-manifest
```

## 2. Environment

```bash
cd /home/omote/MAF07
bash scripts/setup_env.sh
source .venv/bin/activate
```

On Hades, the completed runs used the Python environment at
`/home/omote/granood_ke/.venv/bin/python`. The scripts also work with the local
`.venv` created by `scripts/setup_env.sh`.

## 3. Splits

Create deterministic splits for seeds `0,1,2`:

```bash
bash scripts/make_splits.sh
```

Output:

```text
results/splits/splits_seed0.csv
results/splits/splits_seed1.csv
results/splits/splits_seed2.csv
```

## 4. Feature extraction

The main RSN results use DINOv2 only:

```bash
export MAF07_BACKBONES=dinov2_vitb14,dinov2_vitl14
bash scripts/extract_features_parallel.sh 4
```

Expected feature files:

```text
results/features/dinov2_vitb14.*
results/features/dinov2_vitl14.*
```

## 5. Verified baseline OOD runs

The fair protocol is the main deployable protocol. It never uses OOD train or
OOD validation data.

The paper-table audit has a dedicated single-size entry point. It runs both RSN
profiles and only the methods that can be implemented faithfully from DINOv2
features plus the shared ridge linear probe:

```bash
bash scripts/run_verified_baseline_audit_hades.sh
```

The verified method list is:

```text
rsn_reported, rsn_paper, knn, mahalanobis, mahalanobispp, rmd,
msp, entropy, energy, maxlogit, gen, gradnorm, kl_matching, vim, react,
ashp, ashb, ashs, dice, scale, nci, openmax
```

Run the complete `m=2..7` sweep with:

```bash
export MAF07_AUDIT_ID_SIZES="2 3 4 5 6 7"
export MAF07_AUDIT_COVERAGE_ID_SIZES="2,3,4,5,6,7"
export MAF07_AUDIT_WORKERS=8
bash scripts/run_verified_baseline_sweep_hades.sh
```

Audit and aggregate the fold-level output with:

```bash
python scripts/audit_verified_baseline_sweep.py \
  --input results/hades_results/rsn_baseline_full_20260711/fold_level_verified.csv \
  --id-sizes 2,3,4,5,6,7 \
  --report results/hades_results/rsn_baseline_full_20260711/coverage_report.json

python scripts/aggregate_verified_baseline_sweep.py \
  --verified-fold results/hades_results/rsn_baseline_full_20260711/fold_level_verified.csv \
  --output-dir results/hades_results/rsn_baseline_full_20260711/aggregated
```

The completed run contains 1,476 folds and 32,472 verified jobs, with exact
per-size fold counts `168,336,420,336,168,48`. Its immutable paper artifacts are
under `docs/results/rsn_baseline_full_20260711/`.

`mah_mindist` is not a separate method. ODIN requires image-input gradients and
is not evaluated by the feature-cache runner. MCM, CLIP-Zeroshot-MSP,
CLIP-Text-Energy, and Tip-Adapter require a genuine CLIP image/text encoder and
must not be reported from DINO class prototypes. See `docs/BASELINE_AUDIT.md`.

Oracle runs are separate and should not be reported as deployable main results:

```bash
bash scripts/run_ood_oracle.sh
```

Archived note: the KNN oracle CSV in the current artifacts is incomplete, so
KNN-oracle should be rerun before making a final oracle-vs-KNN claim.

## 6. RSN full sweep

Use the dedicated RSN entry point:

```bash
export MAF07_BACKBONES=dinov2_vitb14,dinov2_vitl14
export MAF07_SEEDS=0,1,2
export MAF07_RSN_WORKERS=4
export MAF07_RSN_PROTOCOLS=fair
export MAF07_RSN_PROFILE=primary
export MAF07_RSN_K=150
export MAF07_RSN_DELTA=1.345
export MAF07_RSN_NORMALIZE=0
export MAF07_RSN_USE_TOPQ=0
export MAF07_RSN_MIN_STD=1e-3
export MAF07_RSN_SCORE_BATCH=96
bash scripts/run_rsn_full.sh
```

Outputs:

```text
results/quick/rsn_full_results.csv
results/quick/rsn_full_summary_by_id_size.csv
results/logs/rsn_full_*_worker*.log
```

The same primary setting also exists under the historical artifact names:

```text
results/quick/diagcard_full_huber_raw_results.csv
results/quick/diagcard_full_huber_raw_summary_by_id_size.csv
```

Those files use `k=150`, no L2 feature normalization, and all candidate classes.
`MAF07_RSN_PROFILE=reported_20260708` remains an alias for `primary` so old run
commands remain reproducible. The `paper` alias now denotes only the historical
L2-normalized, `k=10`, top-3 sensitivity profile.

## 7. Additional analyses

### PSM id-size sweep

PSM is implemented in `src/maf07/methods/psm.py`, but it is not part of the
historical `coverage_report.json` job list. Run it as a quick sweep:

```bash
export MAF07_BACKBONES=dinov2_vitb14,dinov2_vitl14
export MAF07_SEEDS=0,1,2
export MAF07_PSM_WORKERS=4
export MAF07_PSM_CUDA_DEVICES="0 1 2 3"
bash scripts/run_psm_full.sh
```

Outputs:

```text
results/quick/psm_full_results.csv
results/quick/psm_full_summary_by_id_size.csv
```

### KNN oracle completion

The archived KNN-oracle baseline is incomplete for `id_size=6,7`. It can be
completed without changing the expected-job definition:

```bash
export MAF07_BACKBONES=dinov2_vitb14,dinov2_vitl14
export MAF07_METHODS=knn
bash scripts/run_ood_oracle.sh
python -m maf07.cli aggregate
python -m maf07.cli audit-coverage
```

### vitb14-only consistency table

Generate the single-backbone RSN-vs-baseline table:

```bash
python scripts/export_rsn_vitb14_comparison.py \
  --rsn-summary results/quick/diagcard_full_huber_raw_summary_by_id_size.csv \
  --baseline-summary results/ood/fair/summary_by_setting.csv \
  --output results/analysis/rsn_vitb14_vs_knn_card_by_id_size.csv
```

### per-OOD-class evaluation

For methods with sample-level score parquet files from the main harness:

```bash
python scripts/analyze_score_per_ood_class.py \
  --input results/ood/fair/scores/jobs \
  --output results/analysis/per_ood_class_fair_results.csv \
  --summary-output results/analysis/per_ood_class_fair_summary.csv
```

For quick-method comparisons that need RSN included:

```bash
python scripts/run_quick_per_ood_class.py \
  --id-size 2 \
  --protocol fair \
  --methods rsn,knn,card,psm \
  --output results/analysis/per_ood_class_quick_fair.csv \
  --summary-output results/analysis/per_ood_class_quick_fair_summary.csv
```

Run the command for each `id_size` required in the paper.

### Shared-shape validation

Directly test the RSN shared-shape assumption by comparing class-wise
validation distance distributions with pairwise KS statistics:

```bash
python scripts/analyze_rsn_ks.py \
  --protocol fair \
  --id-sizes 2,3,4,5,6,7 \
  --output results/analysis/rsn_validation_ks_summary.csv \
  --pairs-output results/analysis/rsn_validation_ks_pairs.csv
```

### DiagCARD 2x2 ablation

Complete the calibration x Huber ablation over all `id_size` values:

```bash
export MAF07_BACKBONES=dinov2_vitb14,dinov2_vitl14
export MAF07_SEEDS=0,1,2
export MAF07_DIAGCARD_WORKERS=4
export MAF07_DIAGCARD_VARIANTS=diag_calib,diag_huber_calib,diag_raw,diag_huber_raw
bash scripts/run_diagcard_full.sh
```

Outputs:

```text
results/quick/diagcard_full_2x2_results.csv
results/quick/diagcard_full_2x2_summary_by_id_size.csv
```

### Multiple-comparison correction

Generate paired RSN-vs-KNN/CARD tests by `id_size`, with Bonferroni and Holm
corrections over the produced family of tests:

```bash
python scripts/compare_rsn_with_correction.py \
  --rsn-results results/quick/diagcard_full_huber_raw_results.csv \
  --baseline-results results/ood/fair/summary_by_setting.csv \
  --output results/analysis/rsn_vs_baselines_corrected_stats.csv
```

## 8. Archived result tables

The following analysis tables were generated from completed Hades outputs and
copied to the local working artifact directory:

```text
/Users/k.omote/MAF-OOD-v51/hades_results/diagcard_full_huber_raw_results.csv
/Users/k.omote/MAF-OOD-v51/hades_results/diagcard_full_huber_raw_summary_by_id_size.csv
/Users/k.omote/MAF-OOD-v51/hades_results/diagcard_vs_knn_card_by_id_size.csv
/Users/k.omote/MAF-OOD-v51/hades_results/diagcard_vs_knn_card_paired_stats_fair.csv
/Users/k.omote/MAF-OOD-v51/hades_results/id_size2_fair_baselines_no_lar_plus_diagcard_ranked_auroc.csv
/Users/k.omote/MAF-OOD-v51/hades_results/id_size2_fair_baselines_no_lar_plus_diagcard_ranked_fpr95.csv
```

These CSVs are not committed because `results/**` is ignored. The key result
values are copied into `README.md` and `docs/RSN.md`.

## 9. Coverage-based full harness

The original full harness remains available:

```bash
bash scripts/run_all.sh --resume --workers auto
```

Completion is defined by:

```text
results/coverage/coverage_report.json
```

The report must satisfy:

```text
expected_jobs == completed_jobs
missing_jobs == 0
```

RSN is intentionally provided as a dedicated sweep script so that adding the new
method does not invalidate the existing historical coverage accounting.
