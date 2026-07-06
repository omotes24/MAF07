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

## 5. Baseline OOD runs

The fair protocol is the main deployable protocol. It never uses OOD train or
OOD validation data.

```bash
export MAF07_BACKBONES=dinov2_vitb14,dinov2_vitl14
export MAF07_METHODS=knn,card,msp,entropy,energy,maxlogit,mahalanobis,mah_mindist,rmd,mahalanobispp,vim,react,ashp,ashs,ashb,dice,gen,scale,nci,odin,gradnorm,openmax,kl_matching,mcm,clip_zeroshot_msp,clip_text_energy,tip_adapter
bash scripts/run_ood_fair.sh
```

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
export MAF07_RSN_K=150
export MAF07_RSN_DELTA=1.345
export MAF07_RSN_NORMALIZE=0
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

The archived completed experiment was run before the public rename and is stored
under the legacy names:

```text
results/quick/diagcard_full_huber_raw_results.csv
results/quick/diagcard_full_huber_raw_summary_by_id_size.csv
```

Those legacy files are exactly RSN.

## 7. Archived result tables

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

## 8. Coverage-based full harness

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
