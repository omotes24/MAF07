# Verified baseline sweep for all ID-set sizes

This directory contains the completed fair-protocol DINOv2 baseline sweep for
`id_size=2,3,4,5,6,7` on the cleaned 105,554-image CAT dataset.

## Protocol

- Backbones: DINOv2 ViT-B/14 and ViT-L/14
- Seeds: `0,1,2`
- ID sets: every combination of 8 species for each ID-set size
- Verified feature-cache methods: 22 entries, including the reported RSN and
  its sensitivity profile
- PSM: merged from the matching cleaned fold-level run
- Scope: fair only; no OOD train or validation samples are used

| ID classes | folds | verified methods | jobs |
|---:|---:|---:|---:|
| 2 | 168 | 22 | 3,696 |
| 3 | 336 | 22 | 7,392 |
| 4 | 420 | 22 | 9,240 |
| 5 | 336 | 22 | 7,392 |
| 6 | 168 | 22 | 3,696 |
| 7 | 48 | 22 | 1,056 |
| **Total** | **1,476** | **22** | **32,472** |

`coverage_report.json` records `complete=true`, with no missing, duplicate,
extra, or non-finite jobs. The PSM-merged aggregate contains 33,948 method-fold
rows and 23 method/configuration identifiers. The main rankings exclude the
non-primary `rsn_paper` sensitivity profile, leaving 22 methods.

## Main result

RSN has the best mean rank over `m=2..7` for all three metrics: AUROC 2.50,
FPR95 1.33, and AUPR-OUT 2.17. It ranks first on all three metrics for
`m=2,3,4`. At larger ID-set sizes, the ranking changes: OpenMax has the highest
mean AUROC for `m=5,6,7`, while RMD has the highest AUPR-OUT for `m=5,6,7` and
the lowest FPR95 for `m=6,7`. The fold-paired confidence intervals are in
`aggregated/paired_rsn_vs_all_by_id_size.csv`.

## Files

- `fold_level_verified.csv`: raw 32,472-job verified sweep with score hashes
- `summary_verified.csv`: backbone-specific and pooled verified summaries
- `equivalence_audit.csv`: raw-score and metric identity checks
- `coverage_report.json`: exact expected/completed-job audit
- `aggregated/summary_verified_with_psm.csv`: PSM-merged summary
- `aggregated/ranking_by_id_size.csv`: pooled AUROC ranking for each `m`
- `aggregated/metric_ranking_by_id_size.csv`: ranking for every metric and `m`
- `aggregated/mean_metric_rank_m2_m7.csv`: mean metric ranks over all six sizes
- `aggregated/paired_rsn_vs_all_by_id_size.csv`: paired differences and 95% CIs

## Reproduce

```bash
bash scripts/run_verified_baseline_sweep_hades.sh
python scripts/aggregate_verified_baseline_sweep.py \
  --verified-fold results/hades_results/rsn_baseline_full_20260711/fold_level_verified.csv \
  --psm-fold results/hades_results/rsn_cleaned_20260708/fold_level.csv \
  --output-dir results/hades_results/rsn_baseline_full_20260711/aggregated
```
