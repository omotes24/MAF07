# DINOv2 Baseline Implementation Audit

Audit target: the fair-protocol DINOv2 ranking for every ID-set size
`m=2,3,4,5,6,7`, together with the code that generated the original `m=2`
table in `docs/results/rsn_cleaned_20260708/C_full_ranking.csv`.

## Conclusion

The archived full-ranking table is invalid as a paper-faithful baseline table.
Some equal results are mathematically expected for two-class ID heads, but the
audit also found implementation errors and methods that cannot be computed from
cached DINOv2 features under their original names.

Do not overwrite the archived CSV. The corrected full rerun is stored in
`docs/results/rsn_baseline_full_20260711/` and uses
`implementation_version=verified_v1`.

## Findings and actions

| method | archived implementation | audit result | action |
|---|---|---|---|
| MSP | max softmax of a ridge head | valid after the head protocol is disclosed | keep |
| Entropy | negative Shannon entropy | valid | keep; equality with MSP at `m=2` can be rank-equivalence |
| Energy | log-sum-exp of ridge logits | valid | keep |
| MaxLogit | maximum ridge logit | valid | keep |
| GEN | omitted the `(1-p)^gamma` term and used a different exponent | incorrect | replaced with official `gamma=0.1`, top-`M=100` formula |
| GradNorm | used distance from the predicted one-hot distribution and omitted the feature outer product | incorrect | replaced with exact final-layer gradient L1 norm |
| ODIN | temperature scaling only; no input perturbation | not ODIN | excluded from the DINO feature-cache table |
| Mahalanobis | used only the diagonal of the precision matrix because of `einsum(...,dd,...)`; covariance included between-class scatter | incorrect | full quadratic form and tied within-class empirical covariance |
| `mah_mindist` | exact alias of Mahalanobis | duplicate name | removed as a separate row |
| Mahalanobis++ | z-standardized the final Mahalanobis score | incorrect and necessarily rank-identical | replaced with L2 feature normalization before fitting/scoring |
| RMD | reused one covariance for class and background terms and had the same diagonal-only bug | incorrect | separate class-conditional and background covariance estimates |
| KNN | normalized cosine-neighbor distance | rank-equivalent to normalized Euclidean KNN | uses normalized Euclidean distance explicitly |
| ViM | omitted the classifier origin and alpha matching; centered test features by their own mean | incorrect | official origin, principal/null space, alpha, and virtual-logit score |
| ReAct | used per-dimension clipping and class prototypes | incorrect | scalar ID-train quantile, shared ridge head, energy score |
| ASH-P/B/S | selected by absolute magnitude and used non-reference redistribution | incorrect | reference top-value pruning/binarization/sharpening |
| DICE | masked feature dimensions by training variance | incorrect | masks classifier weights by mean-activation contribution |
| SCALE | multiplied MSP by feature norm | incorrect | reference activation scaling and ridge-head energy |
| NCI | used a margin/concentration heuristic | incorrect | reference weight-direction and feature-norm score |
| OpenMax | subtracted a Mahalanobis proxy; no EVT model | incorrect | class MAVs, tail Weibull fits, OpenMax unknown probability |
| KL-Matching | nearest class probability template by KL | consistent with the reference score | keep |
| MCM / CLIP-Zeroshot-MSP | both called the same function on DINO class means | mislabelled; MCM and zero-shot MSP are the same CLIP score by definition, but no CLIP encoder was used | exclude from DINO table |
| CLIP-Text-Energy | used DINO features and DINO class means as text features | not a CLIP method | exclude from DINO table |
| Tip-Adapter | used DINO features and class means without a CLIP text encoder | not Tip-Adapter | exclude from DINO table |

## Why several `m=2` metrics can still match

For two classes, MSP is a strictly monotone function of the absolute logit
margin. Negative entropy and the two-class GEN score are also monotone functions
of that margin. Temperature-only MSP has the same ordering. Therefore these
methods can have identical AUROC, FPR95, and AUPR even when their raw score
vectors are different.

The corrected runner stores SHA-256 fingerprints of raw score vectors. Its
`equivalence_audit.csv` reports raw-score identity separately from metric
identity.

MCM and CLIP zero-shot MSP are not independent algorithms when both mean the
maximum softmax over the same CLIP image-text similarities. One row is enough in
a genuine CLIP experiment.

## Paper/implementation alignment

`docs/paper/RSN_revised.tex` now defines the primary RSN configuration as raw
DINOv2 features, `k=150`, all ID classes, Huber `delta=1.345`, and no empirical
calibration. This matches the `rsn_reported` implementation. The old
L2-normalized, `k=10`, top-3 profile remains only as a sensitivity condition.

The paper also documents the deterministic closed-form ridge linear probe used
by the verified head-based baselines. Its full ranking includes only methods
that can be implemented faithfully from the cached DINOv2 features and the
shared ridge head; unsupported ODIN and CLIP-dependent rows are excluded.

## Primary references used for verification

- [Mahalanobis++](https://openreview.net/pdf?id=vutMcZl50l)
- [GradNorm](https://papers.nips.cc/paper/2021/file/063e26c670d07bb7c4d30e6fc69fe056-Paper.pdf)
- [GEN](https://openaccess.thecvf.com/content/CVPR2023/html/Liu_GEN_Pushing_the_Limits_of_Softmax-Based_Out-of-Distribution_Detection_CVPR_2023_paper.html)
- [ViM](https://ooddetection.github.io/)
- [ReAct](https://openreview.net/forum?id=IBVBtz_sRSm)
- [DICE](https://arxiv.org/abs/2111.09805)
- [ASH](https://andrijazz.github.io/ash/)
- [SCALE](https://openreview.net/pdf?id=RDSTjtnqCg)
- [NCI](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_Detecting_Out-of-Distribution_Through_the_Lens_of_Neural_Collapse_CVPR_2025_paper.html)
- [ODIN](https://openreview.net/pdf?id=H1VGkIxRZ)
- [OpenMax](https://www.cv-foundation.org/openaccess/content_cvpr_2016/app/S07-07.pdf)
- [MCM](https://proceedings.neurips.cc/paper_files/paper/2022/hash/e43a33994a28f746dcfd53eb51ed3c2d-Abstract-Conference.html)

## Rerun

The verified Hades sweep uses the cleaned 105,554-image manifest, fair protocol,
both DINOv2 backbones, seeds `0,1,2`, and every ID-set combination for
`m=2,3,4,5,6,7`:

```bash
bash scripts/run_verified_baseline_sweep_hades.sh
```

The completed coverage is 1,476 folds and 32,472 verified jobs. The coverage
audit reports no missing, duplicate, extra, or non-finite jobs. Outputs:

```text
docs/results/rsn_baseline_full_20260711/fold_level_verified.csv
docs/results/rsn_baseline_full_20260711/summary_verified.csv
docs/results/rsn_baseline_full_20260711/equivalence_audit.csv
docs/results/rsn_baseline_full_20260711/coverage_report.json
docs/results/rsn_baseline_full_20260711/aggregated/metric_ranking_by_id_size.csv
```
