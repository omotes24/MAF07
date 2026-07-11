# RSN: Robust Standardized-Neighbor Distance

RSN is the proposed OOD detector in this repository. The primary configuration
is the raw-feature setting that was previously called `reported_20260708`: raw
DINOv2 CLS features, `k=150`, every ID class as a candidate, Huber aggregation,
and no empirical calibration. The revised paper and the default runner now use
this same configuration.

## Score

Let `h(x)` be the frozen DINOv2 CLS feature and let `B_c` be the ID training
feature bank for class `c`. The primary profile uses `q(x) = h(x)` without L2
normalization. Estimate a diagonal class scale from the corresponding training
bank:

```text
sigma_c = std({q_i : y_i = c})
```

For each candidate class, retrieve the nearest neighbors in the standardized
space and aggregate the coordinate residuals with the Huber function:

```text
u_i^c(q) = (q - q_i^c) / sigma_c
d_c(q) = (1 / k) * sum_i sum_j psi_delta(u_i^c(q)_j)
s_RSN(q) = -min_c d_c(q)
```

Higher `s_RSN` is more ID-like. The primary profile uses `delta=1.345`, `k=150`,
raw features, and every ID class as a candidate. No empirical quantile
calibration is used.

## Reproducibility profiles

| profile | k | L2 feature normalization | candidate classes | purpose |
|---|---:|---|---|---|
| `primary` / `reported_20260708` | 150 | no | all ID classes | Revised-paper main setting |
| `sensitivity` / `paper` | 10 | yes | top-3 ridge logits | Historical draft sensitivity setting |

`rsn_from_env()` and `scripts/run_rsn_full.sh` default to `primary`. Select the
historical draft sensitivity setting only by setting:

```bash
export MAF07_RSN_PROFILE=sensitivity
export MAF07_RSN_K=10
export MAF07_RSN_NORMALIZE=1
export MAF07_RSN_USE_TOPQ=1
```

The old aggregate values under `docs/results/rsn_cleaned_20260708/` remain for
provenance. Use the re-audited full sweep under
`docs/results/rsn_baseline_full_20260711/` for final claims.

## Baseline status

The old full-ranking table used several proxy or incorrect implementations. It
is superseded by the audit in [BASELINE_AUDIT.md](BASELINE_AUDIT.md). In
particular, the old table is not evidence for claims about Mahalanobis++, ViM,
ReAct, DICE, SCALE, NCI, GradNorm, OpenMax, ODIN, MCM, or Tip-Adapter.

The verified all-size rerun entry point is:

```bash
bash scripts/run_verified_baseline_sweep_hades.sh
```

It covers `m=2,3,4,5,6,7`, writes fold-level score fingerprints as well as
metrics, and audits the exact 32,472-job coverage. This distinguishes true score
identity from equal AUROC/FPR95 caused by monotone transforms.
