# RSN: Robust Standardized-Neighbor Distance

RSN is the proposed OOD detector in this repository. The implementation audit
performed in July 2026 found that the PDF method description and the configuration
that produced the archived result table were different. The distinction is now
explicit and the two profiles are rerun side by side.

## Score

Let `h(x)` be the frozen DINOv2 CLS feature and let `B_c` be the ID training
feature bank for class `c`. For the paper profile, use `q(x) = h(x) / ||h(x)||`.
Estimate a diagonal class scale from the corresponding training bank:

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

Higher `s_RSN` is more ID-like. The paper profile uses `delta=1.345`, `k=10`,
L2-normalized features, and the top three classes from the shared ridge linear
probe as candidates. No empirical quantile calibration is used.

## Reproducibility profiles

| profile | k | L2 feature normalization | candidate classes | purpose |
|---|---:|---|---|---|
| `paper` | 10 | yes | top-3 ridge logits | Matches `RSN_preview_2.pdf` |
| `reported_20260708` | 150 | no | all ID classes | Reproduces the archived 2026-07-08 run |

`rsn_from_env()` defaults to `paper`. Reproduce the archived implementation only
by setting:

```bash
export MAF07_RSN_PROFILE=reported_20260708
export MAF07_RSN_K=150
export MAF07_RSN_NORMALIZE=0
export MAF07_RSN_USE_TOPQ=0
```

The archived values under `docs/results/rsn_cleaned_20260708/` must not be mixed
with the paper-profile rerun. They remain in Git only as provenance for the PDF
draft and are not verified final results.

## Baseline status

The old full-ranking table used several proxy or incorrect implementations. It
is superseded by the audit in [BASELINE_AUDIT.md](BASELINE_AUDIT.md). In
particular, the old table is not evidence for claims about Mahalanobis++, ViM,
ReAct, DICE, SCALE, NCI, GradNorm, OpenMax, ODIN, MCM, or Tip-Adapter.

The verified rerun entry point is:

```bash
bash scripts/run_verified_baseline_audit_hades.sh
```

It writes fold-level score fingerprints as well as metrics. This distinguishes
true score identity from equal AUROC/FPR95 caused by monotone transforms.
