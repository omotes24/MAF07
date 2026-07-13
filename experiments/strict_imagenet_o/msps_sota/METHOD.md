# Reliability-Calibrated Multi-Stage Prototype Support

## 1. Scope

RC-MSPS is a score-only, Strict Fair Inductive OOD detector. It does not update the frozen ResNet50d backbone or classifier. Fitting consumes ImageNet train features and ID-only held-out features. Evaluation consumes one query and fixed compressed ID statistics; queries never interact.

## 2. Frozen representation

For stage `l in {1,2,3,4}`, let `z_l(x)` be the global-average-pooled output of ResNet50d `layer1` through `layer4`, with dimensions `(256, 512, 1024, 2048)`. Define

```math
u_l(x) = z_l(x) / ||z_l(x)||_2.
```

For class `c`, 150 deterministic ImageNet train samples form the normalized mean prototype

```math
p_{l,c} = normalize((1/N_c) sum_{i:y_i=c} u_l(x_i)).
```

The raw support is cosine similarity

```math
a_{l,c}(x) = u_l(x)^T p_{l,c}.
```

All prototypes are unit norm to `1e-5` tolerance.

## 3. ID-only calibration

For every stage and class, 50 train images not used by the final prototype provide an own-class reference distribution. For support `a`, the smoothed empirical CDF is

```math
F(a; R) = (1 + sum_{r in R} 1[r <= a]) / (|R| + 1).
```

Let `F_{l,c}` be the class reference and `F_{l,g}` the pooled global stage reference. The locked class-CDF weight is `alpha=0.1`:

```math
q_{l,c}(x) = alpha F_{l,c}(a_{l,c}(x))
             + (1-alpha) F_{l,g}(a_{l,c}(x)).
```

Thus the final calibration is 10% class-specific and 90% global. This shrinkage is important because each final class reference has only 50 observations.

## 4. Candidate verification

The deepest stage proposes the top `K=5` classes:

```math
C_K(x) = TopK_c a_{4,c}(x).
```

The other stages only verify these same classes. This avoids independently maximizing over 1,000 classes at every stage.

The locked ID-derived layer weights are

```text
w = (0.0152057, 0.0285255, 0.0997000, 0.8565688).
```

They are proportional to each stage's held-out-class AUROC above chance, then normalized. No real OOD result is used.

For each candidate, calibrated supports are fused by a weighted harmonic mean:

```math
H_c(x) = (sum_l w_l / max(q_{l,c}(x), 1e-8))^{-1}.
```

Let `c_l(x)=argmax_j a_{l,j}(x)`. The unweighted stage-consensus fraction is

```math
C_c(x) = (1/4) sum_l 1[c_l(x)=c].
```

The final locked confidence is

```math
S_ID(x) = max_{c in C_K(x)} [log H_c(x) + 0.5 C_c(x)],
S_OOD(x) = -S_ID(x).
```

The searched variance penalty and diagonal residual have locked coefficients zero. They are not part of the final method.

## 5. ID-only model selection

The 200 cached ImageNet train examples per class are path-sorted and split deterministically:

| Role | Images/class | Used for |
|---|---:|---|
| Prototype | 150 | Prototype and optional diagonal statistics |
| Calibration | 25 | Search-time CDF references |
| Proxy query | 25 | Five-fold class-holdout pseudo-ID/OOD metrics |

For fold `f`, classes satisfying `class_id mod 5 == f` are pseudo-OOD and removed from the candidate prototype bank. The remaining classes are pseudo-ID. All 200 held-out classes appear exactly once as pseudo-OOD. This proxy is only a selection device: the backbone was pretrained on all 1,000 classes, so the proxy metrics are not final OOD performance estimates.

The staged search evaluates 80 candidate records and 395 fold evaluations. It uses this order, with AUROC differences below `0.001` treated as ties:

1. maximize mean five-fold pseudo-AUROC;
2. maximize worst-fold pseudo-AUROC;
3. minimize mean pseudo-FPR95;
4. minimize hyperparameter complexity;
5. use lexicographic serialized configuration as the deterministic final tie-break.

No real OOD cache path is accepted by the search entry point. Failed variants remain in `results/id_proxy_search.csv` and `results/id_proxy_ablation.csv`.

After selection, the 25 calibration and 25 proxy samples are combined into the final 50-per-class CDF set. This is valid because proxy labels are no longer used after the configuration is fixed.

## 6. Strict Fair Inductive data flow

```text
ImageNet train only
  -> frozen ResNet50d stage features
  -> prototypes + ID CDFs + fixed layer weights
  -> immutable locked state

one test image
  -> same frozen ResNet50d and preprocessing
  -> support against locked prototypes/CDFs
  -> one scalar OOD score
```

The implementation has no API for an OOD validation set during search, no trainable parameter update, no test-batch normalization, no test-time adaptation, and no test-to-test nearest-neighbor operation. Permuting or splitting a test set cannot change an image's score.

## 7. Storage and complexity

The final state stores 4,000 prototypes plus empirical CDF references. It does not retain the 200,000-image feature bank for inference. For one query, prototype similarity costs `O(sum_l D_l C)`, while cross-stage calibration and fusion operate only on `K=5` candidates. The serialized state is 31 MiB on Hades.

## 8. Rejected extensions

The ID-only proxy selected normalized mean over 5% trimmed mean and coordinate-wise Huber centers. Diagonal residual-only and cosine-plus-residual variants were also worse. The final method therefore remains a calibrated prototype detector rather than adding unsupported covariance complexity.

## 9. Known limitation

The ID-only proxy strongly emphasizes the deepest stage, and its gains do not transfer uniformly. Final locked evaluation improves iNaturalist and OpenImage-O but degrades ImageNet-O and Texture. The method is consequently supported as a comparison-set-best macro detector under this exact protocol, not as a detector that dominates every OOD shift.
