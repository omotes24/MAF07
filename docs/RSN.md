# RSN: Robust Scale-Normalized kNN

RSN is the main proposed OOD detector in this repository. It is the promoted
name for the legacy result label `diagcard_huber_raw`.

## Method

Let `z(x) in R^D` be the frozen backbone feature for image `x`. For each ID
class `c`, estimate a diagonal scale vector from the ID training features:

```text
sigma_c = std({z_i : y_i = c})
```

with elementwise clamping by `min_std=1e-3`. For a query feature `z`, compute
class-conditional scaled differences against the class bank:

```text
u_i^c(z) = (z - z_i^c) / sigma_c
```

and retrieve the `k=150` nearest class-bank neighbors in the scaled space. RSN
uses a Huberized residual:

```text
psi_delta(t) = t^2                 if |t| <= delta
             = 2 * delta * |t| - delta^2  otherwise
```

with `delta=1.345`. The class distance is:

```text
d_c^RSN(z) = (1 / k) * sum_{i in NN_k^c(z)} sum_l psi_delta(u_i^c(z)_l)
```

The ID score used by the metric code is:

```text
s_RSN(z) = - min_c d_c^RSN(z)
```

Equivalently, the OOD score is `min_c d_c^RSN(z)`. Higher `s_RSN` means more
ID-like.

## Relation to CARD and the legacy name

CARD/CQS uses per-class kNN distances followed by class-wise empirical quantile
calibration. RSN keeps the class-conditional kNN architecture but changes the
metric:

- class-wise diagonal scaling removes most class-dependent distance-scale
  mismatch inside the distance itself;
- Huberization suppresses single-coordinate accidental outliers;
- empirical quantile calibration is not used.

The experimental variant previously called `diagcard_huber_raw` is exactly RSN:

```text
diagcard_huber_raw == RSN
```

The older `diag_raw` variant is the same class-wise diagonal scaling without
Huberization. The older `diag_calib` and `diag_huber_calib` variants include
CARD-style empirical quantile calibration.

## Calibration redundancy intuition

Assume class features can be approximated by

```text
z_c = mu_c + sigma_c o u
```

where `o` denotes elementwise multiplication and the latent residual `u` has a
class-independent distribution. After diagonal scaling by `sigma_c`, the ID
distance distribution of `d_c^RSN` becomes approximately class-independent.

If all classes share the same limiting distance CDF `F`, a calibration map
`q_c(z) = 1 - F_c(d_c(z))` becomes approximately:

```text
q_c(z) ~= 1 - F(d_c(z))
```

which is a shared monotone transform. Therefore:

```text
arg min_c d_c(z) == arg max_c q_c(z)
```

and ranking by raw distance is preserved. Empirical calibration replaces `F`
with `F_hat_c`, adding finite-validation noise of order
`O_p(sqrt(1 / m_c))`. Once diagonal scaling removes the dominant class-scale
shift, this finite-sample noise can dominate the remaining benefit of
calibration. This matches the observed result: raw RSN variants outperform the
calibrated variants.

## Main DINOv2 results

The completed full sweep uses `dinov2_vitb14` and `dinov2_vitl14`, seeds
`0,1,2`, protocols `fair` and `oracle`, and ID sizes `2..7`. The trial count is
`6 * C(8, k)` per ID size across the two DINOv2 backbones and three seeds.

Fair protocol, pooled over the two DINOv2 backbones:

| id_size | n | RSN AUROC | RSN FPR95 | RSN AUPR_OUT | AUROC vs KNN | FPR95 vs KNN | AUROC vs CARD | FPR95 vs CARD |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 168 | 0.888020 | 0.527700 | 0.944890 | +0.005021 | -0.028476 | +0.010547 | -0.013240 |
| 3 | 336 | 0.865054 | 0.603194 | 0.887647 | +0.004508 | -0.027255 | +0.014811 | -0.046004 |
| 4 | 420 | 0.846675 | 0.653460 | 0.807276 | +0.004302 | -0.024446 | +0.017366 | -0.065790 |
| 5 | 336 | 0.830355 | 0.694429 | 0.696770 | +0.004054 | -0.023011 | +0.018530 | -0.072362 |
| 6 | 168 | 0.813942 | 0.732347 | 0.547123 | +0.003624 | -0.018921 | +0.018480 | -0.070645 |
| 7 | 48 | 0.792711 | 0.776005 | 0.348381 | +0.002910 | -0.013564 | +0.016266 | -0.056380 |

Oracle protocol produced the same RSN aggregate values. Existing KNN-oracle
baseline rows are incomplete in the archived baseline CSV, so the strongest
KNN comparison should use the complete fair protocol unless KNN-oracle is
rerun.

## LAR status

LAR is not a shipped baseline from the original method set. It is an additional
experimental method that was added during exploration. It does not show an
obvious evaluation-data leakage in code inspection, but its archived full sweep
is incomplete after `id_size=3`. For paper tables, report LAR separately or
exclude it from the original-baseline table.
