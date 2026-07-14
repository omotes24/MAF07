# PULSE

PULSE is the primary OOD method in MAF07. It is a score-only postprocessor for
the frozen `timm/resnet50d.ra2_in1k` checkpoint. It uses four independently
ID-standardized evidence sources, fixed unit weights, no fine-tuning, no
target-OOD fit or calibration, and no sharing between test images.

## Scope of the claim

The eligible-SOTA claim is deliberately protocol-specific: every comparator is
rerun with the same ResNet50d weights, direct `224x224` bilinear resize,
ImageNet normalization, ID split, and OOD split. Published values using another
checkpoint, preprocessing pipeline, outlier exposure, test-time adaptation, or
transductive test statistics are reference values, not eligible comparisons.

| Property | Locked PULSE setting |
|---|---|
| Backbone | frozen `timm/resnet50d.ra2_in1k` |
| Model-state SHA-256 | `6f0b5ee0bea90806aa619a32ccea114c80b528965fecf0d07a31f6ac3f5b9e9b` |
| Fit data | ID train for prototypes/index; independent ID validation for scalar normalization |
| OOD fit or calibration | none |
| Test-image sharing | none |
| Component weights | four fixed unit weights; no learned fusion |
| Free hyperparameters | perturb fraction `0.05`, perturb magnitude `0.5`, KNN `k=200` |
| Score direction | larger means more OOD-like |

## Score

For an ID-validation atom `a`, PULSE freezes the robust standardization

```text
z_ID(a) = (a - median_ID(a)) / max(1.4826 * MAD_ID(a), 1e-6).
```

For a query image `x`, the final score is

```text
S_PULSE(x) = ReLU(z_shift(x))
           + ReLU(z_uniform(x))
           + z_localized(x)
           + z_knn(x).
```

The two ReLUs prevent unusually ID-like shift or uniformity values from
cancelling positive OOD evidence. Localized and KNN evidence remains signed.

### 1. Activation shift

PULSE differentiates the predicted logit with respect to the current image,
selects the 5% input coordinates with the smallest absolute gradient, and
perturbs only those coordinates by `0.5 * sign(gradient)`. Let `h` and `h'` be
the original and perturbed final pooled features. Over the 5% coordinates with
the largest original activation, it computes

```text
r_shift = sum |h - h'| / max(sum ReLU(h), 1e-12),
z_shift = z_ID(log1p(r_shift)).
```

This is own-image stability evidence: no other test image enters the score.

### 2. Class-center uniformity

Let `p_c` be each of the 1,000 train-only class prototypes and
`s_c = cosine(h, p_c)`. PULSE computes

```text
u = mean_c(s_c) / sqrt(mean_c(s_c^2)),
z_uniform = z_ID(u).
```

A query with similarly distributed support across many classes has a larger
uniformity atom than a query concentrated around a known class center.

### 3. Localized prototype support

Four deterministic views are generated from the query's own frozen activation
maps: `cam_mass50`, `energy_mass50`, `energy_mass80`, and `energy_peak3`.
Their features are compared with all class prototypes:

```text
l = max over views v and classes c of cosine(h_v, p_c),
z_localized = z_ID(-l).
```

Weak support even in the most favorable object-focused crop is OOD evidence.

### 4. Compact multiview KNN support

PULSE averages final features from `full`, `full_hflip`, `zoom90`,
`zoom90_hflip`, and `zoom80`, then queries a train-only cosine FAISS
`IndexIVFPQ`. With `k=200` and `nprobe=64`,

```text
q = mean cosine similarity of the top-200 approximate neighbors,
z_knn = z_ID(-q).
```

The serialized index is 71,690,932 bytes and replaces the raw train-feature
bank.

## Frozen scalar state

The following `(center, scale)` values are stored in
`locked_models/cycle87_pulse/state.npz`. Localized and KNN rows apply to the
negative confidence atoms shown above.

| Atom | Center | Scale |
|---|---:|---:|
| `log1p(shift_ratio)` | 0.209302842 | 0.050444290 |
| `uniform_cosine` | 0.986108750 | 0.003290186 |
| `-localized_support` | -0.722089082 | 0.100534663 |
| `-knn_support` | -0.570493937 | 0.087235133 |

## Implementation map

| File | Responsibility |
|---|---|
| `methods/pulse.py` | frozen state, four standardized components, final sum |
| `methods/activation_shift.py` | low-gradient perturbation and shift ratio |
| `methods/bilateral_distribution.py` | class-center uniformity |
| `methods/localized_views_v2.py` | deterministic CAM/energy crops |
| `methods/compact_knn.py` | IVF-PQ construction and top-k support |
| `extract_final_locked.py` | aligned per-image extraction of all four atoms |
| `final_lock.py` | config, code, model, manifest, and artifact hash verification |
| `run_final_locked.py` | immutable four-GPU final launcher |
| `evaluate_final_locked.py` | metrics and post-lock error analysis |

The executable aggregation is intentionally short:

```python
result = {
    "shift": np.maximum((np.log1p(shift) - state.shift_center) / state.shift_scale, 0.0),
    "uniform": np.maximum((uniform - state.uniform_center) / state.uniform_scale, 0.0),
    "localized": (-localized - state.localized_center) / state.localized_scale,
    "knn": (-knn - state.knn_center) / state.knn_scale,
}
pulse_ood_score = sum(result.values())
```

## Legacy development benchmark

These four datasets were used during research and are reported separately from
the untouched final suite.

| Dataset | PULSE AUROC | RC AUROC | PULSE FPR95 | RC FPR95 | PULSE AUPR | RC AUPR |
|---|---:|---:|---:|---:|---:|---:|
| Texture | 0.961150 | 0.960210 | 0.166473 | 0.198062 | 0.819861 | 0.806981 |
| iNaturalist | 0.954128 | 0.952078 | 0.248600 | 0.250400 | 0.752528 | 0.814861 |
| OpenImage-O | 0.909334 | 0.867307 | 0.474308 | 0.556205 | 0.766402 | 0.708284 |
| ImageNet-O | 0.821093 | 0.770950 | 0.710500 | 0.733000 | 0.165644 | 0.146966 |
| **Macro** | **0.911426** | 0.887636 | **0.399970** | 0.434417 | **0.626109** | 0.619273 |

The 1,000-draw paired macro differences versus RC-MSPS are AUROC `+0.023783`
with 95% CI `[+0.021864,+0.025760]`, FPR95 `-0.033918` with CI
`[-0.044841,-0.023158]`, and AUPR-OUT `+0.006904` with CI
`[+0.000812,+0.013150]`.

## Preregistered untouched final suite

The code and configuration were locked before any final image was decoded.
NINCO and SSB-hard were then evaluated once with the same settings.

| Dataset | Method | AUROC | FPR95 | AUPR-OUT |
|---|---|---:|---:|---:|
| NINCO (5,879 OOD) | PULSE | **0.836081** | **0.666950** | **0.371468** |
|  | RC-MSPS | 0.792441 | 0.736520 | 0.310765 |
| SSB-hard (49,000 OOD) | PULSE | **0.610486** | **0.884653** | **0.601776** |
|  | RC-MSPS | 0.581932 | 0.900551 | 0.578357 |
| **Macro** | **PULSE** | **0.723284** | **0.775802** | **0.486622** |
|  | RC-MSPS | 0.687187 | 0.818535 | 0.444561 |

The 2,000-draw paired macro differences are:

| Metric | Mean difference | 95% CI | Favorable draws |
|---|---:|---:|---:|
| AUROC | +0.036102 | `[+0.033709,+0.038405]` | 1.000 |
| FPR95 | -0.042785 | `[-0.051957,-0.033954]` | 1.000 |
| AUPR-OUT | +0.042045 | `[+0.036883,+0.047003]` | 1.000 |

## Component ablation

Every leave-one-component-out variant lowers legacy Macro AUROC and worsens
Macro FPR95.

| Variant | Macro AUROC | AUROC change | Macro FPR95 | FPR95 change |
|---|---:|---:|---:|---:|
| Full PULSE | **0.911426** | 0 | **0.399970** | 0 |
| minus activation shift | 0.903050 | -0.008377 | 0.415144 | +0.015174 |
| minus uniformity | 0.901461 | -0.009966 | 0.460360 | +0.060390 |
| minus localized support | 0.884268 | -0.027158 | 0.571392 | +0.171422 |
| minus compact KNN | 0.902425 | -0.009002 | 0.411956 | +0.011986 |

Localized support is the largest single contribution, but its standalone
AUROC is only `0.898731`; the full result requires complementary evidence.

## Cost and integrity

- Persistent method state: 79,885,074 bytes (71,690,932-byte IVF-PQ index,
  8,192,000-byte prototypes, and 2,142-byte scalar state).
- Raw train feature bank: not required at inference.
- Frozen classifier prediction: unchanged; ImageNet top-1 delta is exactly 0.
- Current extractor: approximately 12 forward-equivalent image views plus one
  input-gradient pass per query, batched and sharded over four GPUs.
- Final wall time: 56.27 seconds for NINCO and 192.84 seconds for SSB-hard.
- Integrity audit: all 54,879 final paths are unique and exactly match the
  preregistered manifest; all scores are finite; formula recomputation error is
  below `4.8e-7`.
- Verification: 22 targeted lock, leakage, formula, manifest, extraction, and
  summary tests pass after integration into MAF07.

## Reproduction and immutable record

```bash
cd experiments/strict_imagenet_o/autonomous_sota
python run_final_locked.py --locked-config configs/final_locked_config.json
```

The archived final runner refuses to overwrite existing score shards. The
immutable record is identified by:

- code commit: `f18785ecfcc09e64b91ce2b23afb1aeb9ec79500`
- MAF07 locked-tree import merge: `21b99d17b5a9d8a36cfc4a8c1dcbf33014473402`
- locked-config SHA-256: `f43edfd5e1770c61ca2810dac167c0025fe24155e27fc630b8a096de42c6d977`
- final-manifest SHA-256: `6dbd4c0a9b4bc9db40a0e430c82dd3c95e2dd82ae116f0761db57d6243c401e5`

Raw score shards, bootstrap draws, failure cases, runtime, and integrity output
are under `experiments/strict_imagenet_o/autonomous_sota/final_results/pulse_locked/`.

## Limitations

1. The claim is same-condition SOTA, not a claim over every published OOD
   checkpoint or training regime.
2. PULSE improves legacy Macro AUPR-OUT, but iNaturalist AUPR-OUT alone is lower
   than RC-MSPS (`0.752528` versus `0.814861`).
3. Absolute performance on SSB-hard remains difficult (`0.610486` AUROC and
   `0.884653` FPR95), despite a significant improvement over RC-MSPS.
4. The score equation is simple, but current inference is more expensive than
   a one-forward baseline because it uses a gradient perturbation, five global
   views, and four localized views.
