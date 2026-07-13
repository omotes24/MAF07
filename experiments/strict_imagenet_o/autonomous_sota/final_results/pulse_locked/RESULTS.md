# PULSE final locked evaluation

## Outcome

**same-condition eligible SOTA achieved**

- Locked configuration SHA-256: `f43edfd5e1770c61ca2810dac167c0025fe24155e27fc630b8a096de42c6d977`
- Code commit: `f18785ecfcc09e64b91ce2b23afb1aeb9ec79500`
- Final suite was untouched until this lock: yes
- Target OOD used for fitting, calibration, or selection: no
- Test-image sharing: no
- Same score equation and settings for every dataset: yes

## Legacy gate

PULSE passed the preregistered legacy gate before final access: Macro AUROC
0.911426, Macro FPR95
0.399970, ImageNet-O AUROC
0.821093. The paired legacy AUROC and
FPR95 confidence intervals excluded zero in the favorable direction.

## Untouched final suite

| method   | dataset   |   n_id |   n_ood |    AUROC |    FPR95 |   AUPR_OUT |
|:---------|:----------|-------:|--------:|---------:|---------:|-----------:|
| PULSE    | NINCO     |  50000 |    5879 | 0.836081 | 0.666950 |   0.371468 |
| RC-MSPS  | NINCO     |  50000 |    5879 | 0.792441 | 0.736520 |   0.310765 |
| PULSE    | SSB-hard  |  50000 |   49000 | 0.610486 | 0.884653 |   0.601776 |
| RC-MSPS  | SSB-hard  |  50000 |   49000 | 0.581932 | 0.900551 |   0.578357 |

| method   |   Macro_AUROC |   Macro_FPR95 |   Macro_AUPR_OUT |
|:---------|--------------:|--------------:|-----------------:|
| PULSE    |      0.723284 |      0.775802 |         0.486622 |
| RC-MSPS  |      0.687187 |      0.818535 |         0.444561 |

| comparison          |   Macro_AUROC_diff |   Macro_FPR95_diff |   Macro_AUPR_OUT_diff |
|:--------------------|-------------------:|-------------------:|----------------------:|
| PULSE_minus_RC-MSPS |           0.036097 |          -0.042734 |              0.042061 |

| method   |   Worst_Dataset_AUROC | Worst_Dataset   |
|:---------|----------------------:|:----------------|
| PULSE    |              0.610486 | SSB-hard        |
| RC-MSPS  |              0.581932 | SSB-hard        |

## Paired bootstrap

| dataset   | metric   |   iterations |   pulse_mean |   rc_msps_mean |   mean_diff |    ci_low |   ci_high |   good_rate |
|:----------|:---------|-------------:|-------------:|---------------:|------------:|----------:|----------:|------------:|
| macro     | AUROC    |         2000 |     0.723260 |       0.687158 |    0.036102 |  0.033709 |  0.038405 |    1.000000 |
| macro     | FPR95    |         2000 |     0.775594 |       0.818379 |   -0.042785 | -0.051957 | -0.033954 |    1.000000 |
| macro     | AUPR_OUT |         2000 |     0.486798 |       0.444753 |    0.042045 |  0.036883 |  0.047003 |    1.000000 |

The final comparison used 2000 paired,
stratified image-level bootstrap draws with seed
20260713.

## Runtime and storage

| dataset   |   images |   parallel_wall_seconds |   aggregate_gpu_process_seconds |   images_per_wall_second |
|:----------|---------:|------------------------:|--------------------------------:|-------------------------:|
| NINCO     |     5879 |               56.274583 |                      212.655894 |               104.469900 |
| SSB-hard  |    49000 |              192.838718 |                      758.332729 |               254.098350 |

| method   |   compact_index_bytes |   final_prototypes_bytes |   additional_persistent_bytes | raw_train_feature_bank_required   | classifier_prediction_changed   |   id_top1_accuracy_delta |
|:---------|----------------------:|-------------------------:|------------------------------:|:----------------------------------|:--------------------------------|-------------------------:|
| PULSE    |              71690932 |                  8192000 |                      79885074 | False                             | False                           |                 0.000000 |

PULSE is a score-only postprocessor and does not alter the frozen classifier's
prediction, so the ImageNet top-1 accuracy delta is exactly 0. The persistent
support is the compact IVF-PQ index plus 1,000 final-stage class prototypes; no
raw train feature bank is retained.

## Error analysis

`error_analysis.csv` records false-accept and false-reject counts at 95% ID
acceptance. `failure_cases.csv` records the 50 strongest cases per method and
split with immutable relative image paths. These files were generated only
after the method and final suite were locked and were never used to tune PULSE.
