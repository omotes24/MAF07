# SOTA Eligibility Audit

## Fixed Protocol

- Backbone: frozen `timm/resnet50d.ra2_in1k`.
- Weight SHA-256: `9c170e1af55e7ca051be5f147e86b7ddcf2e9ec2b4836ffe85e0bb940ec62827`.
- Input: direct bilinear resize to `224x224`, then ImageNet normalization.
- ID train statistics: at most the fixed 200-images-per-class ImageNet mirror.
- ID test: official ImageNet validation, 50,000 images.
- Legacy OOD: Texture 5,160; iNaturalist 10,000; OpenImage-O 17,632; ImageNet-O 2,000.
- No fine-tuning, OOD validation, test-time adaptation, or test-batch dependency.

## Current Same-Condition Reproductions

| Method | Extra ID statistics | Additional training | Inductive | Macro AUROC | Macro FPR95 | Eligible |
|---|---|---|---|---:|---:|---|
| RC-MSPS | class score distributions | No | Yes | 0.887636 | 0.434417 | Yes, current provisional SOTA |
| MSPS | class prototypes | No | Yes | 0.881466 | 0.485732 | Yes |
| NNGuide | deterministic 10k energy-weighted ID bank | No | Yes | 0.853549 | 0.546116 | Yes; official formula, below SOTA |
| ViM | covariance and classifier | No | Yes | 0.836215 | 0.688679 | Yes |
| KNN-k50 | 200k feature bank | No | Yes | 0.833174 | 0.654297 | Eligible comparison, violates final compact-bank preference |
| Mahalanobis | class means/shared covariance | No | Yes | 0.797049 | 0.805085 | Yes |
| MSP | None | No | Yes | 0.779758 | 0.680140 | Yes |
| MaxLogit | None | No | Yes | 0.753500 | 0.694002 | Yes |
| Energy | None | No | Yes | 0.710313 | 0.784704 | Yes |
| ASH-S | None | No | Yes | 0.632004 | 0.829121 | Yes |
| ReAct | ID-train clipping quantile | No | Yes | 0.625164 | 0.923555 | Yes |

## External Methods Under Audit

| Method/source | Published setup | Mismatch or action | Eligible status |
|---|---|---|---|
| NNGuide, ICCV 2023, `roomo7time/nnguide` commit `c123cac...` | SupCon ResNet50 checkpoint; ID bank; energy guidance | Published number is ineligible because checkpoint differs. Official formula was rerun with the fixed checkpoint, deterministic 10k ID bank, and `k=10`. | Local reproduction eligible; published number reference only |
| ViM, CVPR 2022 official repository | ViM checkpoint/data pipeline | Published Table 6 is not byte-identical to this model and preprocessing. | Reference only; local reproduction is eligible |
| OpenOOD v1.5 | torchvision ResNet50 or BiT checkpoints depending configuration | Different checkpoint and often center-crop preprocessing. Use implementations as references, not their numbers. | Reference only unless rerun locally |
| NINCO paper model table | Multiple architectures and checkpoints | Different models; useful only for final-benchmark context. | Reference only |
| Training-based methods and outlier exposure | Updated weights or auxiliary OOD data | Violates frozen-backbone/no-OOD-training constraints. | Ineligible |
| Test-time adaptation or transductive methods | Shared test-batch statistics or adaptation | Violates per-image independent inference. | Ineligible |

## Sources

- OpenOOD official repository: https://github.com/Jingkang50/OpenOOD
- OpenOOD v1.5 method/benchmark overview: https://github.com/Jingkang50/OpenOOD/wiki/OpenOOD-v1.5-methods-%26-benchmarks-overview
- NNGuide official repository: https://github.com/roomo7time/nnguide
- NNGuide paper: https://openaccess.thecvf.com/content/ICCV2023/html/Park_Nearest_Neighbor_Guidance_for_Out-of-Distribution_Detection_ICCV_2023_paper.html
- NINCO official repository: https://github.com/j-cb/NINCO
- NINCO paper: https://proceedings.mlr.press/v202/bitterwolf23a.html

The closest published nearest-neighbor alternative has now been rerun under the exact fixed model and cache. RC-MSPS remains the eligible same-condition SOTA; any further external method must meet the same protocol before its number can displace it.
