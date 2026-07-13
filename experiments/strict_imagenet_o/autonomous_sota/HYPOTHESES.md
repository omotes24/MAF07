# Hypothesis Registry

Variants that only change a coefficient do not receive a new hypothesis ID. Every entry below changes the scoring principle.

| Priority | ID | Principle | Main components | Expected cost | Status |
|---:|---|---|---|---|---|
| 1 | P01 | Multi-proxy validation | class holdout, interpolation, corruption, extrapolation ranks | cache-only | completed; cycle-specific proxy panels locked |
| 2 | B01 | Exact NNGuide reproduction | energy confidence times ID-bank cosine guidance | high bank memory | completed; below RC-MSPS |
| 3 | H04 | Compact centroid guidance | energy confidence times nearest class-centroid support | low | rejected by ID-only proxy |
| 4 | H05 | Multi-centroid guidance | nearest compressed within-class centroid | medium | rejected; proxy selected one centroid |
| 5 | H01 | Relative prototype evidence | nearest-class support minus global/background support | low | rejected by ID-only proxy |
| 6 | H10 | Logit-prototype relation residual | deviation from ID-only relation between logit and prototype evidence | low | rejected by ID-only proxy |
| 7 | H11 | Logit-feature class disagreement | class-matched logit and prototype quantiles | low | rejected by ID-only proxy |
| 8 | H02 | Class-conditional diagonal residual | nearest predicted-class standardized residual | low | rejected by ID-only proxy |
| 9 | H03 | Relative diagonal residual | class diagonal residual minus global diagonal residual | low | rejected by ID-only proxy |
| 10 | H06 | Multi-stage Cauchy tail | ID-calibrated stage tail probabilities with Cauchy aggregation | low | failed legacy full |
| 11 | H23 | Truncated stage-tail product | product of only strongly anomalous stage tails | low | failed legacy full |
| 12 | H24 | Robust stage-tail median | median calibrated anomaly across stages | low | failed legacy full |
| 13 | H08 | Cross-stage top-class agreement | stage top-1 vote concentration | low | rejected by ID-only proxy |
| 14 | H09 | Cross-stage reciprocal-rank stability | reciprocal rank of deep candidate in shallow stages | low | rejected by ID-only proxy |
| 15 | H12 | Confusion-normalized prototype margin | top-1/top-2 margin normalized by train-only class confusion | low | rejected by ID-only proxy |
| 16 | H13 | Class radial-shell deviation | two-sided deviation from class-conditional feature radius | low | rejected by ID-only proxy |
| 17 | H14 | Class low-rank subspace residual | residual outside compressed class tangent subspace | medium | rejected; positive ranks worsened every proxy summary |
| 18 | H15 | Global-to-class residual ratio | class residual divided by global residual | low | rejected by ID-only proxy |
| 19 | H16 | Fisher-scaled prototype evidence | prototype evidence after global diagonal whitening | low | rejected by ID-only proxy |
| 20 | H17 | Boundary-radius ratio | own-class radius relative to nearest rival-class boundary | low | rejected by ID-only proxy |
| 21 | H18 | Coreset/prototype disagreement | compressed local support versus prototype support | medium | tested as H58-H60; failed legacy gate |
| 22 | H19 | Deterministic same-class view mean | maximum class support after averaging five deterministic views | 5x inference | clean validation failed legacy gate |
| 23 | H20 | Deterministic view support floor | maximum same-class floor over five deterministic views | 5x inference | clean validation failed legacy gate |
| 24 | H21 | Conditional feature-norm residual | feature norm deviation conditioned on classifier confidence | low | rejected by ID-only proxy |
| 25 | H22 | Energy-support geometric evidence | harmonic ID-normalized energy and prototype support | low | rejected by ID-only proxy |
| 26 | H25 | Classifier-weight alignment residual | feature energy orthogonal to the predicted classifier weight | low | rejected by ID-only proxy |
| 27 | H26 | Prototype barycentric reconstruction | residual from reconstruction by top class prototypes | medium | rejected by ID-only proxy |
| 28 | H27 | Class-center correction | shrink class-specific expected prototype support | low | failed legacy full |
| 29 | H28 | Prototype hubness correction | penalize prototypes close to many other class prototypes | low | rejected by ID-only proxy |
| 30 | H29 | Local contrastive support | top support minus nearby rival support | low | rejected by ID-only proxy |
| 31 | H30 | Center plus hubness correction | joint class-center and prototype-hubness correction | low | failed legacy full |
| 32 | H32 | Per-view mean support | average nearest-prototype support over five views | 5x inference | clean result 0.891390/0.433548; failed gate |
| 33 | H33 | View stability penalty | mean view support minus fixed instability penalty | 5x inference | proxy-passed; not top-three promoted |
| 34 | H34 | View winner consensus | support gated by class consistency over views | 5x inference | proxy-passed; not top-three promoted |
| 35 | H35 | Clean multiview-RC fusion | ID-only fixed mean of flip support and RC-MSPS | 2x inference | 0.892172/0.430616; failed AUROC gain and ImageNet-O |
| 36 | H36 | Feature-view cohesion | cosine stability of deterministic feature views | 5x inference | rejected by proxy |
| 37 | H37 | Logit-shape cohesion | stability of centered logit profiles across views | 5x inference | rejected by proxy |
| 38 | H38 | View Jensen-Shannon | negative JS divergence between view predictions | 5x inference | rejected by proxy |
| 39 | H39 | Class activation profile | cross-view predicted-class activation shape | 5x inference | rejected by proxy |
| 40 | H40 | Activation-view stability | activation-vector stability across views | 5x inference | rejected by proxy |
| 41 | H41 | TTA classifier confidence | MSP of mean deterministic-view logits | 5x inference | failed legacy: 0.782537/0.677161 |
| 42 | H42 | Prototype-feature stability | view stability fused with deep support | 5x inference | rejected by proxy |
| 43 | H43 | Prototype-activation profile | activation profile fused with deep support | 5x inference | rejected by proxy |
| 44 | H45 | Class confusion profile | top-class-conditioned 1000-logit profile likelihood | low | rejected by multi-proxy gate |
| 45 | H46 | Full confusion profile | all-class logit-shape likelihood | low | rejected by multi-proxy gate |
| 46 | H47 | Confusion/deep conjunction | geometric, harmonic, and minimum profile support | low | rejected by multi-proxy gate |
| 47 | H48 | Profile winner conjunction | agreement between profile and prototype winners | low | rejected by multi-proxy gate |
| 48 | H51 | Tangent conformal residual | class tangent residual with empirical calibration | medium | only rank-zero control survived; no tangent effect |
| 49 | H52 | Tangent/deep geometric | calibrated tangent and deep support | medium | failed legacy: 0.866254/0.494737 |
| 50 | H53 | Tangent/deep minimum | lower confidence bound of tangent and deep support | medium | failed legacy: 0.866750/0.470613 |
| 51 | H54 | Compact ActSub prototype | one prototype per class in classifier-insignificant subspace | medium | failed legacy: 0.865217/0.498182 |
| 52 | H55 | Decisive SCALE component | ID-selected SCALE percentile in classifier row space | low | rejected by proxy |
| 53 | H56 | Compact ActSub product | decisive energy times insignificant prototype support | low | rejected by proxy |
| 54 | H57 | Quantile ActSub conjunction | quantile geometric decisive/insignificant support | low | failed worst-proxy stability gate |
| 55 | H58 | Nullspace coreset support | up to four centroids per class in insignificant subspace | medium | local variants rejected by proxy |
| 56 | H59 | Nullspace/deep geometric | nullspace representative support fused with deep support | medium | failed legacy: 0.890103/0.432228 |
| 57 | H60 | Nullspace/deep agreement | fixed disagreement-penalized nullspace/deep support | medium | failed legacy: 0.889880/0.434811 |
| 58 | H61 | Train-only local radius | query distance divided by reference-point class radius | medium | rejected by proxy |
| 59 | H62 | Neighborhood label purity | local support weighted by neighbor-class purity | medium | rejected by proxy |
| 60 | H63 | Exact GradNorm | vectorized official last-layer gradient L1 norm | low | rejected by proxy; mean AUROC 0.600036 |
| 61 | H64 | Cross-stage local radius | geometric support from stage-3 and stage-4 local radii | medium | rejected by proxy |
| 62 | H65 | Local excess distance | query distance minus reference local radius | medium | rejected by proxy |
| 63 | H66 | Local-radius/deep geometric | calibrated local radius fused with deep prototype | medium | failed legacy: 0.834331/0.571839 |
| 64 | H67 | GradNorm/deep geometric | exact GradNorm fused with deep prototype | low | rejected by worst-proxy gate |
| 65 | H68 | GradNorm/deep agreement | disagreement-penalized GradNorm/deep confidence | low | rejected by worst-proxy gate |
| 66 | H69 | Exact train-bank ActSub | official decisive product with 20k insignificant bank | medium | rejected by proxy |
| 67 | H70 | Bank-tail/deep geometric | full insignificant-bank support fused with deep support | medium | failed legacy: 0.883775/0.471344 |
| 68 | H71 | Bank ActSub tail | official insignificant 10-NN score with 20k bank | medium | failed legacy: 0.805563/0.637802 |
| 69 | H72 | Bank-tail/decisive geometric | quantile conjunction of official ActSub components | medium | rejected by worst-proxy gate |
| 70 | H73 | Classifier-restricted KNN | search only frozen classifier's top candidate classes | medium | rejected by proxy |
| 71 | H74 | Classifier-restricted radius | restricted KNN normalized by train-only class radii | medium | rejected by proxy |
| 72 | H75 | Restricted neighbor agreement | restricted support weighted by top-class neighbor agreement | medium | rejected by proxy |
| 73 | H77 | Restricted KNN/deep geometric | restricted local support fused with deep prototype | medium | rejected by proxy |
| 74 | H78 | RankFeat | remove leading singular feature direction at layers 3 and 4 | 1.5x inference | failed ImageNet-O gate: 0.743203/0.811500 |
| 75 | H79 | Spatial singular dominance | negative leading-singular-value energy fraction | 1.5x inference | failed; best layer-3 ratio 0.763850/0.746000 |
| 76 | H80 | Rank-one energy drop | confidence from energy sensitivity to rank-one removal | 1.5x inference | failed ImageNet-O gate: 0.485849/0.921500 |
| 77 | H81 | CAM concentration | predicted-class positive CAM mass concentration | 1x inference | failed; best CAM statistic 0.685882/0.841000 |
| 78 | H82 | Cross-layer rank collapse | maximum singular dominance across layers 3 and 4 | 1.5x inference | failed ImageNet-O gate: 0.551602/0.958500 |
| 79 | H83 | Mahalanobis++ | normalized class means and shared empirical covariance | medium fit, low state | failed ImageNet-O gate: 0.639495/0.949000 |
| 80 | H84 | ClaFR | frozen-classifier SVD projection length | low | failed ImageNet-O gate: 0.346248/0.992500 |
| 81 | H85 | ExCeL | max logit plus train rank-frequency evidence | high state | eligibility audit failed: official APS uses OOD validation and full ImageNet-1k rank tensor is oversized |
| 82 | H86 | Patch class consensus | fraction of layer-4 patches voting for global prediction | 1x inference | failed ImageNet-O gate |
| 83 | H87 | Patch vote concentration | modal local-class vote fraction | 1x inference | failed ImageNet-O gate |
| 84 | H88 | Patch class-support floor | lower quantile of local cosine to global class weight | 1x inference | failed ImageNet-O gate |
| 85 | H89 | Patch-global coherence | lower quantile of local cosine to pooled feature | 1x inference | failed ImageNet-O gate |
| 86 | H90 | Patch margin floor | lower quantile of local top-logit margin | 1x inference | failed ImageNet-O gate |
| 87 | H91 | Patch predictive agreement | negative mutual information across patch predictions | 1x inference | failed ImageNet-O gate; best patch atom still only 0.525748 |
| 88 | H92 | Patch energy floor | lower quantile of local logit energy | 1x inference | failed ImageNet-O gate |
| 89 | H93 | CAM crop support | prototype support after per-image CAM crop | 2x inference | failed ImageNet-O gate |
| 90 | H94 | CAM soft-mask support | prototype support after per-image CAM attenuation | 2x inference | failed ImageNet-O gate |
| 91 | H95 | CAM box-mask support | prototype support after hard CAM box masking | 2x inference | failed ImageNet-O gate |
| 92 | H98 | Signed atomic contrast | fixed positive/negative contrasts among four ID atoms | low | rejected by ID-only proxy |
| 93 | H103 | Proxy Fisher discriminant | one linear direction learned from locked ID-only proxies | low | failed leave-one-proxy stability |
| 94 | H104 | Proxy lower envelope | minimum support across proxy-specific linear heads | low | failed legacy: 0.504859/0.996918 |
| 95 | H105 | Proxy harmonic envelope | harmonic support across proxy-specific linear heads | low | failed legacy: 0.476722/0.995614 |
| 96 | H109 | Class-holdout mean direction | mean discriminant over class-holdout folds | low | rejected; diagnostic split was protocol-invalid and result was weak |
| 97 | H113 | Copula support | quadratic support in calibrated four-atom space | low | failed legacy: 0.703655/0.848854 |
| 98 | H118 | Official ODIN | fixed temperature and input perturbation | 2x inference | failed ImageNet-O: 0.510093/0.898500 |
| 99 | H120 | Negative gradient norm | input-gradient L1 confidence diagnostic | 2x inference | failed ImageNet-O: 0.633585/0.922500 |
| 100 | H121 | Relative Mahalanobis distance | exact OpenOOD RMDS with empirical covariances | medium | failed ImageNet-O: 0.721833/0.890500 |
| 101 | H123 | Robust spatial moments | class-conditional median/MAD of four-stage diagonal Gram moments | 1x inference | failed ImageNet-O: 0.537141/0.940000 |
| 102 | H124 | Spatial moment envelope | excursion outside per-class moment envelopes | 1x inference | failed ImageNet-O: at most 0.469295 |
| 103 | H125 | Conditional KNN calibration | predicted/true-class empirical calibration of full-bank KNN | high bank memory | failed; raw KNN remained best |
| 104 | H126 | Neighborhood label structure | KNN support combined with purity, entropy, or predicted-class agreement | high bank memory | failed legacy; best 0.870008/0.559745 |
| 105 | H127 | Joint ID support | one-class support in the two-dimensional RC/KNN plane | low | failed legacy; best 0.885911/0.470224 |
| 106 | H128 | Conditional conformal RC/KNN | symmetric conditional lower-tail quantiles | low | failed legacy; best 0.882905/0.464891 |
| 107 | H129 | Diagonal Fisher KNN | exact KNN after ID-supervised diagonal feature scaling | high bank memory | ImageNet-O improved to 0.823774 but macro stayed 0.842876 |
| 108 | H130 | Multi-view KNN | exact KNN on deterministic-view mean feature | 5x inference, high bank memory | ImageNet-O 0.830220 but macro only 0.842621 |
| 109 | H131 | Classifier-space KNN | exact KNN in fixed logit/probability embeddings | high bank memory | failed ImageNet-O; best 0.780819 |
| 110 | H132 | Class residual-direction KNN | local support among per-class residual directions | high bank memory | failed ImageNet-O; best residual 0.599557 |
| 111 | H133 | Corruption-response consistency | blur/grayscale response of prototype and logit support | 3x inference | failed ImageNet-O; best support floor 0.804494 |
| 112 | H134 | Validation hubness-corrected KNN | downweight neighbors frequently retrieved by independent ID validation | high bank memory | failed ImageNet-O; best 0.768519 |
| 113 | H135 | Full-bank local-density normalization | divide or subtract query radius by neighbors' train-only radii | high bank memory | failed ImageNet-O; best corrected score 0.592719 |
| 114 | H136 | ID classification-failure score | four-component linear score selected by ID class-held-out error AUROC | high bank memory | failed ImageNet-O: 0.543789/0.972000 |
| 115 | H137 | Complementary tail consensus | coefficient-free lower-tail consensus of prototype, MSPS, RC, and KNN | high bank memory | failed ImageNet-O: 0.798652/0.702500 |
| 116 | H138 | Activation-shift evidence | low-gradient own-image perturbation response | 2x canonical inference | useful but insufficient alone; retained as complementary atom |
| 117 | H139 | Localized prototype support | four deterministic CAM/energy crops with class prototypes | 4 crop forwards | standalone macro AUROC 0.898731 |
| 118 | H140 | Compact multi-view locality | five-view mean feature with fixed IVF-PQ support | 5 views, 71.7 MB index | ImageNet-O support retained without the 819 MB raw bank |
| 119 | H141 | PULSE | equal robust-ID sum of shift, class uniformity, localized support, and compact KNN | four components, no learned weights | success: 0.911426/0.399970; ImageNet-O 0.821093 |
| 120 | H142 | Cauchy minimum consensus | Cauchy stage-tail anchor plus three-cue minimum | four components | passed old synthetic proxy, failed legacy at 0.864317 |
| 121 | H143 | Cauchy raw-scale OR envelope | max of stage-tail and three-cue geometric branch | four components | passed old synthetic proxy, failed legacy; best 0.864921 |
| 122 | H144 | Cauchy quantile envelope | ID-ECDF aligned OR envelope | four components | passed old synthetic proxy, failed legacy; best 0.886053 |
| 123 | H145 | Image-space proxy repair | patch permutation and center CutMix ranked against frozen methods | infrastructure | Spearman 0.9, Kendall 0.8; both promote PULSE over MSPS |
| 124 | H146 | Locked untouched final evaluation | one frozen PULSE configuration on NINCO and SSB-hard | final-only | completed; broad and statistically significant gains over RC-MSPS |

## Cycle 1 Ranking Rationale

The dominant bottleneck is proxy transfer, not scoring throughput. `P01` is first because past class-holdout winners collapsed on ImageNet-O. `B01` audits the closest strong published alternative. `H04`, `H05`, and `H01` test manifold-relative evidence with compact summaries. `H10` and `H11` test a distinct disagreement principle. Tail, residual, rank-consistency, and deterministic-stability hypotheses remain available if the first group fails.

## Cycles 4-10 Diagnosis

The clean Pareto point is `MVRC-mean` at macro AUROC `0.892172` and FPR95 `0.430616`, still far below the locked success thresholds. Full-bank KNN diagnosed a genuine ImageNet-O-locality effect (`k=200`: AUROC `0.821822`) but macro AUROC remained `0.839136`. Every compact/global/class-restricted local variant failed to preserve that benefit while retaining the other three datasets. The active bottleneck is therefore representation of spatial evidence, not another scalar fusion or neighborhood coefficient.

## Cycles 11-13 Diagnosis

RankFeat, singular-dominance, rank-removal sensitivity, and CAM concentration all failed the ImageNet-O gate before full legacy evaluation. Two recent published post-hoc methods were reproduced from their primary specifications: official-form Mahalanobis++ reached only `0.639495/0.949000`, while paper-fixed ClaFR reached `0.346248/0.992500` (AUROC/FPR95). ExCeL's official automatic parameter search reads validation OOD and its full ImageNet-1k rank-frequency state is too large for the final compact-method constraint, so it was not promoted. The current cycle tests patch-local semantic evidence that needs neither an inference bank nor target-set sharing.

All proposed final forms can be expressed with at most four score components and three free hyperparameters. `B01` is an eligible baseline audit but cannot be the final method if its full train bank is required.

## Cycles 14-25 Diagnosis

Proxy envelopes, signed contrasts, copula support, conditional conformal scores, and generic one-class models all confirmed that the bottleneck is not scalar calibration. Official ODIN and exact RMDS also failed the ImageNet-O gate. Spatial moments and classifier-space locality discarded the feature geometry that makes raw full-bank KNN useful. Two new variants modestly improved ImageNet-O without target fitting: uncentered within-class diagonal scaling reached `0.823774`, and five-view mean-feature KNN reached `0.830220`. Neither transferred to the other three legacy datasets, with macro AUROC `0.842876` and `0.842621`, respectively. The clean research Pareto therefore remains `MVRC-mean` at `0.892172/0.430616`; no final benchmark access is authorized.

## Cycles 26-29 Diagnosis

Blur/grayscale stability, validation hubness, and train-local density all failed to improve the exact KNN signal. An ID-only linear predictor detected held-out ImageNet classification errors well (`0.908687` mean error AUROC) but transferred to ImageNet-O at only `0.543789`, proving that ordinary ID difficulty is not the required near-OOD axis. Four complementary single scores have a dataset-oracle macro ceiling near `0.908`, but coefficient-free lower-tail consensus selected on three locked synthetic proxies reached only `0.798652` on ImageNet-O. The active diagnosis now isolates nonlinear cross-score interactions with a protocol-ineligible legacy LODO model; its coefficients cannot be promoted.

## Cycles 30-96 Outcome

The old feature-space proxy suite was invalid for methods using own-image perturbation and localized views: it ranked CauchyStageTail first although Cauchy reached only `0.855591` legacy macro AUROC, while it ranked PULSE last although PULSE reached `0.911426`. Three separately preregistered Cauchy minimum, raw-envelope, and quantile-envelope families passed that proxy and all failed legacy, confirming an infrastructure defect rather than a coefficient issue.

The proxy was repaired with deterministic image-space pseudo-OOD. A 4x4 patch permutation and cross-class center CutMix reproduced the frozen five-method legacy ordering with Spearman `0.9` and Kendall `0.8`; Fourier phase mixing reversed the ordering and was rejected. PULSE exceeds MSPS on both selected proxies (`0.709716` vs `0.661103`; `0.690661` vs `0.599584`). On the four legacy datasets PULSE reaches macro AUROC/FPR95/AUPR-OUT `0.911426/0.399970/0.626109` and ImageNet-O AUROC `0.821093`. The 1000-replicate paired macro AUROC difference over RC-MSPS is `+0.023783`, 95% CI `[0.021864, 0.025760]`; the FPR95 difference is `-0.033918`, CI `[-0.044841, -0.023158]`. Every leave-one-component-out ablation lowers AUROC and worsens FPR95. PULSE is ready for immutable final lock; NINCO and SSB-hard remain untouched.

## Cycle 97 Final Outcome

The code was fixed at commit `f18785ecfcc09e64b91ce2b23afb1aeb9ec79500` and the final configuration was fixed at SHA-256 `f43edfd5e1770c61ca2810dac167c0025fe24155e27fc630b8a096de42c6d977` before any final image was decoded. On untouched NINCO and SSB-hard, PULSE improved AUROC over RC-MSPS on both datasets. The final macro differences were AUROC `+0.036097`, FPR95 `-0.042734`, and AUPR-OUT `+0.042061`. Their 2,000-draw paired bootstrap intervals were respectively `[0.033709, 0.038405]`, `[-0.051957, -0.033954]`, and `[0.036883, 0.047003]`. H146 is accepted and the research stopping condition is satisfied.
