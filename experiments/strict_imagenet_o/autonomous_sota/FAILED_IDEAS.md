# Failed Ideas

## B01 - Protocol-Matched NNGuide

- Fixed setting: official scoring equation, `k=10`, deterministic 10-per-class ImageNet train bank, frozen `timm/resnet50d` features and classifier.
- Legacy result: macro AUROC `0.853549`, macro FPR95 `0.546116`, macro AUPR-OUT `0.538988`; ImageNet-O AUROC `0.754202`.
- Diagnosis: energy-weighted local cosine guidance improves Texture and iNaturalist but does not transfer to OpenImage-O or ImageNet-O strongly enough. It is below RC-MSPS by `0.034087` macro AUROC and is not the eligible leader.
- This closes the missing near-neighbor baseline audit. Published NNGuide numbers from a different checkpoint/preprocessing remain ineligible.

## Cycle 1 - Stage-Tail Aggregation

- Cauchy stage tails, the product of the two worst stage tails, and the stage median all passed the first synthetic proxy panel.
- Legacy macro AUROC/FPR95 was `0.855692/0.487968`, `0.843200/0.526777`, and `0.783274/0.653764`, respectively.
- Diagnosis: channel/radial feature corruption over-rewarded shallow-stage anomaly evidence. Real OpenImage-O and ImageNet-O require preserving the strong deepest-stage prototype signal.
- Retry only with a proxy panel containing independently weighted semantic class holdouts; do not retune these same aggregators.

## H05 - Multi-Centroid Support

- Spherical centroids `k={2,4,8}` were fit per class from ID train only.
- The ID-only panel selected `k=1`; class-holdout AUROC fell monotonically from `0.879080` at one centroid to `0.837038` at eight.
- Additional centroids enlarge class support and also accept held-out semantic classes, so the hypothesis was stopped before legacy evaluation.

## H27/H30 - Class Difficulty and Hubness Corrections

- ID-only selection fixed class-center shrinkage `alpha=0.5`; the joint version added hubness `beta=0.005`.
- Legacy macro AUROC/FPR95 was `0.874300/0.474809` and `0.875925/0.470560`, both below the uncorrected deep prototype (`0.889929/0.443199`).
- Diagnosis: ImageNet class holdout rewards equalizing class difficulty, but real OOD ranking benefits from the raw absolute support scale. No further coefficient sweep is allowed.

## Official Modern Baselines

- Protocol-matched NCI: `0.852595/0.623451`; fDBD: `0.845315/0.667872`; SHE: `0.692006/0.685057`; NECO: `0.453251/0.946505` (macro AUROC/FPR95).
- All were fit from ID train only using the fixed classifier/features. None displaced RC-MSPS.

## F00 - Fixed MSPS plus ViM fusion

- Formula family: robust-z linear fusion, ECDF fusion, and linear fusion with prototype margin.
- Selection: five-fold ImageNet class holdout only.
- Locked winner: linear margin, alpha 0.7, beta 0.3.
- Result: macro AUROC 0.859668, FPR95 0.595740.
- Failure reason: ViM and MSPS errors are strongly correlated; class holdout overweights a ViM-heavy mixture that does not transfer to real dataset shift.
- Retry only if: a new proxy family predicts known legacy method ordering without using candidate legacy metrics for coefficient tuning. Do not retry by changing alpha or beta.

## F01 - PAVE and multi-view proxy winners

- Proxy result: class-holdout AUROC up to roughly 0.969.
- Frozen ImageNet-O result: PAVE AUROC 0.640; mean-logit view energy AUROC 0.532.
- Failure reason: held-out ImageNet classes and deterministic view consistency are not faithful proxies for ImageNet-O semantic/covariate structure.
- Retry only if: independently motivated deterministic stability passes the new multi-proxy gate and uses no proxy-specific coefficient proliferation.

## F02 - Class-typicality residuals

- Tested squared and Huberized class-conditional feature/classifier residuals.
- ImageNet-O AUROC remained about 0.58 to 0.63.
- Failure reason: high-dimensional class-conditional residual magnitude suppresses useful prototype support and creates broad ID tails.
- Retry only if: residual is relative to a global background model or compressed to a demonstrably stable low-rank subspace.

## F03 - Naive diagonal and Huber residual support

- Diagonal and Huber top-class residual variants underperformed prototype cosine support on ImageNet-O.
- Failure reason: low-variance coordinates amplify benign class variation; Huberization reduces but does not remove the mismatch.
- Retry only if: the distance is relative/background-corrected or gated by an independently validated class-support signal.

## F04 - Full-bank KNN as a standalone score

- Same-condition macro AUROC 0.833174, FPR95 0.654297.
- Failure reason: strong Texture performance does not offset weak iNaturalist/OpenImage-O performance; the full 200k bank is also undesirable for the final method.
- Retry only as: official NNGuide baseline or a compressed guidance component, not standalone KNN.

## F05 - Deterministic Multi-View Stability

- The original all-five-view selection was contaminated because official ID test data influenced view choice. It is retained only as historical diagnosis.
- Independent validation selected the canonical plus horizontal-flip pair. Clean `PerViewMeanSupport` reached `0.891390/0.433548`; the fixed RC fusion reached `0.892172/0.430616` with ImageNet-O AUROC `0.780376`.
- Mean-logit TTA failed at `0.782537/0.677161`.
- Diagnosis: deterministic support reduces FPR slightly but does not recover the adversarially close ImageNet-O cases. Retry only with a materially new view that changes object/background evidence, not another aggregation coefficient.

## F06 - Confusion Profiles and Class Tangents

- Full 1000-logit class profiles, top-class profiles, profile/prototype conjunctions, and winner agreement all failed the multi-proxy gate.
- Class tangent ranks `4,8,16` worsened proxy performance; only the rank-zero controls survived. Their legacy results were `0.866254/0.494737` and `0.866750/0.470613`.
- Diagnosis: class-specific covariance/profile shape is not stable with respect to the target shifts. Retry only with spatial evidence or an external semantic structure fixed before target evaluation.

## F07 - Compact and Bank ActSub

- One insignificant-subspace prototype per class reached only `0.865217/0.498182`.
- Up to four nullspace centroids did not help; nullspace/deep fusions reached at most `0.890103/0.432228`.
- The official-form 20k insignificant train bank used only `43,720,000` bytes but its tail score reached `0.805563/0.637802`; its deep fusion reached `0.883775/0.471344` and ImageNet-O `0.809763`.
- The exact ActSub product and decisive/tail conjunction failed the worst-proxy gate. Retry only if the classifier decomposition itself changes for a principled reason; do not resweep `lambda` or percentile.

## F08 - Local-Radius and Restricted-Neighbor Families

- A 20k balanced bank with train-only within-class radii failed proxy screening. Its only promoted deep fusion collapsed on legacy to `0.834331/0.571839`, ImageNet-O `0.715698`.
- Classifier-restricted KNN, restricted local radius, neighbor-label agreement, and deep geometric fusion all failed the proxy gate for bank sizes 20 and 50 per class.
- Full-bank KNN diagnosis showed monotonic improvement through `k=100` in macro AUROC and ImageNet-O AUROC `0.821822` at `k=200`, but macro AUROC stayed at `0.839136` and FPR95 at `0.638119`.
- Diagnosis: local support explains the ImageNet-O-specific gap but conflicts with iNaturalist/OpenImage-O evidence. Another KNN coefficient or class restriction is not justified. Reopen only if a new representation makes local support independently strong across the proxy panel.

## F09 - Exact GradNorm

- The vectorized score was unit-tested against autograd and the OpenOOD formula.
- Exact GradNorm had proxy mean AUROC `0.600036` and worst AUROC `0.067363`. Prototype fusions also failed the worst-proxy gate.
- Retry only under a different fixed backbone; score scaling or temperature cannot repair the observed rank failure.

## F10 - RankFeat and Compact Spatial Statistics

- Exact/power-iteration RankFeat, layer-3/layer-4 singular dominance, rank-removal energy sensitivity, predicted-class CAM concentration, and CAM entropy were evaluated on all 50,000 ID test images and 2,000 ImageNet-O images.
- Best ImageNet-O AUROC was only `0.763850` from negative layer-3 singular dominance; RankFeat itself was `0.743203`. All remained below the already-known local-support ceiling.
- Diagnosis: generic spatial rank and CAM compactness describe image composition but do not identify whether the composition belongs to an ImageNet class. Retry only with spatial statistics tied directly to semantic class evidence.

## F11 - Mahalanobis++ and ClaFR

- Mahalanobis++ was reproduced from the official ICML 2025 implementation using all 200,000 balanced ID-train features. Its accelerated score had identical rank order to the float64 reference, but ImageNet-O was `0.639495/0.949000`.
- ClaFR was reproduced from Algorithm 1 with the paper's fixed `alpha=0.9`; its 775-dimensional classifier subspace produced ImageNet-O `0.346248/0.992500`.
- Diagnosis: feature normalization plus Gaussian distance and global classifier-subspace retention both erase the local support signal that makes full-bank KNN useful on ImageNet-O. Retry only under another backbone geometry, not by sweeping covariance ridge or subspace percentage on target OOD.

## F12 - Proxy Discriminants, Envelopes, and Copula Support

- Signed contrasts and Fisher directions did not survive the locked ID-only proxy panel.
- Proxy minimum and harmonic envelopes collapsed to `0.504859/0.996918` and `0.476722/0.995614` on legacy data.
- Four-atom quadratic copula support reached only `0.703655/0.848854`.
- Diagnosis: synthetic feature proxies have incompatible directionality across real shifts. Do not add another proxy-specific head or scalar envelope unless a new proxy first predicts the ordering of held-out existing methods.

## F13 - ODIN and Relative Mahalanobis

- Fixed official ODIN reached ImageNet-O `0.510093/0.898500`; negative input-gradient L1 reached `0.633585/0.922500`.
- Exact OpenOOD RMDS reached `0.721833/0.890500`; a float64 spot audit and the accelerated implementation had identical ranking.
- Retry only with a different backbone. Temperature, perturbation magnitude, covariance ridge, and score sign are not the observed bottleneck.

## F14 - Spatial Moment Typicality

- Four-stage log diagonal-Gram and spatial-contrast features were fitted with per-class median/MAD and class envelopes.
- Best ImageNet-O result was `0.537141/0.940000`; envelope variants were below chance.
- Diagnosis: channelwise spatial moments capture style and activation scale but do not preserve near-class semantic locality. Off-diagonal Gram expansion is not justified at ImageNet-1k state size.

## F15 - Conditional KNN and Neighborhood Labels

- Predicted-class and true-class empirical KNN calibration degraded the raw score.
- Purity, label entropy, predicted-class agreement, majority support, and fixed conjunctions reached at most macro `0.870008/0.559745`, while losing the ImageNet-O gate.
- Diagnosis: ImageNet-O neighbors can remain label-coherent; label structure is useful for broad shifts but conflicts with the locality signal. Do not retry with another purity coefficient.

## F16 - Joint RC/KNN One-Class Support

- Nearest support, LOF, isolation forest, one-class SVM, KDE, Gaussian support, and symmetric conditional conformal quantiles were fitted using only independent ID validation.
- Best result was the plain lower envelope at `0.885911/0.470224`; learned support estimators were worse.
- Diagnosis: a two-score ID manifold cannot recover the missing ordering. Do not retry with a more complex two-dimensional boundary.

## F17 - Diagonal Metric KNN

- Uncentered within-class scaling improved ImageNet-O to `0.823774`, and uncentered Fisher scaling retained `0.821612`.
- Their macro results were only `0.842876/0.630928` and `0.859079/0.599758`.
- Diagnosis: supervised channel scaling helps near-OOD locally but loses the broad-shift geometry. Reopen only if the metric can be compressed and independently improves all four ID-only proxy regimes.

## F18 - Multi-View Full-Bank KNN

- Five-view mean-feature KNN improved ImageNet-O to `0.830220/0.658500`.
- Macro remained `0.842621/0.630854` because iNaturalist and OpenImage-O declined.
- Diagnosis: deterministic zoom/flip stabilizes adversarially close samples but does not repair the cross-domain weakness of KNN. Another view aggregation coefficient is not justified.

## F19 - Classifier-Space KNN

- Raw logits, centered logits, probabilities, Hellinger coordinates, and log-probabilities all failed the ImageNet-O gate; best AUROC was `0.780819`.
- Diagnosis: the classifier projection discards the residual geometry that supports local near-OOD separation.

## F20 - Class Residual-Direction KNN

- Full 200-example-per-class raw restricted KNN reached ImageNet-O `0.788095`; residual-direction KNN reached only `0.599557`.
- Diagnosis: subtracting the class prototype removes absolute semantic evidence while normalizing residual directions amplifies benign within-class variation. Do not retry with residual-neighbor k or candidate-count sweeps.

## F21 - Corruption Response and Neighbor Normalization

- Blur/grayscale prototype response reached ImageNet-O AUROC `0.804494`, below raw KNN.
- Validation hubness correction reduced AUROC to `0.768519`; local-radius excess reached only `0.592719`, and ratio variants were near or below chance.
- Diagnosis: exact raw cosine support, rather than generic image stability, hub rarity, or local density normalization, carries the useful near-OOD ordering. Do not retry another hubness exponent or radius ratio.

## F22 - ID Classification-Failure Transfer

- A four-component logistic score selected only by five-fold ImageNet class-held-out error detection reached mean/worst ID-error AUROC `0.908687/0.897344`.
- The locked score collapsed to ImageNet-O `0.543789/0.972000`.
- Diagnosis: supervised ImageNet difficulty and semantic OOD novelty are different axes. Do not train another compact scorer on correctness, loss, or margin errors without a new ID-only proxy that predicts real OOD ordering.

## F23 - Coefficient-Free Tail Consensus

- Exact KNN was added to the three locked synthetic proxy families. Proxy selection chose the geometric mean of the two weakest among deep support, RC-MSPS, MSPS, and KNN.
- Worst proxy AUROC was already only `0.476901`; the locked ImageNet-O score reached `0.798652/0.702500`.
- Diagnosis: complementarity is domain-dependent rather than a universal conjunction. Do not retry another generalized-mean exponent over the same four components.

## F24 - Feature-Space Proxy Overgeneralization

- The locked radial/channel-mask/class-direction proxies ranked CauchyStageTail above MSPS and PULSE, but their corresponding legacy order was reversed.
- Three Cauchy-based families passed this proxy and failed legacy: minimum consensus `0.864317`, raw-scale envelope `0.864921`, and ID-quantile envelope `0.886053` macro AUROC at best.
- Diagnosis: feature-space pseudo-OOD does not define activation-shift behavior and does not preserve localized image evidence. It may be used for methods operating only on pooled features, but not as a universal promotion gate.
- Reopen only for a candidate whose complete scoring inputs are transformed consistently by the proxy.

## F25 - Fourier Phase-Mix Proxy

- Source-amplitude/partner-phase mixing strongly separated all known methods but reversed their frozen legacy ranking (Spearman `-0.9`, Kendall `-0.8`).
- Diagnosis: phase mixing rewards generic stage support and penalizes PULSE's real-OOD complementarity. It is rejected as selection infrastructure, not treated as a failed OOD detector.
