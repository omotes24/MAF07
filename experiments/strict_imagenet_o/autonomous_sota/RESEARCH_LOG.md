# Research Log

## 2026-07-13 - Cycle 1 Start

- Read the completed MSPS, RC-MSPS, and fixed MSPS-ViM fusion results.
- Confirmed four idle RTX 2080 Ti GPUs and 934 GB free disk on Hades.
- Confirmed reusable train, ID-validation, ID-test, and four legacy OOD ResNet50d stage-feature caches.
- Current same-condition Pareto leader is RC-MSPS: macro AUROC 0.887636 and FPR95 0.434417.
- The fixed ViM fusion failed at 0.859668/0.595740. MSPS and ViM ID scores had Pearson correlation 0.759, and new fusion false accepts exceeded rescued errors by 3,430.
- Found a larger infrastructure failure: previous class-holdout proxy winners above AUROC 0.96 produced ImageNet-O AUROC around 0.53 to 0.64. Proxy transfer is the current bottleneck.
- Preregistered NINCO and SSB-hard as the untouched final benchmark suite. No feature extraction or score computation is permitted before final lock.
- Audited official OpenOOD, NINCO, and NNGuide repositories at fixed commits.
- Downloaded but did not decode NINCO (5,879 images) and SSB-hard (49,000 images). Locked a byte-only 54,879-file manifest with aggregate SHA-256 `6dbd4c0a9b4bc9db40a0e430c82dd3c95e2dd82ae116f0761db57d6243c401e5`; final feature extraction and evaluation remain prohibited until method lock.
- Verified the preregistration and manifest with an automated test (`1 passed`).
- Next action: validate multiple ID-only proxies against the known ordering of existing methods, then reproduce NNGuide and screen compact manifold-guidance and relative-evidence hypotheses.

## 2026-07-13 - B01 NNGuide Audit

- Reimplemented the official NNGuide formula from commit `c123cac961b17a6c4f11adefd9ad861298be1469` on the fixed ResNet50d cache and classifier.
- Used a deterministic 10-example-per-class ID-train bank (10,000 total) and official `k=10`; no OOD sample was used in fit or selection.
- Local and Hades unit tests passed (`6 passed`), including a direct formula equality test, score orientation, and untouched-final access rejection.
- Legacy macro result: AUROC `0.853549`, FPR95 `0.546116`, AUPR-OUT `0.538988`; ImageNet-O AUROC `0.754202`.
- NNGuide does not displace RC-MSPS. The eligible SOTA remains RC-MSPS (`0.887636 / 0.434417`) after the closest published-neighbor audit.

## 2026-07-13 - Cycles 1-3

- Built a 30-component atomic cache from disjoint ID-only fit/calibration/query splits. No final-benchmark image was decoded.
- Screened 21 distinct compact score principles. Cauchy, truncated-product, and median stage tails passed proxy screening but all failed legacy full; the best reached only `0.855692/0.487968`.
- Diagnosed all individual atoms on legacy development data. Deepest-stage prototype support was strongest at `0.889932/0.444920`, slightly above RC-MSPS AUROC but below the required gain and FPR target.
- Tested spherical multi-centroid support (`k=2,4,8`); ID-only selection retained one centroid.
- Tested class-center, hubness, and local-contrast corrections. Two proxy-promoted fixed formulas failed legacy; raw deep support remained best.
- Audited and reran official-form fDBD, SHE, NECO, and NCI under the fixed ResNet50d. NCI was best among them at `0.852595/0.623451`, below RC-MSPS.
- Found deterministic five-view features for ID, Texture, and ImageNet-O. Five-view mean and support-floor hypotheses passed five-fold class holdout. Missing iNaturalist/OpenImage-O view extraction is running on four GPUs.

## 2026-07-13 - Cycles 4-6

- Repeated multi-view selection with a disjoint validation cache. The clean two-view support result was `0.891390/0.433548`; fixed RC fusion was `0.892172/0.430616`. Both missed the locked AUROC and ImageNet-O gates.
- Tested feature/logit/view cohesion, Jensen-Shannon stability, class activation profiles, and mean-logit TTA. Only TTA classifier confidence was promoted and it failed legacy at `0.782537/0.677161`.
- Screened full class-conditioned logit profiles and profile/prototype conjunctions; none passed the multi-proxy gate.
- Tested class-local tangent ranks `0,4,8,16`. Positive ranks consistently hurt proxy performance. Rank-zero control fusions failed legacy at `0.866254/0.494737` and `0.866750/0.470613`.
- Reproduced compact ActSub's classifier SVD and automatic split (`c=955`). One prototype per class failed legacy at `0.865217/0.498182`; decisive SCALE and product variants failed proxy stability.

## 2026-07-13 - Cycles 7-10

- Tested up to four per-class coreset representatives in ActSub's insignificant subspace. Local support was rejected; the two promoted deep fusions reached only `0.890103/0.432228` and `0.889880/0.434811`.
- Implemented train-only local radius normalization with a balanced 20k bank at stages 3 and 4. Raw radius, excess, neighbor purity, and cross-stage variants were rejected. The promoted deep fusion collapsed to `0.834331/0.571839`.
- Reproduced exact GradNorm in closed form and verified it against autograd. It failed the proxy gate with mean/worst AUROC `0.600036/0.067363`.
- Diagnosed exact full-bank KNN for `k={1,5,10,20,50,100,200}` in one pass. ImageNet-O reached `0.821822` at `k=200`, but macro performance remained `0.839136/0.638119`. Fixed KNN/prototype/MSPS conjunctions did not exceed macro AUROC `0.892`.
- Evaluated official-form 20k-bank ActSub. Its compact bank occupied `43.72 MB`; tail/deep fusion reached `0.883775/0.471344` and tail-only `0.805563/0.637802`. Exact product variants were rejected by ID-only proxies.
- Tested classifier-restricted KNN, restricted local radii, neighbor agreement, and deep fusion with 20/50 samples per class and top-1/top-3 classifier candidates. No configuration passed the proxy gate.
- Current clean Pareto point remains `MVRC-mean` (`0.892172/0.430616`), below all minimum success conditions.

## 2026-07-13 - Cycle 11 Start

- Identified pooled representation as the current bottleneck after exhausting prototype, residual, local-density, classifier-shape, and deterministic-view families.
- Started one-pass spatial extraction for RankFeat, layer-3/layer-4 singular dominance, rank-removal energy drop, and CAM concentration.
- Verified deterministic 20-step power iteration against exact SVD on eight images: RankFeat correlation `0.9999999991`, maximum absolute difference `1.67e-5`.
- Four GPUs are extracting the 50,000 ID and 2,000 ImageNet-O scores. The remaining three legacy datasets will run only if this stage reaches the ImageNet-O ceiling needed to justify promotion.

## 2026-07-13 - Cycles 11-12 Outcome

- Completed spatial extraction and verified the deterministic power-iteration approximation against exact SVD. No spatial atom reached the ImageNet-O promotion gate: RankFeat `0.743203/0.811500`, layer-3 singular dominance `0.763850/0.746000`, rank-removal energy drop `0.485849/0.921500`, CAM concentration `0.685882/0.841000`, and cross-layer rank collapse `0.551602/0.958500` (AUROC/FPR95).
- Audited the ICML 2025 Mahalanobis++ primary implementation at commit `53de550be7f88acad959f5fdd8f2430fb9a8b6ea`. Reimplemented exact L2 normalization, class means, and shared `EmpiricalCovariance`; direct float64 and accelerated scores had Pearson correlation effectively 1.0 and identical rank order. ImageNet-O was only `0.639495/0.949000`.
- Audited the ClaFR paper and implemented its Algorithm 1 exactly with paper-fixed cumulative singular-value threshold `alpha=0.9`. The resulting classifier subspace had dimension 775. ImageNet-O was `0.346248/0.992500`; score reversal was also noncompetitive.
- Audited ExCeL's official repository. Its automatic parameter search explicitly reads validation OOD, violating Strict Fair Inductive selection. At ImageNet-1k scale its class-by-rank-by-class collective state is also incompatible with the compact inference-state requirement. It was not promoted.

## 2026-07-13 - Cycle 13 Start

- The remaining bottleneck is retaining full-bank KNN's ImageNet-O locality without a train bank or sacrificing the three broad OOD sets.
- Ranked seven patch-local hypotheses from one frozen layer-4 forward: global-class consensus, modal vote concentration, class-cosine floor, patch-to-global coherence, local margin floor, predictive mutual information, and local energy floor.
- Added three unit tests covering exact high-consensus behavior, degradation under mixed semantics, finite output, and batch alignment.
- Four GPUs are extracting fixed atomic scores for ImageNet-val and ImageNet-O. No legacy target was used to select a coefficient; every atom will first be judged independently.

## 2026-07-13 - Cycles 13-17 Outcome

- All seven patch-local atoms failed the ImageNet-O gate; the best patch result remained below the established full-bank KNN ceiling.
- CAM crop, soft-mask, and hard-mask views also failed. A later class-holdout selection accidentally used the final ID test split; it is explicitly retained only as a protocol-invalid diagnostic and was noncompetitive anyway.
- Tested signed atomic contrasts, proxy-specific Fisher directions, minimum/harmonic proxy envelopes, and class-holdout discriminants. The envelopes passed their own proxy check but collapsed to macro AUROC `0.504859` and `0.476722`.
- A four-atom copula support model reached only `0.703655/0.848854`.
- Reproduced fixed-form ODIN and exact OpenOOD RMDS. ODIN reached ImageNet-O `0.510093/0.898500`; RMDS reached `0.721833/0.890500`. Numerical audits confirmed the RMDS accelerated score preserved the reference ranking.
- A contaminated ceiling diagnostic showed that OOD-supervised LODO can exceed macro AUROC `0.91` with many atoms, but those coefficients are ineligible and were not promoted. It confirms that ID-only model selection, not raw information availability, is the limiting factor.

## 2026-07-13 - Cycles 18-21

- Extracted four-stage channelwise diagonal-Gram energy and spatial contrast for 102,000 legacy gate images using four GPUs. Class median/MAD typicality reached only ImageNet-O AUROC `0.537141`; class envelopes were worse.
- Computed independent-validation full-bank KNN scores and tested predicted-class and true-class conditional empirical calibration. Every conditional calibration degraded raw KNN.
- Extracted exact top-200 neighbor label structure. Purity, label entropy, predicted-class agreement, and their fixed conjunctions improved macro AUROC to at most `0.870008`, while ImageNet-O fell below `0.81`.
- Fitted compact two-dimensional RC/KNN support using nearest-support distance, LOF, isolation forest, one-class SVM, KDE, Gaussian support, and symmetric conditional conformal quantiles. The best remained the ordinary lower envelope at `0.885911/0.470224`; all support estimators failed.

## 2026-07-13 - Cycles 22-25

- Tested exact KNN after centered/global-z/within-z/Fisher diagonal metrics. Uncentered within-class scaling improved ImageNet-O from `0.821822` to `0.823774`, but full legacy macro AUROC was only `0.842876`; uncentered Fisher reached `0.859079` macro.
- Tested full-bank KNN over five deterministic views. Mean-feature support improved ImageNet-O to `0.830220/0.658500`, but iNaturalist and OpenImage-O declined; macro was `0.842621/0.630854`.
- Tested KNN in raw logits, centered logits, probability, Hellinger, and log-probability spaces. Best ImageNet-O AUROC was `0.780819`.
- Tested full per-class raw KNN and residual-direction KNN using all 200 train samples per class. The best raw class-restricted score reached `0.788095`; residual-direction support reached only `0.599557`.
- Four GPUs are idle. The untouched NINCO/SSB-hard final suite remains byte-manifested but unextracted and unevaluated.

## 2026-07-13 - Cycles 26-29

- Extracted deterministic blur and grayscale responses for ID and ImageNet-O. The best prototype support-floor score improved the prototype-only near-OOD value to `0.804494` but stayed below exact KNN's `0.821822` gate.
- Computed top-200 neighbor identities, independent-validation hub counts, and exact same-class local radii over the 200k train bank. Hub rarity reduced ImageNet-O AUROC to `0.768519`; local excess reached only `0.592719`. Raw KNN remained best.
- Selected a four-component linear ID-failure score using five ImageNet class-held-out folds and no OOD samples. Its held-class error AUROC was `0.908687` (worst fold `0.897344`), but ImageNet-O AUROC was only `0.543789`.
- Added exact KNN scores for the three locked synthetic proxy families on four GPUs. Proxy-only selection locked the lower-two geometric tail consensus over deep support, RC-MSPS, MSPS, and KNN. It failed the ImageNet-O gate at `0.798652/0.702500`.
- Started an explicitly protocol-ineligible leave-one-OOD-domain-out ceiling diagnosis. It may identify reusable interactions, but no fitted coefficient or tree is eligible for promotion. The final benchmark remains untouched.

## 2026-07-13 - Cycles 30-96

- Built deterministic CAM/activation-energy local views, five-view mean features, and a fixed 71.7 MB IVF-PQ train-support index. Combined four robust-ID standardized signals without learned weights as PULSE.
- PULSE reached legacy macro AUROC/FPR95/AUPR-OUT `0.911426/0.399970/0.626109`; all four dataset AUROC differences versus RC-MSPS were positive and ImageNet-O reached `0.821093`.
- A 1000-replicate paired bootstrap confirmed macro AUROC `+0.023783 [0.021864, 0.025760]` and FPR95 `-0.033918 [-0.044841, -0.023158]` versus RC-MSPS.
- Detected that the old three feature-space proxies inverted PULSE and Cauchy. Preserved this failure explicitly and stopped using those proxies for image-derived methods.
- Extracted three image-space ID-only pseudo-OOD families from 1,000 fixed ImageNet validation images on four GPUs. Patch permutation and center CutMix each achieved frozen-method Spearman `0.9` and Kendall `0.8`; phase mixing scored `-0.9/-0.8` and was rejected.
- PULSE passed both selected image proxies and the worst-case requirement. Leave-one-component-out ablation reduced macro AUROC by `0.008377` (shift), `0.009966` (uniformity), `0.027158` (localized support), and `0.009002` (compact KNN); every removal worsened FPR95.
- No NINCO or SSB-hard feature, score, prediction, or decoded image has been accessed. The next action is immutable code/config lock, followed by one final-suite evaluation.
