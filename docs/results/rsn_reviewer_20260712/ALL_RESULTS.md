# RSN Reviewer Suite: Complete Available Results

Generated from the cleaned 105,554-image experiment output. Core results use equal weight across m=2,...,7. Sensitivity tables use equal weight across m={2,4,7}; they must not be compared directly with the six-m aggregate.

## Coverage

| block | method_count | expected_jobs | completed_jobs | missing_jobs | duplicate_job_rows | expected_per_ood_rows | completed_per_ood_rows | strict_pass |
|---|---|---|---|---|---|---|---|---|
| core | 16 | 23616 | 23616 | 0 | 0 | 92160 | 92160 | True |
| sensitivity | 17 | 10812 | 10812 | 0 | 0 | 46512 | 46512 | True |

Unique fold-method rows: **34,428**. Per-OOD rows: **138,672**.

## Core 16-condition ranking (m=2,...,7 equal weight; n=1,476 each)

| method | n | AUROC | FPR95 | AUPR_OUT | AUPR_OUT_BALANCED |
|---|---|---|---|---|---|
| knn_raw_classwise_k150_mean | 1476 | 0.873581 | 0.570593 | 0.740230 | 0.832092 |
| raw_classwise_squared_k150 | 1476 | 0.873432 | 0.569222 | 0.739820 | 0.831597 |
| raw_classwise_huber_k150 | 1476 | 0.873389 | 0.570921 | 0.739873 | 0.831755 |
| global_std_huber_k150 | 1476 | 0.871262 | 0.587041 | 0.734054 | 0.826330 |
| knn_raw_pooled_k150_kth | 1476 | 0.871043 | 0.569051 | 0.739149 | 0.829210 |
| knn_raw_pooled_k150_mean | 1476 | 0.870837 | 0.575745 | 0.738717 | 0.830581 |
| raw_pooled_huber_k150 | 1476 | 0.870664 | 0.575971 | 0.738474 | 0.830338 |
| rsn_class_std_huber_k150 | 1476 | 0.870392 | 0.566419 | 0.739969 | 0.830133 |
| class_std_squared_k150 | 1476 | 0.869914 | 0.571160 | 0.737731 | 0.828081 |
| class_mad_huber_k150 | 1476 | 0.869528 | 0.567229 | 0.738949 | 0.829224 |
| class_iqr_huber_k150 | 1476 | 0.869523 | 0.567355 | 0.738949 | 0.829147 |
| nnguide_k50 | 1476 | 0.868711 | 0.630199 | 0.725000 | 0.820465 |
| knn_l2_pooled_k50_kth | 1476 | 0.864983 | 0.609024 | 0.728049 | 0.820105 |
| rsn_equalbank3000 | 1476 | 0.864696 | 0.602482 | 0.729383 | 0.820119 |
| rsn_class_std_huber_calibrated | 1476 | 0.864502 | 0.637795 | 0.722609 | 0.816538 |
| nnguide_k10 | 1476 | 0.862561 | 0.636504 | 0.713163 | 0.813523 |

## All 33 registered conditions

Rows with n=1,476 cover all six m values. Rows with n=636 are sensitivity conditions covering m={2,4,7} only.

| method | n | AUROC | FPR95 | AUPR_OUT | AUPR_OUT_BALANCED |
|---|---|---|---|---|---|
| knn_raw_classwise_k150_mean | 1476 | 0.873581 | 0.570593 | 0.740230 | 0.832092 |
| raw_classwise_squared_k150 | 1476 | 0.873432 | 0.569222 | 0.739820 | 0.831597 |
| raw_classwise_huber_k150 | 1476 | 0.873389 | 0.570921 | 0.739873 | 0.831755 |
| global_std_euclidean_k150 | 636 | 0.872962 | 0.583501 | 0.723391 | 0.826047 |
| rsn_k300 | 636 | 0.872860 | 0.567061 | 0.728040 | 0.827935 |
| rsn_delta075 | 636 | 0.872799 | 0.555161 | 0.730554 | 0.831324 |
| rsn_delta100 | 636 | 0.872762 | 0.555639 | 0.730368 | 0.831140 |
| global_std_squared_k150 | 636 | 0.872757 | 0.582332 | 0.722877 | 0.825349 |
| rsn_tau1e1 | 636 | 0.872713 | 0.556376 | 0.730091 | 0.830863 |
| rsn_tau1e2 | 636 | 0.872701 | 0.556410 | 0.730073 | 0.830844 |
| rsn_tau1e4 | 636 | 0.872701 | 0.556410 | 0.730073 | 0.830844 |
| rsn_delta200 | 636 | 0.872559 | 0.558101 | 0.729466 | 0.830232 |
| rsn_k100 | 636 | 0.872525 | 0.562078 | 0.730661 | 0.832071 |
| class_std_euclidean_k150 | 636 | 0.872369 | 0.561889 | 0.728440 | 0.829347 |
| rsn_delta300 | 636 | 0.872355 | 0.560039 | 0.728697 | 0.829451 |
| rsn_k50 | 636 | 0.871752 | 0.567707 | 0.729599 | 0.832601 |
| global_std_huber_k150 | 1476 | 0.871262 | 0.587041 | 0.734054 | 0.826330 |
| knn_raw_pooled_k150_kth | 1476 | 0.871043 | 0.569051 | 0.739149 | 0.829210 |
| knn_raw_pooled_k150_mean | 1476 | 0.870837 | 0.575745 | 0.738717 | 0.830581 |
| raw_pooled_huber_k150 | 1476 | 0.870664 | 0.575971 | 0.738474 | 0.830338 |
| rsn_class_std_huber_k150 | 1476 | 0.870392 | 0.566419 | 0.739969 | 0.830133 |
| rsn_k25 | 636 | 0.870110 | 0.570486 | 0.725827 | 0.830734 |
| l2_class_std_huber_k150 | 636 | 0.870026 | 0.581808 | 0.723722 | 0.823810 |
| class_std_squared_k150 | 1476 | 0.869914 | 0.571160 | 0.737731 | 0.828081 |
| l2_class_std_euclidean_k150 | 636 | 0.869718 | 0.587060 | 0.722284 | 0.822434 |
| l2_class_std_squared_k150 | 636 | 0.869551 | 0.586165 | 0.721967 | 0.821946 |
| class_mad_huber_k150 | 1476 | 0.869528 | 0.567229 | 0.738949 | 0.829224 |
| class_iqr_huber_k150 | 1476 | 0.869523 | 0.567355 | 0.738949 | 0.829147 |
| nnguide_k50 | 1476 | 0.868711 | 0.630199 | 0.725000 | 0.820465 |
| knn_l2_pooled_k50_kth | 1476 | 0.864983 | 0.609024 | 0.728049 | 0.820105 |
| rsn_equalbank3000 | 1476 | 0.864696 | 0.602482 | 0.729383 | 0.820119 |
| rsn_class_std_huber_calibrated | 1476 | 0.864502 | 0.637795 | 0.722609 | 0.816538 |
| nnguide_k10 | 1476 | 0.862561 | 0.636504 | 0.713163 | 0.813523 |

## Key methods by ID class count

| id_size | method | n | AUROC | FPR95 | AUPR_OUT_BALANCED |
|---|---|---|---|---|---|
| 2 | knn_raw_classwise_k150_mean | 168 | 0.918889 | 0.426834 | 0.896361 |
| 2 | raw_classwise_huber_k150 | 168 | 0.918738 | 0.428023 | 0.895970 |
| 2 | raw_classwise_squared_k150 | 168 | 0.918692 | 0.427866 | 0.895597 |
| 2 | rsn_class_std_huber_k150 | 168 | 0.916862 | 0.430698 | 0.891472 |
| 2 | nnguide_k10 | 168 | 0.914209 | 0.447322 | 0.888647 |
| 2 | knn_l2_pooled_k50_kth | 168 | 0.912603 | 0.462615 | 0.884702 |
| 3 | knn_raw_classwise_k150_mean | 336 | 0.898774 | 0.501114 | 0.870208 |
| 3 | raw_classwise_squared_k150 | 336 | 0.898615 | 0.501074 | 0.869457 |
| 3 | raw_classwise_huber_k150 | 336 | 0.898605 | 0.502047 | 0.869814 |
| 3 | rsn_class_std_huber_k150 | 336 | 0.896488 | 0.500462 | 0.866807 |
| 3 | nnguide_k10 | 336 | 0.894210 | 0.542517 | 0.859007 |
| 3 | knn_l2_pooled_k50_kth | 336 | 0.891039 | 0.537261 | 0.857371 |
| 4 | knn_raw_classwise_k150_mean | 420 | 0.881616 | 0.555496 | 0.847057 |
| 4 | raw_classwise_squared_k150 | 420 | 0.881479 | 0.554348 | 0.846438 |
| 4 | raw_classwise_huber_k150 | 420 | 0.881430 | 0.556178 | 0.846677 |
| 4 | rsn_class_std_huber_k150 | 420 | 0.878812 | 0.552805 | 0.844983 |
| 4 | nnguide_k10 | 420 | 0.873933 | 0.619843 | 0.829773 |
| 4 | knn_l2_pooled_k50_kth | 420 | 0.872917 | 0.595092 | 0.834023 |
| 5 | knn_raw_classwise_k150_mean | 336 | 0.865823 | 0.599755 | 0.824368 |
| 5 | raw_classwise_squared_k150 | 336 | 0.865699 | 0.597450 | 0.823938 |
| 5 | raw_classwise_huber_k150 | 336 | 0.865624 | 0.599969 | 0.824034 |
| 5 | rsn_class_std_huber_k150 | 336 | 0.862379 | 0.595234 | 0.823536 |
| 5 | knn_l2_pooled_k50_kth | 336 | 0.856499 | 0.640101 | 0.811732 |
| 5 | nnguide_k10 | 336 | 0.853594 | 0.682876 | 0.800909 |
| 6 | knn_raw_classwise_k150_mean | 168 | 0.849447 | 0.642369 | 0.798107 |
| 6 | raw_classwise_squared_k150 | 168 | 0.849324 | 0.639702 | 0.797865 |
| 6 | raw_classwise_huber_k150 | 168 | 0.849235 | 0.641786 | 0.797830 |
| 6 | rsn_class_std_huber_k150 | 168 | 0.845382 | 0.633589 | 0.797922 |
| 6 | knn_l2_pooled_k50_kth | 168 | 0.839741 | 0.683533 | 0.786405 |
| 6 | nnguide_k10 | 168 | 0.832564 | 0.736688 | 0.770755 |
| 7 | knn_raw_classwise_k150_mean | 48 | 0.826935 | 0.697988 | 0.756450 |
| 7 | raw_classwise_squared_k150 | 48 | 0.826782 | 0.694891 | 0.756284 |
| 7 | raw_classwise_huber_k150 | 48 | 0.826701 | 0.697520 | 0.756203 |
| 7 | rsn_class_std_huber_k150 | 48 | 0.822429 | 0.685728 | 0.756079 |
| 7 | knn_l2_pooled_k50_kth | 48 | 0.817101 | 0.735543 | 0.746396 |
| 7 | nnguide_k10 | 48 | 0.806853 | 0.789780 | 0.732046 |

## Matched factor decomposition (m={2,4,7} equal weight)

| method | AUROC | FPR95 | AUPR_OUT_BALANCED |
|---|---|---|---|
| knn_raw_classwise_k150_mean | 0.875813 | 0.560106 | 0.833290 |
| raw_classwise_squared_k150 | 0.875651 | 0.559035 | 0.832773 |
| raw_classwise_huber_k150 | 0.875623 | 0.560574 | 0.832950 |
| global_std_huber_k150 | 0.873497 | 0.576656 | 0.827598 |
| global_std_euclidean_k150 | 0.872962 | 0.583501 | 0.826047 |
| global_std_squared_k150 | 0.872757 | 0.582332 | 0.825349 |
| rsn_class_std_huber_k150 | 0.872701 | 0.556410 | 0.830844 |
| class_std_euclidean_k150 | 0.872369 | 0.561889 | 0.829347 |
| class_std_squared_k150 | 0.872204 | 0.561291 | 0.828825 |
| class_mad_huber_k150 | 0.871868 | 0.557470 | 0.829961 |
| class_iqr_huber_k150 | 0.871864 | 0.557618 | 0.829858 |
| l2_class_std_huber_k150 | 0.870026 | 0.581808 | 0.823810 |
| l2_class_std_euclidean_k150 | 0.869718 | 0.587060 | 0.822434 |
| l2_class_std_squared_k150 | 0.869551 | 0.586165 | 0.821946 |
| rsn_class_std_huber_calibrated | 0.867210 | 0.621966 | 0.818849 |
| rsn_equalbank3000 | 0.866827 | 0.593762 | 0.820348 |

## Independent k, delta, and tau sensitivity (m={2,4,7} equal weight)

| method | AUROC | FPR95 | AUPR_OUT_BALANCED |
|---|---|---|---|
| rsn_k300 | 0.872860 | 0.567061 | 0.827935 |
| rsn_delta075 | 0.872799 | 0.555161 | 0.831324 |
| rsn_delta100 | 0.872762 | 0.555639 | 0.831140 |
| rsn_tau1e1 | 0.872713 | 0.556376 | 0.830863 |
| rsn_class_std_huber_k150 | 0.872701 | 0.556410 | 0.830844 |
| rsn_tau1e2 | 0.872701 | 0.556410 | 0.830844 |
| rsn_tau1e4 | 0.872701 | 0.556410 | 0.830844 |
| rsn_delta200 | 0.872559 | 0.558101 | 0.830232 |
| rsn_k100 | 0.872525 | 0.562078 | 0.832071 |
| rsn_delta300 | 0.872355 | 0.560039 | 0.829451 |
| rsn_k50 | 0.871752 | 0.567707 | 0.832601 |
| rsn_k25 | 0.870110 | 0.570486 | 0.830734 |

## Backbone consistency (m=2,...,7 equal weight)

| backbone | method | AUROC | FPR95 | AUPR_OUT_BALANCED |
|---|---|---|---|---|
| dinov2_vitb14 | knn_raw_classwise_k150_mean | 0.860550 | 0.582498 | 0.820218 |
| dinov2_vitb14 | raw_classwise_huber_k150 | 0.860400 | 0.582930 | 0.820109 |
| dinov2_vitb14 | raw_classwise_squared_k150 | 0.860249 | 0.584553 | 0.819370 |
| dinov2_vitb14 | rsn_class_std_huber_k150 | 0.858555 | 0.579614 | 0.821100 |
| dinov2_vitb14 | knn_l2_pooled_k50_kth | 0.850941 | 0.621744 | 0.805225 |
| dinov2_vitb14 | nnguide_k10 | 0.850487 | 0.663557 | 0.801171 |
| dinov2_vitl14 | raw_classwise_squared_k150 | 0.886615 | 0.553890 | 0.843823 |
| dinov2_vitl14 | knn_raw_classwise_k150_mean | 0.886612 | 0.558687 | 0.843966 |
| dinov2_vitl14 | raw_classwise_huber_k150 | 0.886377 | 0.558912 | 0.843401 |
| dinov2_vitl14 | rsn_class_std_huber_k150 | 0.882229 | 0.553225 | 0.839166 |
| dinov2_vitl14 | knn_l2_pooled_k50_kth | 0.879026 | 0.596304 | 0.834985 |
| dinov2_vitl14 | nnguide_k10 | 0.874634 | 0.609452 | 0.825874 |

## Per-OOD-species results (m and backbone equal weight)

| ood_class | RSN_AUROC | dAUROC_vs_raw_classwise | RSN_FPR95 | FPR95_improvement_vs_raw | RSN_balanced_AUPR |
|---|---|---|---|---|---|
| leopard | 0.723238 | -0.020581 | 0.785896 | 0.024417 | 0.666788 |
| jaguar | 0.734717 | 0.009875 | 0.805789 | 0.015143 | 0.676952 |
| ocelot | 0.778301 | -0.019657 | 0.881440 | -0.000840 | 0.683827 |
| serval | 0.908542 | 0.011740 | 0.595069 | 0.042851 | 0.864174 |
| puma | 0.909352 | -0.006611 | 0.634267 | -0.069271 | 0.841046 |
| cheetah | 0.920381 | 0.003196 | 0.629207 | 0.008661 | 0.842427 |
| lion | 0.950528 | 0.001360 | 0.330801 | 0.046067 | 0.903249 |
| tiger | 0.973946 | -0.003081 | 0.061384 | -0.009729 | 0.960497 |

## ID-set cluster bootstrap and sign-flip effects

Positive effect means RSN is better. FPR95 is direction-adjusted, so positive means RSN has lower FPR95.

| method_b | metric | mean_effect | ci_low | ci_high | permutation_p | good_rate |
|---|---|---|---|---|---|---|
| class_iqr_huber_k150 | AUPR_OUT_BALANCED | 0.000985 | -0.000140 | 0.002012 | 0.000400 | 0.626016 |
| class_iqr_huber_k150 | AUROC | 0.000869 | 0.000093 | 0.001677 | 0.002000 | 0.609756 |
| class_iqr_huber_k150 | FPR95 | 0.000936 | -0.004295 | 0.005176 | 0.367963 | 0.585366 |
| class_mad_huber_k150 | AUPR_OUT_BALANCED | 0.000909 | -0.000289 | 0.001982 | 0.003000 | 0.621951 |
| class_mad_huber_k150 | AUROC | 0.000864 | 0.000057 | 0.001719 | 0.002200 | 0.613821 |
| class_mad_huber_k150 | FPR95 | 0.000810 | -0.005017 | 0.005414 | 0.445055 | 0.577236 |
| class_std_squared_k150 | AUPR_OUT_BALANCED | 0.002052 | 0.001917 | 0.002194 | 0.000100 | 0.995935 |
| class_std_squared_k150 | AUROC | 0.000478 | 0.000428 | 0.000531 | 0.000100 | 0.971545 |
| class_std_squared_k150 | FPR95 | 0.004741 | 0.004153 | 0.005392 | 0.000100 | 0.987805 |
| knn_l2_pooled_k50_kth | AUPR_OUT_BALANCED | 0.010028 | 0.007406 | 0.012640 | 0.000100 | 0.780488 |
| knn_l2_pooled_k50_kth | AUROC | 0.005409 | 0.002710 | 0.007729 | 0.000100 | 0.715447 |
| knn_l2_pooled_k50_kth | FPR95 | 0.042605 | 0.032344 | 0.055650 | 0.000100 | 0.800813 |
| knn_raw_classwise_k150_mean | AUPR_OUT_BALANCED | -0.001959 | -0.004452 | 0.000568 | 0.006499 | 0.357724 |
| knn_raw_classwise_k150_mean | AUROC | -0.003189 | -0.005713 | -0.000877 | 0.000100 | 0.369919 |
| knn_raw_classwise_k150_mean | FPR95 | 0.004173 | -0.001428 | 0.009984 | 0.242276 | 0.455285 |
| nnguide_k10 | AUPR_OUT_BALANCED | 0.016610 | 0.008850 | 0.024815 | 0.000100 | 0.613821 |
| nnguide_k10 | AUROC | 0.007831 | 0.001454 | 0.014463 | 0.000300 | 0.573171 |
| nnguide_k10 | FPR95 | 0.070085 | 0.046999 | 0.094682 | 0.000100 | 0.638211 |
| raw_classwise_huber_k150 | AUPR_OUT_BALANCED | -0.001622 | -0.004104 | 0.000914 | 0.021998 | 0.361789 |
| raw_classwise_huber_k150 | AUROC | -0.002997 | -0.005504 | -0.000677 | 0.000100 | 0.378049 |
| raw_classwise_huber_k150 | FPR95 | 0.004501 | -0.001138 | 0.010122 | 0.163384 | 0.463415 |
| raw_classwise_squared_k150 | AUPR_OUT_BALANCED | -0.001464 | -0.003912 | 0.001053 | 0.039996 | 0.365854 |
| raw_classwise_squared_k150 | AUROC | -0.003040 | -0.005525 | -0.000734 | 0.000100 | 0.382114 |
| raw_classwise_squared_k150 | FPR95 | 0.002803 | -0.002322 | 0.008224 | 0.454755 | 0.443089 |
| rsn_class_std_huber_calibrated | AUPR_OUT_BALANCED | 0.013595 | 0.008669 | 0.018984 | 0.000100 | 0.731707 |
| rsn_class_std_huber_calibrated | AUROC | 0.005890 | 0.002402 | 0.009617 | 0.000100 | 0.646341 |
| rsn_class_std_huber_calibrated | FPR95 | 0.071375 | 0.053896 | 0.090260 | 0.000100 | 0.768293 |
| rsn_equalbank3000 | AUPR_OUT_BALANCED | 0.010014 | 0.008010 | 0.011832 | 0.000100 | 0.861789 |
| rsn_equalbank3000 | AUROC | 0.005696 | 0.004178 | 0.007285 | 0.000100 | 0.821138 |
| rsn_equalbank3000 | FPR95 | 0.036063 | 0.029827 | 0.043358 | 0.000100 | 0.878049 |

## Verified 21-baseline ranking with ID-set cluster 95% CIs

| method | AUROC | AUROC_95CI | FPR95 | FPR95_95CI | AUPR_OUT | AUPR_OUT_95CI |
|---|---|---|---|---|---|---|
| rsn_reported | 0.870392 | [0.853575, 0.886328] | 0.566419 | [0.523621, 0.605239] | 0.739969 | [0.712061, 0.770558] |
| knn | 0.864983 | [0.849407, 0.879807] | 0.609024 | [0.567922, 0.643904] | 0.728049 | [0.700795, 0.758374] |
| openmax | 0.857563 | [0.846409, 0.868108] | 0.628280 | [0.598497, 0.654538] | 0.737993 | [0.714555, 0.764682] |
| msp | 0.854203 | [0.844062, 0.863629] | 0.661230 | [0.642310, 0.679450] | 0.719258 | [0.703070, 0.736288] |
| maxlogit | 0.852838 | [0.842707, 0.862083] | 0.660157 | [0.640928, 0.678582] | 0.718280 | [0.701736, 0.735052] |
| entropy | 0.846635 | [0.836462, 0.856354] | 0.656329 | [0.636360, 0.676324] | 0.712854 | [0.696953, 0.730760] |
| gradnorm | 0.844290 | [0.833518, 0.854863] | 0.658320 | [0.636993, 0.678196] | 0.710458 | [0.693648, 0.729030] |
| gen | 0.843824 | [0.833662, 0.853945] | 0.656563 | [0.635620, 0.676334] | 0.710862 | [0.694844, 0.728853] |
| energy | 0.842311 | [0.832006, 0.852615] | 0.656411 | [0.636009, 0.675973] | 0.709969 | [0.694111, 0.727611] |
| rmd | 0.840294 | [0.820561, 0.857579] | 0.594871 | [0.565845, 0.623400] | 0.748619 | [0.722083, 0.774814] |
| vim | 0.838506 | [0.823772, 0.853032] | 0.733731 | [0.708165, 0.756539] | 0.677899 | [0.657045, 0.700755] |
| mahalanobis | 0.838197 | [0.821769, 0.853591] | 0.725925 | [0.698131, 0.750832] | 0.681982 | [0.660039, 0.706957] |
| scale | 0.836913 | [0.826829, 0.846771] | 0.662286 | [0.642898, 0.680982] | 0.705800 | [0.690577, 0.723134] |
| mahalanobispp | 0.835210 | [0.819035, 0.851171] | 0.735972 | [0.708924, 0.760637] | 0.678402 | [0.656483, 0.702832] |
| kl_matching | 0.827142 | [0.811577, 0.841030] | 0.634326 | [0.609561, 0.659376] | 0.723517 | [0.700496, 0.747343] |
| nci | 0.815260 | [0.799384, 0.830025] | 0.746092 | [0.727812, 0.764091] | 0.673925 | [0.655237, 0.694022] |
| dice | 0.781545 | [0.758967, 0.803343] | 0.812879 | [0.793303, 0.830373] | 0.642725 | [0.621644, 0.665963] |
| react | 0.670571 | [0.659029, 0.682170] | 0.853814 | [0.840707, 0.866236] | 0.562323 | [0.549139, 0.576260] |
| ashs | 0.523921 | [0.515108, 0.533128] | 0.942649 | [0.937824, 0.947393] | 0.452938 | [0.445997, 0.459831] |
| ashp | 0.518681 | [0.508354, 0.528476] | 0.950117 | [0.945070, 0.954682] | 0.445768 | [0.438678, 0.452774] |
| ashb | 0.510140 | [0.499337, 0.520761] | 0.935417 | [0.929550, 0.940455] | 0.453793 | [0.445071, 0.462715] |

## m confounds and prevalence

| id_size | id_train_n_mean | id_n_mean | ood_n_mean | ood_prevalence_mean | AUROC_mean | FPR95_mean | AUPR_OUT_mean | AUPR_OUT_BALANCED_mean | MACRO_AUROC_mean |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 17598.083333 | 4397.750000 | 13193.250000 | 0.750000 | 0.916862 | 0.430698 | 0.959028 | 0.891472 | 0.909538 |
| 3 | 26397.125000 | 6596.625000 | 10994.375000 | 0.625000 | 0.896488 | 0.500462 | 0.912542 | 0.866807 | 0.887051 |
| 4 | 35196.166667 | 8795.500000 | 8795.500000 | 0.500000 | 0.878812 | 0.552805 | 0.843133 | 0.844983 | 0.867958 |
| 5 | 43995.208333 | 10994.375000 | 6596.625000 | 0.375000 | 0.862379 | 0.595234 | 0.741777 | 0.823536 | 0.851173 |
| 6 | 52794.250000 | 13193.250000 | 4397.750000 | 0.250000 | 0.845382 | 0.633589 | 0.595276 | 0.797922 | 0.836105 |
| 7 | 61593.291667 | 15392.125000 | 2198.875000 | 0.125000 | 0.822429 | 0.685728 | 0.388058 | 0.756079 | 0.822429 |

## Calibration diagnostic (representative cheetah|jaguar fold)

| variant | AUPR_OUT | AUROC | FPR95 |
|---|---|---|---|
| raw | 0.962751 | 0.880019 | 0.372866 |
| calibrated | 0.963818 | 0.887554 | 0.366695 |

| sample_n | spearman_rho | kendall_tau | discordant_pair_fraction | mean_abs_rank_displacement | max_abs_rank_displacement |
|---|---|---|---|---|---|
| 17608 | 0.990159 | 0.938275 | 0.030862 | 472.543787 | 2725.000000 |

The representative fold improves after calibration, but the all-fold core aggregate deteriorates from AUROC 0.870392 / FPR95 0.566419 to AUROC 0.864502 / FPR95 0.637795.

## Runtime and peak GPU memory (representative fold)

| method | query_n | bank_n | bank_bytes | runtime_sec_median | runtime_sec_min | queries_per_sec | peak_gpu_bytes |
|---|---|---|---|---|---|---|---|
| rsn_class_std_huber_k150 | 17608 | 14501 | 44547072 | 0.669188 | 0.668972 | 26312.481773 | 382966272 |
| knn_l2_pooled_k50_kth | 17608 | 14501 | 44547072 | 0.130656 | 0.130505 | 134766.222238 | 295023616 |
| nnguide_k10 | 17608 | 14501 | 44547072 | 0.083328 | 0.083113 | 211309.879745 | 211558400 |

## Closed-set species confusion matrix

| true_class | cheetah | jaguar | leopard | lion | ocelot | puma | serval | tiger |
|---|---|---|---|---|---|---|---|---|
| cheetah | 0.921703 | 0.001576 | 0.012086 | 0.042039 | 0.002102 | 0.004729 | 0.014188 | 0.001576 |
| jaguar | 0.001132 | 0.813809 | 0.031126 | 0.001132 | 0.084890 | 0.056593 | 0.002264 | 0.009055 |
| leopard | 0.021942 | 0.048272 | 0.839276 | 0.012068 | 0.021942 | 0.040592 | 0.010422 | 0.005485 |
| lion | 0.005665 | 0.001511 | 0.001133 | 0.963369 | 0.000378 | 0.021148 | 0.005287 | 0.001511 |
| ocelot | 0.001464 | 0.018058 | 0.002928 | 0.000976 | 0.933138 | 0.034163 | 0.007321 | 0.001952 |
| puma | 0.000638 | 0.013070 | 0.004144 | 0.005100 | 0.016258 | 0.952184 | 0.004144 | 0.004463 |
| serval | 0.023474 | 0.006036 | 0.008048 | 0.022803 | 0.028169 | 0.038229 | 0.849095 | 0.024145 |
| tiger | 0.000358 | 0.011828 | 0.005735 | 0.009677 | 0.004659 | 0.022581 | 0.009677 | 0.935484 |

Macro diagonal accuracy: **0.901007**.

## Source files

- `fold_level.csv`: every fold-method result
- `per_ood_class.csv`: every fold-method-OOD-species result
- `analysis/factor_summary.csv`: aggregate factor table
- `analysis/paired_idset_cluster_bootstrap.csv`: dependence-aware comparisons
- `analysis/baseline_21_cluster_ci.csv`: verified baseline intervals
- `representative_fold/`: qualitative panels, score distributions, dimensions,   confusion matrix, and runtime
- `calibration_assumptions/`: rank reversals and DKW union-bound diagnostics

Candidate-count counterfactual, data audits, external backbones, and saliency ablation are queued separately and are not included in this frozen report.
