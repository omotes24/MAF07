# CW-kNN Mean

CW-kNN Mean is the promoted name for the reviewer-suite condition
`knn_raw_classwise_k150_mean`. It uses raw backbone features and no learned
classifier, normalization, scale estimation, robust penalty, or calibration.

For class `c` with training feature bank `B_c`, define

```text
d_c(z) = (1 / k_c) * sum_{x in NN_k(z; B_c)} ||z - x||_2
k_c = min(k, |B_c|)
```

The OOD and ID scores are

```text
s_OOD(z) = min_c d_c(z)
s_ID(z)  = -s_OOD(z)
```

The reported configuration uses raw DINOv2 features, all available ID classes,
and `k=150`. On the cleaned 105,554-image dataset, averaged equally over
`m=2,...,7`, two DINOv2 backbones, three seeds, and every ID-class combination,
it obtained:

| AUROC | FPR95 | Balanced AUPR-OUT |
|---:|---:|---:|
| 0.873581 | 0.570593 | 0.832092 |

## Python API

```python
from maf07.methods.cwknn import CWKNNMeanDetector

detector = CWKNNMeanDetector(k=150, device="cuda").fit(
    train_features,
    train_labels,
)

id_scores = detector.id_scores(test_features)    # larger means more ID-like
ood_scores = detector.ood_scores(test_features)  # larger means more OOD-like
nearest_classes = detector.predict(test_features)
```

Environment defaults can be loaded with `cwknn_mean_from_env()` using
`MAF07_CWKNN_K`, `MAF07_CWKNN_DEVICE`, and `MAF07_CWKNN_SCORE_BATCH`.
The standard MAF07 scoring dispatch accepts both `cwknn_mean` and `cwknn` as
method aliases.

This method is intentionally simple. Its empirical advantage establishes that
raw features, class-wise candidate banks, and mean-neighbor aggregation explain
more of the previous KNN gap than RSN's class-conditional scaling. A separate
novelty review is required before presenting class-wise kNN itself as a new
algorithmic contribution.
