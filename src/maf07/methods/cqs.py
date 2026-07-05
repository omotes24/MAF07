from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
from sklearn.neighbors import NearestNeighbors

EPS = 1e-12


def _l2_normalize(x: np.ndarray, eps: float = EPS) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    return arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), eps)


def _class_logits(logits: np.ndarray) -> np.ndarray:
    arr = np.asarray(logits, dtype=np.float64)
    if arr.ndim == 1:
        arr = np.stack([-arr, arr], axis=1)
    return arr


@dataclass
class CQSDetector:
    k: int = 10
    topq: int = 3
    normalize: bool = True
    eps: float = EPS

    classes_: np.ndarray | None = None
    nn_: dict[int, NearestNeighbors] = field(default_factory=dict)
    cal_scores_: dict[int, np.ndarray] = field(default_factory=dict)
    train_features_: np.ndarray | None = None
    train_labels_: np.ndarray | None = None

    def fit(
        self,
        z_train: np.ndarray,
        y_train: np.ndarray,
        *,
        z_cal: np.ndarray | None = None,
        y_cal: np.ndarray | None = None,
    ) -> "CQSDetector":
        train_x = np.asarray(z_train, dtype=np.float64)
        train_y = np.asarray(y_train, dtype=int)
        if self.normalize:
            train_x = _l2_normalize(train_x, self.eps)
        self.train_features_ = train_x
        self.train_labels_ = train_y
        self.classes_ = np.asarray(sorted(np.unique(train_y)))
        self.nn_ = {}
        self.cal_scores_ = {}

        for cls in self.classes_:
            class_id = int(cls)
            class_x = train_x[train_y == class_id]
            n_neighbors = min(int(self.k), len(class_x))
            self.nn_[class_id] = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(class_x)

        if z_cal is not None and y_cal is not None:
            self._fit_calibration_from_val(z_cal, y_cal)
        else:
            self._fit_calibration_from_train_leave_one_out()
        return self

    def _mean_knn_distance(self, x: np.ndarray, class_id: int) -> np.ndarray:
        dists, _ = self.nn_[int(class_id)].kneighbors(x)
        return np.mean(dists, axis=1)

    def _fit_calibration_from_val(self, z_cal: np.ndarray, y_cal: np.ndarray) -> None:
        cal_x = np.asarray(z_cal, dtype=np.float64)
        cal_y = np.asarray(y_cal, dtype=int)
        if self.normalize:
            cal_x = _l2_normalize(cal_x, self.eps)
        for cls in self.classes_ if self.classes_ is not None else []:
            class_id = int(cls)
            values = self._mean_knn_distance(cal_x[cal_y == class_id], class_id) if np.any(cal_y == class_id) else np.array([])
            if len(values) == 0:
                values = np.array([0.0], dtype=np.float64)
            self.cal_scores_[class_id] = np.sort(values.astype(np.float64))

    def _fit_calibration_from_train_leave_one_out(self) -> None:
        if self.train_features_ is None or self.train_labels_ is None or self.classes_ is None:
            raise RuntimeError("CQSDetector is not fitted")
        for cls in self.classes_:
            class_id = int(cls)
            class_x = self.train_features_[self.train_labels_ == class_id]
            if len(class_x) <= 1:
                self.cal_scores_[class_id] = np.array([0.0], dtype=np.float64)
                continue
            n_neighbors = min(int(self.k) + 1, len(class_x))
            nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(class_x)
            dists, _ = nn.kneighbors(class_x)
            self.cal_scores_[class_id] = np.sort(np.mean(dists[:, 1:], axis=1).astype(np.float64))

    def _right_tail_pvalues(self, values: np.ndarray, sorted_values: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(sorted_values, values, side="left")
        tail_count = len(sorted_values) - idx
        return (tail_count + 1.0) / (len(sorted_values) + 1.0)

    def score_samples(self, z: np.ndarray, logits: np.ndarray) -> np.ndarray:
        if not self.nn_:
            raise RuntimeError("CQSDetector is not fitted")
        x = np.asarray(z, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if self.normalize:
            x = _l2_normalize(x, self.eps)
        class_logits = _class_logits(logits)
        top = np.argsort(class_logits, axis=1)[:, ::-1][:, : min(int(self.topq), class_logits.shape[1])]
        best_p = np.zeros(len(x), dtype=np.float64)

        for class_id in self.nn_:
            rows = np.where(np.any(top == int(class_id), axis=1))[0]
            if len(rows) == 0:
                continue
            distances = self._mean_knn_distance(x[rows], class_id)
            pvals = self._right_tail_pvalues(distances, self.cal_scores_[class_id])
            best_p[rows] = np.maximum(best_p[rows], pvals)

        return -np.log(np.maximum(best_p, self.eps))

    def id_scores(self, z: np.ndarray, logits: np.ndarray) -> np.ndarray:
        return -self.score_samples(z, logits)


def cqs_from_env() -> CQSDetector:
    return CQSDetector(
        k=int(os.environ.get("MAF07_CQS_K", "10")),
        topq=int(os.environ.get("MAF07_CQS_TOPQ", "3")),
        normalize=os.environ.get("MAF07_CQS_NORMALIZE", "1") != "0",
    )
