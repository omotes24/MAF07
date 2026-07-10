from __future__ import annotations

import os

from .diagcard import DiagCARDDetector


class RSNDetector(DiagCARDDetector):
    """Robust Scale-Normalized kNN OOD detector.

    RSN is the promoted name for the former ``diag_huber_raw`` condition:
    class-conditional diagonal feature scaling, Huberized neighbor residuals,
    and no empirical quantile calibration.
    """

    def id_scores(self, z, logits=None) -> object:
        return super().id_scores(z, logits=logits, delta=float(self.delta), calibrated=False)


def rsn_profile() -> str:
    profile = os.environ.get("MAF07_RSN_PROFILE", "paper").strip().lower()
    aliases = {
        "legacy": "reported_20260708",
        "reported": "reported_20260708",
        "reported_20260708": "reported_20260708",
        "paper": "paper",
    }
    if profile not in aliases:
        raise ValueError(f"Unknown MAF07_RSN_PROFILE: {profile}")
    return aliases[profile]


def rsn_uses_topq() -> bool:
    default = "1" if rsn_profile() == "paper" else "0"
    return os.environ.get("MAF07_RSN_USE_TOPQ", default) == "1"


def rsn_from_env() -> RSNDetector:
    profile = rsn_profile()
    paper = profile == "paper"
    return RSNDetector(
        k=int(
            os.environ.get(
                "MAF07_RSN_K",
                os.environ.get("MAF07_DIAGCARD_K", "10" if paper else "150"),
            )
        ),
        delta=float(
            os.environ.get(
                "MAF07_RSN_DELTA",
                os.environ.get("MAF07_DIAGCARD_DELTA", "1.345"),
            )
        ),
        topq=int(
            os.environ.get(
                "MAF07_RSN_TOPQ",
                os.environ.get("MAF07_DIAGCARD_TOPQ", os.environ.get("MAF07_CARD_TOPQ", "3")),
            )
        ),
        normalize=os.environ.get(
            "MAF07_RSN_NORMALIZE",
            os.environ.get("MAF07_DIAGCARD_NORMALIZE", "1" if paper else "0"),
        )
        == "1",
        min_std=float(
            os.environ.get(
                "MAF07_RSN_MIN_STD",
                os.environ.get("MAF07_DIAGCARD_MIN_STD", "1e-3"),
            )
        ),
        device=(
            os.environ.get("MAF07_RSN_DEVICE")
            or os.environ.get("MAF07_DIAGCARD_DEVICE")
            or None
        ),
        score_batch=int(
            os.environ.get(
                "MAF07_RSN_SCORE_BATCH",
                os.environ.get("MAF07_DIAGCARD_SCORE_BATCH", "256"),
            )
        ),
    )
