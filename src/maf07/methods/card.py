from __future__ import annotations

import os

from .cqs import CQSDetector


class CARDDetector(CQSDetector):
    pass


def card_from_env() -> CARDDetector:
    return CARDDetector(
        k=int(os.environ.get("MAF07_CARD_K", os.environ.get("MAF07_CQS_K", "10"))),
        topq=int(os.environ.get("MAF07_CARD_TOPQ", os.environ.get("MAF07_CQS_TOPQ", "3"))),
        normalize=os.environ.get(
            "MAF07_CARD_NORMALIZE",
            os.environ.get("MAF07_CQS_NORMALIZE", "1"),
        )
        != "0",
    )
