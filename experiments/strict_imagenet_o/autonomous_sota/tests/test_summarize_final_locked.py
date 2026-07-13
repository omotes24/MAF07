import pandas as pd

from summarize_final_locked import decide_outcome


def test_final_outcome_requires_broad_and_significant_gains() -> None:
    per_dataset = pd.DataFrame(
        [
            {"dataset": "A", "method": "PULSE", "AUROC": 0.91},
            {"dataset": "A", "method": "RC-MSPS", "AUROC": 0.89},
            {"dataset": "B", "method": "PULSE", "AUROC": 0.90},
            {"dataset": "B", "method": "RC-MSPS", "AUROC": 0.88},
        ]
    )
    bootstrap = pd.DataFrame(
        [
            {"dataset": "macro", "metric": "AUROC", "ci_low": 0.01, "ci_high": 0.03},
            {"dataset": "macro", "metric": "FPR95", "ci_low": -0.05, "ci_high": -0.01},
            {"dataset": "macro", "metric": "AUPR_OUT", "ci_low": 0.00, "ci_high": 0.02},
        ]
    )
    outcome = decide_outcome(per_dataset, bootstrap)
    assert outcome["same_condition_final_sota"] is True
    assert outcome["macro_aupr_out_gain_significant"] is False


def test_final_outcome_rejects_one_dataset_regression() -> None:
    per_dataset = pd.DataFrame(
        [
            {"dataset": "A", "method": "PULSE", "AUROC": 0.91},
            {"dataset": "A", "method": "RC-MSPS", "AUROC": 0.89},
            {"dataset": "B", "method": "PULSE", "AUROC": 0.87},
            {"dataset": "B", "method": "RC-MSPS", "AUROC": 0.88},
        ]
    )
    bootstrap = pd.DataFrame(
        [
            {"dataset": "macro", "metric": "AUROC", "ci_low": 0.01, "ci_high": 0.03},
            {"dataset": "macro", "metric": "FPR95", "ci_low": -0.05, "ci_high": -0.01},
            {"dataset": "macro", "metric": "AUPR_OUT", "ci_low": 0.01, "ci_high": 0.02},
        ]
    )
    assert decide_outcome(per_dataset, bootstrap)["same_condition_final_sota"] is False
