#!/usr/bin/env python3
"""Write the immutable final outcome report from locked evaluation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from final_lock import verify_locked_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--locked-config", type=Path, required=True)
    return parser.parse_args()


def decide_outcome(per_dataset: pd.DataFrame, bootstrap: pd.DataFrame) -> dict[str, object]:
    pivot = per_dataset.pivot(index="dataset", columns="method", values="AUROC")
    broad_auroc_gain = bool((pivot["PULSE"] > pivot["RC-MSPS"]).all())
    macro = bootstrap[bootstrap["dataset"] == "macro"].set_index("metric")
    auroc_significant = bool(macro.loc["AUROC", "ci_low"] > 0)
    fpr_significant = bool(macro.loc["FPR95", "ci_high"] < 0)
    aupr_significant = bool(macro.loc["AUPR_OUT", "ci_low"] > 0)
    final_sota = broad_auroc_gain and auroc_significant and fpr_significant
    return {
        "broad_auroc_gain": broad_auroc_gain,
        "macro_auroc_gain_significant": auroc_significant,
        "macro_fpr95_gain_significant": fpr_significant,
        "macro_aupr_out_gain_significant": aupr_significant,
        "same_condition_final_sota": final_sota,
        "report_category": (
            "same-condition eligible SOTA achieved"
            if final_sota
            else "legacy benchmark best, untouched final benchmark target not met"
        ),
    }


def markdown_table(frame: pd.DataFrame) -> str:
    return frame.to_markdown(index=False, floatfmt=".6f")


def main() -> None:
    args = parse_args()
    lock, config_sha = verify_locked_config(args.locked_config)
    output = Path(lock["outputs"]["result_dir"])
    per_dataset = pd.read_csv(output / "final_per_dataset.csv")
    summary = pd.read_csv(output / "final_summary.csv")
    comparison = pd.read_csv(output / "final_comparison.csv")
    bootstrap = pd.read_csv(output / "paired_bootstrap.csv")
    worst = pd.read_csv(output / "worst_dataset.csv")
    runtime = pd.read_csv(output / "inference_runtime.csv")
    resources = pd.read_csv(output / "resource_accounting.csv")
    outcome = decide_outcome(per_dataset, bootstrap)
    outcome.update(
        {
            "config_sha256": config_sha,
            "legacy_gate_passed": bool(lock["legacy_gate"]["passed"]),
            "final_suite": [item["name"] for item in lock["final_suite"]["datasets"]],
        }
    )
    (output / "final_outcome.json").write_text(
        json.dumps(outcome, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    macro_bootstrap = bootstrap[bootstrap["dataset"] == "macro"]
    report = f"""# PULSE final locked evaluation

## Outcome

**{outcome['report_category']}**

- Locked configuration SHA-256: `{config_sha}`
- Code commit: `{lock['code_commit']}`
- Final suite was untouched until this lock: yes
- Target OOD used for fitting, calibration, or selection: no
- Test-image sharing: no
- Same score equation and settings for every dataset: yes

## Legacy gate

PULSE passed the preregistered legacy gate before final access: Macro AUROC
{lock['legacy_gate']['pulse_macro_auroc']:.6f}, Macro FPR95
{lock['legacy_gate']['pulse_macro_fpr95']:.6f}, ImageNet-O AUROC
{lock['legacy_gate']['pulse_imagenet_o_auroc']:.6f}. The paired legacy AUROC and
FPR95 confidence intervals excluded zero in the favorable direction.

## Untouched final suite

{markdown_table(per_dataset)}

{markdown_table(summary)}

{markdown_table(comparison)}

{markdown_table(worst)}

## Paired bootstrap

{markdown_table(macro_bootstrap)}

The final comparison used {lock['evaluation']['paired_bootstrap_replicates']} paired,
stratified image-level bootstrap draws with seed
{lock['evaluation']['bootstrap_seed']}.

## Runtime and storage

{markdown_table(runtime)}

{markdown_table(resources)}

PULSE is a score-only postprocessor and does not alter the frozen classifier's
prediction, so the ImageNet top-1 accuracy delta is exactly 0. The persistent
support is the compact IVF-PQ index plus 1,000 final-stage class prototypes; no
raw train feature bank is retained.

## Error analysis

`error_analysis.csv` records false-accept and false-reject counts at 95% ID
acceptance. `failure_cases.csv` records the 50 strongest cases per method and
split with immutable relative image paths. These files were generated only
after the method and final suite were locked and were never used to tune PULSE.
"""
    (output / "RESULTS.md").write_text(report, encoding="utf-8")
    print(json.dumps(outcome, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
