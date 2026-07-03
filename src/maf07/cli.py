from __future__ import annotations

import argparse
import json
from pathlib import Path

from .aggregate import aggregate_fair_summaries, aggregate_scores, make_tables
from .audit import write_dataset_audit
from .data import build_manifest
from .jobs import audit_coverage, write_expected_jobs
from .plots import make_figures
from .runner import extract_configured_features, run_ablation_jobs, run_closed_jobs, run_ood_jobs
from .splits import make_splits


def _print_json(obj: object) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="maf07")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("make-manifest")
    p.add_argument("--dataset-config", default="configs/dataset.yaml")

    p = sub.add_parser("verify-dataset")
    p.add_argument("--dataset-config", default="configs/dataset.yaml")
    p.add_argument("--strict", action="store_true")

    p = sub.add_parser("make-splits")
    p.add_argument("--dataset-config", default="configs/dataset.yaml")

    sub.add_parser("generate-expected-jobs")

    p = sub.add_parser("extract-features")
    p.add_argument("--tiers", nargs="*", default=None)
    p.add_argument("--no-resume", action="store_true")

    p = sub.add_parser("run-closed")
    p.add_argument("--max-jobs", type=int, default=None)

    p = sub.add_parser("run-ood-fair")
    p.add_argument("--max-jobs", type=int, default=None)

    p = sub.add_parser("run-ood-oracle")
    p.add_argument("--max-jobs", type=int, default=None)

    p = sub.add_parser("run-ablation")
    p.add_argument("--max-jobs", type=int, default=None)

    sub.add_parser("aggregate")
    sub.add_parser("make-tables")
    sub.add_parser("make-figures")

    p = sub.add_parser("audit-coverage")
    p.add_argument("--strict", action="store_true")

    args = parser.parse_args(argv)
    if args.cmd == "make-manifest":
        df = build_manifest(args.dataset_config)
        _print_json({"manifest_rows": int(len(df))})
        return 0
    if args.cmd == "verify-dataset":
        report = write_dataset_audit(args.dataset_config)
        _print_json(report)
        if args.strict and not report.get("ok", False):
            return 2
        return 0
    if args.cmd == "make-splits":
        paths = make_splits(args.dataset_config)
        _print_json({"splits": [str(Path(p)) for p in paths]})
        return 0
    if args.cmd == "generate-expected-jobs":
        df = write_expected_jobs()
        _print_json({"expected_jobs": int(len(df))})
        return 0
    if args.cmd == "extract-features":
        count = extract_configured_features(tiers=args.tiers, resume=not args.no_resume)
        _print_json({"feature_caches": count})
        return 0
    if args.cmd == "run-closed":
        _print_json({"completed": run_closed_jobs(max_jobs=args.max_jobs)})
        return 0
    if args.cmd == "run-ood-fair":
        _print_json({"completed": run_ood_jobs("fair", max_jobs=args.max_jobs)})
        return 0
    if args.cmd == "run-ood-oracle":
        _print_json({"completed": run_ood_jobs("oracle", max_jobs=args.max_jobs)})
        return 0
    if args.cmd == "run-ablation":
        _print_json({"completed": run_ablation_jobs(max_jobs=args.max_jobs)})
        return 0
    if args.cmd == "aggregate":
        paths = {
            "fair_scores": str(aggregate_scores("fair") or ""),
            "oracle_scores": str(aggregate_scores("oracle") or ""),
            "summaries": {k: str(v) for k, v in aggregate_fair_summaries().items()},
        }
        _print_json(paths)
        return 0
    if args.cmd == "make-tables":
        _print_json({"tables": [str(p) for p in make_tables()]})
        return 0
    if args.cmd == "make-figures":
        _print_json({"figures": [str(p) for p in make_figures()]})
        return 0
    if args.cmd == "audit-coverage":
        _print_json(audit_coverage(strict=args.strict))
        return 0
    raise AssertionError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())

