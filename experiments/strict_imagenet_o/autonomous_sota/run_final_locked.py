#!/usr/bin/env python3
"""Run the immutable final extraction, evaluation, and bootstrap end to end."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from final_lock import sha256_file, verify_locked_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--locked-config", type=Path, required=True)
    return parser.parse_args()


def complete_shard(path: Path, config_sha: str) -> bool:
    manifest_path = path.with_suffix(".manifest.json")
    if not path.is_file() or not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return (
        manifest.get("config_sha256") == config_sha
        and manifest.get("output_sha256") == sha256_file(path)
    )


def run_parallel_extraction(
    lock: dict, config_path: Path, config_sha: str, dataset: str
) -> None:
    code_root = Path(lock["code_root"])
    python = lock["execution"]["python"]
    score_dir = Path(lock["outputs"]["score_shard_dir"])
    score_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    log_streams = []
    for rank, device in enumerate(lock["execution"]["devices"]):
        output = score_dir / f"{dataset}.rank{rank}.npz"
        if complete_shard(output, config_sha):
            print(f"reuse verified shard: {output}", flush=True)
            continue
        if output.exists() or output.with_suffix(".manifest.json").exists():
            raise RuntimeError(f"incomplete or changed final shard exists: {output}")
        log_path = score_dir / f"{dataset}.rank{rank}.log"
        stream = log_path.open("x", encoding="utf-8")
        command = [
            python,
            str(code_root / "extract_final_locked.py"),
            "--locked-config",
            str(config_path),
            "--dataset",
            dataset,
            "--shard-index",
            str(rank),
            "--device",
            device,
        ]
        processes.append((rank, subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)))
        log_streams.append(stream)
    failures = []
    for rank, process in processes:
        returncode = process.wait()
        if returncode:
            failures.append((rank, returncode))
    for stream in log_streams:
        stream.close()
    if failures:
        raise RuntimeError(f"final extraction failed for {dataset}: {failures}")


def run_checked(command: list[str]) -> None:
    print("run:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    started = time.time()
    config_path = args.locked_config.resolve()
    lock, config_sha = verify_locked_config(config_path)
    for spec in lock["final_suite"]["datasets"]:
        run_parallel_extraction(lock, config_path, config_sha, str(spec["key"]))

    code_root = Path(lock["code_root"])
    python = lock["execution"]["python"]
    run_checked(
        [python, str(code_root / "evaluate_final_locked.py"), "--locked-config", str(config_path)]
    )
    run_checked(
        [python, str(code_root / "bootstrap_final_locked.py"), "--locked-config", str(config_path)]
    )
    run_checked(
        [python, str(code_root / "summarize_final_locked.py"), "--locked-config", str(config_path)]
    )
    output = Path(lock["outputs"]["result_dir"])
    completion = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": config_sha,
        "final_suite_evaluated_after_lock": True,
        "runtime_seconds": time.time() - started,
    }
    (output / "COMPLETED.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
