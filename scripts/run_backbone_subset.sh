#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
source "${MAF07_VENV:-.venv}/bin/activate" 2>/dev/null || true

BACKBONES="${1:-}"
WORKERS="${2:-${MAF07_WORKERS:-1}}"
if [ -z "$BACKBONES" ]; then
  echo "usage: scripts/run_backbone_subset.sh <comma-separated-backbones> [workers]" >&2
  exit 2
fi
export MAF07_BACKBONES="$BACKBONES"

python -m maf07.cli verify-dataset --strict
python -m maf07.cli make-splits
python -m maf07.cli generate-expected-jobs

python - <<'PY'
import os
from maf07.features import load_feature_cache

missing = []
for backbone in [b.strip() for b in os.environ["MAF07_BACKBONES"].split(",") if b.strip()]:
    try:
        load_feature_cache(backbone)
    except Exception as exc:
        missing.append(f"{backbone}: {exc}")
if missing:
    raise SystemExit("Missing feature caches:\n" + "\n".join(missing))
PY

bash scripts/run_jobs_parallel.sh run-closed "$WORKERS"
bash scripts/run_jobs_parallel.sh run-ood-fair "$WORKERS"
bash scripts/run_jobs_parallel.sh run-ood-oracle "$WORKERS"
bash scripts/run_jobs_parallel.sh run-ablation "$WORKERS"
python -m maf07.cli aggregate
python -m maf07.cli make-tables
python -m maf07.cli make-figures
python -m maf07.cli audit-coverage
