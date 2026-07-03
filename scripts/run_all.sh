#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

RESUME=0
WORKERS="auto"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --resume)
      RESUME=1
      shift
      ;;
    --workers)
      WORKERS="${2:-auto}"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done
export MAF07_RESUME="$RESUME"
export MAF07_WORKERS="$WORKERS"

source .venv/bin/activate 2>/dev/null || true

python -m maf07.cli verify-dataset --strict
python -m maf07.cli make-splits
python -m maf07.cli generate-expected-jobs
python -m maf07.cli extract-features
python -m maf07.cli run-closed
python -m maf07.cli run-ood-fair
python -m maf07.cli run-ood-oracle
python -m maf07.cli run-ablation
python -m maf07.cli aggregate
python -m maf07.cli make-tables
python -m maf07.cli make-figures
python -m maf07.cli audit-coverage --strict
