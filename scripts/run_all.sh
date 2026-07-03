#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

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

source "${MAF07_VENV:-.venv}/bin/activate" 2>/dev/null || true

run_with_retries() {
  local attempts="${MAF07_RETRY_ATTEMPTS:-6}"
  local delay="${MAF07_RETRY_DELAY:-300}"
  local n=1
  while true; do
    "$@" && return 0
    if [ "$n" -ge "$attempts" ]; then
      return 1
    fi
    echo "command failed, retry $n/$attempts after ${delay}s: $*" >&2
    sleep "$delay"
    n=$((n + 1))
  done
}

python -m maf07.cli verify-dataset --strict
python -m maf07.cli make-splits
python -m maf07.cli generate-expected-jobs
if [ "$WORKERS" = "1" ]; then
  run_with_retries python -m maf07.cli extract-features
else
  run_with_retries bash scripts/extract_features_parallel.sh "$WORKERS"
fi
python -m maf07.cli run-closed
python -m maf07.cli run-ood-fair
python -m maf07.cli run-ood-oracle
python -m maf07.cli run-ablation
python -m maf07.cli aggregate
python -m maf07.cli make-tables
python -m maf07.cli make-figures
python -m maf07.cli audit-coverage --strict
