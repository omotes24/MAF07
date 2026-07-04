#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
source "${MAF07_VENV:-.venv}/bin/activate" 2>/dev/null || true

STAGE="${1:-}"
WORKERS="${2:-${MAF07_WORKERS:-1}}"
if [ -z "$STAGE" ]; then
  echo "usage: scripts/run_jobs_parallel.sh <run-closed|run-ood-fair|run-ood-oracle|run-ablation> [workers]" >&2
  exit 2
fi
case "$STAGE" in
  run-closed|run-ood-fair|run-ood-oracle|run-ablation)
    ;;
  *)
    echo "unknown stage: $STAGE" >&2
    exit 2
    ;;
esac

if [ "$WORKERS" = "auto" ]; then
  if command -v nproc >/dev/null 2>&1; then
    WORKERS="$(nproc)"
  else
    WORKERS=1
  fi
fi
if [ "$WORKERS" -lt 1 ]; then
  WORKERS=1
fi

mkdir -p results/logs/jobs
echo "parallel job stage=$STAGE workers=$WORKERS"

pids=()
for worker_index in $(seq 0 $((WORKERS - 1))); do
  log="results/logs/jobs/${STAGE}_worker${worker_index}.log"
  (
    export MAF07_JOB_WORKER_INDEX="$worker_index"
    export MAF07_JOB_WORKER_COUNT="$WORKERS"
    python -m maf07.cli "$STAGE" --worker-index "$worker_index" --worker-count "$WORKERS"
  ) > "$log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
