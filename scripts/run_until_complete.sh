#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
source "${MAF07_VENV:-.venv}/bin/activate" 2>/dev/null || true

WORKERS="auto"
RESUME=1
while [ "$#" -gt 0 ]; do
  case "$1" in
    --workers)
      WORKERS="${2:-auto}"
      shift 2
      ;;
    --resume)
      RESUME=1
      shift
      ;;
    --no-resume)
      RESUME=0
      shift
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

delay="${MAF07_SUPERVISOR_DELAY:-300}"
max_attempts="${MAF07_SUPERVISOR_MAX_ATTEMPTS:-0}"
attempt=1

while true; do
  echo "supervisor attempt=$attempt workers=$WORKERS resume=$RESUME"
  if [ "$RESUME" = "1" ]; then
    bash scripts/run_all.sh --resume --workers "$WORKERS"
  else
    bash scripts/run_all.sh --workers "$WORKERS"
  fi
  status=$?

  if python -m maf07.cli audit-coverage --strict; then
    echo "supervisor complete: coverage is 100%"
    exit 0
  fi

  echo "supervisor run_all_status=$status coverage incomplete; retrying after ${delay}s" >&2
  if [ "$max_attempts" != "0" ] && [ "$attempt" -ge "$max_attempts" ]; then
    echo "supervisor exhausted attempts=$max_attempts" >&2
    exit 1
  fi
  attempt=$((attempt + 1))
  sleep "$delay"
done
