#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
source "${MAF07_VENV:-.venv}/bin/activate" 2>/dev/null || true
python -m maf07.cli make-splits
