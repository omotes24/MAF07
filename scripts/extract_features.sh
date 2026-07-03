#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
source "${MAF07_VENV:-.venv}/bin/activate" 2>/dev/null || true
python -m maf07.cli extract-features
