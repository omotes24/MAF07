#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
source .venv/bin/activate 2>/dev/null || true
python -m maf07.cli extract-features
