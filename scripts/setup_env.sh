#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

VENV_DIR="${MAF07_VENV:-.venv}"

if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
if [ "${MAF07_NO_DEPS:-0}" = "1" ]; then
  python -m pip install -e . --no-deps
else
  python -m pip install --upgrade pip wheel setuptools
  python -m pip install -e ".[dev]"
fi
