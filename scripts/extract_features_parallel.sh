#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
source "${MAF07_VENV:-.venv}/bin/activate" 2>/dev/null || true

WORKERS="${1:-${MAF07_WORKERS:-1}}"
if [ "$WORKERS" = "auto" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    WORKERS="$(nvidia-smi -L | wc -l | tr -d ' ')"
  else
    WORKERS=1
  fi
fi
if [ "$WORKERS" -lt 1 ]; then
  WORKERS=1
fi

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
else
  mapfile -t GPUS < <(python - <<'PY'
try:
    import torch
    n = torch.cuda.device_count()
except Exception:
    n = 0
if n:
    print("\n".join(str(i) for i in range(n)))
else:
    print("cpu")
PY
)
fi

GPU_COUNT="${#GPUS[@]}"
if [ "$GPU_COUNT" -lt "$WORKERS" ]; then
  WORKERS="$GPU_COUNT"
fi

mapfile -t BACKBONES < <(python - <<'PY'
from maf07.config import load_yaml

bcfg = load_yaml("configs/backbones.yaml")
ecfg = load_yaml("configs/experiments.yaml")
tiers = ecfg.get("default_backbone_tiers", ["tier0"])
for tier in tiers:
    for item in bcfg.get(tier, []):
        print(item["name"])
PY
)

mkdir -p results/logs/features
echo "parallel feature extraction workers=$WORKERS gpus=${GPUS[*]} backbones=${BACKBONES[*]}"

run_worker() {
  local worker_index="$1"
  local device="${GPUS[$worker_index]}"
  export CUDA_VISIBLE_DEVICES="$device"
  local idx="$worker_index"
  while [ "$idx" -lt "${#BACKBONES[@]}" ]; do
    local backbone="${BACKBONES[$idx]}"
    local log="results/logs/features/${backbone}.log"
    echo "worker=$worker_index device=$device backbone=$backbone start"
    python - <<PY > "$log" 2>&1
from maf07.features import extract_feature_cache

extract_feature_cache("${backbone}", resume=True)
PY
    echo "worker=$worker_index device=$device backbone=$backbone done"
    idx=$((idx + WORKERS))
  done
}

pids=()
for worker_index in $(seq 0 $((WORKERS - 1))); do
  run_worker "$worker_index" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"

