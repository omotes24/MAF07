#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source "${MAF07_VENV:-.venv}/bin/activate" 2>/dev/null || true

PYBIN="${MAF07_PYTHON:-python}"
WORKERS="${MAF07_BASELINE_WORKERS:-${MAF07_WORKERS:-4}}"
OUT="${MAF07_BASELINE_OUTPUT:-results/quick/baseline_full_results.csv}"
SUM="${MAF07_BASELINE_SUMMARY:-results/quick/baseline_full_summary_by_id_size.csv}"
SPLIT_DIR="${MAF07_SPLIT_DIR:-results/splits}"
PROTOCOLS="${MAF07_BASELINE_PROTOCOLS:-fair oracle}"
ID_SIZES="${MAF07_BASELINE_ID_SIZES:-2 3 4 5 6 7}"
IFS=',' read -r -a BASELINE_GPUS <<< "${MAF07_BASELINE_GPUS:-0,1,2,3}"

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export MAF07_BACKBONES="${MAF07_BACKBONES:-dinov2_vitb14,dinov2_vitl14}"
export MAF07_SEEDS="${MAF07_SEEDS:-0,1,2}"
export MAF07_BASELINE_METHODS="${MAF07_BASELINE_METHODS:-knn,card}"
export MAF07_CARD_K="${MAF07_CARD_K:-10}"
export MAF07_CARD_TOPQ="${MAF07_CARD_TOPQ:-3}"
export MAF07_CARD_NORMALIZE="${MAF07_CARD_NORMALIZE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

mkdir -p results/logs
echo "Baseline full start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "backbones=$MAF07_BACKBONES seeds=$MAF07_SEEDS workers=$WORKERS methods=$MAF07_BASELINE_METHODS split_dir=$SPLIT_DIR gpus=${BASELINE_GPUS[*]}"

run_group() {
  local protocol="$1"
  local idsize="$2"
  echo "GROUP start protocol=$protocol id_size=$idsize $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  local pids=()
  for ((wi=0; wi<WORKERS; wi++)); do
    local gpu="${BASELINE_GPUS[$((wi % ${#BASELINE_GPUS[@]}))]}"
    (
      CUDA_VISIBLE_DEVICES="$gpu" nice -n 10 "$PYBIN" scripts/run_baseline_id_size.py \
        --id-size "$idsize" \
        --protocol "$protocol" \
        --worker-index "$wi" \
        --worker-count "$WORKERS" \
        --split-dir "$SPLIT_DIR" \
        --output "$OUT" \
        --summary-output "$SUM"
    ) > "results/logs/baseline_full_${protocol}_id${idsize}_worker${wi}.log" 2>&1 &
    pids+=("$!")
    echo "  worker=$wi gpu=$gpu pid=${pids[-1]}"
  done

  local status=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  "$PYBIN" scripts/run_baseline_id_size.py \
    --id-size "$idsize" \
    --protocol "$protocol" \
    --split-dir "$SPLIT_DIR" \
    --output "$OUT" \
    --summary-output "$SUM" \
    --summarize-only || true
  echo "GROUP end protocol=$protocol id_size=$idsize status=$status $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  return "$status"
}

status=0
for protocol in $PROTOCOLS; do
  for idsize in $ID_SIZES; do
    if ! run_group "$protocol" "$idsize"; then
      status=1
    fi
  done
done

"$PYBIN" scripts/run_baseline_id_size.py --split-dir "$SPLIT_DIR" --output "$OUT" --summary-output "$SUM" --summarize-only || true
echo "Baseline full end $(date -u +%Y-%m-%dT%H:%M:%SZ) status=$status"
exit "$status"
