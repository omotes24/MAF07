#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source "${MAF07_VENV:-.venv}/bin/activate" 2>/dev/null || true

PYBIN="${MAF07_PYTHON:-python}"
WORKERS="${MAF07_RSN_WORKERS:-${MAF07_WORKERS:-4}}"
OUT="${MAF07_RSN_OUTPUT:-results/quick/rsn_full_results.csv}"
SUM="${MAF07_RSN_SUMMARY:-results/quick/rsn_full_summary_by_id_size.csv}"
PROTOCOLS="${MAF07_RSN_PROTOCOLS:-fair oracle}"
ID_SIZES="${MAF07_RSN_ID_SIZES:-2 3 4 5 6 7}"

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export MAF07_BACKBONES="${MAF07_BACKBONES:-dinov2_vitb14,dinov2_vitl14}"
export MAF07_SEEDS="${MAF07_SEEDS:-0,1,2}"
export MAF07_RSN_K="${MAF07_RSN_K:-150}"
export MAF07_RSN_DELTA="${MAF07_RSN_DELTA:-1.345}"
export MAF07_RSN_NORMALIZE="${MAF07_RSN_NORMALIZE:-0}"
export MAF07_RSN_MIN_STD="${MAF07_RSN_MIN_STD:-1e-3}"
export MAF07_RSN_SCORE_BATCH="${MAF07_RSN_SCORE_BATCH:-96}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

mkdir -p results/logs
echo "RSN full start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "backbones=$MAF07_BACKBONES seeds=$MAF07_SEEDS workers=$WORKERS k=$MAF07_RSN_K delta=$MAF07_RSN_DELTA"

run_group() {
  local protocol="$1"
  local idsize="$2"
  echo "GROUP start protocol=$protocol id_size=$idsize $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  local pids=()
  for ((wi=0; wi<WORKERS; wi++)); do
    (
      if [ "${MAF07_RSN_SET_CUDA_VISIBLE:-1}" = "1" ]; then
        export CUDA_VISIBLE_DEVICES="$wi"
        export MAF07_RSN_DEVICE="${MAF07_RSN_DEVICE:-cuda}"
      fi
      nice -n 10 "$PYBIN" scripts/run_rsn_id_size.py \
        --id-size "$idsize" \
        --protocol "$protocol" \
        --worker-index "$wi" \
        --worker-count "$WORKERS" \
        --output "$OUT" \
        --summary-output "$SUM"
    ) > "results/logs/rsn_full_${protocol}_id${idsize}_worker${wi}.log" 2>&1 &
    pids+=("$!")
    echo "  worker=$wi pid=${pids[-1]}"
  done

  local status=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  "$PYBIN" scripts/run_rsn_id_size.py \
    --id-size "$idsize" \
    --protocol "$protocol" \
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

"$PYBIN" scripts/run_rsn_id_size.py --output "$OUT" --summary-output "$SUM" --summarize-only || true
echo "RSN full end $(date -u +%Y-%m-%dT%H:%M:%SZ) status=$status"
exit "$status"
