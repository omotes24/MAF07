#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source "${MAF07_VENV:-/home/omote/granood_ke/.venv}/bin/activate"

PYBIN="${MAF07_PYTHON:-python}"
WORKERS="${MAF07_AUDIT_WORKERS:-4}"
ID_SIZE="${MAF07_AUDIT_ID_SIZE:-2}"
OUT_DIR="${MAF07_AUDIT_OUT_DIR:-results/hades_results/rsn_baseline_audit_20260710}"
SPLIT_DIR="${MAF07_SPLIT_DIR:-results/splits_content_cleaned_conservative}"
MANIFEST="${MAF07_MANIFEST:-results/manifest.content_cleaned_conservative.csv}"
METHODS="${MAF07_AUDIT_METHODS:-rsn_reported,rsn_paper,knn,mahalanobis,mahalanobispp,rmd,msp,entropy,energy,maxlogit,gen,gradnorm,kl_matching,vim,react,ashp,ashb,ashs,dice,scale,nci,openmax}"
DEVICES=(${MAF07_AUDIT_CUDA_DEVICES:-0 1 2 3})

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$OUT_DIR/logs"
RESULTS="$OUT_DIR/fold_level_verified.csv"
SUMMARY="$OUT_DIR/summary_verified.csv"
EQUIVALENCE="$OUT_DIR/equivalence_audit.csv"
LOG_PREFIX="${MAF07_AUDIT_LOG_PREFIX:-id_size${ID_SIZE}}"

echo "verified baseline audit start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "commit=$(git rev-parse HEAD) id_size=$ID_SIZE workers=$WORKERS methods=$METHODS"

pids=()
for ((worker=0; worker<WORKERS; worker++)); do
  (
    export CUDA_VISIBLE_DEVICES="${DEVICES[$((worker % ${#DEVICES[@]}))]}"
    export MAF07_TORCH_DEVICE=cuda
    export MAF07_DIAGCARD_DEVICE=cuda
    nice -n 10 "$PYBIN" scripts/run_verified_baseline_audit.py \
      --id-size "$ID_SIZE" \
      --methods "$METHODS" \
      --worker-index "$worker" \
      --worker-count "$WORKERS" \
      --split-dir "$SPLIT_DIR" \
      --manifest "$MANIFEST" \
      --output "$RESULTS" \
      --summary-output "$SUMMARY" \
      --equivalence-output "$EQUIVALENCE"
  ) > "$OUT_DIR/logs/${LOG_PREFIX}_worker${worker}.log" 2>&1 &
  pids+=("$!")
  echo "worker=$worker gpu=${DEVICES[$((worker % ${#DEVICES[@]}))]} pid=${pids[-1]}"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

"$PYBIN" scripts/run_verified_baseline_audit.py \
  --id-size "$ID_SIZE" \
  --methods "$METHODS" \
  --split-dir "$SPLIT_DIR" \
  --manifest "$MANIFEST" \
  --output "$RESULTS" \
  --summary-output "$SUMMARY" \
  --equivalence-output "$EQUIVALENCE" \
  --summarize-only

echo "verified baseline audit end $(date -u +%Y-%m-%dT%H:%M:%SZ) status=$status"
exit "$status"
