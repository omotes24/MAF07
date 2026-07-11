#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source "${MAF07_VENV:-/home/omote/granood_ke/.venv}/bin/activate"

PYBIN="${MAF07_PYTHON:-python}"
WORKERS="${MAF07_AUDIT_WORKERS:-8}"
ID_SIZES=(${MAF07_AUDIT_ID_SIZES:-3 4 5 6 7})
AUDIT_ID_SIZES="${MAF07_AUDIT_COVERAGE_ID_SIZES:-2,3,4,5,6,7}"
OUT_DIR="${MAF07_AUDIT_OUT_DIR:-results/hades_results/rsn_baseline_full_20260711}"
SEED_RESULTS="${MAF07_AUDIT_SEED_RESULTS:-results/hades_results/rsn_baseline_audit_20260710/fold_level_verified.csv}"
METHODS="${MAF07_AUDIT_METHODS:-rsn_reported,rsn_paper,knn,mahalanobis,mahalanobispp,rmd,msp,entropy,energy,maxlogit,gen,gradnorm,kl_matching,vim,react,ashp,ashb,ashs,dice,scale,nci,openmax}"

mkdir -p "$OUT_DIR/logs"
RESULTS="$OUT_DIR/fold_level_verified.csv"
SUMMARY="$OUT_DIR/summary_verified.csv"
EQUIVALENCE="$OUT_DIR/equivalence_audit.csv"
REPORT="$OUT_DIR/coverage_report.json"
MISSING="$OUT_DIR/missing_jobs.csv"

if [[ ! -s "$RESULTS" && -s "$SEED_RESULTS" ]]; then
  cp "$SEED_RESULTS" "$RESULTS"
fi

echo "verified baseline sweep start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "commit=$(git rev-parse HEAD) id_sizes=${ID_SIZES[*]} workers=$WORKERS"

for id_size in "${ID_SIZES[@]}"; do
  echo "id_size=$id_size start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  MAF07_AUDIT_ID_SIZE="$id_size" \
  MAF07_AUDIT_WORKERS="$WORKERS" \
  MAF07_AUDIT_OUT_DIR="$OUT_DIR" \
  MAF07_AUDIT_METHODS="$METHODS" \
  MAF07_AUDIT_LOG_PREFIX="id_size${id_size}" \
    bash scripts/run_verified_baseline_audit_hades.sh
  echo "id_size=$id_size end $(date -u +%Y-%m-%dT%H:%M:%SZ)"
done

"$PYBIN" scripts/run_verified_baseline_audit.py \
  --id-size 2 \
  --methods "$METHODS" \
  --output "$RESULTS" \
  --summary-output "$SUMMARY" \
  --equivalence-output "$EQUIVALENCE" \
  --summarize-only

"$PYBIN" scripts/audit_verified_baseline_sweep.py \
  --input "$RESULTS" \
  --id-sizes "$AUDIT_ID_SIZES" \
  --methods "$METHODS" \
  --report "$REPORT" \
  --missing-output "$MISSING"

echo "verified baseline sweep end $(date -u +%Y-%m-%dT%H:%M:%SZ)"
