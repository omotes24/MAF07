#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source "${MAF07_VENV:-/home/omote/granood_ke/.venv}/bin/activate"

PYBIN="${MAF07_PYTHON:-python}"
WORKERS="${MAF07_REVIEWER_WORKERS:-4}"
GPU_COUNT="${MAF07_GPU_COUNT:-4}"
OUT_DIR="${MAF07_REVIEWER_OUT_DIR:-results/hades_results/rsn_reviewer_20260712}"
SPLIT_DIR="${MAF07_SPLIT_DIR:-results/splits_content_cleaned_conservative}"
MANIFEST="${MAF07_MANIFEST:-results/manifest.content_cleaned_conservative.csv}"
BASELINE_RESULTS="${MAF07_BASELINE_RESULTS:-docs/results/rsn_baseline_full_20260711/fold_level_verified.csv}"

CORE_METHODS="knn_l2_pooled_k50_kth,knn_raw_pooled_k150_kth,knn_raw_pooled_k150_mean,raw_pooled_huber_k150,knn_raw_classwise_k150_mean,raw_classwise_squared_k150,raw_classwise_huber_k150,global_std_huber_k150,class_std_squared_k150,rsn_class_std_huber_k150,class_mad_huber_k150,class_iqr_huber_k150,rsn_equalbank3000,rsn_class_std_huber_calibrated,nnguide_k10,nnguide_k50"
SENSITIVITY_METHODS="global_std_euclidean_k150,global_std_squared_k150,class_std_euclidean_k150,l2_class_std_euclidean_k150,l2_class_std_squared_k150,l2_class_std_huber_k150,rsn_k25,rsn_k50,rsn_k100,rsn_k300,rsn_delta075,rsn_delta100,rsn_delta200,rsn_delta300,rsn_tau1e4,rsn_tau1e2,rsn_tau1e1"
EXTERNAL_METHODS="knn_l2_pooled_k50_kth,knn_raw_pooled_k150_kth,class_std_squared_k150,rsn_class_std_huber_k150,nnguide_k10"
SALIENCY_METHODS="knn_l2_pooled_k50_kth,knn_raw_pooled_k150_kth,class_std_squared_k150,rsn_class_std_huber_k150"

RESULTS="$OUT_DIR/fold_level.csv"
PER_OOD="$OUT_DIR/per_ood_class.csv"
SUMMARY="$OUT_DIR/summary.csv"
SCORE_SAMPLES="$OUT_DIR/representative_score_samples.csv"
mkdir -p "$OUT_DIR/logs" "$OUT_DIR/analysis"

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export MAF07_REVIEWER_SCORE_BATCH="${MAF07_REVIEWER_SCORE_BATCH:-32}"
export MAF07_NNGUIDE_BATCH="${MAF07_NNGUIDE_BATCH:-64}"

phase_record() {
  printf '%s\t%s\t%s\n' "$1" "$2" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >> "$OUT_DIR/phase_status.tsv"
}

run_sharded() {
  local phase="$1"
  local id_sizes="$2"
  local backbones="$3"
  local methods="$4"
  local output="$5"
  local per_ood="$6"
  local summary="$7"
  local capture_samples="$8"
  local id_size
  phase_record "$phase" "started"
  for id_size in $id_sizes; do
    local pids=()
    local worker
    for ((worker=0; worker<WORKERS; worker++)); do
      local gpu=$((worker % GPU_COUNT))
      local sample_args=()
      if [[ "$capture_samples" == "1" && "$id_size" == "2" ]]; then
        sample_args=(--score-sample-output "$SCORE_SAMPLES")
      fi
      (
        export CUDA_VISIBLE_DEVICES="$gpu"
        export MAF07_REVIEWER_DEVICE="cuda"
        nice -n 10 "$PYBIN" scripts/run_rsn_reviewer_suite.py \
          --id-size "$id_size" \
          --methods "$methods" \
          --backbones "$backbones" \
          --worker-index "$worker" \
          --worker-count "$WORKERS" \
          --split-dir "$SPLIT_DIR" \
          --output "$output" \
          --per-ood-output "$per_ood" \
          --summary-output "$summary" \
          --no-final-summary \
          "${sample_args[@]}"
      ) > "$OUT_DIR/logs/${phase}_m${id_size}_w${worker}.log" 2>&1 &
      pids+=("$!")
    done
    local failed=0
    local pid
    for pid in "${pids[@]}"; do
      if ! wait "$pid"; then
        failed=1
      fi
    done
    if [[ "$failed" != "0" ]]; then
      phase_record "$phase" "failed_m${id_size}"
      return 1
    fi
    "$PYBIN" scripts/run_rsn_reviewer_suite.py \
      --id-size "$id_size" \
      --output "$output" \
      --per-ood-output "$per_ood" \
      --summary-output "$summary" \
      --summarize-only
  done
  phase_record "$phase" "complete"
}

echo "RSN reviewer suite start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "commit=$(git rev-parse HEAD) workers=$WORKERS gpu_count=$GPU_COUNT out=$OUT_DIR"

# Priority 1: the exact factor-matched controls, NNGuide, robust scales, and
# dependency-aware raw material for every m=2,...,7 fold.
run_sharded \
  "core" "2 3 4 5 6 7" "dinov2_vitb14,dinov2_vitl14" "$CORE_METHODS" \
  "$RESULTS" "$PER_OOD" "$SUMMARY" "1"

"$PYBIN" scripts/audit_rsn_reviewer_coverage.py \
  --input "$RESULTS" \
  --per-ood-input "$PER_OOD" \
  --backbones "dinov2_vitb14,dinov2_vitl14" \
  --id-sizes "2,3,4,5,6,7" \
  --methods "$CORE_METHODS" \
  --report "$OUT_DIR/core_coverage_report.json" \
  --missing-output "$OUT_DIR/core_missing_jobs.csv" \
  --strict

# Independent k, delta, minimum-scale, normalization, and aggregation
# sensitivity at low, middle, and high ID-set sizes.
run_sharded \
  "sensitivity" "2 4 7" "dinov2_vitb14,dinov2_vitl14" "$SENSITIVITY_METHODS" \
  "$RESULTS" "$PER_OOD" "$SUMMARY" "0"

"$PYBIN" scripts/audit_rsn_reviewer_coverage.py \
  --input "$RESULTS" \
  --per-ood-input "$PER_OOD" \
  --backbones "dinov2_vitb14,dinov2_vitl14" \
  --id-sizes "2,4,7" \
  --methods "$SENSITIVITY_METHODS" \
  --report "$OUT_DIR/sensitivity_coverage_report.json" \
  --missing-output "$OUT_DIR/sensitivity_missing_jobs.csv" \
  --strict

phase_record "analysis" "started"
"$PYBIN" scripts/analyze_rsn_reviewer_suite.py \
  --fold-results "$RESULTS" \
  --per-ood-results "$PER_OOD" \
  --baseline-results "$BASELINE_RESULTS" \
  --output-dir "$OUT_DIR/analysis"

CUDA_VISIBLE_DEVICES=0 "$PYBIN" scripts/analyze_rsn_representative_fold.py \
  --seed 0 \
  --backbone dinov2_vitb14 \
  --id-set cheetah\|jaguar \
  --split-dir "$SPLIT_DIR" \
  --output-dir "$OUT_DIR/representative_fold" \
  --device cuda

"$PYBIN" scripts/analyze_rsn_calibration_assumptions.py \
  --score-samples "$SCORE_SAMPLES" \
  --split "$SPLIT_DIR/splits_seed0.csv" \
  --id-set cheetah\|jaguar \
  --output-dir "$OUT_DIR/calibration_assumptions"
phase_record "analysis" "complete"

# Candidate-count counterfactual: hold the anchor ID species, target OOD
# species, evaluated images, and prevalence fixed while adding distractor banks.
phase_record "candidate_counterfactual" "started"
CANDIDATE_RESULTS="$OUT_DIR/candidate_counterfactual.csv"
CANDIDATE_SUMMARY="$OUT_DIR/candidate_counterfactual_summary.csv"
candidate_specs=(
  "dinov2_vitb14 0" "dinov2_vitb14 1" "dinov2_vitb14 2"
  "dinov2_vitl14 0" "dinov2_vitl14 1" "dinov2_vitl14 2"
)
candidate_index=0
while [[ "$candidate_index" -lt "${#candidate_specs[@]}" ]]; do
  pids=()
  for ((slot=0; slot<GPU_COUNT && candidate_index<${#candidate_specs[@]}; slot++)); do
    read -r backbone seed <<< "${candidate_specs[$candidate_index]}"
    (
      export CUDA_VISIBLE_DEVICES="$slot"
      "$PYBIN" scripts/run_rsn_candidate_count_counterfactual.py \
        --backbone "$backbone" \
        --seed "$seed" \
        --split-dir "$SPLIT_DIR" \
        --output "$CANDIDATE_RESULTS" \
        --summary-output "$CANDIDATE_SUMMARY" \
        --device cuda \
        --no-final-summary
    ) > "$OUT_DIR/logs/candidate_${backbone}_s${seed}.log" 2>&1 &
    pids+=("$!")
    candidate_index=$((candidate_index + 1))
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
done
"$PYBIN" scripts/run_rsn_candidate_count_counterfactual.py \
  --backbone dinov2_vitb14 \
  --seed 0 \
  --output "$CANDIDATE_RESULTS" \
  --summary-output "$CANDIDATE_SUMMARY" \
  --summarize-only
phase_record "candidate_counterfactual" "complete"

# Data quality and image shortcut audits. The human-label audit intentionally
# emits an annotation sheet and remains incomplete until blinded review.
phase_record "data_audits" "started"
"$PYBIN" scripts/audit_rsn_dataset_leakage.py \
  --manifest "$MANIFEST" \
  --split-dir "$SPLIT_DIR" \
  --output-dir "$OUT_DIR/dataset_leakage_audit"
"$PYBIN" scripts/audit_rsn_visual_shortcuts.py \
  --manifest "$MANIFEST" \
  --split "$SPLIT_DIR/splits_seed0.csv" \
  --output-dir "$OUT_DIR/visual_shortcut_audit"
"$PYBIN" scripts/build_rsn_human_label_audit.py \
  --manifest "$MANIFEST" \
  --output-dir "$OUT_DIR/human_label_audit"
phase_record "data_audits" "complete"

# Representation-family robustness: OpenAI CLIP, BioCLIP, and supervised
# ImageNet ViT. Feature extraction is resume-safe.
phase_record "external_backbones" "started"
external_backbones=(openai_clip_vitb16 bioclip imagenet_vitb16)
pids=()
for index in "${!external_backbones[@]}"; do
  backbone="${external_backbones[$index]}"
  (
    export CUDA_VISIBLE_DEVICES="$index"
    "$PYBIN" scripts/extract_selected_features.py \
      --backbones "$backbone" \
      --manifest "$MANIFEST" \
      --device cuda
  ) > "$OUT_DIR/logs/features_${backbone}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

EXTERNAL_DIR="$OUT_DIR/external_backbones"
mkdir -p "$EXTERNAL_DIR"
run_sharded \
  "external" "2 4 7" "openai_clip_vitb16,bioclip,imagenet_vitb16" "$EXTERNAL_METHODS" \
  "$EXTERNAL_DIR/fold_level.csv" "$EXTERNAL_DIR/per_ood_class.csv" \
  "$EXTERNAL_DIR/summary.csv" "0"
phase_record "external_backbones" "complete"

# Foreground/background masking audit for DINOv2 ViT-B/14.
phase_record "saliency_ablation" "started"
CUDA_VISIBLE_DEVICES=0 "$PYBIN" scripts/extract_dino_saliency_ablation_features.py \
  --backbone dinov2_vitb14 \
  --manifest "$MANIFEST" \
  --mode both \
  --batch-size 8 \
  --device cuda

SALIENCY_DIR="$OUT_DIR/saliency_ablation"
mkdir -p "$SALIENCY_DIR"
run_sharded \
  "saliency" "2 4 7" \
  "dinov2_vitb14_saliency_fg,dinov2_vitb14_saliency_bg" "$SALIENCY_METHODS" \
  "$SALIENCY_DIR/fold_level.csv" "$SALIENCY_DIR/per_ood_class.csv" \
  "$SALIENCY_DIR/summary.csv" "0"
phase_record "saliency_ablation" "complete"

echo "RSN reviewer suite end $(date -u +%Y-%m-%dT%H:%M:%SZ)"
