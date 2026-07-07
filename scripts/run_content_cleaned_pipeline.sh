#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${MAF07_PYTHON:-/home/omote/granood_ke/.venv/bin/python}"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MAF07_SPLIT_DIR="${MAF07_SPLIT_DIR:-results/splits_content_cleaned}"
export MAF07_BACKBONES="${MAF07_BACKBONES:-dinov2_vitb14,dinov2_vitl14}"
export MAF07_SEEDS="${MAF07_SEEDS:-0,1,2}"
export MAF07_WORKERS="${MAF07_WORKERS:-4}"

echo "pipeline start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
while pgrep -f "scripts/score_image_content.py" >/dev/null; do
  echo "waiting content scoring $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  sleep 60
done

echo "combine scores $(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$PY" - <<'PY'
from pathlib import Path
import pandas as pd

root = Path("results/content_filter")
files = sorted(root.glob("imagenet_cat_scores_worker*.csv"))
if not files:
    raise SystemExit("no worker score files")
df = pd.concat([pd.read_csv(path) for path in files], ignore_index=True).drop_duplicates("rel_path", keep="last")
df = df.sort_values(["class_name", "rel_path"]).reset_index(drop=True)
df.to_csv(root / "imagenet_cat_scores.csv", index=False)

manual = []
for line in Path("configs/excluded_images.txt").read_text(encoding="utf-8").splitlines():
    item = line.strip()
    if item and not item.startswith("#"):
        manual.append(item)

exclude = sorted(set(df.loc[df["cat_score"] < 0.20, "rel_path"].astype(str)) | set(manual))
out = root / "auto_excluded_cat_lt020.txt"
out.write_text(
    "# Auto-generated: manual exclusions plus ImageNet cat_score < 0.20\n"
    + "# Paths are relative to the CAT dataset root.\n"
    + "\n".join(exclude)
    + "\n",
    encoding="utf-8",
)

summary = (
    df.assign(excluded=df["rel_path"].astype(str).isin(exclude))
    .groupby("class_name")
    .agg(n=("rel_path", "size"), excluded=("excluded", "sum"), cat_score_mean=("cat_score", "mean"))
    .reset_index()
)
summary["kept"] = summary["n"] - summary["excluded"]
summary.to_csv(root / "auto_excluded_cat_lt020_summary.csv", index=False)
print(summary.to_string(index=False))
print("total_excluded", len(exclude), "total_scores", len(df))
PY

echo "apply exclusions $(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$PY" scripts/apply_image_exclusions.py \
  --exclude-list results/content_filter/auto_excluded_cat_lt020.txt \
  --manifest results/manifest.csv \
  --manifest-output results/manifest.content_cleaned.csv \
  --split-dir results/splits \
  --split-output-dir results/splits_content_cleaned \
  --seeds 0,1,2

echo "run baseline fair $(date -u +%Y-%m-%dT%H:%M:%SZ)"
MAF07_BASELINE_OUTPUT=results/quick/baseline_content_cleaned_fair_results.csv \
MAF07_BASELINE_SUMMARY=results/quick/baseline_content_cleaned_fair_summary_by_id_size.csv \
MAF07_BASELINE_PROTOCOLS=fair \
bash scripts/run_baseline_full.sh

echo "run rsn fair $(date -u +%Y-%m-%dT%H:%M:%SZ)"
MAF07_RSN_OUTPUT=results/quick/rsn_content_cleaned_fair_results.csv \
MAF07_RSN_SUMMARY=results/quick/rsn_content_cleaned_fair_summary_by_id_size.csv \
MAF07_RSN_PROTOCOLS=fair \
bash scripts/run_rsn_full.sh

echo "run psm fair $(date -u +%Y-%m-%dT%H:%M:%SZ)"
MAF07_PSM_OUTPUT=results/quick/psm_content_cleaned_fair_results.csv \
MAF07_PSM_SUMMARY=results/quick/psm_content_cleaned_fair_summary_by_id_size.csv \
MAF07_PSM_PROTOCOLS=fair \
bash scripts/run_psm_full.sh

echo "collect summary $(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$PY" - <<'PY'
from pathlib import Path
import pandas as pd

paths = {
    "RSN": "results/quick/rsn_content_cleaned_fair_summary_by_id_size.csv",
    "PSM": "results/quick/psm_content_cleaned_fair_summary_by_id_size.csv",
    "baseline": "results/quick/baseline_content_cleaned_fair_summary_by_id_size.csv",
}
frames = []
for label, path in paths.items():
    p = Path(path)
    if not p.exists():
        print("missing", path)
        continue
    df = pd.read_csv(p)
    df["source_file"] = label
    if "protocol" in df.columns and "scope" not in df.columns:
        df = df.rename(columns={"protocol": "scope"})
    frames.append(df)
if frames:
    out = pd.concat(frames, ignore_index=True, sort=False)
    out.to_csv("results/quick/content_cleaned_fair_core_summary.csv", index=False)
    cols = ["scope", "id_size", "backbone", "method", "variant", "n", "AUROC_mean", "FPR95_mean", "AUPR_OUT_mean"]
    print(out[out["backbone"].eq("ALL")][cols].to_string(index=False))
PY

echo "pipeline end $(date -u +%Y-%m-%dT%H:%M:%SZ)"
