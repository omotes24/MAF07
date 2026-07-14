# MAF07

The primary OOD method in this repository is **PULSE**, a four-component,
strict-fair-inductive score for a frozen ResNet50d. The historical 8-class
wild-cat RSN experiments remain available as a separate secondary track.

## Primary method: PULSE

PULSE is implemented under
[`experiments/strict_imagenet_o/autonomous_sota/`](experiments/strict_imagenet_o/autonomous_sota/).
PULSE was selected without target-OOD access and then evaluated once on the
preregistered untouched NINCO and SSB-hard suite. It improves over the
same-condition RC-MSPS comparator on every macro metric:

| Metric | PULSE | RC-MSPS | Difference |
|---|---:|---:|---:|
| Macro AUROC | **0.723284** | 0.687187 | **+0.036097** |
| Macro FPR95 | **0.775802** | 0.818535 | **-0.042734** |
| Macro AUPR-OUT | **0.486622** | 0.444561 | **+0.042061** |

The complete lock, implementation, raw scores, 2,000-draw paired bootstrap,
and integrity audit are linked from the PULSE
[`README`](experiments/strict_imagenet_o/autonomous_sota/README.md) and final
[`RESULTS`](experiments/strict_imagenet_o/autonomous_sota/final_results/pulse_locked/RESULTS.md).
The full method definition, code map, ablations, efficiency measurements, and
limitations are documented in [docs/PULSE.md](docs/PULSE.md).

The archived one-shot locked command is:

```bash
cd experiments/strict_imagenet_o/autonomous_sota
python run_final_locked.py --locked-config configs/final_locked_config.json
```

The runner verifies every code and artifact hash and intentionally refuses to
overwrite the archived final result.

## Secondary track: RSN wild-cat benchmark

The RSN track is an 8-class wild-cat closed-classification and OOD experiment.

RSN is the promoted name for the legacy experiment label
`diagcard_huber_raw`. It applies class-conditional diagonal feature scaling,
Huberizes the neighbor residual, and uses the raw minimum class distance without
empirical quantile calibration.

Target classes:

- cheetah
- jaguar
- leopard
- lion
- ocelot
- puma
- serval
- tiger

The Hades dataset layout expected by default is:

```text
/home/omote/CAT/
  cheetah/
  jaguar/
  leopard/
  lion/
  ocelot/
  puma/
  serval/
  tiger/
```

The image dataset itself is not committed to this Git repository. Keep the
images on Hades, or materialize the same directory layout locally and update
`configs/dataset.yaml`.

## RSN result audit status

The July 2026 audit found that the archived full-ranking table used several
incorrect or proxy baseline implementations. The corrected paper and RSN
implementation now agree on raw features, `k=150`, all ID classes, Huber
`delta=1.345`, and no empirical calibration.

The archived CSVs remain under `docs/results/rsn_cleaned_20260708/` for
provenance, but must not be cited as verified final results. See
[docs/BASELINE_AUDIT.md](docs/BASELINE_AUDIT.md) for the method-by-method audit.
The corrected runner evaluates the primary and sensitivity RSN profiles side by
side and stores raw-score fingerprints in addition to metrics. The verified
fair sweep for `m=2,3,4,5,6,7` completed all 32,472 jobs over 1,476 folds; its
coverage report and aggregated rankings are in
`docs/results/rsn_baseline_full_20260711/`.

More details are in [docs/RSN.md](docs/RSN.md) and
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

## RSN reproduction

```bash
cd MAF07
bash scripts/setup_env.sh
bash scripts/run_all.sh --resume --workers auto
```

Useful individual commands:

```bash
make verify
make splits
make features
make closed
make ood_fair
make ood_oracle
make ablation
make aggregate
make tables
make figures
make audit
```

Run the RSN DINOv2 full sweep:

```bash
export MAF07_BACKBONES=dinov2_vitb14,dinov2_vitl14
export MAF07_SEEDS=0,1,2
export MAF07_RSN_WORKERS=4
bash scripts/run_rsn_full.sh
```

Run the verified all-method baseline sweep for every ID-set size:

```bash
bash scripts/run_verified_baseline_sweep_hades.sh
```

Completion is defined by `results/coverage/coverage_report.json`: `expected_jobs` and
`completed_jobs` must match exactly and `missing_jobs.csv` must have zero rows.

The main fair OOD protocol never loads OOD train or OOD validation data. It uses
ID train, ID val, ID test, and OOD test only. Oracle runs are written separately
under `results/ood/oracle/` and are not deployable main results.
