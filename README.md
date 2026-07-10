# MAF07

MAF07 is an 8-class wild-cat closed classification and OOD detection experiment.
The main proposed OOD method in this repository is **RSN: Robust
Scale-Normalized kNN**.

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

## Result audit status

The July 2026 audit found that the archived full-ranking table used several
incorrect or proxy baseline implementations. It also found that the RSN settings
in the PDF (`L2`, `k=10`, top-3 candidates) differ from the settings that produced
the archived numbers (raw features, `k=150`, all classes).

The archived CSVs remain under `docs/results/rsn_cleaned_20260708/` for
provenance, but must not be cited as verified final results. See
[docs/BASELINE_AUDIT.md](docs/BASELINE_AUDIT.md) for the method-by-method audit.
The corrected runner evaluates the PDF and archived RSN profiles side by side and
stores raw-score fingerprints in addition to metrics.

More details are in [docs/RSN.md](docs/RSN.md) and
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

## Reproduction

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

Run the verified `m=2` baseline audit used to replace the PDF ranking table:

```bash
bash scripts/run_verified_baseline_audit_hades.sh
```

Completion is defined by `results/coverage/coverage_report.json`: `expected_jobs` and
`completed_jobs` must match exactly and `missing_jobs.csv` must have zero rows.

The main fair OOD protocol never loads OOD train or OOD validation data. It uses
ID train, ID val, ID test, and OOD test only. Oracle runs are written separately
under `results/ood/oracle/` and are not deployable main results.
