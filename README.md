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

## RSN result summary

The completed DINOv2 full sweep uses:

- backbones: `dinov2_vitb14`, `dinov2_vitl14`
- seeds: `0,1,2`
- protocols: `fair`, `oracle`
- ID sizes: `2,3,4,5,6,7`
- trials: `6 * C(8, k)` per id size across two DINOv2 backbones and three seeds
- legacy result label: `diagcard_huber_raw`

Fair protocol, DINOv2 pooled over both backbones:

| id_size | n | RSN AUROC | RSN FPR95 | RSN AUPR_OUT | AUROC vs KNN | FPR95 vs KNN | AUROC vs CARD | FPR95 vs CARD |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 168 | 0.888020 | 0.527700 | 0.944890 | +0.005021 | -0.028476 | +0.010547 | -0.013240 |
| 3 | 336 | 0.865054 | 0.603194 | 0.887647 | +0.004508 | -0.027255 | +0.014811 | -0.046004 |
| 4 | 420 | 0.846675 | 0.653460 | 0.807276 | +0.004302 | -0.024446 | +0.017366 | -0.065790 |
| 5 | 336 | 0.830355 | 0.694429 | 0.696770 | +0.004054 | -0.023011 | +0.018530 | -0.072362 |
| 6 | 168 | 0.813942 | 0.732347 | 0.547123 | +0.003624 | -0.018921 | +0.018480 | -0.070645 |
| 7 | 48 | 0.792711 | 0.776005 | 0.348381 | +0.002910 | -0.013564 | +0.016266 | -0.056380 |

For the fair protocol paired comparison over all 1476 DINOv2 settings, RSN
improves over KNN by `+0.004252` AUROC, `-0.024234` FPR95, and `+0.008914`
AUPR_OUT. It improves over CARD by `+0.016364` AUROC, `-0.057047` FPR95, and
`+0.024043` AUPR_OUT.

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

Completion is defined by `results/coverage/coverage_report.json`: `expected_jobs` and
`completed_jobs` must match exactly and `missing_jobs.csv` must have zero rows.

The main fair OOD protocol never loads OOD train or OOD validation data. It uses
ID train, ID val, ID test, and OOD test only. Oracle runs are written separately
under `results/ood/oracle/` and are not deployable main results.
