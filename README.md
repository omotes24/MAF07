# MAF07

MAF07 is an 8-class wild-cat closed classification and OOD detection experiment.

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

Completion is defined by `results/coverage/coverage_report.json`: `expected_jobs` and
`completed_jobs` must match exactly and `missing_jobs.csv` must have zero rows.

The main fair OOD protocol never loads OOD train or OOD validation data. It uses
ID train, ID val, ID test, and OOD test only. Oracle runs are written separately
under `results/ood/oracle/` and are not deployable main results.

