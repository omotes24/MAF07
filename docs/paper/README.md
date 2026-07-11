# RSN paper rewrite

`RSN_revised.tex` is the Japanese RSN paper source. Its main configuration is:

- raw DINOv2 CLS features (`normalize=false`)
- class-conditional diagonal scaling
- `k=150` neighbors per ID class
- Huber aggregation with `delta=1.345`
- all ID classes considered
- no empirical calibration

The L2-normalized, `k=10`, top-3 configuration is reported as a sensitivity
condition.

## Data sources

- Cleaned dataset and full ID-size sweep:
  `docs/results/rsn_cleaned_20260708/`
- Corrected all-method baseline sweep for `m=2,3,4,5,6,7`:
  `docs/results/rsn_baseline_full_20260711/`
- Baseline implementation audit:
  `docs/BASELINE_AUDIT.md`

The all-method sweep covers both DINOv2 backbones, three seeds, every ID-set
combination, 1,476 folds, and 32,472 verified jobs. The PSM-merged ranking and
paired statistics are under the result directory's `aggregated/` subdirectory.

## Rebuild figures

```bash
python3 scripts/make_rsn_paper_figures.py
```

## Build PDF

Run from `docs/paper`:

```bash
uplatex -interaction=nonstopmode -halt-on-error RSN_revised.tex
uplatex -interaction=nonstopmode -halt-on-error RSN_revised.tex
dvipdfmx -o ../../output/pdf/RSN_revised.pdf RSN_revised.dvi
```

The final PDF is `output/pdf/RSN_revised.pdf`.
