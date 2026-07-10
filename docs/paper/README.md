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
- Corrected `m=2` baseline audit:
  `docs/results/rsn_baseline_audit_20260710/`
- Baseline implementation audit:
  `docs/BASELINE_AUDIT.md`

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
