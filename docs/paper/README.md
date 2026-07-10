# RSN paper rewrite

`RSN_old_setting_revised.tex` is the rewritten Japanese paper that promotes the
original reported RSN configuration to the main method:

- raw DINOv2 CLS features (`normalize=false`)
- class-conditional diagonal scaling
- `k=150` neighbors per ID class
- Huber aggregation with `delta=1.345`
- all ID classes considered
- no empirical calibration

The paper deliberately treats the L2-normalized, `k=10`, top-3 configuration as
a sensitivity condition rather than the proposed main configuration.

## Data sources

- Cleaned dataset and full ID-size sweep:
  `docs/results/rsn_cleaned_20260708/`
- Corrected `m=2` baseline audit:
  `docs/results/rsn_baseline_audit_20260710/`
- Baseline implementation audit:
  `docs/BASELINE_AUDIT.md`

The invalidated archived ranking is not read by the figure builder.

## Rebuild figures

```bash
python3 scripts/make_rsn_old_setting_paper_figures.py
```

## Build PDF

Run from `docs/paper`:

```bash
uplatex -interaction=nonstopmode -halt-on-error RSN_old_setting_revised.tex
uplatex -interaction=nonstopmode -halt-on-error RSN_old_setting_revised.tex
dvipdfmx -o ../../output/pdf/RSN_old_setting_revised.pdf RSN_old_setting_revised.dvi
```

The final PDF is `output/pdf/RSN_old_setting_revised.pdf`.
