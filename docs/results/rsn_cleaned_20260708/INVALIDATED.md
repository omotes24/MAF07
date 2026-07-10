# Archived, not a verified final result

These files reproduce the results used by the July 2026 PDF draft. They are kept
for provenance and must not be cited as a paper-faithful full-baseline ranking.

The baseline audit found incorrect or proxy implementations in the old table and
an RSN configuration mismatch:

- archived RSN: raw features, `k=150`, all ID classes;
- PDF RSN: L2-normalized features, `k=10`, top-3 ridge-head candidates.

See `docs/BASELINE_AUDIT.md`. Corrected results are written to
`results/hades_results/rsn_baseline_audit_20260710/` and are intentionally not
merged into these CSVs.
