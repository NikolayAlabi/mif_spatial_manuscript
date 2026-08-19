# Corrected Stage 2A-4 / Stage 2A-5 v9 workflow

This bundle replaces the old Stage 2A-4 rescue and Stage 2A-5
microcompression ontology with `stage2_feature_parser_v8.py`.

## Important design changes

- Cell identity, checkpoint/state identity, tissue, metric subtype, and summary
  statistic are separate.
- State rescue requires a known underlying cell identity.
- Metric-summary rescue is deliberately restricted to Median <-> Mean.
- Q1/Q3, Min/Max, SD, triad count/fraction, ATHENA interaction flavors, and
  diversity parameters remain distinct during microcompression.
- Residual microcompression only compares exact structured biological identities
  across provenance/prep-root duplication.
- Exact vector duplicates are only collapsed when their corrected semantic
  identity also matches.
- Stage 2A-4 can reuse the previous v8 patient matrices and constructs only
  newly required rescue columns.
- The new grid evaluates caps 10/15/20 and semantic rho 0.85/0.90/0.95.
- No Slurm submitter in this bundle uses a #SBATCH partition line.
- No dependency/automatic aggregate job is submitted.

## Recommended sequence

1. Install scripts and configs.
2. Run corrected Stage 2A-4 DRY RUN first.
3. Inspect parser failures and rescue counts.
4. If parsing looks good, run full Stage 2A-4 with incremental matrix reuse.
5. Aggregate Stage 2A-4 manually.
6. Run Stage 2A-5 grid inventory manually.
7. Submit Stage 2A-5 grid workers.
8. Aggregate Stage 2A-5 manually.
9. Review cap/rho results before rebuilding Stage 2B.

Do not rerun Stage 1 or Stage 2A steps 1-3.
