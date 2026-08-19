#!/usr/bin/env python3
"""
stage2b_v9_aggregate_shared_cache.py

Lightweight aggregation/QC of the eight independent v9 shared-cache workers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = read_json(args.config)
    root = Path(cfg["stage2b_root"])
    setup_root = root / "setup"
    cache_root = root / "shared_matrix_cache"
    work = pd.read_csv(setup_root / "shared_cache_worker_index.csv")

    summary_rows = []
    failure_parts = []
    audit_parts = []

    for _, r in work.iterrows():
        cohort = str(r["cohort"])
        panel = str(r["panel"])
        sample_type = str(r["sample_type"])
        subset = str(r["patient_subset"])
        agg = str(r["agg"])

        pdir = cache_root / "patient_matrices" / panel
        stem = f"{cohort}__{sample_type}__{subset}__agg-{agg}"

        matrix_path = pdir / f"{stem}.parquet"
        summary_path = pdir / f"{stem}__summary.json"
        failure_path = pdir / f"{stem}__failures.csv"
        audit_path = pdir / f"{stem}__build_audit.csv"

        if summary_path.exists():
            with open(summary_path, "r") as f:
                s = json.load(f)
        else:
            s = {
                "cohort": cohort,
                "panel": panel,
                "n_requested": r.get("n_requested_features"),
                "n_present_final": None,
                "n_missing_final": None,
                "n_patients": None,
                "matrix_path": str(matrix_path),
                "status": "missing_worker_summary",
            }

        s["matrix_exists"] = matrix_path.exists()
        summary_rows.append(s)

        if failure_path.exists():
            x = pd.read_csv(failure_path)
            if not x.empty:
                failure_parts.append(x)

        if audit_path.exists():
            x = pd.read_csv(audit_path)
            if not x.empty:
                x["cohort"] = cohort
                x["panel"] = panel
                audit_parts.append(x)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(cache_root / "shared_cache_context_summary.csv", index=False)

    failures = (
        pd.concat(failure_parts, ignore_index=True, sort=False)
        if failure_parts else pd.DataFrame()
    )
    failures.to_csv(cache_root / "shared_cache_build_failures.csv", index=False)

    audits = (
        pd.concat(audit_parts, ignore_index=True, sort=False)
        if audit_parts else pd.DataFrame()
    )
    audits.to_csv(cache_root / "shared_cache_build_audit.csv", index=False)

    print("[DONE] Shared cache aggregation")
    print(summary.to_string(index=False))
    print(f"\nTotal unavailable feature/cohort rows: {len(failures)}")
    print(
        "NOTE: unavailable features in individual cohorts are allowed; "
        "Stage 2B pair-support filtering determines whether a feature has enough "
        "cross-cohort correlation support."
    )


if __name__ == "__main__":
    main()
