#!/usr/bin/env python3
"""
stage2b_v9_build_shared_cache_worker.py

Build one v9 shared Stage 2B patient matrix for one cohort x panel context.

The worker:
1. loads the all-cap cross-cohort-expanded candidate union;
2. reuses matching feature_uid columns from earlier Stage 2B caches when present;
3. reconstructs ONLY missing candidate columns from raw Stage 1 inputs;
4. writes one independent context matrix and audit files.

One worker = one CPU. The eight workers never write a common audit file, avoiding
race conditions.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def ensure_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def import_module_from_path(name, path):
    path = Path(path)
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def read_table(path: Path):
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def find_old_cache_matrix(cfg, cohort, panel, sample_type, patient_subset, agg):
    filename = f"{cohort}__{sample_type}__{patient_subset}__agg-{agg}.parquet"

    candidates = []
    for root_text in cfg.get("reuse_shared_cache_roots", []) or []:
        root = Path(root_text)
        candidates.extend([
            root / "patient_matrices" / panel / filename,
            root / panel / filename,
            root / filename,
        ])

    for p in candidates:
        if p.exists():
            return p

    return None


def zscore(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan)
    mu = x.mean(skipna=True)
    sd = x.std(skipna=True)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=x.index)
    return (x - mu) / sd


def resolve_array_id(x):
    if x is not None:
        return int(x)
    env = os.environ.get("SLURM_ARRAY_TASK_ID")
    if env is None:
        raise ValueError("Provide --array-id or SLURM_ARRAY_TASK_ID")
    return int(env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--array-id", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = read_json(args.config)
    stage2b_root = Path(cfg["stage2b_root"])
    setup_root = stage2b_root / "setup"
    cache_root = ensure_dir(stage2b_root / "shared_matrix_cache")

    work = pd.read_csv(setup_root / "shared_cache_worker_index.csv")
    array_id = resolve_array_id(args.array_id)
    match = work[work["cache_array_id"].astype(int).eq(array_id)]
    if match.empty:
        raise IndexError(f"cache_array_id={array_id} not found")
    row = match.iloc[0]

    cohort = str(row["cohort"])
    panel = str(row["panel"])
    sample_type = str(row["sample_type"])
    patient_subset = str(row["patient_subset"])
    agg = str(row["agg"])

    outdir = ensure_dir(cache_root / "patient_matrices" / panel)
    outfile = outdir / f"{cohort}__{sample_type}__{patient_subset}__agg-{agg}.parquet"
    meta_file = outdir / f"{cohort}__{sample_type}__{patient_subset}__agg-{agg}__feature_meta.csv"
    audit_file = outdir / f"{cohort}__{sample_type}__{patient_subset}__agg-{agg}__build_audit.csv"
    failure_file = outdir / f"{cohort}__{sample_type}__{patient_subset}__agg-{agg}__failures.csv"
    done_file = outdir / f"{cohort}__{sample_type}__{patient_subset}__agg-{agg}.done"

    expanded = pd.read_csv(setup_root / "shared_union_cross_cohort_expanded_manifest.csv")
    req = expanded[
        expanded["cohort"].astype(str).eq(cohort)
        & expanded["panel"].astype(str).eq(panel)
    ].copy()
    req = req.drop_duplicates("feature_uid").reset_index(drop=True)

    required_uids = req["feature_uid"].astype(str).tolist()

    if outfile.exists() and done_file.exists() and not args.force:
        existing = pd.read_parquet(outfile)
        missing = [u for u in required_uids if u not in existing.columns]
        if not missing:
            print(f"[CACHE COMPLETE] {cohort} {panel}: {len(required_uids)} requested")
            return

    old_path = find_old_cache_matrix(
        cfg, cohort, panel, sample_type, patient_subset, agg
    )
    old = pd.DataFrame()
    if old_path is not None:
        old = pd.read_parquet(old_path)
        print(f"[REUSE SOURCE] {old_path}")

    # Also reuse an incomplete v9 matrix if one already exists. If v9 is
    # incomplete, supplement it from the older v8 cache before rebuilding raw.
    current = pd.DataFrame()
    if outfile.exists():
        current = pd.read_parquet(outfile)

    current_uids = [
        u for u in required_uids
        if not current.empty and u in current.columns
    ]
    old_uids = [
        u for u in required_uids
        if u not in set(current_uids)
        and not old.empty
        and u in old.columns
    ]

    reusable = current_uids + old_uids
    missing_uids = [u for u in required_uids if u not in set(reusable)]

    matrix = pd.DataFrame()
    if current_uids:
        matrix = current[["patient_id"] + current_uids].copy()

    if old_uids:
        old_part = old[["patient_id"] + old_uids].copy()
        if matrix.empty:
            matrix = old_part
        else:
            matrix = matrix.merge(
                old_part,
                on="patient_id",
                how="outer",
                validate="one_to_one",
            )

    audit_rows: List[dict] = []
    for u in current_uids:
        audit_rows.append({
            "feature_uid": u,
            "status": "reused",
            "source": str(outfile),
        })
    for u in old_uids:
        audit_rows.append({
            "feature_uid": u,
            "status": "reused",
            "source": str(old_path),
        })
    meta_rows: List[dict] = []
    failures: List[dict] = []

    if missing_uids:
        stage1 = import_module_from_path(
            "stage1_v6_stage2b_v9_cache",
            cfg["stage1_script_path"],
        )

        missing_req = req[req["feature_uid"].astype(str).isin(missing_uids)].copy()

        for (feature_source, feature_group), g in missing_req.groupby(
            ["feature_source", "feature_group"], dropna=False
        ):
            feature_source = str(feature_source)
            feature_group = str(feature_group)
            features = g["feature"].astype(str).drop_duplicates().tolist()

            try:
                data_dict = stage1.load_data_dict(
                    feature_group=feature_group,
                    feature_source=feature_source,
                    panels=[panel],
                    cohorts=[cohort],
                    spatial_root=cfg.get("spatial_root"),
                    cell_features_path=cfg.get("cell_features_path"),
                    triads_path=cfg.get("triads_path"),
                )
                harm_df = stage1.load_harmonized_df(cfg["harmonized_path"])

                kwargs = dict(
                    data_dict=data_dict,
                    feature_group=feature_group,
                    cohort=cohort,
                    panel=panel,
                    qc_acceptability=str(
                        cfg.get("qc_acceptability", "acceptable_or_borderline")
                    ),
                    min_epi_fraction=cfg.get("min_epi_fraction", 0.05),
                    sample_type=sample_type,
                )
                if cfg.get("koll_metadata_csv") is not None:
                    kwargs["koll_metadata_csv"] = cfg["koll_metadata_csv"]

                core_df = stage1.prepare_core_level_feature_table(**kwargs)
                if core_df.empty:
                    raise ValueError("No cores after filters")

                core_df = stage1.merge_harmonized_to_core_df(core_df, harm_df)
                core_df = stage1.replace_with_harmonized_columns(core_df)
                core_df = stage1.simplify_clinical_vars(core_df)
                core_df = stage1.ensure_patient_id_column(core_df)

                present = [f for f in features if f in core_df.columns]
                if not present:
                    raise ValueError("None of requested features present")

                patient_df = stage1.aggregate_core_to_patient(
                    core_df,
                    feature_cols=present,
                    agg=agg,
                )
                if "cohort" in patient_df.columns:
                    patient_df = patient_df[
                        patient_df["cohort"].astype(str).eq(cohort)
                    ].copy()

                if (
                    cohort in {"No-NAC", "KOLL"}
                    and patient_subset in {"no_adj_chemo", "adj_chemo"}
                ):
                    patient_df = stage1.apply_patient_subset(
                        patient_df,
                        patient_subset=patient_subset,
                    )

                tmp = patient_df[["patient_id"]].copy()

                feature_to_uid = (
                    g.drop_duplicates("feature")
                    .set_index("feature")["feature_uid"]
                    .astype(str)
                    .to_dict()
                )

                for feat, uid in feature_to_uid.items():
                    if feat not in patient_df.columns:
                        failures.append({
                            "cohort": cohort,
                            "panel": panel,
                            "feature_source": feature_source,
                            "feature_group": feature_group,
                            "feature": feat,
                            "feature_uid": uid,
                            "reason": "feature_missing_from_patient_df",
                        })
                        continue

                    vec = zscore(patient_df[feat])
                    tmp[uid] = vec

                    meta_rows.append({
                        "cohort": cohort,
                        "panel": panel,
                        "sample_type": sample_type,
                        "patient_subset": patient_subset,
                        "agg": agg,
                        "feature_uid": uid,
                        "feature": feat,
                        "feature_source": feature_source,
                        "feature_group": feature_group,
                        "selected_transform_mode": "zscore",
                        "n_patients": int(len(vec)),
                        "nonmissing_fraction": float(vec.notna().mean()),
                        "n_unique": int(vec.dropna().nunique()),
                        "cache_source": "newly_built_v9",
                    })
                    audit_rows.append({
                        "feature_uid": uid,
                        "status": "newly_built",
                        "source": "raw_stage1",
                    })

                if matrix.empty:
                    matrix = tmp
                else:
                    matrix = matrix.merge(
                        tmp,
                        on="patient_id",
                        how="outer",
                        validate="one_to_one",
                    )

            except Exception as exc:
                for _, r in g.iterrows():
                    failures.append({
                        "cohort": cohort,
                        "panel": panel,
                        "feature_source": feature_source,
                        "feature_group": feature_group,
                        "feature": r["feature"],
                        "feature_uid": r["feature_uid"],
                        "reason": f"{type(exc).__name__}: {exc}",
                    })

    if matrix.empty:
        matrix = pd.DataFrame(columns=["patient_id"])

    # Preserve only requested candidate columns and deterministic order.
    present_final = [u for u in required_uids if u in matrix.columns]
    matrix = matrix[["patient_id"] + present_final].copy()
    matrix.to_parquet(outfile, index=False)

    pd.DataFrame(meta_rows).to_csv(meta_file, index=False)
    pd.DataFrame(audit_rows).to_csv(audit_file, index=False)
    pd.DataFrame(
        failures,
        columns=[
            "cohort", "panel", "feature_source", "feature_group",
            "feature", "feature_uid", "reason",
        ],
    ).to_csv(failure_file, index=False)

    missing_final = [u for u in required_uids if u not in matrix.columns]

    summary = {
        "cohort": cohort,
        "panel": panel,
        "n_requested": len(required_uids),
        "n_reused": len(reusable),
        "n_newly_built": int(
            sum(r.get("status") == "newly_built" for r in audit_rows)
        ),
        "n_present_final": len(present_final),
        "n_missing_final": len(missing_final),
        "n_patients": int(len(matrix)),
        "old_cache_path": str(old_path) if old_path is not None else None,
        "matrix_path": str(outfile),
    }
    with open(outdir / f"{cohort}__{sample_type}__{patient_subset}__agg-{agg}__summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    done_file.write_text("complete\n")

    print(json.dumps(summary, indent=2))
    print(f"[SAVE] {outfile}")


if __name__ == "__main__":
    main()
