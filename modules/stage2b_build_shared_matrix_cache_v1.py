#!/usr/bin/env python3
"""
stage2b_build_shared_matrix_cache_v1.py

Build/augment one shared all-cap union patient-matrix cache for Stage 2B sensitivity runs.

The script first reuses matching feature columns from the existing Stage 2A-4
patient matrices. It reconstructs only missing feature/cohort combinations through
the existing Stage 1 v6 loaders. The saved matrices use global feature_uid columns
and can be subset cheaply by cap 10, 15, and 20 Stage 2B runs.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd


def log(msg: str) -> None:
    print(msg, flush=True)


def read_json(path: str | Path) -> dict:
    with open(path, "r") as handle:
        return json.load(handle)


def import_module(name: str, path: str | Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        alternatives = [path.with_suffix(".csv.gz"), path.with_suffix(".csv")]
        for alt in alternatives:
            if alt.exists():
                path = alt
                break
    if not path.exists():
        raise FileNotFoundError(path)
    if path.name.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def normalize_key(df: pd.DataFrame) -> pd.Series:
    return (
        df["feature_source"].astype(str)
        + "||"
        + df["feature_group"].astype(str)
        + "||"
        + df["feature"].astype(str)
    )


def load_stage2a4_candidates(
    stage2a4_manifest: pd.DataFrame,
    cohort: str,
    panel: str,
    sample_type: str,
    patient_subset: str,
    agg: str,
) -> List[dict]:
    rows = stage2a4_manifest[
        stage2a4_manifest["cohort"].astype(str).eq(cohort)
        & stage2a4_manifest["panel"].astype(str).eq(panel)
        & stage2a4_manifest["sample_type"].astype(str).eq(sample_type)
        & stage2a4_manifest["patient_subset"].astype(str).eq(patient_subset)
        & stage2a4_manifest["agg"].astype(str).eq(agg)
    ]
    out = []
    for _, row in rows.iterrows():
        mpath = Path(str(row["matrix_path"]))
        rpath = Path(str(row["candidate_registry_path"]))
        meta_path = Path(str(row.get("matrix_feature_meta_path", "")))
        if not mpath.exists() or not rpath.exists():
            continue
        reg = pd.read_csv(rpath)
        if reg.empty:
            continue
        reg = reg.copy()
        reg["feature_key"] = normalize_key(reg)
        matrix = read_table(mpath)
        if "patient_id" not in matrix.columns:
            continue
        meta = pd.read_csv(meta_path) if meta_path.exists() else pd.DataFrame()
        out.append({
            "context_id": row.get("context_id"),
            "matrix_path": str(mpath),
            "registry_path": str(rpath),
            "matrix": matrix,
            "registry": reg,
            "meta": meta,
        })
    return out


def reuse_existing_features(
    required: pd.DataFrame,
    stage2a4_sources: List[dict],
) -> Tuple[pd.DataFrame, pd.DataFrame, set]:
    candidates_by_global: Dict[str, List[dict]] = {}
    required = required.copy()
    required["feature_key"] = normalize_key(required)
    required_by_key = {
        str(r["feature_key"]): r
        for _, r in required.drop_duplicates("feature_key").iterrows()
    }

    for src in stage2a4_sources:
        matrix = src["matrix"]
        registry = src["registry"]
        for _, rr in registry.iterrows():
            key = str(rr["feature_key"])
            if key not in required_by_key:
                continue
            local_uid = str(rr["feature_uid"])
            if local_uid not in matrix.columns:
                continue
            req = required_by_key[key]
            global_uid = str(req["feature_uid"])
            values = pd.to_numeric(matrix[local_uid], errors="coerce")
            candidates_by_global.setdefault(global_uid, []).append({
                "global_uid": global_uid,
                "local_uid": local_uid,
                "feature_key": key,
                "patient_id": matrix["patient_id"].astype(str),
                "values": values,
                "nonmissing_fraction": float(values.notna().mean()),
                "n_unique": int(values.dropna().nunique()),
                "source_context_id": src["context_id"],
                "source_matrix_path": src["matrix_path"],
                "selected_transform_mode": rr.get("selected_transform_mode", rr.get("transform_mode", "unknown")),
            })

    chosen = {}
    audit_rows = []
    patient_union = set()
    for global_uid, options in candidates_by_global.items():
        options = sorted(
            options,
            key=lambda x: (
                str(x.get("selected_transform_mode")) != "zscore",
                -x["nonmissing_fraction"],
                -x["n_unique"],
                str(x["source_context_id"]),
            ),
        )
        best = options[0]
        s = pd.Series(best["values"].to_numpy(), index=best["patient_id"].to_numpy(), name=global_uid)
        s = s[~s.index.duplicated(keep="first")]
        chosen[global_uid] = s
        patient_union.update(s.index.astype(str))
        audit_rows.append({
            "feature_uid": global_uid,
            "cache_source": "reused_stage2a4",
            "local_feature_uid": best["local_uid"],
            "source_context_id": best["source_context_id"],
            "source_matrix_path": best["source_matrix_path"],
            "source_transform_mode": best["selected_transform_mode"],
            "n_reuse_options": len(options),
            "nonmissing_fraction": best["nonmissing_fraction"],
            "n_unique": best["n_unique"],
        })

    if patient_union:
        index = pd.Index(sorted(patient_union), name="patient_id")
        matrix = pd.DataFrame(index=index)
        for uid, s in chosen.items():
            matrix[uid] = s.reindex(index)
        matrix = matrix.reset_index()
    else:
        matrix = pd.DataFrame(columns=["patient_id"])
    return matrix, pd.DataFrame(audit_rows), set(chosen)


def rebuild_missing_features(
    missing: pd.DataFrame,
    cohort: str,
    panel: str,
    sample_type: str,
    patient_subset: str,
    agg: str,
    stage1_mod,
    gm,
    cfg: Mapping,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[dict]]:
    if missing.empty:
        return pd.DataFrame(columns=["patient_id"]), pd.DataFrame(), []

    parts = []
    audit = []
    failures: List[dict] = []
    for (source, group), g in missing.groupby(["feature_source", "feature_group"], dropna=False):
        features = g["feature"].astype(str).drop_duplicates().tolist()
        try:
            raw = gm.build_patient_matrix_for_source_group(
                stage1_mod=stage1_mod,
                cohort=cohort,
                panel=panel,
                feature_source=str(source),
                feature_group=str(group),
                features=features,
                sample_type=sample_type,
                patient_subset=patient_subset,
                agg=agg,
                qc_acceptability=str(cfg.get("qc_acceptability", "acceptable_or_borderline")),
                min_epi_fraction=cfg.get("min_epi_fraction", 0.05),
                harmonized_path=cfg["harmonized_path"],
                spatial_root=cfg.get("spatial_root"),
                cell_features_path=cfg.get("cell_features_path"),
                triads_path=cfg.get("triads_path"),
                koll_metadata_csv=cfg.get("koll_metadata_csv"),
            )
        except Exception as exc:
            for _, row in g.iterrows():
                failures.append({
                    "cohort": cohort,
                    "panel": panel,
                    "feature_uid": row["feature_uid"],
                    "feature_source": source,
                    "feature_group": group,
                    "feature": row["feature"],
                    "reason": f"{type(exc).__name__}: {exc}",
                })
            continue

        tmp = raw[["patient_id"]].copy()
        for _, row in g.iterrows():
            uid = str(row["feature_uid"])
            feature = str(row["feature"])
            if feature not in raw.columns:
                failures.append({
                    "cohort": cohort,
                    "panel": panel,
                    "feature_uid": uid,
                    "feature_source": source,
                    "feature_group": group,
                    "feature": feature,
                    "reason": "feature_missing_from_stage1_patient_matrix",
                })
                continue
            values = gm.fit_apply_transform_full(raw[feature], "zscore")
            tmp[uid] = values
            audit.append({
                "feature_uid": uid,
                "cache_source": "rebuilt_missing",
                "local_feature_uid": "",
                "source_context_id": "",
                "source_matrix_path": "",
                "source_transform_mode": "zscore",
                "n_reuse_options": 0,
                "nonmissing_fraction": float(values.notna().mean()),
                "n_unique": int(values.dropna().nunique()),
            })
        parts.append(tmp)

    merged = None
    for part in parts:
        merged = part if merged is None else merged.merge(part, on="patient_id", how="outer")
    if merged is None:
        merged = pd.DataFrame(columns=["patient_id"])
    return merged, pd.DataFrame(audit), failures


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = read_json(args.config)

    output_root = Path(cfg["output_root"])
    matrix_root = output_root / "patient_matrices"
    matrix_root.mkdir(parents=True, exist_ok=True)

    gm = import_module("stage2_global_module_utils_for_shared_cache", cfg["stage2_global_module_utils_path"])
    stage1_mod = import_module("stage1_runtime_v6_for_shared_cache", cfg["stage1_script_path"])

    manifest = pd.read_csv(cfg["union_manifest"])
    stage2a4_root = Path(cfg["stage2a4_root"])
    stage2a4_manifest_path = stage2a4_root / "stage2a4_matrix_manifest.csv"
    stage2a4_manifest = pd.read_csv(stage2a4_manifest_path) if stage2a4_manifest_path.exists() else pd.DataFrame()
    if stage2a4_manifest.empty:
        log(f"[WARN] No Stage 2A-4 matrix manifest found at {stage2a4_manifest_path}; all features will be rebuilt")

    overall_audit = []
    failures = []
    context_rows = []
    feature_meta_parts = []

    group_cols = ["cohort", "panel", "sample_type", "patient_subset", "agg"]
    for keys, required in manifest.groupby(group_cols, dropna=False):
        cohort, panel, sample_type, patient_subset, agg = [str(x) for x in keys]
        required = required.drop_duplicates("feature_uid").copy()
        outdir = matrix_root / panel
        outdir.mkdir(parents=True, exist_ok=True)
        outpath = outdir / f"{cohort}__{sample_type}__{patient_subset}__agg-{agg}.parquet"
        metapath = outdir / f"{cohort}__{sample_type}__{patient_subset}__agg-{agg}__feature_meta.csv"
        auditpath = outdir / f"{cohort}__{sample_type}__{patient_subset}__agg-{agg}__reuse_audit.csv"

        # Incremental cache logic: existing matrices are reused, but we still
        # check whether the current all-cap union requires additional feature_uids.
        # This avoids a full rebuild when a smaller-cap microcompression run chose
        # a representative that was absent from the original cap-20 cache.
        existing_matrix = None
        existing_meta = pd.DataFrame()
        existing_uids = set()
        if outpath.exists() and metapath.exists() and not args.force:
            existing_matrix = pd.read_parquet(outpath)
            existing_meta = pd.read_csv(metapath)
            existing_uids = set(existing_matrix.columns) - {"patient_id"}
            log(f"[LOAD] {outpath} shape={existing_matrix.shape}")

        required_uids = set(required["feature_uid"].astype(str))
        missing_required = required[~required["feature_uid"].astype(str).isin(existing_uids)].copy()

        if existing_matrix is not None and missing_required.empty:
            matrix = existing_matrix
            meta = existing_meta
            log(f"[CACHE COMPLETE] {cohort} {panel}: all {len(required_uids)} required features already present")
        else:
            log(f"[CONTEXT] {cohort} {panel} {sample_type} {patient_subset} {agg} | required={len(required_uids)} existing={len(existing_uids)} add={len(missing_required)}")
            to_build = required if existing_matrix is None else missing_required
            sources = []
            if not stage2a4_manifest.empty and bool(cfg.get("prefer_reuse", True)):
                sources = load_stage2a4_candidates(
                    stage2a4_manifest,
                    cohort=cohort,
                    panel=panel,
                    sample_type=sample_type,
                    patient_subset=patient_subset,
                    agg=agg,
                )
            reused, reuse_audit, reused_uids = reuse_existing_features(to_build, sources)
            missing = to_build[~to_build["feature_uid"].astype(str).isin(reused_uids)].copy()
            log(f"[REUSE] reused_new={len(reused_uids)} rebuild_new={len(missing)} source_contexts={len(sources)}")

            rebuilt = pd.DataFrame(columns=["patient_id"])
            rebuild_audit = pd.DataFrame()
            build_failures: List[dict] = []
            if not missing.empty and bool(cfg.get("rebuild_missing", True)):
                rebuilt, rebuild_audit, build_failures = rebuild_missing_features(
                    missing,
                    cohort=cohort,
                    panel=panel,
                    sample_type=sample_type,
                    patient_subset=patient_subset,
                    agg=agg,
                    stage1_mod=stage1_mod,
                    gm=gm,
                    cfg=cfg,
                )

            additions = reused
            if rebuilt.shape[1] > 1:
                additions = rebuilt if additions.shape[1] <= 1 else additions.merge(rebuilt, on="patient_id", how="outer")
            if additions.empty:
                additions = pd.DataFrame(columns=["patient_id"])
            additions["patient_id"] = additions["patient_id"].astype(str)

            if existing_matrix is None:
                matrix = additions
            else:
                matrix = existing_matrix.copy()
                matrix["patient_id"] = matrix["patient_id"].astype(str)
                if additions.shape[1] > 1:
                    matrix = matrix.merge(additions, on="patient_id", how="outer")

            audit_new = pd.concat([reuse_audit, rebuild_audit], ignore_index=True, sort=False)
            built_uids = set(matrix.columns) - {"patient_id"}
            missing_after = sorted(required_uids - built_uids)
            for uid in missing_after:
                rr = required[required["feature_uid"].astype(str).eq(uid)].iloc[0]
                build_failures.append({
                    "cohort": cohort,
                    "panel": panel,
                    "feature_uid": uid,
                    "feature_source": rr["feature_source"],
                    "feature_group": rr["feature_group"],
                    "feature": rr["feature"],
                    "reason": "missing_after_reuse_and_rebuild",
                })
            if missing_after and bool(cfg.get("fail_if_any_missing", True)):
                pd.DataFrame(build_failures).to_csv(output_root / "shared_cache_build_failures.csv", index=False)
                raise RuntimeError(
                    f"{cohort}/{panel}: {len(missing_after)} required features remain missing; see shared_cache_build_failures.csv"
                )

            # Preserve all union-required columns in manifest order.
            ordered = [uid for uid in required["feature_uid"].astype(str) if uid in matrix.columns]
            matrix = matrix[["patient_id", *ordered]].copy()
            matrix.to_parquet(outpath, index=False)

            meta_required = required[[
                c for c in [
                    "cohort", "panel", "sample_type", "patient_subset", "agg",
                    "feature_uid", "feature", "feature_source", "feature_group",
                    "selected_transform_mode", "canonical_feature_id",
                ] if c in required.columns
            ]].copy()
            meta_required = meta_required[meta_required["feature_uid"].astype(str).isin(ordered)]
            audit_cols = [c for c in ["feature_uid", "cache_source", "nonmissing_fraction", "n_unique"] if c in audit_new.columns]
            if audit_cols:
                meta_required = meta_required.merge(audit_new[audit_cols], on="feature_uid", how="left")
            if not existing_meta.empty:
                # Keep prior cache provenance when the new incremental audit has no row.
                prior_cols = [c for c in ["feature_uid", "cache_source", "nonmissing_fraction", "n_unique"] if c in existing_meta.columns]
                if len(prior_cols) > 1:
                    prior = existing_meta[prior_cols].drop_duplicates("feature_uid")
                    meta_required = meta_required.merge(prior, on="feature_uid", how="left", suffixes=("", "_prior"))
                    for c in ["cache_source", "nonmissing_fraction", "n_unique"]:
                        pc = f"{c}_prior"
                        if pc in meta_required.columns:
                            if c not in meta_required.columns:
                                meta_required[c] = meta_required[pc]
                            else:
                                meta_required[c] = meta_required[c].combine_first(meta_required[pc])
                            meta_required = meta_required.drop(columns=[pc])
            meta = meta_required
            meta.to_csv(metapath, index=False)

            if auditpath.exists() and existing_matrix is not None:
                prior_audit = pd.read_csv(auditpath)
                audit = pd.concat([prior_audit, audit_new], ignore_index=True, sort=False).drop_duplicates("feature_uid", keep="last")
            else:
                audit = audit_new
            audit.to_csv(auditpath, index=False)
            failures.extend(build_failures)
            log(f"[SAVE/AUGMENT] {outpath} shape={matrix.shape}")

        meta = pd.read_csv(metapath)
        feature_meta_parts.append(meta)
        if auditpath.exists():
            tmp_audit = pd.read_csv(auditpath)
            tmp_audit["cohort"] = cohort
            tmp_audit["panel"] = panel
            overall_audit.append(tmp_audit)
        context_rows.append({
            "context_matrix_id": "__".join([cohort, panel, sample_type, patient_subset, agg]),
            "cohort": cohort,
            "panel": panel,
            "sample_type": sample_type,
            "patient_subset": patient_subset,
            "agg": agg,
            "path": str(outpath),
            "feature_meta_path": str(metapath),
            "n_patients": int(matrix.shape[0]),
            "n_feature_uid_columns": int(max(matrix.shape[1] - 1, 0)),
        })

    pd.DataFrame(context_rows).to_csv(matrix_root / "context_matrix_manifest.csv", index=False)
    pd.concat(feature_meta_parts, ignore_index=True, sort=False).to_csv(matrix_root / "context_feature_meta.csv", index=False)
    pd.concat(overall_audit, ignore_index=True, sort=False).to_csv(output_root / "shared_cache_reuse_summary.csv", index=False)
    pd.DataFrame(failures).to_csv(output_root / "shared_cache_build_failures.csv", index=False)

    summary = pd.DataFrame(context_rows)
    summary.to_csv(output_root / "shared_cache_context_summary.csv", index=False)
    log(f"[DONE] Shared cache built under {output_root}")


if __name__ == "__main__":
    main()