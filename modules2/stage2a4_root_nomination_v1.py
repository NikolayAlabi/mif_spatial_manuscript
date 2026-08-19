#!/usr/bin/env python3
"""
stage2a4_root_nomination_v1.py

Finalized root-aware Stage 2A-4.

Purpose
-------
1. Consume the already thresholded/ranked candidate shards and shared raw
   patient matrices produced by stage2a_candidate_cap_sensitivity_v1.
2. Apply a FIXED panel x prep-root candidate cap.
3. Select the top candidates independently within each prep root using the
   frozen root-specific evidence rank from Stage 2A steps 1-3.
4. Build transformed patient x feature matrices for downstream within-root
   redundancy compression.
5. Perform NO cross-root rescue, simplification, deduplication, or correlation
   compression. Those operations are intentionally deferred to the root-aware
   Stage 2A-5 / Stage 2B workflow.

The cap is a maximum, not a quota: if a context/root has fewer eligible
candidates than its fixed cap, all eligible candidates are retained.

Commands
--------
validate   Validate cap coverage, upstream cache completion, and candidate files.
worker     Process one endpoint context (one CPU; SLURM_ARRAY_TASK_ID supported).
aggregate  Aggregate context/root nomination and matrix manifests.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


CONTEXT_COLS = [
    "cohort", "panel", "endpoint", "sample_type", "patient_subset", "agg"
]
MATRIX_COLS = [
    "cohort", "panel", "sample_type", "patient_subset", "agg"
]
ROOT_COL = "feature_source"
CAP_KEY_COLS = ["panel", ROOT_COL]


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: Union[str, Path]) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def write_json(obj: Mapping, path: Union[str, Path]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def read_table(path: Union[str, Path]) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        alternatives = [
            p.with_suffix(".csv.gz"),
            p.with_suffix(".csv"),
        ]
        for alt in alternatives:
            if alt.exists():
                p = alt
                break
    if not p.exists():
        raise FileNotFoundError(str(path))
    low = p.name.lower()
    if low.endswith(".parquet"):
        return pd.read_parquet(p)
    if low.endswith(".tsv") or low.endswith(".tsv.gz"):
        return pd.read_csv(p, sep="\t")
    return pd.read_csv(p)


def save_table(df: pd.DataFrame, path: Union[str, Path]) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    if p.suffix.lower() == ".parquet":
        try:
            df.to_parquet(p, index=False)
            return p
        except (ImportError, ModuleNotFoundError):
            p = p.with_suffix(".csv.gz")
            df.to_csv(p, index=False, compression="gzip")
            return p
    df.to_csv(p, index=False)
    return p


def parse_bool(v: object) -> bool:
    if isinstance(v, bool):
        return v
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return False
    return str(v).strip().lower() in {
        "1", "true", "t", "yes", "y", "include", "included"
    }


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def fit_apply_transform_full(x: pd.Series, mode: str) -> pd.Series:
    """Match the full-data transformation used in the established pipeline."""
    x = safe_numeric(x)
    mode = str(mode) if pd.notna(mode) else "zscore"
    if mode == "raw":
        return x
    if mode == "log1p_zscore":
        if (x.dropna() < 0).any():
            return pd.Series(np.nan, index=x.index, dtype=float)
        x = np.log1p(x)
    mu = x.mean(skipna=True)
    sd = x.std(skipna=True)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=x.index, dtype=float)
    return (x - mu) / sd


def upstream_root(cfg: Mapping) -> Path:
    return Path(cfg["cap_sensitivity_output_root"])


def output_root(cfg: Mapping) -> Path:
    return Path(cfg["output_root"])


def load_context_index(cfg: Mapping) -> pd.DataFrame:
    p = upstream_root(cfg) / "context_index.csv"
    idx = pd.read_csv(p)
    required = {"array_id", "context_slug", "candidate_shard", "cache_id", "cache_slug", *CONTEXT_COLS}
    missing = sorted(required - set(idx.columns))
    if missing:
        raise ValueError(f"Context index missing columns: {missing}")
    if idx["array_id"].duplicated().any():
        raise ValueError("Duplicate array_id values in upstream context_index.csv")
    return idx


def load_caps(cfg: Mapping) -> pd.DataFrame:
    p = Path(cfg["root_caps_csv"])
    caps = pd.read_csv(p)
    required = {"panel", ROOT_COL, "max_candidates"}
    missing = sorted(required - set(caps.columns))
    if missing:
        raise ValueError(f"Root caps CSV missing columns: {missing}")
    if caps.duplicated(CAP_KEY_COLS).any():
        dup = caps.loc[caps.duplicated(CAP_KEY_COLS, keep=False), CAP_KEY_COLS]
        raise ValueError(f"Duplicate panel/root caps:\n{dup.to_string(index=False)}")
    caps["max_candidates"] = pd.to_numeric(caps["max_candidates"], errors="raise").astype(int)
    if (caps["max_candidates"] < 1).any():
        raise ValueError("All max_candidates values must be >= 1")
    return caps


def cap_map(caps: pd.DataFrame) -> Dict[Tuple[str, str], int]:
    return {
        (str(r["panel"]), str(r[ROOT_COL])): int(r["max_candidates"])
        for _, r in caps.iterrows()
    }


def expected_roots_by_panel(caps: pd.DataFrame) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for panel, g in caps.groupby("panel", sort=False):
        out[str(panel)] = g[ROOT_COL].astype(str).tolist()
    return out


def context_output_dir(cfg: Mapping, index_row: pd.Series) -> Path:
    return output_root(cfg) / "contexts" / str(index_row["context_slug"])


def root_output_dir(context_dir: Path, root: str) -> Path:
    safe = str(root).replace("/", "_")
    return context_dir / "roots" / safe


def get_index_row(cfg: Mapping, array_id: int) -> pd.Series:
    idx = load_context_index(cfg)
    m = idx[idx["array_id"].astype(int) == int(array_id)]
    if m.empty:
        raise IndexError(f"array_id={array_id} not found")
    return m.iloc[0]


def resolve_array_id(v: Optional[int]) -> int:
    if v is not None:
        return int(v)
    env = os.environ.get("SLURM_ARRAY_TASK_ID")
    if env is None:
        raise ValueError("Provide --array-id or run in a Slurm array task")
    return int(env)


def locate_cache_matrix(cfg: Mapping, index_row: pd.Series) -> Path:
    cdir = upstream_root(cfg) / "shared_cache" / str(index_row["cache_slug"])
    p = cdir / "patient_feature_matrix.parquet"
    if p.exists():
        return p
    for alt in [cdir / "patient_feature_matrix.csv.gz", cdir / "patient_feature_matrix.csv"]:
        if alt.exists():
            return alt
    raise FileNotFoundError(f"No upstream patient matrix in {cdir}")


def validate_upstream_cache(cfg: Mapping, index_row: pd.Series) -> Tuple[bool, str]:
    cdir = upstream_root(cfg) / "shared_cache" / str(index_row["cache_slug"])
    done = cdir / ".done"
    if not done.exists():
        return False, f"missing .done in {cdir}"
    try:
        locate_cache_matrix(cfg, index_row)
    except Exception as exc:
        return False, str(exc)
    return True, "ok"


def select_root_candidates(
    shard: pd.DataFrame,
    panel: str,
    caps: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Select top candidates within each root using fixed panel x root caps."""
    cmap = cap_map(caps)
    roots_expected = expected_roots_by_panel(caps).get(str(panel), [])

    if not shard.empty:
        required = {
            ROOT_COL, "feature_uid", "feature", "feature_group",
            "eligible_root_rank", "root_candidate_evidence_score",
            "oof_metric", "fold_sd", "selected_transform_mode",
        }
        missing = sorted(required - set(shard.columns))
        if missing:
            raise ValueError(f"Candidate shard missing columns: {missing}")

        observed_roots = sorted(shard[ROOT_COL].dropna().astype(str).unique())
        unexpected = [r for r in observed_roots if (str(panel), r) not in cmap]
        if unexpected:
            raise ValueError(
                f"Observed roots without a fixed cap for panel={panel}: {unexpected}"
            )

    selected_parts: List[pd.DataFrame] = []
    summary_rows: List[dict] = []

    for root in roots_expected:
        cap = int(cmap[(str(panel), str(root))])
        if shard.empty:
            g = shard.copy()
        else:
            g = shard[shard[ROOT_COL].astype(str).eq(str(root))].copy()

        if not g.empty:
            g["eligible_root_rank"] = pd.to_numeric(g["eligible_root_rank"], errors="coerce")
            g = g.sort_values(
                [
                    "eligible_root_rank",
                    "root_candidate_evidence_score",
                    "oof_metric",
                    "fold_sd",
                    "nonmissing_fraction" if "nonmissing_fraction" in g.columns else "feature_uid",
                ],
                ascending=[True, False, False, True, False],
                na_position="last",
            )
            # eligible_root_rank was frozen upstream. Use it as the primary rank,
            # then defensively deduplicate feature_uid.
            g = g.drop_duplicates("feature_uid", keep="first")
            take = g.head(cap).copy()
            take["fixed_root_cap"] = cap
            take["selected_stage2a4"] = True
            take["stage2a4_root_rank"] = np.arange(1, len(take) + 1)
            selected_parts.append(take)
        else:
            take = g

        summary_rows.append({
            "panel": str(panel),
            ROOT_COL: str(root),
            "fixed_root_cap": cap,
            "n_eligible_top_depth_available": int(len(g)),
            "n_selected": int(len(take)),
            "cap_filled": bool(len(take) >= cap),
            "underfill_n": int(max(cap - len(take), 0)),
            "max_eligible_rank_available": (
                int(pd.to_numeric(g["eligible_root_rank"], errors="coerce").max())
                if not g.empty and pd.to_numeric(g["eligible_root_rank"], errors="coerce").notna().any()
                else 0
            ),
            "worst_selected_oof": (
                float(pd.to_numeric(take["oof_metric"], errors="coerce").min())
                if not take.empty else np.nan
            ),
            "worst_selected_fold_sd": (
                float(pd.to_numeric(take["fold_sd"], errors="coerce").max())
                if not take.empty else np.nan
            ),
            "lowest_selected_root_evidence": (
                float(pd.to_numeric(take["root_candidate_evidence_score"], errors="coerce").min())
                if not take.empty else np.nan
            ),
        })

    selected = (
        pd.concat(selected_parts, ignore_index=True, sort=False)
        if selected_parts else pd.DataFrame()
    )
    return selected, pd.DataFrame(summary_rows)


def build_transformed_context_matrix(
    raw_matrix: pd.DataFrame,
    selected: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Subset raw cache to selected UIDs and apply context-specific transforms."""
    if "patient_id" not in raw_matrix.columns:
        raise ValueError("Upstream raw patient matrix has no patient_id column")

    matrix = raw_matrix[["patient_id"]].copy()
    meta_rows: List[dict] = []
    fail_rows: List[dict] = []

    for _, r in selected.iterrows():
        uid = str(r["feature_uid"])
        mode = str(r.get("selected_transform_mode", "zscore"))
        if uid not in raw_matrix.columns:
            fail_rows.append({
                "feature_uid": uid,
                ROOT_COL: r.get(ROOT_COL),
                "feature_group": r.get("feature_group"),
                "feature": r.get("feature"),
                "reason": "feature_uid_missing_from_shared_raw_cache",
            })
            continue

        transformed = fit_apply_transform_full(raw_matrix[uid], mode)
        if transformed.notna().sum() == 0:
            fail_rows.append({
                "feature_uid": uid,
                ROOT_COL: r.get(ROOT_COL),
                "feature_group": r.get("feature_group"),
                "feature": r.get("feature"),
                "reason": f"transform_returned_all_nan:{mode}",
            })
            continue

        matrix[uid] = transformed
        meta_rows.append({
            "feature_uid": uid,
            ROOT_COL: r.get(ROOT_COL),
            "feature_group": r.get("feature_group"),
            "feature": r.get("feature"),
            "selected_transform_mode": mode,
            "fixed_root_cap": r.get("fixed_root_cap"),
            "stage2a4_root_rank": r.get("stage2a4_root_rank"),
            "eligible_root_rank": r.get("eligible_root_rank"),
            "root_candidate_evidence_score": r.get("root_candidate_evidence_score"),
            "oof_metric": r.get("oof_metric"),
            "fold_sd": r.get("fold_sd"),
            "n_patients": int(len(transformed)),
            "n_nonmissing": int(transformed.notna().sum()),
            "nonmissing_fraction_matrix": float(transformed.notna().mean()),
            "n_unique_transformed": int(transformed.dropna().nunique()),
        })

    return matrix, pd.DataFrame(meta_rows), pd.DataFrame(fail_rows)


def command_validate(cfg: Mapping) -> None:
    out = ensure_dir(output_root(cfg))
    idx = load_context_index(cfg)
    caps = load_caps(cfg)

    problems: List[str] = []
    audit_rows: List[dict] = []

    # Require caps for all roots visible in the top-depth candidate universe.
    observed: Dict[str, set] = {}
    for _, row in idx.iterrows():
        shard_path = Path(str(row["candidate_shard"]))
        if not shard_path.exists() and not shard_path.with_suffix(".csv.gz").exists():
            problems.append(f"missing candidate shard: {shard_path}")
            continue
        shard = read_table(shard_path)
        panel = str(row["panel"])
        roots = set(shard[ROOT_COL].dropna().astype(str)) if (not shard.empty and ROOT_COL in shard.columns) else set()
        observed.setdefault(panel, set()).update(roots)

        ok, reason = validate_upstream_cache(cfg, row)
        audit_rows.append({
            "array_id": int(row["array_id"]),
            **{c: row[c] for c in CONTEXT_COLS},
            "candidate_shard_exists": True,
            "n_shard_rows": int(len(shard)),
            "n_roots_in_shard": int(len(roots)),
            "cache_ok": bool(ok),
            "cache_reason": reason,
        })
        if not ok:
            problems.append(f"array_id={row['array_id']} cache not ready: {reason}")

    cmap = cap_map(caps)
    for panel, roots in observed.items():
        for root in sorted(roots):
            if (str(panel), str(root)) not in cmap:
                problems.append(f"missing fixed cap for {panel}/{root}")

    # Snapshot frozen decisions/config for auditability.
    caps.to_csv(out / "root_caps_snapshot.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(out / "stage2a4_validation_context_audit.csv", index=False)
    write_json(dict(cfg), out / "config.resolved.json")

    report = {
        "n_contexts": int(len(idx)),
        "n_cap_rows": int(len(caps)),
        "observed_roots_by_panel": {k: sorted(v) for k, v in observed.items()},
        "problems": problems,
    }
    write_json(report, out / "stage2a4_validation_report.json")

    if problems:
        raise RuntimeError("Stage 2A-4 validation failed:\n- " + "\n- ".join(problems))
    log(f"[VALID] contexts={len(idx)} cap_rows={len(caps)}")


def command_worker(cfg: Mapping, array_id: int) -> None:
    row = get_index_row(cfg, array_id)
    cdir = ensure_dir(context_output_dir(cfg, row))
    caps = load_caps(cfg)

    log("=" * 90)
    log(
        f"[STAGE2A4 ROOT NOMINATION] array={array_id} "
        + " | ".join(f"{c}={row[c]}" for c in CONTEXT_COLS)
    )

    shard = read_table(Path(str(row["candidate_shard"])))
    selected, root_summary = select_root_candidates(shard, str(row["panel"]), caps)

    for c in CONTEXT_COLS:
        root_summary[c] = row[c]
    root_summary["array_id"] = int(array_id)
    root_summary["context_slug"] = row["context_slug"]
    root_summary.to_csv(cdir / "root_nomination_summary.csv", index=False)

    if selected.empty:
        pd.DataFrame().to_csv(cdir / "candidate_registry.csv", index=False)
        summary = pd.DataFrame([{
            "array_id": int(array_id),
            **{c: row[c] for c in CONTEXT_COLS},
            "context_slug": row["context_slug"],
            "status": "zero_selected_candidates",
            "n_selected_candidates": 0,
            "n_matrix_patients": 0,
            "n_matrix_features": 0,
            "n_matrix_failures": 0,
        }])
        summary.to_csv(cdir / "context_stage2a4_summary.csv", index=False)
        (cdir / ".done").write_text("zero_selected_candidates\n")
        log("[DONE] zero selected candidates")
        return

    # Attach context identity explicitly and freeze the candidate registry.
    for c in CONTEXT_COLS:
        if c not in selected.columns:
            selected[c] = row[c]
    selected["array_id"] = int(array_id)
    selected["context_slug"] = row["context_slug"]
    selected["candidate_role"] = "root_seed"
    selected["cross_root_rescue"] = False
    selected["stage2a4_selection_policy"] = "fixed_panel_root_cap"

    selected = selected.sort_values(
        [ROOT_COL, "stage2a4_root_rank"], ascending=[True, True]
    ).reset_index(drop=True)
    selected.to_csv(cdir / "candidate_registry.csv", index=False)

    raw_matrix = read_table(locate_cache_matrix(cfg, row))
    matrix, matrix_meta, failures = build_transformed_context_matrix(raw_matrix, selected)

    # Combined context matrix.
    matrix_path = save_table(matrix, cdir / "patient_feature_matrix.parquet")
    matrix_meta.to_csv(cdir / "matrix_feature_meta.csv", index=False)
    failures.to_csv(cdir / "matrix_build_failures.csv", index=False)

    # Root-specific views are the canonical inputs for root-aware Stage 2A-5.
    root_manifest_rows: List[dict] = []
    for root, reg in selected.groupby(ROOT_COL, sort=False):
        rdir = ensure_dir(root_output_dir(cdir, str(root)))
        built_uids = [u for u in reg["feature_uid"].astype(str) if u in matrix.columns]
        root_matrix = matrix[["patient_id"] + built_uids].copy()
        root_matrix_path = save_table(root_matrix, rdir / "patient_feature_matrix.parquet")
        reg.to_csv(rdir / "candidate_registry.csv", index=False)
        rmeta = matrix_meta[matrix_meta[ROOT_COL].astype(str).eq(str(root))].copy() if not matrix_meta.empty else pd.DataFrame()
        rmeta.to_csv(rdir / "matrix_feature_meta.csv", index=False)

        root_manifest_rows.append({
            "array_id": int(array_id),
            **{c: row[c] for c in CONTEXT_COLS},
            "context_slug": row["context_slug"],
            ROOT_COL: str(root),
            "fixed_root_cap": int(reg["fixed_root_cap"].iloc[0]),
            "n_selected": int(len(reg)),
            "n_built": int(len(built_uids)),
            "matrix_path": str(root_matrix_path),
            "candidate_registry_path": str(rdir / "candidate_registry.csv"),
            "matrix_feature_meta_path": str(rdir / "matrix_feature_meta.csv"),
        })

    pd.DataFrame(root_manifest_rows).to_csv(cdir / "root_matrix_manifest.csv", index=False)

    built_uids = set(matrix_meta["feature_uid"].astype(str)) if not matrix_meta.empty else set()
    selected_uids = set(selected["feature_uid"].astype(str))
    build_audit = selected[[
        c for c in [
            "feature_uid", ROOT_COL, "feature_group", "feature",
            "fixed_root_cap", "stage2a4_root_rank", "eligible_root_rank",
            "root_candidate_evidence_score", "oof_metric", "fold_sd"
        ] if c in selected.columns
    ]].copy()
    build_audit["matrix_build_success"] = build_audit["feature_uid"].astype(str).isin(built_uids)
    build_audit["matrix_build_failure_reason"] = np.where(
        build_audit["matrix_build_success"], "", "not_returned_by_matrix_builder"
    )
    build_audit.to_csv(cdir / "matrix_feature_build_audit.csv", index=False)

    status = "complete" if len(built_uids) > 0 else "zero_matrix_features"
    summary = pd.DataFrame([{
        "array_id": int(array_id),
        **{c: row[c] for c in CONTEXT_COLS},
        "context_slug": row["context_slug"],
        "status": status,
        "n_roots_with_selected_candidates": int(selected[ROOT_COL].nunique()),
        "n_selected_candidates": int(len(selected)),
        "n_matrix_patients": int(matrix.shape[0]),
        "n_matrix_features": int(max(matrix.shape[1] - 1, 0)),
        "n_matrix_build_success": int(len(built_uids)),
        "n_matrix_failures": int(len(selected_uids - built_uids)),
        "combined_matrix_path": str(matrix_path),
    }])
    summary.to_csv(cdir / "context_stage2a4_summary.csv", index=False)
    (cdir / ".done").write_text(status + "\n")
    log(
        f"[DONE] selected={len(selected)} built={len(built_uids)} "
        f"roots={selected[ROOT_COL].nunique()} patients={matrix.shape[0]}"
    )


def command_aggregate(cfg: Mapping) -> None:
    out = ensure_dir(output_root(cfg))
    idx = load_context_index(cfg)

    summary_parts: List[pd.DataFrame] = []
    registry_parts: List[pd.DataFrame] = []
    root_summary_parts: List[pd.DataFrame] = []
    root_manifest_parts: List[pd.DataFrame] = []
    build_parts: List[pd.DataFrame] = []
    context_matrix_rows: List[dict] = []
    missing: List[dict] = []

    for _, row in idx.sort_values("array_id").iterrows():
        cdir = context_output_dir(cfg, row)
        sp = cdir / "context_stage2a4_summary.csv"
        if not sp.exists():
            missing.append({
                "array_id": int(row["array_id"]),
                "context_slug": row["context_slug"],
                "reason": "missing_context_stage2a4_summary",
            })
            continue

        summary_parts.append(pd.read_csv(sp))
        for filename, collection in [
            ("candidate_registry.csv", registry_parts),
            ("root_nomination_summary.csv", root_summary_parts),
            ("root_matrix_manifest.csv", root_manifest_parts),
            ("matrix_feature_build_audit.csv", build_parts),
        ]:
            p = cdir / filename
            if p.exists() and p.stat().st_size > 0:
                try:
                    d = pd.read_csv(p)
                except pd.errors.EmptyDataError:
                    d = pd.DataFrame()
                if not d.empty:
                    collection.append(d)

        mp = cdir / "patient_feature_matrix.parquet"
        if not mp.exists() and mp.with_suffix(".csv.gz").exists():
            mp = mp.with_suffix(".csv.gz")
        if mp.exists():
            context_matrix_rows.append({
                "array_id": int(row["array_id"]),
                **{c: row[c] for c in CONTEXT_COLS},
                "context_slug": row["context_slug"],
                "matrix_path": str(mp),
                "candidate_registry_path": str(cdir / "candidate_registry.csv"),
                "matrix_feature_meta_path": str(cdir / "matrix_feature_meta.csv"),
            })

    summary = pd.concat(summary_parts, ignore_index=True, sort=False) if summary_parts else pd.DataFrame()
    registry = pd.concat(registry_parts, ignore_index=True, sort=False) if registry_parts else pd.DataFrame()
    root_summary = pd.concat(root_summary_parts, ignore_index=True, sort=False) if root_summary_parts else pd.DataFrame()
    root_manifest = pd.concat(root_manifest_parts, ignore_index=True, sort=False) if root_manifest_parts else pd.DataFrame()
    build = pd.concat(build_parts, ignore_index=True, sort=False) if build_parts else pd.DataFrame()

    summary.to_csv(out / "all_context_stage2a4_summary.csv", index=False)
    save_table(registry, out / "all_context_root_seed_candidates.parquet")
    root_summary.to_csv(out / "all_context_root_nomination_summary.csv", index=False)
    root_manifest.to_csv(out / "stage2a4_root_matrix_manifest.csv", index=False)
    pd.DataFrame(context_matrix_rows).to_csv(out / "stage2a4_context_matrix_manifest.csv", index=False)
    build.to_csv(out / "all_context_matrix_feature_build_audit.csv", index=False)
    pd.DataFrame(missing).to_csv(out / "stage2a4_missing_context_outputs.csv", index=False)

    if not registry.empty:
        comp = (
            registry.groupby(["panel", ROOT_COL, "feature_group"], dropna=False)
            .agg(
                n_nominations=("feature_uid", "size"),
                n_unique_features=("feature_uid", "nunique"),
                n_contexts=("context_slug", "nunique"),
                median_oof=("oof_metric", "median"),
                median_fold_sd=("fold_sd", "median"),
            )
            .reset_index()
        )
        comp.to_csv(out / "stage2a4_candidate_composition.csv", index=False)

        support = (
            registry.groupby(["panel", ROOT_COL, "feature_uid"], dropna=False)
            .agg(
                feature=("feature", "first"),
                feature_group=("feature_group", "first"),
                n_context_nominations=("context_slug", "nunique"),
                n_cohorts=("cohort", "nunique"),
                endpoints=("endpoint", lambda x: ";".join(sorted(set(map(str, x))))),
            )
            .reset_index()
            .sort_values(["panel", ROOT_COL, "n_context_nominations"], ascending=[True, True, False])
        )
        support.to_csv(out / "stage2a4_feature_nomination_support.csv", index=False)

    write_json({
        "n_expected_contexts": int(len(idx)),
        "n_completed_contexts": int(len(summary)),
        "n_missing_context_outputs": int(len(missing)),
        "n_root_matrix_rows": int(len(root_manifest)),
        "n_seed_nomination_rows": int(len(registry)),
        "cross_root_rescue_performed": False,
        "cross_root_compression_performed": False,
    }, out / "stage2a4_aggregate_summary.json")

    if missing:
        log(f"[WARN] missing context outputs={len(missing)}; see stage2a4_missing_context_outputs.csv")
    log(f"[DONE] contexts={len(summary)}/{len(idx)} root_matrices={len(root_manifest)} nominations={len(registry)}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ["validate", "worker", "aggregate"]:
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        if name == "worker":
            p.add_argument("--array-id", type=int, default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = read_json(args.config)
    if args.command == "validate":
        command_validate(cfg)
    elif args.command == "worker":
        command_worker(cfg, resolve_array_id(args.array_id))
    elif args.command == "aggregate":
        command_aggregate(cfg)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
