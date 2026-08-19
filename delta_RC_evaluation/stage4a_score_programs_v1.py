#!/usr/bin/env python3
"""Stage 4A: build frozen root/meta scores for TURBT-reference delta work and RC-only work."""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from delta_rc_common_v1 import *


def outroot(cfg): return Path(cfg["stage4_output_root"])
def cache_dir(cfg, cohort, panel): return outroot(cfg) / "score_cache" / f"{safe_slug(cohort)}__{safe_slug(panel)}"


def validate(cfg):
    validate_frozen_definitions(cfg)
    reg = program_registry(cfg)
    print(f"[VALID] frozen programs={len(reg)} root={(reg.program_level=='root_module').sum()} meta={(reg.program_level=='meta_module').sum()}", flush=True)


def setup(cfg):
    validate(cfg)
    root = ensure_dir(outroot(cfg))
    program_registry(cfg).to_csv(root / "frozen_program_registry.csv", index=False)
    rows=[]; aid=0
    for cohort in cfg["cohorts"]:
        for panel in cfg["panels"]:
            aid += 1
            rows.append({"array_id": aid, "cohort": cohort, "panel": panel})
    pd.DataFrame(rows).to_csv(root / "stage4a_score_worker_index.csv", index=False)
    write_json(cfg, root / "stage4_config.resolved.json")
    print(f"[SETUP] score workers={aid}", flush=True)


def worker(cfg, array_id):
    row = load_index_row(outroot(cfg) / "stage4a_score_worker_index.csv", array_id)
    cohort, panel = str(row["cohort"]), str(row["panel"])
    wdir = ensure_dir(cache_dir(cfg, cohort, panel))

    s1 = import_from_path(f"stage1_stage4_{safe_slug(cohort)}_{panel}", cfg["stage1_script_path"])
    rm = root_membership(cfg, panel)
    mm = meta_membership(cfg, panel)

    raw_by_sample = {"TURBT": {}, "RC": {}}
    matrix_audits=[]
    for root_name, mem_root in rm.groupby(ROOT_COL, sort=True):
        for sample in ["TURBT", "RC"]:
            X, audit = build_root_raw_matrix(s1, cfg, cohort, panel, sample, mem_root)
            raw_by_sample[sample][str(root_name)] = X
            matrix_audits.append(audit)

    # A) Comparable TURBT-reference scale for matched delta.
    ref_root_parts=[]; ref_feature_qc=[]; ref_module_qc=[]
    if any(len(x) > 0 for x in raw_by_sample["TURBT"].values()):
        for root_name, mem_root in rm.groupby(ROOT_COL, sort=True):
            tur = raw_by_sample["TURBT"][str(root_name)]
            rc = raw_by_sample["RC"][str(root_name)]
            if tur.empty:
                continue
            scores, fq, mq = score_root_modules_reference(
                tur, rc, mem_root, cfg, cohort, panel, "TURBT", "RC"
            )
            ref_root_parts.append(scores); ref_feature_qc.append(fq); ref_module_qc.append(mq)

    ref_root = pd.concat(ref_root_parts, ignore_index=True, sort=False) if ref_root_parts else pd.DataFrame()
    ref_meta = score_meta_modules_reference(ref_root, mm, cfg, "TURBT", "RC") if not ref_root.empty else pd.DataFrame()
    ref_program = unified_program_long(ref_root, ref_meta) if not ref_root.empty else pd.DataFrame()

    # B) RC-internal scale for cross-sectional RC-only survival evaluation.
    rc_root_parts=[]; rc_feature_qc=[]; rc_module_qc=[]
    for root_name, mem_root in rm.groupby(ROOT_COL, sort=True):
        rc = raw_by_sample["RC"][str(root_name)]
        if rc.empty:
            continue
        scores, fq, mq = score_root_modules_reference(
            rc, rc, mem_root, cfg, cohort, panel, "RC", "RC"
        )
        # target==reference gives duplicate RC rows; retain first set by patient/program.
        if not scores.empty:
            scores = scores.drop_duplicates(["patient_id", "root_module_id", "sample_type"])
        rc_root_parts.append(scores); rc_feature_qc.append(fq); rc_module_qc.append(mq)

    rc_root = pd.concat(rc_root_parts, ignore_index=True, sort=False) if rc_root_parts else pd.DataFrame()
    rc_meta = score_meta_modules_reference(rc_root, mm, cfg, "RC", "RC") if not rc_root.empty else pd.DataFrame()
    if not rc_meta.empty:
        rc_meta = rc_meta.drop_duplicates(["patient_id", "meta_module_id", "sample_type"])
    rc_program = unified_program_long(rc_root, rc_meta) if not rc_root.empty else pd.DataFrame()
    if not rc_program.empty:
        rc_program = rc_program.drop_duplicates(["patient_id", "program_id", "program_level", "sample_type"])

    # Save.
    pd.concat(matrix_audits, ignore_index=True, sort=False).to_csv(wdir / "raw_matrix_build_audit.csv", index=False)
    if ref_feature_qc: pd.concat(ref_feature_qc, ignore_index=True).to_csv(wdir / "turbt_reference_feature_qc.csv", index=False)
    if ref_module_qc: pd.concat(ref_module_qc, ignore_index=True).to_csv(wdir / "turbt_reference_root_module_qc.csv", index=False)
    if rc_feature_qc: pd.concat(rc_feature_qc, ignore_index=True).to_csv(wdir / "rc_internal_feature_qc.csv", index=False)
    if rc_module_qc: pd.concat(rc_module_qc, ignore_index=True).to_csv(wdir / "rc_internal_root_module_qc.csv", index=False)

    ref_program.to_parquet(wdir / "turbt_reference_program_scores_long.parquet", index=False)
    rc_program.to_parquet(wdir / "rc_internal_program_scores_long.parquet", index=False)

    summary = {
        "cohort": cohort, "panel": panel,
        "n_turbt_reference_patients": int(ref_program.loc[ref_program.sample_type.eq("TURBT"), "patient_id"].nunique()) if not ref_program.empty else 0,
        "n_rc_reference_patients": int(ref_program.loc[ref_program.sample_type.eq("RC"), "patient_id"].nunique()) if not ref_program.empty else 0,
        "n_rc_internal_patients": int(rc_program["patient_id"].nunique()) if not rc_program.empty else 0,
        "n_programs_turbt_reference": int(ref_program["program_id"].nunique()) if not ref_program.empty else 0,
        "n_programs_rc_internal": int(rc_program["program_id"].nunique()) if not rc_program.empty else 0,
    }
    pd.DataFrame([summary]).to_csv(wdir / "score_cache_summary.csv", index=False)
    (wdir / ".done").write_text("complete\n")
    print(f"[DONE 4A] {cohort}/{panel}: {summary}", flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("command", choices=["validate","setup","worker"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--array-id", type=int)
    a=ap.parse_args(); cfg=load_json(a.config)
    if a.command=="validate": validate(cfg)
    elif a.command=="setup": setup(cfg)
    else: worker(cfg, resolve_array_id(a.array_id))

if __name__=="__main__": main()
