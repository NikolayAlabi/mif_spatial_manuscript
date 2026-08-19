#!/usr/bin/env python3
"""
stage2c_finalize_global_modules_v7.py

Finalize frozen global module memberships after reviewing Stage 2B k diagnostics.

This can be run from the command line or called from the review notebook. It reads
prepared Stage 2B outputs and writes final module membership/summary files in an
evaluation-script-friendly format.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd

import stage2_global_module_utils_v7 as gm


def parse_final_k(final_k_text: str | None, ar_k: int | None, bt_k: int | None) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if final_k_text:
        for item in final_k_text.split(","):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                raise ValueError("--final-k should look like AR=24,BT=12")
            panel, k = item.split("=", 1)
            out[panel.strip()] = int(k)
    if ar_k is not None:
        out["AR"] = int(ar_k)
    if bt_k is not None:
        out["BT"] = int(bt_k)
    if not out:
        raise ValueError("Provide --final-k AR=<k>,BT=<k> or --ar-k/--bt-k.")
    return out


def run(args) -> None:
    prepared_root = Path(args.prepared_root)
    final_k = parse_final_k(args.final_k, args.ar_k, args.bt_k)
    gm.log("=" * 80)
    gm.log(f"[START] Finalizing modules from {prepared_root}")
    gm.log(f"[INFO] final_k={final_k}")

    all_summaries = []
    all_memberships = []
    all_reps = []
    for panel, k in final_k.items():
        res = gm.finalize_panel_modules(prepared_root, panel, int(k))
        ms = res["module_summary"].copy(); ms["panel"] = panel; ms["final_k"] = int(k)
        mem = res["membership"].copy(); mem["panel"] = panel; mem["final_k"] = int(k)
        rep = res["representatives"].copy(); rep["panel"] = panel; rep["final_k"] = int(k)
        all_summaries.append(ms)
        all_memberships.append(mem)
        all_reps.append(rep)

    final_root = gm.ensure_dir(prepared_root / "final_modules")
    if all_summaries:
        pd.concat(all_summaries, ignore_index=True, sort=False).to_csv(final_root / "all_panel_module_summary.csv", index=False)
    if all_memberships:
        pd.concat(all_memberships, ignore_index=True, sort=False).to_csv(final_root / "all_panel_global_module_memberships.csv", index=False)
    if all_reps:
        pd.concat(all_reps, ignore_index=True, sort=False).to_csv(final_root / "all_panel_module_representatives.csv", index=False)

    gm.write_json({"final_k": final_k, "prepared_root": str(prepared_root)}, final_root / "final_module_config.json")
    gm.log(f"[DONE] Final module files written to: {final_root}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared-root", required=True, help="Stage 2B prepared output root.")
    ap.add_argument("--final-k", default=None, help="Comma-separated, e.g. AR=24,BT=12")
    ap.add_argument("--ar-k", type=int, default=None)
    ap.add_argument("--bt-k", type=int, default=None)
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
