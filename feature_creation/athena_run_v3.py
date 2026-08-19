import os
import sys
import json
import math
import argparse
from pathlib import Path
from copy import deepcopy
from typing import Dict, Any, List, Tuple, Optional
import re
import numpy as np
import pandas as pd
from tqdm import tqdm

# ==== Your stack ====
import athena as ath
from spatialOmics import SpatialOmics
from athena.graph_builder.constants import GRAPH_BUILDER_DEFAULT_PARAMS

## HELPERS FOR CLEANING
def _exclude_unwanted_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    bad = {"artifact", "unresolved", "mixed_lineage"}
    ph = df["phenotype"].fillna("").astype(str).str.strip().str.lower()
    return df.loc[~ph.isin(bad)].copy()

def _clean_phenotype_string(x: str) -> str:
    if isinstance(x, str):
        return x.strip()
    return x

def _region_filtered_df(df, region: str) -> pd.DataFrame:
    r = region.lower()
    if r == 'tumor':
        return df[df['analysisregion'].str.lower() == 'tumor']
    elif r == 'stroma':
        return df[df['analysisregion'].str.lower() == 'stroma']
    elif r == 'all':
        # keep Tumor or Stroma; DO NOT drop Unknown/Other here (we recode in _phenotype_use_column)
        m = df['analysisregion'].str.lower().isin({'tumor', 'stroma'})
        return df[m].copy()
    else:
        return df

def _build_graphs(so, spl: str, radius: float):
    """Build a radius graph at the requested radius.

    The previous v2 script accepted ``--radius`` but did not reliably pass it
    into ATHENA's graph builder. This helper tries the commonly used ATHENA
    keyword variants and falls back to the package default only if needed.
    """
    try:
        ath.graph.build_graph(
            so,
            spl,
            builder_type='radius',
            mask_key=None,
            coordinate_keys=('Xcenter', 'Ycenter'),
            builder_params={'radius': radius},
        )
        return
    except TypeError:
        pass

    try:
        ath.graph.build_graph(
            so,
            spl,
            builder_type='radius',
            mask_key=None,
            coordinate_keys=('Xcenter', 'Ycenter'),
            graph_builder_params={'radius': radius},
        )
        return
    except TypeError:
        pass

    try:
        conf_rad = deepcopy(GRAPH_BUILDER_DEFAULT_PARAMS['radius'])
        conf_rad['builder_params']['radius'] = radius
        ath.graph.build_graph(
            so,
            spl,
            builder_type='radius',
            mask_key=None,
            coordinate_keys=('Xcenter', 'Ycenter'),
            params=conf_rad,
        )
        return
    except TypeError:
        pass

    print(
        f"[WARN] Could not pass radius={radius} through this ATHENA API; "
        "falling back to ath.graph.build_graph defaults.",
        flush=True,
    )
    ath.graph.build_graph(
        so,
        spl,
        builder_type='radius',
        mask_key=None,
        coordinate_keys=('Xcenter', 'Ycenter'),
    )

def _make_categorical_ids(df: pd.DataFrame, col: str = 'phenotype', col_id: str = 'phenotypes_id') -> pd.DataFrame:
    """Assign a global consistent categorical id to phenotypes."""
    df = df.copy()
    df[col] = df[col].astype('category')
    df[col_id] = df.groupby(col).ngroup()
    return df

def _phenotype_use_column(df: pd.DataFrame, region: str) -> pd.DataFrame:
    """Create a region-aware phenotype column used for metrics/infiltration.

    - Always standardize blanks to 'Marker-'
    - STROMA: map 'Unknown'/'Other' → 'StromaCell' (everywhere in stroma slice)
    - ALL:    map 'Unknown'/'Other' → 'StromaCell' **only where analysisregion=='Stroma'**
              (leave Tumor-side Unknown/Other unchanged)
    - TUMOR:  keep as-is
    """
    df = df.copy()
    df['phenotype'] = df['phenotype'].fillna('Marker-').map(_clean_phenotype_string)

    r = str(region).lower()

    if r == 'stroma':
        # full recode in stroma slice
        df['phenotype_use'] = df['phenotype'].where(
            ~df['phenotype'].str.lower().isin({'unknown', 'other'}),
            other='StromaCell'
        )

    elif r == 'all':
        # recode only for rows *located in stroma*
        # (requires an 'analysisregion' column)
        df['phenotype_use'] = df['phenotype']
        if 'analysisregion' in df.columns:
            is_stroma = df['analysisregion'].astype(str).str.lower().eq('stroma')
            is_unkoth = df['phenotype'].str.lower().isin({'unknown', 'other'})
            df.loc[is_stroma & is_unkoth, 'phenotype_use'] = 'StromaCell'
        # else: leave as-is (can't localize to stroma)

    else:
        # tumor or any other region: keep phenotype
        df['phenotype_use'] = df['phenotype']

    # categorical ids
    df['phenotype_use'] = df['phenotype_use'].astype('category')
    df['phenotypes_id_use'] = df.groupby('phenotype_use').ngroup()
    return df


def _name_sets_for_infiltration(df: pd.DataFrame, region: str):
    """
    Returns:
      base_id: the 'denominator' phenotype for infiltration
               - Tumor/All: tumor
               - Stroma:    StromaCell
      immune_dict: {phenotype_id -> clean_name} for numerators (excludes Unknown/Other/Marker-/StromaCell/Tumor)
      id_to_name:  full id->name mapping (phenotype_use)
    """
    id_to_name = (
        df.groupby('phenotypes_id_use')['phenotype_use']
          .first()
          .astype(str).str.strip()
    )

    if region.lower() == 'stroma':
        # base = StromaCell
        base_candidates = id_to_name[id_to_name.str.lower() == 'stromacell'].index.tolist()
    else:
        # base = tumor
        base_candidates = id_to_name[id_to_name.str.lower().isin({'tumor','tumour','cancer','panck+'})].index.tolist()

    if not base_candidates:
        return None, None, id_to_name.to_dict()

    base_id = base_candidates[0]

    # Immune (numerators) exclude these buckets from consideration:
    exclude = {'marker-','unknown','other','stromacell','tumor','tumour','cancer'}
    immune_dict = {
        pid: nm.replace(' ', '')
        for pid, nm in id_to_name.items()
        if isinstance(nm, str) and nm.lower() not in exclude and pid != base_id
    }
    return base_id, immune_dict, id_to_name.to_dict()

def _ensure_spl_window_from_lookup(so, sample_id: str, df_s: pd.DataFrame, area_lookup_um2: dict):
    """
    Set so.spl.loc[sample_id, ['area','width','height']] in µm/µm².
    We only know total area, so we assume a square: width = height = sqrt(area).
    """
    import numpy as np
    import pandas as pd

    if not isinstance(getattr(so, "spl", None), pd.DataFrame):
        so.spl = pd.DataFrame()

    # 1) area in µm² (prefer lookup; else fallback to bbox)
    if sample_id in area_lookup_um2 and np.isfinite(area_lookup_um2[sample_id]):
        area_um2 = float(area_lookup_um2[sample_id])
    else:
        w = float(df_s["Xcenter"].max() - df_s["Xcenter"].min())
        h = float(df_s["Ycenter"].max() - df_s["Ycenter"].min())
        area_um2 = max(w, 0.0) * max(h, 0.0)

    # 2) synthesize a square window
    side_um = float(np.sqrt(area_um2))
    so.spl.loc[sample_id, "area"]   = area_um2
    so.spl.loc[sample_id, "width"]  = side_um
    so.spl.loc[sample_id, "height"] = side_um

def _drop_names_for(region_name: str):
    """Return phenotype names to EXCLUDE for Ripley/Interactions, per region."""
    base = {"marker-", "unknown", "other"}
    # In STROMA we KEEP StromaCell; elsewhere we can drop it.
    if str(region_name).lower() == "stroma":
        return base
    else:
        return base | {"stromacell"}

## HELPERS FOR HETEROGENEITY AND INFILTRATION METRICS
def _safe_infiltration_ratio(so, spl: str, attr_id: str, base_id: int, immune_id: int,
                             graph_key: str = 'radius',
                             alpha_num: float = 0.0,
                             alpha_den: float = 1.0) -> pd.Series:
    """
    Per-node infiltration for *base* nodes only:
        ratio(node) = ( #edges base–immune + alpha_num ) / ( #edges base–base + alpha_den )
    Returns NaN for non-base nodes.
    """
    G = so.G[spl][graph_key]
    phenos = so.obs[spl][attr_id]
    out = pd.Series(index=so.obs[spl].index, dtype=float)
    out[:] = np.nan

    base_nodes = so.obs[spl].index[phenos == base_id]
    for node in base_nodes:
        neigh = list(G.neighbors(node))
        if not neigh:
            out.loc[node] = 0.0
            continue
        neigh_labels = phenos.loc[neigh].values
        num = np.count_nonzero(neigh_labels == immune_id)
        den = np.count_nonzero(neigh_labels == base_id)
        out.loc[node] = (num + alpha_num) / (den + alpha_den)
    return out


def _heterogeneity_metrics(so: "SpatialOmics",
                           spl_id: str,
                           region: str):
    """
    Build a single-sample SpatialOmics object, compute graphs, local metrics,
    and stabilized infiltration features. Returns a DataFrame (cell-level).
    """
    # diversity metrics (local only)
    # key_added names include attr and graph_key already, so runs at different params won't overwrite
    ath.metrics.richness(so, spl_id, 'phenotype_use', local=True, graph_key='radius')
    ath.metrics.shannon(so, spl_id, 'phenotype_use', local=True, graph_key='radius')
    # Three Renyi entropies (distinct keys): q=0.5, q=2, q=3
    ath.metrics.renyi_entropy(so, spl_id, 'phenotype_use', q=0.5, local=True, graph_key='radius')
    ath.metrics.renyi_entropy(so, spl_id, 'phenotype_use', q=2.0, local=True, graph_key='radius')
    ath.metrics.renyi_entropy(so, spl_id, 'phenotype_use', q=3.0, local=True, graph_key='radius')

    # stabilized infiltration per immune cell type
    base_id, immune_dict, _id2name = _name_sets_for_infiltration(so.obs[spl_id], region=region)
    if base_id is not None and immune_dict:
        for immune_id, immune_name in immune_dict.items():
            # print(f'{immune_name} index: {immune_id} and base id: {base_id}')
            series = _safe_infiltration_ratio(so, spl_id, 'phenotypes_id_use',
                                              base_id=base_id, immune_id=immune_id,
                                              graph_key='radius', alpha_num=0.0, alpha_den=1.0)
            col = f"infiltration_{immune_name}"
            so.obs[spl_id][col] = series.values

    # return the augmented cell-level DF
    return so.obs[spl_id].copy()

def _norm_name(x: str) -> str:
    """Sanitize phenotype names for column keys."""
    if x is None:
        return "NA"
    x = str(x)
    x = x.strip()
    x = re.sub(r"\s+", "", x)
    x = x.replace("+", "p").replace("-", "neg").replace("/", "_")
    return x

def _id2name_from_cell_df(cell_df: pd.DataFrame) -> Dict[int, str]:
    """Build id->name mapping from the cell_df."""
    if "phenotypes_id_use" in cell_df.columns and "phenotype_use" in cell_df.columns:
        m = (cell_df.groupby("phenotypes_id_use")["phenotype_use"]
                     .first().astype(str))
        return {int(k): _norm_name(v) for k, v in m.to_dict().items()}
    return {}

def _pick_metric_cols(df: pd.DataFrame) -> List[str]:
    prefixes = ("richness_", "shannon_", "simpson_", "hill_number_", "renyi_", "infiltration_")
    return [c for c in df.columns if c.startswith(prefixes)]

# ---------------------------
# 1) CELL-LEVEL Heterogeneity_metrics DF -> one sample row
# ---------------------------

def aggregate_cell_df_to_sample_row(
    cell_df: pd.DataFrame,
    sample_id: Optional[str] = None,
    region_name: Optional[str] = None,
    add_coverage: bool = True,
    stats: Tuple[str, ...] = ("min", "mean", "median", "max"),
) -> pd.DataFrame:
    """
    Compute per-sample summaries from a single cell-level DF (output of _process_sample_region).
    Returns a 1-row DataFrame (wide).
    """
    metric_cols = _pick_metric_cols(cell_df)
    if not metric_cols:
        raise ValueError("No metric columns (richness_/shannon_/renyi_/infiltration_...) found in cell_df.")

    suf = f"__{region_name}" if region_name else ""
    row = {}
    if sample_id is None:
        # infer from df if constant
        if "sample_id" in cell_df.columns and cell_df["sample_id"].nunique() == 1:
            sample_id = str(cell_df["sample_id"].iloc[0])
    if sample_id is not None:
        row["sample_id"] = sample_id

    for col in metric_cols:
        s = cell_df[col]
        # nan-safe stats
        if "min" in stats:
            row[f"{col}{suf}__min"] = float(np.nanmin(s.values)) if s.notna().any() else np.nan
        if "mean" in stats:
            row[f"{col}{suf}__mean"] = float(np.nanmean(s.values)) if s.notna().any() else np.nan
        if "median" in stats:
            row[f"{col}{suf}__median"] = float(np.nanmedian(s.values)) if s.notna().any() else np.nan
        if "max" in stats:
            row[f"{col}{suf}__max"] = float(np.nanmax(s.values)) if s.notna().any() else np.nan

    if add_coverage:
        infil_cols = [c for c in metric_cols if c.startswith("infiltration_")]
        if infil_cols:
            for c in infil_cols:
                row[f"{c}{suf}__pct_non_na"] = float(cell_df[c].notna().mean())

    return pd.DataFrame([row])


## RUN INTERACTIONS
from pandas.api import types as pdt

def _can_run_interactions(so: "SpatialOmics", spl: str, attr: str,
                          min_labels: int = 2, min_cells_per_label: int = 3,
                          graph_key: str = "radius") -> bool:
    # graph exists and has edges
    G = so.G.get(spl, {}).get(graph_key, None)
    if G is None or G.number_of_edges() == 0:
        return False

    # obs exists & has attr
    if spl not in so.obs or attr not in so.obs[spl].columns:
        return False

    # node-label alignment: graph nodes must be in obs index
    obs_index = so.obs[spl].index
    try:
        nodes = set(G.nodes())
    except Exception:
        return False
    if not nodes.issubset(set(obs_index)):
        return False

    # labels dtype can be anything; we'll factorize upstream
    labs = so.obs[spl][attr]

    # at least `min_labels` classes with enough support
    vc = pd.Series(labs).value_counts()
    ok = (vc >= min_cells_per_label).sum()
    return ok >= min_labels

def _run_interactions_one_sample(
    soX: "SpatialOmics",
    df_s: pd.DataFrame,
    sample_id: str,
    pairs_mode: str = "both",
    n_perm: int = 100,
    immune_whitelist: Optional[List[str]] = None,
    compute_pvals: bool = False,
):
    # --- basics & names
    region_name = df_s.attrs.get("region_name", "All")

    # ---- SAFETY: make labels contiguous ints WITHOUT touching the node index
    # factorize current labels (works for int/object/categorical)
    labs = soX.obs[sample_id]["phenotypes_id_use"]
    codes, uniques = pd.factorize(labs, sort=True)        # contiguous 0..K-1; -1 if NaN (won’t happen here)
    soX.obs[sample_id] = soX.obs[sample_id].copy()
    soX.obs[sample_id]["phenotypes_id_use"] = codes.astype(int)

    # refresh id->name map after remap
    id2name = (
        soX.obs[sample_id]
        .assign(_pid=soX.obs[sample_id]["phenotypes_id_use"].astype(int))
        .groupby("_pid")["phenotype_use"].first().astype(str)
    )

    # ---- quick preflight before calling ATHENA
    if not _can_run_interactions(soX, sample_id, "phenotypes_id_use", graph_key="radius"):
        return None

    # tumor ids (for pairs_mode filtering)
    id2name_l = id2name.str.lower()
    tumor_alias = {"tumor", "tumour", "cancer", "panck+"}
    tumor_ids = [pid for pid, nm in id2name_l.items() if nm in tumor_alias]

    # --- run estimator on *soX* (graph & obs must already be aligned)
    ath.neighborhood.estimators.interactions(
        soX, sample_id, attr="phenotypes_id_use",
        mode="proportion", prediction_type="diff",
        n_permutations=n_perm, graph_key="radius", inplace=True
    )

    # --- extract DF from uns
    inter_uns = getattr(soX, "uns", {}).get(sample_id, {}).get("interactions", {})
    dfI = None
    for v in inter_uns.values():
        if isinstance(v, pd.DataFrame):
            dfI = v.copy()
            break
    if dfI is None:
        return None

    if isinstance(dfI.index, pd.MultiIndex):
        dfI = dfI.reset_index()

    if "diff" not in dfI.columns:
        if {"score", "perm_mean"}.issubset(dfI.columns):
            dfI["diff"] = dfI["score"] - dfI["perm_mean"]
        else:
            raise ValueError("Interactions DF missing 'diff' (and 'score'/'perm_mean' to compute it).")

    # --- pairs_mode filter using tumor ids
    src = dfI["source_label"].astype(int)
    tgt = dfI["target_label"].astype(int)
    src_is_tumor = src.isin(tumor_ids)
    tgt_is_tumor = tgt.isin(tumor_ids)

    if pairs_mode == "immune_only":
        mask = (~src_is_tumor) & (~tgt_is_tumor)
    elif pairs_mode == "tumor_only":
        mask = (src_is_tumor) | (tgt_is_tumor)
    else:
        mask = pd.Series(True, index=dfI.index)

    dfI = dfI.loc[mask].copy()
    if dfI.empty:
        return None

    # --- map ids -> names
    dfI["src_id"] = dfI["source_label"].astype(int)
    dfI["tgt_id"] = dfI["target_label"].astype(int)
    dfI["src"] = dfI["src_id"].map(id2name).fillna(dfI["src_id"].astype(str))
    dfI["tgt"] = dfI["tgt_id"].map(id2name).fillna(dfI["tgt_id"].astype(str))

    # drop Unknown/Other pairs post-hoc
    src_name_l = dfI["src"].str.lower()
    tgt_name_l = dfI["tgt"].str.lower()
    drop_pairs = src_name_l.isin({"unknown", "other"}) | tgt_name_l.isin({"unknown", "other"})
    dfI = dfI.loc[~drop_pairs].copy()
    if dfI.empty:
        return None

    # clean display names
    dfI["src"] = dfI["src"].map(_norm_name)
    dfI["tgt"] = dfI["tgt"].map(_norm_name)

    # z and approx p
    dfI["z"] = np.nan
    if {"perm_std", "diff"}.issubset(dfI.columns):
        std = dfI["perm_std"].replace(0.0, np.nan)
        dfI.loc[std.notna(), "z"] = dfI.loc[std.notna(), "diff"] / std[std.notna()]
    if compute_pvals and "z" in dfI.columns:
        from math import erfc, sqrt
        dfI["p"] = dfI["z"].abs().map(lambda z: erfc(z / sqrt(2.0)) if pd.notna(z) else np.nan)

    keep_cols = ["src_id", "tgt_id", "src", "tgt", "diff", "z"]
    for c in ["p", "score", "perm_mean", "perm_std", "perm_median"]:
        if c in dfI.columns:
            keep_cols.append(c)

    dfI = dfI[keep_cols].copy()
    dfI.insert(0, "region", region_name)
    dfI.insert(0, "sample_id", sample_id)
    return dfI


def interactions_uns_to_sample_row(
    inter_df: pd.DataFrame,
    sample_id: Optional[str] = None,
    region_name: Optional[str] = None,
    *,
    include_symmetric: bool = False,
    prefix: str = "inter",
    add_p_from_z: bool = False  # normal approx two-sided p from z
) -> pd.DataFrame:
    """
    Flatten the DataFrame returned by `_run_interactions_one_sample` into ONE wide row.

    Expected columns in `inter_df` (exactly what your runner outputs):
      sample_id, region, src_id, tgt_id, src, tgt, diff, z, score, perm_mean, perm_std, perm_median

    Output columns (examples; region suffix added if provided/available):
      inter_diff_<SRC>__<TGT>[__Region]
      inter_z_<SRC>__<TGT>[__Region]
      inter_p_<SRC>__<TGT>[__Region]     (optional, from z)
      inter_symdiff_<A>__<B>[__Region]   (optional, undirected average)

    Notes:
      - Assumes any region-aware dropping (e.g., keep StromaCell in Stroma) was already done upstream.
      - If `sample_id` / `region_name` are not passed, they are inferred from the DataFrame if unique.
    """
    required = {"src", "tgt", "diff"}
    if not required.issubset(inter_df.columns):
        raise ValueError(f"inter_df must contain columns {sorted(required)}; got {sorted(inter_df.columns)}")

    df = inter_df.copy()

    # infer sample/region if not provided (and unique)
    if sample_id is None and "sample_id" in df.columns and df["sample_id"].nunique() == 1:
        sample_id = str(df["sample_id"].iloc[0])
    if region_name is None and "region" in df.columns and df["region"].nunique() == 1:
        region_name = str(df["region"].iloc[0])

    # sanitize names (idempotent; safe even if already sanitized)
    df["src"] = df["src"].map(_norm_name)
    df["tgt"] = df["tgt"].map(_norm_name)

    # if duplicate (src,tgt) rows slipped in, average their stats
    by = ["src", "tgt"]
    stats = {"diff": "mean"}
    if "z" in df.columns:
        stats["z"] = "mean"
    if "score" in df.columns:
        stats["score"] = "mean"
    if "perm_mean" in df.columns:
        stats["perm_mean"] = "mean"
    if "perm_std" in df.columns:
        stats["perm_std"] = "mean"
    if "perm_median" in df.columns:
        stats["perm_median"] = "median"

    df = df.groupby(by, as_index=False).agg(stats)

    # optional p from z (normal approx)
    if add_p_from_z and "z" in df.columns:
        from math import erfc, sqrt
        df["p"] = df["z"].abs().map(lambda z: erfc(z / sqrt(2.0)) if pd.notna(z) else np.nan)

    # build the single wide row
    suf = f"__{region_name}" if region_name else ""
    row = {}
    if sample_id is not None:
        row["sample_id"] = sample_id

    for _, r in df.iterrows():
        row[f"{prefix}_diff_{r['src']}__{r['tgt']}{suf}"] = float(r["diff"]) if pd.notna(r["diff"]) else np.nan
        if "z" in df.columns and pd.notna(r.get("z", np.nan)):
            row[f"{prefix}_z_{r['src']}__{r['tgt']}{suf}"] = float(r["z"])
        if "p" in df.columns and pd.notna(r.get("p", np.nan)):
            row[f"{prefix}_p_{r['src']}__{r['tgt']}{suf}"] = float(r["p"])

    # optional symmetric average of diff (undirected)
    if include_symmetric and not df.empty:
        piv = df.pivot(index="src", columns="tgt", values="diff").astype(float)
        sym = (piv + piv.T) / 2.0
        for a in sym.index:
            for b in sym.columns:
                if a < b and pd.notna(sym.loc[a, b]):
                    row[f"{prefix}_symdiff_{a}__{b}{suf}"] = float(sym.loc[a, b])

    return pd.DataFrame([row])

## RUN RIPLEY
def run_ripley_rescue(
    so, sample_id, *,
    attr='phenotypes_id_use',
    base_radii=(20, 40, 60),
    min_points=3,
    region_name="All",
    try_corrections=('translation','border','ripley','none'),
    try_modes=('csr-deviation','K','L')
):
    import numpy as np, pandas as pd
    # window sanity
    area  = float(so.spl.loc[sample_id, 'area'])
    width = float(so.spl.loc[sample_id, 'width'])
    height= float(so.spl.loc[sample_id, 'height'])
    if not (np.isfinite(area) and np.isfinite(width) and np.isfinite(height) and area>0 and width>0 and height>0):
        raise ValueError("Invalid window geometry (area/width/height).")

    cap = 0.45 * min(width, height)
    radii = [r for r in base_radii if 0 < r <= cap] or [min(width,height)*0.025, min(width,height)*0.05, min(width,height)*0.1]

    # region-aware keep set
    drop_names = _drop_names_for(region_name)  # <- key line
    id2nm = (so.obs[sample_id].groupby(attr)['phenotype_use'].first().astype(str).str.lower())
    counts = so.obs[sample_id].groupby(attr).size()
    keep_ids = [pid for pid, nm in id2nm.items() if nm not in drop_names and counts.get(pid,0) >= min_points]

    out = {}
    for pid in keep_ids:
        got = False
        for corr in try_corrections:
            for mode in try_modes:
                key = f"{pid}_{attr}_{mode}_{corr}"
                try:
                    ath.neighborhood.estimators.ripleysK(
                        so=so, spl=sample_id, attr=attr, id=pid,
                        mode=mode, radii=radii, correction=corr,
                        inplace=True, key_added=None
                    )
                    series = so.uns[sample_id]['ripleysK'][key]
                    if isinstance(series, pd.Series) and np.isfinite(series.values).any():
                        out[key] = series
                        got = True
                        break
                except Exception:
                    pass
            if got:
                break
    return out

def ripley_rescue_to_sample_row(
    ripley_res: Dict[str, pd.Series],
    cell_df_for_names: Optional[pd.DataFrame] = None,
    sample_id: Optional[str] = None,
    prefer_mode: Optional[str] = "csr-deviation",
    prefer_correction: Optional[str] = "translation",
    at_radius: float = 40.0,
    do_peak_abs: bool = True,
    do_auc: bool = True,
    region_name: Optional[str] = None,   # <-- add this
) -> pd.DataFrame:
    id2name = _id2name_from_cell_df(cell_df_for_names) if cell_df_for_names is not None else {}
    suf = f"__{region_name}" if region_name else ""   # <-- ensure region is appended once

    row = {}
    if sample_id is not None:
        row["sample_id"] = sample_id

    for k, series in ripley_res.items():
        parts = str(k).split("_")
        if len(parts) < 4:
            pid = parts[0]; mode = parts[-2] if len(parts) >= 2 else "mode"; corr = parts[-1] if len(parts) >= 1 else "corr"
        else:
            pid = parts[0]; mode = parts[-2]; corr = parts[-1]

        if prefer_mode and mode != prefer_mode:         continue
        if prefer_correction and corr != prefer_correction: continue

        try:    pid_int = int(pid)
        except: pid_int = None
        pname = _norm_name(id2name.get(pid_int, pid))

        s = pd.Series(series).copy()
        s.index = pd.to_numeric(s.index, errors="coerce")
        s = s.sort_index()
        vals = s.values.astype(float)

        v_at = s.loc[s.index == float(at_radius)]
        row[f"ripley_{mode}_{corr}_{pname}{suf}__at{int(at_radius)}"] = float(v_at.iloc[0]) if len(v_at) else np.nan
        if do_peak_abs and np.isfinite(vals).any():
            i = np.nanargmax(np.abs(vals))
            row[f"ripley_{mode}_{corr}_{pname}{suf}__peak_abs"] = float(vals[i])
        if do_auc and len(s) >= 2 and np.isfinite(vals).any():
            row[f"ripley_{mode}_{corr}_{pname}{suf}__auc"] = float(np.trapz(y=vals, x=s.index.values))
    return pd.DataFrame([row])



def build_sample_row(
    cell_df: pd.DataFrame,
    ripley_res: Optional[dict] = None,
    inter_uns: Optional[pd.DataFrame] = None,   # <-- now a DataFrame
    sample_id: Optional[str] = None,
    region_name: Optional[str] = None
) -> pd.DataFrame:
    if sample_id is None and "sample_id" in cell_df.columns and cell_df["sample_id"].nunique() == 1:
        sample_id = str(cell_df["sample_id"].iloc[0])

    rows = []

    rows.append(aggregate_cell_df_to_sample_row(
        cell_df=cell_df, sample_id=sample_id, region_name=region_name, add_coverage=True
    ))

    # inside build_sample_row(...)
    if ripley_res:
        rows.append(ripley_rescue_to_sample_row(
            ripley_res=ripley_res,
            cell_df_for_names=cell_df,
            sample_id=sample_id,
            prefer_mode="csr-deviation",
            prefer_correction="translation",
            at_radius=40.0,
            do_peak_abs=True,
            do_auc=True,
            region_name=region_name,                # <-- add this
        ))

    if inter_uns is not None and not inter_uns.empty:
        rows.append(interactions_uns_to_sample_row(
            inter_df=inter_uns,                   # <-- pass the DF
            sample_id=sample_id,
            region_name=region_name,
            include_symmetric=False,              # True to also add undirected averages
            prefix="inter",
            add_p_from_z=False
        ))

    out = rows[0]
    for r in rows[1:]:
        out = out.merge(r, on="sample_id", how="outer")
    return out

def main():
    ap = argparse.ArgumentParser(description="Compute local spatial biomarkers (parallel-friendly).")
    ap.add_argument("--root", type=Path, default=None, help="Legacy root directory. Not required when --prep-file is supplied.")
    ap.add_argument("--filter", type=str, default=None, help="Legacy qc_filter. Not required when --prep-file is supplied.")
    ap.add_argument("--typemode", type=str, default=None, help="Legacy cell specificity. Not required when --prep-file is supplied.")
    ap.add_argument("--panel", type=str, default=None, help="Panel name, e.g. AR. Required in legacy mode; optional in direct --prep-file mode.")
    ap.add_argument("--surgery", type=str, default=None, help="Legacy surgery name. Not required when --prep-file is supplied.")
    ap.add_argument("--radius", type=float, default=40.0, help="Radius (microns) for radius graph (default: 40)")
    ap.add_argument("--jobs", type=int, default=1, help="Parallel processes (default: 1)")
    ap.add_argument("--min_cells_sample", type=int, default=200, help="Skip samples with total cells < this (default: 200)")
    ap.add_argument("--regions", nargs='+', default=["Tumor","Stroma","All"], help='Regions to analyze: Tumor Stroma All')
    ap.add_argument("--prep-file", type=Path, default=None, help="Direct path to 1NN_prep.tsv. Overrides legacy root layout when provided.")
    ap.add_argument("--tissue-file", type=Path, default=None, help="Direct path to tissue_prep.tsv for Ripley window lookup.")
    ap.add_argument("--out-file", type=Path, default=None, help="Direct output path for athena_features.csv.")
    ap.add_argument("--run-interactions", action="store_true", help="Run ATHENA interaction permutation features. Off by default in v3 because it is expensive.")
    ap.add_argument("--interaction-permutations", type=int, default=100, help="Number of permutations for interactions when enabled.")
    ap.add_argument("--run-ripley", action="store_true", help="Run Ripley features. Off by default in v3 because it depends on tissue area/window assumptions.")
    ap.add_argument("--disable-heterogeneity", action="store_true", help="Disable local richness/shannon/renyi/infiltration features.")
    args = ap.parse_args()

    # Direct mode requires only --prep-file. Legacy mode requires the old wrapper layout args.
    if args.prep_file is None:
        missing_legacy = [
            name for name in ["root", "filter", "typemode", "panel", "surgery"]
            if getattr(args, name) is None
        ]
        if missing_legacy:
            ap.error(
                "legacy mode requires --root, --filter, --typemode, --panel, and --surgery; "
                "or use direct mode with --prep-file [--tissue-file --out-file]. "
                f"Missing: {missing_legacy}"
            )

    # Paths: direct mode or legacy wrapper mode
    if args.prep_file is not None:
        spatial_data_file = args.prep_file
        if args.tissue_file is None:
            tissue_prep_path = spatial_data_file.parent / "tissue_prep.tsv"
        else:
            tissue_prep_path = args.tissue_file
        output_file = args.out_file if args.out_file is not None else spatial_data_file.parent / "athena_features.csv"
        run_dir = output_file.parent
    else:
        run_dir = Path(args.root) / args.filter / args.typemode / args.panel / args.surgery
        spatial_data_file = run_dir / "1NN_prep.tsv"
        tissue_prep_path  = run_dir / "tissue_prep.tsv"
        output_file       = run_dir / "athena_features.csv"
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- tissue prep for Ripley window
    tp = pd.read_csv(tissue_prep_path, sep="\t")
    tp = tp.dropna(subset=["sample_id", "total_area"]).copy()
    tp["area_um2"] = tp["total_area"].astype(float) * 1_000_000.0
    tp_lookup_um2 = tp.groupby("sample_id")["area_um2"].first().to_dict()

    # --- load spatial data
    obs = pd.read_csv(spatial_data_file, sep='\t')
    obs = _exclude_unwanted_labels(obs)
    obs['phenotype'] = obs['phenotype'].fillna('Marker-').map(_clean_phenotype_string)
    obs['Xcenter'] = obs['Xcenter'].astype(float)
    obs['Ycenter'] = obs['Ycenter'].astype(float)
    obs['analysisregion'] = obs['tumor_stroma']

    samples = obs['sample_id'].dropna().astype(str).unique().tolist()

    # master collector (one merged row per sample)
    master_rows: List[pd.DataFrame] = []

    for s in tqdm(samples, desc="Samples"):
        df_s_orig = obs[obs['sample_id'] == s].copy()
        if df_s_orig.shape[0] < args.min_cells_sample:
            continue

        region_results: List[pd.DataFrame] = []

        for region in args.regions:
            # --- region slice + recode phenotype_use
            df_region = _region_filtered_df(df_s_orig, region=region)
            if df_region.empty:
                continue
            df_region = _phenotype_use_column(df_region, region=region)
            df_region.attrs['region_name'] = region

            # --- SpatialOmics for this region
            so = SpatialOmics()
            so.obs[s] = df_region.copy()
            so.obs[s]['x'] = so.obs[s]['Xcenter']
            so.obs[s]['y'] = so.obs[s]['Ycenter']

            # graphs for metrics/interaction
            _build_graphs(so, s, radius=args.radius)

            # local metrics & stabilized infiltrations
            if not args.disable_heterogeneity:
                _heterogeneity_metrics(so=so, spl_id=s, region=region)

            # interactions (tidy DF; Unknown/Other dropped post-hoc in _run_interactions_one_sample)
            inter_tbl = None
            if args.run_interactions:
                inter_tbl = _run_interactions_one_sample(
                    soX=so, df_s=df_region, sample_id=s,
                    pairs_mode="both", n_perm=args.interaction_permutations, compute_pvals=False
                )

            # Ripley: ONLY for All region and only when explicitly requested
            rip_dict = None
            if args.run_ripley and str(region).lower() == "all":
                _ensure_spl_window_from_lookup(so, s, df_region, tp_lookup_um2)
                rip_dict = run_ripley_rescue(
                    so=so, sample_id=s, region_name=region,
                    base_radii=(20, 40, 60)
                )

            # assemble one wide row for this region (region suffixes applied inside)
            row_reg = build_sample_row(
                cell_df=so.obs[s],
                ripley_res=rip_dict,            # None for Tumor/Stroma
                inter_uns=inter_tbl,
                sample_id=s,
                region_name=region
            )
            region_results.append(row_reg)

        # --- merge regional rows into one final row for this sample
        if not region_results:
            continue
        final_row = region_results[0]
        for rr in region_results[1:]:
            final_row = final_row.merge(rr, on="sample_id", how="outer")

        master_rows.append(final_row)

    # --- stitch all samples to a single master DataFrame and write once
    if not master_rows:
        # nothing to write
        print("[WARN] No qualifying samples produced rows; nothing written.")
        return

    master_df = pd.concat(master_rows, axis=0, ignore_index=True)
    # de-dup any accidental duplicate columns (shouldn't happen with region suffixing, but safe)
    master_df = master_df.loc[:, ~master_df.columns.duplicated(keep="first")]
    # put sample_id first
    cols = ["sample_id"] + [c for c in master_df.columns if c != "sample_id"]
    master_df = master_df.reindex(columns=cols)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    master_df.to_csv(output_file, index=False)
    metadata = {
        "script": Path(__file__).name,
        "spatial_data_file": str(spatial_data_file),
        "tissue_prep_path": str(tissue_prep_path),
        "output_file": str(output_file),
        "radius": args.radius,
        "regions": args.regions,
        "min_cells_sample": args.min_cells_sample,
        "run_interactions": bool(args.run_interactions),
        "interaction_permutations": int(args.interaction_permutations),
        "run_ripley": bool(args.run_ripley),
        "disable_heterogeneity": bool(args.disable_heterogeneity),
        "n_output_samples": int(master_df.shape[0]),
        "n_output_features": int(master_df.shape[1] - 1),
    }
    with open(output_file.with_name("athena_run_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[INFO] Wrote {output_file} with shape {master_df.shape}")


if __name__ == "__main__":
    main()