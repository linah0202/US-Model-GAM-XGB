"""
Shared configuration for the GAM+XGB pipeline and all aggregation scripts.

Single source of truth for: the BA -> interconnect hierarchy, train/test dates,
the coverage rule, and the list of BAs kept in the hierarchy. The kept-BA list is
decided ONCE in GAM_XGB.ipynb (before the aggregate series are built) and written to
KEPT_BAS_PATH; every downstream script reads that file instead of recomputing it, so
bottom level, aggregate targets and all reconciliation methods use the same BA set.
"""
import json
import os

import numpy as np
import pandas as pd

RESULTS_DIR = ""        
TRAIN_RESULTS_DIR = ""  
TRAIN_OOS_DIR = ""      
OUTPUT_DIR_ROOT = ""    
TEMP_PATH = "/home/s13q981/HTL/ba_hourly_temp.parquet"
KEPT_BAS_PATH = os.path.join(RESULTS_DIR, "kept_bas.json")
DATASET_CACHE = os.path.join(TRAIN_OOS_DIR, "dataset.parquet")

# training-period forecasts MinT and the CNN learn from
TRAIN_FORECAST_DIR = TRAIN_OOS_DIR

TRAIN_START = "2016-01-01"
TRAIN_END = "2019-12-31 23:00:00"
TEST_START = "2020-01-01"
TEST_END = "2022-09-30 23:00:00"

# rolling-origin folds for out-of-sample training-period forecasts:
OOS_FOLDS = [
    ("2016-01-01", "2017-12-31 23:00:00", "2018-01-01", "2018-12-31 23:00:00"),
    ("2016-01-01", "2018-12-31 23:00:00", "2019-01-01", "2019-12-31 23:00:00"),
]

# shared evaluation window: drop this many leading common test hours so every method
# has a forecast at every hour
EVAL_WARMUP_HOURS = 191
SUBPERIODS = ["all", "pre_shock", "acute_covid", "recovery", "stable"]

BA_COL = "balancing_authority_code_eia"
TARGET = "demand_adjusted_mwh"
MIN_COVERAGE_FRAC = 0.95  # BAs below this share of the best-covered BA's test hours are dropped

BA_TO_INTERCONNECT = {
    # Western (28)
    "BANC": "WESTERN", "CISO": "WESTERN", "IID": "WESTERN", "LDWP": "WESTERN", "TIDC": "WESTERN",
    "AVA": "WESTERN", "BPAT": "WESTERN", "CHPD": "WESTERN", "DOPD": "WESTERN", "GCPD": "WESTERN",
    "IPCO": "WESTERN", "NEVP": "WESTERN", "NWMT": "WESTERN", "PACE": "WESTERN", "PACW": "WESTERN",
    "PGE": "WESTERN", "PSCO": "WESTERN", "PSEI": "WESTERN", "SCL": "WESTERN", "TPWR": "WESTERN",
    "WACM": "WESTERN", "WAUW": "WESTERN", "AZPS": "WESTERN", "EPE": "WESTERN", "PNM": "WESTERN",
    "SRP": "WESTERN", "TEPC": "WESTERN", "WALC": "WESTERN",
    # Eastern (26)
    "AEC": "EASTERN", "SOCO": "EASTERN", "AECI": "EASTERN", "LGEE": "EASTERN", "MISO": "EASTERN",
    "CPLE": "EASTERN", "CPLW": "EASTERN", "DUK": "EASTERN", "SC": "EASTERN", "SCEG": "EASTERN",
    "FMPP": "EASTERN", "FPC": "EASTERN", "FPL": "EASTERN", "GVL": "EASTERN", "HST": "EASTERN",
    "JEA": "EASTERN", "NSB": "EASTERN", "SEC": "EASTERN", "TAL": "EASTERN", "TEC": "EASTERN",
    "ISNE": "EASTERN", "NYIS": "EASTERN", "PJM": "EASTERN", "TVA": "EASTERN", "SPA": "EASTERN", "SWPP": "EASTERN",
    # ERCOT (1)
    "ERCO": "ERCOT",
}
INTERCONNECTS = ["EASTERN", "ERCOT", "WESTERN"]
AGGREGATE_NODES = INTERCONNECTS + ["NATIONAL"]


# kept-BA selection
def select_kept_bas(df, time_col="datetime_utc"):
    """Decide the bottom level of the hierarchy from TEST-period coverage of observed
    demand. Returns (kept, coverage) where coverage is a per-BA hour count."""
    test = df[(df[time_col] >= TEST_START) & (df[time_col] <= TEST_END)]
    test = test[test[BA_COL].isin(BA_TO_INTERCONNECT) & test[TARGET].notna()]
    coverage = test.groupby(BA_COL)[time_col].nunique().reindex(sorted(BA_TO_INTERCONNECT), fill_value=0)
    kept = sorted(coverage[coverage >= MIN_COVERAGE_FRAC * coverage.max()].index)
    return kept, coverage


def save_kept_bas(kept, coverage, path=KEPT_BAS_PATH):
    with open(path, "w") as fh:
        json.dump({
            "min_coverage_frac": MIN_COVERAGE_FRAC,
            "kept": kept,
            "dropped": {ba: int(c) for ba, c in coverage.items() if ba not in kept},
            "max_test_hours": int(coverage.max()),
        }, fh, indent=2)


def load_kept_bas(path=KEPT_BAS_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found -- run GAM_XGB.ipynb first; it decides the kept BAs.")
    with open(path) as fh:
        info = json.load(fh)
    if info["dropped"]:
        print(f"Hierarchy excludes {len(info['dropped'])} BAs (<{info['min_coverage_frac']:.0%} test coverage): "
              + ", ".join(f"{b} ({c}/{info['max_test_hours']} h)" for b, c in sorted(info["dropped"].items())))
    return info["kept"]


def members(node, kept):
    """Kept BAs belonging to an aggregate node."""
    if node == "NATIONAL":
        return sorted(kept)
    return sorted(ba for ba in kept if BA_TO_INTERCONNECT[ba] == node)


# lag features
def add_hourly_lags(df, lags=(24, 168), time_col="datetime_utc"):
    """Lag by TIME, not by row: load_h at (ba, t) is demand at (ba, t - h hours), NaN if
    that hour is missing. groupby().shift(h) would silently pull the wrong hour across gaps."""
    s = df.set_index([BA_COL, time_col])[TARGET]
    if not s.index.is_unique:
        raise ValueError("Duplicate (BA, timestamp) rows -- deduplicate before computing lags.")
    for h in lags:
        key = pd.MultiIndex.from_arrays([df[BA_COL], df[time_col] - pd.Timedelta(hours=h)])
        df[f"load_{h}"] = s.reindex(key).to_numpy()
    return df


# 
def build_eval_index(kept, directory=None):
    """Timestamps at which EVERY node (kept BAs + 4 aggregates) has both an actual and a
    forecast, minus the first EVAL_WARMUP_HOURS of them. Every method is scored on exactly
    these hours, so RMSE%/MAPE are comparable and errors can be paired for DM tests.
    Returns a Series of subperiod labels indexed by datetime_utc."""
    directory = directory or RESULTS_DIR
    common, labels = None, None
    for node in list(kept) + AGGREGATE_NODES:
        df = pd.read_parquet(os.path.join(directory, f"{node}.parquet"),
                             columns=["datetime_utc", "y_true", "y_pred", "subperiod"])
        df = df.dropna(subset=["y_true", "y_pred"]).set_index("datetime_utc")
        common = df.index if common is None else common.intersection(df.index)
        if labels is None:
            labels = df["subperiod"]
    common = common.sort_values()[EVAL_WARMUP_HOURS:]
    idx = labels.loc[common]
    idx.name = "subperiod"
    return idx


def save_or_check_eval_index(idx, path):
    """Write the index the first time; afterwards assert every script gets the same one."""
    if os.path.exists(path):
        prev = pd.read_parquet(path)["datetime_utc"]
        if not prev.reset_index(drop=True).equals(pd.Series(idx.index, name="datetime_utc")):
            raise ValueError(f"Evaluation index differs from {path} -- inputs changed since it was written.")
    else:
        idx.reset_index().to_parquet(path, index=False)



def score(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_true - y_pred
    rmse, mae, mean_demand = np.sqrt(np.mean(err ** 2)), np.mean(np.abs(err)), y_true.mean()
    return {"rmse": rmse, "mae": mae, "mean_demand": mean_demand,
            "rmse_pct": 100 * rmse / mean_demand, "mae_pct": 100 * mae / mean_demand,
            "mape": 100 * np.mean(np.abs(err / y_true)), "n_hours": len(y_true)}


def summary_rows(df, group, subperiods=SUBPERIODS):
    """df: columns y_true, y_pred, subperiod for one node. One row per subperiod."""
    rows = []
    for sp in subperiods:
        d = df if sp == "all" else df[df["subperiod"] == sp]
        if len(d):
            rows.append({"subperiod": sp, "level": "national" if group == "NATIONAL" else "interconnect",
                         "group": group, **score(d["y_true"], d["y_pred"])})
    return rows


def check_hierarchy_consistency(directory, kept, value_col="y_true", rtol=1e-6):
    """Assert each aggregate node's actuals equal the sum of its kept members' actuals
    on shared hours. Fails loudly if aggregates were built from a different BA set."""
    ba = {b: pd.read_parquet(os.path.join(directory, f"{b}.parquet"),
                             columns=["datetime_utc", value_col]).set_index("datetime_utc")[value_col]
          for b in kept}
    ba = pd.DataFrame(ba)
    for node in AGGREGATE_NODES:
        path = os.path.join(directory, f"{node}.parquet")
        if not os.path.exists(path):
            continue
        agg = pd.read_parquet(path, columns=["datetime_utc", value_col]).set_index("datetime_utc")[value_col]
        bottom = ba[members(node, kept)].dropna().sum(axis=1)
        idx = agg.index.intersection(bottom.index)
        gap = (agg.loc[idx] - bottom.loc[idx]).abs()
        rel = (gap / agg.loc[idx].abs()).max()
        if not np.isfinite(rel) or rel > rtol:
            raise ValueError(f"{node}: aggregate actuals differ from sum of kept members "
                             f"(max relative gap {rel:.2e}, max abs gap {gap.max():.1f} MWh). "
                             f"Rebuild the aggregates in GAM_XGB.ipynb.")
    print("Hierarchy check passed: every aggregate equals the sum of its kept members.")
