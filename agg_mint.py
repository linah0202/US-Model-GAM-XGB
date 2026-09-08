"""
MinT (Minimum Trace) hierarchical reconciliation.

REQUIRES two things:
  1. EASTERN/WESTERN/ERCOT/NATIONAL.parquet in RESULTS_DIR (test-period, from the GAM+XGB
     aggregation cell + rerun) -- the independent top-level base forecasts MinT needs.
  2. The matching *.parquet files in TRAIN_RESULTS_DIR (training-period GAM+XGB
     predictions, from the updated XGB cell in GAM+XGB.ipynb) -- used to fit W on
     genuinely held-out history (2016-2019) instead of carving a chunk out of the test
     period itself.

Evaluated over the FULL test period (2020-2022), broken down by subperiod (pre_shock /
acute_covid / recovery / stable) to compare adaptiveness to the COVID demand shock
against agg_simple_sum.py / agg_cnn.py.

BAs with severely truncated test coverage (AEC, NSB, PSEI) are excluded from the
hierarchy.

Hierarchy: ~52 BAs (bottom) -> Eastern / Western / ERCOT (middle) -> National (top).

"""
import os

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

RESULTS_DIR = ""
TRAIN_RESULTS_DIR = ""
OUTPUT_DIR = ""
MIN_COVERAGE_FRAC = 0.95
SUBPERIODS = ["all", "pre_shock", "acute_covid", "recovery", "stable"]

BA_TO_INTERCONNECT = {
    "BANC": "WESTERN", "CISO": "WESTERN", "IID": "WESTERN", "LDWP": "WESTERN", "TIDC": "WESTERN",
    "AVA": "WESTERN", "BPAT": "WESTERN", "CHPD": "WESTERN", "DOPD": "WESTERN", "GCPD": "WESTERN",
    "IPCO": "WESTERN", "NEVP": "WESTERN", "NWMT": "WESTERN", "PACE": "WESTERN", "PACW": "WESTERN",
    "PGE": "WESTERN", "PSCO": "WESTERN", "PSEI": "WESTERN", "SCL": "WESTERN", "TPWR": "WESTERN",
    "WACM": "WESTERN", "WAUW": "WESTERN", "AZPS": "WESTERN", "EPE": "WESTERN", "PNM": "WESTERN",
    "SRP": "WESTERN", "TEPC": "WESTERN", "WALC": "WESTERN",
    "AEC": "EASTERN", "SOCO": "EASTERN", "AECI": "EASTERN", "LGEE": "EASTERN", "MISO": "EASTERN",
    "CPLE": "EASTERN", "CPLW": "EASTERN", "DUK": "EASTERN", "SC": "EASTERN", "SCEG": "EASTERN",
    "FMPP": "EASTERN", "FPC": "EASTERN", "FPL": "EASTERN", "GVL": "EASTERN", "HST": "EASTERN",
    "JEA": "EASTERN", "NSB": "EASTERN", "SEC": "EASTERN", "TAL": "EASTERN", "TEC": "EASTERN",
    "ISNE": "EASTERN", "NYIS": "EASTERN", "PJM": "EASTERN", "TVA": "EASTERN", "SPA": "EASTERN", "SWPP": "EASTERN",
    "ERCO": "ERCOT",
}
INTERCONNECTS = ["EASTERN", "ERCOT", "WESTERN"]


def filter_well_covered_bas():
    counts = {}
    for ba in sorted(BA_TO_INTERCONNECT):
        path = os.path.join(RESULTS_DIR, f"{ba}.parquet")
        counts[ba] = pd.read_parquet(path, columns=["datetime_utc"])["datetime_utc"].nunique()
    counts = pd.Series(counts)
    max_count = counts.max()
    keep = sorted(counts[counts >= MIN_COVERAGE_FRAC * max_count].index)
    dropped = counts[counts < MIN_COVERAGE_FRAC * max_count]
    if len(dropped):
        print(f"Excluding {len(dropped)} BAs with <{MIN_COVERAGE_FRAC:.0%} test coverage "
              f"from the MinT hierarchy (would bottleneck the common-timestamp window):")
        for ba, c in dropped.sort_values().items():
            print(f"  {ba}: {c} / {max_count} hours ({c / max_count:.1%})")
    return keep


def build_summing_matrix(bottom_bas):
    m = len(bottom_bas)
    n = m + len(INTERCONNECTS) + 1
    S = np.zeros((n, m), dtype=float)
    for j in range(m):
        S[j, j] = 1.0
    for i, node in enumerate(INTERCONNECTS):
        row = m + i
        for j, ba in enumerate(bottom_bas):
            if BA_TO_INTERCONNECT[ba] == node:
                S[row, j] = 1.0
    S[m + len(INTERCONNECTS), :] = 1.0
    return S


def load_train_node(node):
    path = os.path.join(TRAIN_RESULTS_DIR, f"{node}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Rerun GAM+XGB.ipynb (the updated GAM cell) so training-"
            f"period predictions get exported, then retry this script."
        )
    df = pd.read_parquet(path, columns=["datetime_utc", "y_true", "pred_train"])
    return df.set_index("datetime_utc").rename(columns={"pred_train": "y_pred"})[["y_true", "y_pred"]]


def load_test_node(node):
    path = os.path.join(RESULTS_DIR, f"{node}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. MinT needs independent top-level base forecasts -- "
            f"rerun GAM+XGB.ipynb after the interconnect/national aggregation cell "
            f"so {node}.parquet gets produced, then retry this script."
        )
    return pd.read_parquet(path, columns=["datetime_utc", "y_true", "y_pred", "subperiod"]).set_index("datetime_utc")


def build_aligned_matrices(all_nodes, loader, extra_cols=None):
    per_node = {node: loader(node) for node in all_nodes}
    common_index = per_node[all_nodes[0]].index
    for node in all_nodes[1:]:
        common_index = common_index.intersection(per_node[node].index)
    common_index = common_index.sort_values()

    Y = np.column_stack([per_node[node].loc[common_index, "y_true"].to_numpy() for node in all_nodes])
    Y_hat = np.column_stack([per_node[node].loc[common_index, "y_pred"].to_numpy() for node in all_nodes])
    extras = {}
    if extra_cols:
        for col in extra_cols:
            extras[col] = per_node[all_nodes[0]].loc[common_index, col]
    return common_index, Y, Y_hat, extras


def fit_W(Y_train, Y_hat_train):
    errors = Y_train - Y_hat_train
    return LedoitWolf().fit(errors).covariance_


def reconcile_matrix(S, W):
    W_inv = np.linalg.pinv(W)
    StWinv = S.T @ W_inv
    G = np.linalg.pinv(StWinv @ S) @ StWinv
    return S @ G


def summarize(y_true_vec, y_pred_vec, label, subperiod):
    if len(y_true_vec) == 0:
        return None
    rmse = np.sqrt(np.mean((y_true_vec - y_pred_vec) ** 2))
    mae = np.mean(np.abs(y_true_vec - y_pred_vec))
    mean_demand = y_true_vec.mean()
    return {
        "subperiod": subperiod,
        "level": "interconnect" if label != "NATIONAL" else "national",
        "group": label, "rmse": rmse, "mae": mae, "mean_demand": mean_demand,
        "rmse_pct": 100 * rmse / mean_demand, "mae_pct": 100 * mae / mean_demand,
        "n_hours": len(y_true_vec),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    bottom_bas = filter_well_covered_bas()
    print(f"\nKept {len(bottom_bas)} / {len(BA_TO_INTERCONNECT)} BAs for the MinT hierarchy")
    all_nodes = bottom_bas + INTERCONNECTS + ["NATIONAL"]
    m, n = len(bottom_bas), len(all_nodes)

    S = build_summing_matrix(bottom_bas)
    print(f"Summing matrix S: {S.shape} ({n} nodes = {m} BAs + {len(INTERCONNECTS)} interconnects + 1 national)")

    train_index, Y_train, Y_hat_train, _ = build_aligned_matrices(all_nodes, load_train_node)
    print(f"Training-period (2016-2019) common timestamps across all {n} nodes: {len(train_index)}")
    print(f"  {train_index[0]} to {train_index[-1]}")

    W = fit_W(Y_train, Y_hat_train)
    P = reconcile_matrix(S, W)

    test_index, Y_test, Y_hat_test, extras = build_aligned_matrices(
        all_nodes, load_test_node, extra_cols=["subperiod"]
    )
    print(f"Test-period common timestamps across all {n} nodes: {len(test_index)}")
    subperiod = extras["subperiod"]

    Y_hat_reconciled = Y_hat_test @ P.T

    rows = []
    for sp in SUBPERIODS:
        mask = np.ones(len(test_index), dtype=bool) if sp == "all" else (subperiod.to_numpy() == sp)
        for node in INTERCONNECTS + ["NATIONAL"]:
            i = all_nodes.index(node)
            row = summarize(Y_test[mask, i], Y_hat_reconciled[mask, i], node, sp)
            if row:
                rows.append(row)
    summary = pd.DataFrame(rows)
    print()
    pd.set_option("display.width", 140)
    print(summary.to_string(index=False))

    out = pd.DataFrame(Y_hat_reconciled, columns=all_nodes, index=test_index)
    out.index.name = "datetime_utc"
    out["subperiod"] = subperiod.to_numpy()
    out.reset_index().to_parquet(os.path.join(OUTPUT_DIR, "reconciled_forecasts.parquet"), index=False)
    summary.to_csv(os.path.join(OUTPUT_DIR, "mint_summary.csv"), index=False)
    print(f"\nSaved results to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
