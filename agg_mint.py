"""
MinT (Minimum Trace) hierarchical reconciliation.
"""
import os

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from config import (RESULTS_DIR, TRAIN_FORECAST_DIR, OUTPUT_DIR_ROOT, BA_TO_INTERCONNECT,
                    INTERCONNECTS, load_kept_bas, check_hierarchy_consistency,
                    build_eval_index, save_or_check_eval_index, summary_rows)

OUTPUT_DIR = os.path.join(OUTPUT_DIR_ROOT, "mint")


def build_summing_matrix(bottom_bas):
    m = len(bottom_bas)
    S = np.zeros((m + len(INTERCONNECTS) + 1, m))
    S[:m] = np.eye(m)
    for i, node in enumerate(INTERCONNECTS):
        for j, ba in enumerate(bottom_bas):
            S[m + i, j] = float(BA_TO_INTERCONNECT[ba] == node)
    S[-1, :] = 1.0
    return S


def load_matrix(directory, nodes, pred_col, index=None):
    """Aligned (T x n) actual and forecast matrices; restricted to `index` if given,
    otherwise to timestamps where every node has both."""
    Y, Yh = {}, {}
    for node in nodes:
        df = pd.read_parquet(os.path.join(directory, f"{node}.parquet"),
                             columns=["datetime_utc", "y_true", pred_col]).set_index("datetime_utc")
        Y[node], Yh[node] = df["y_true"], df[pred_col]
    Y, Yh = pd.DataFrame(Y)[nodes], pd.DataFrame(Yh)[nodes]
    if index is not None:
        return Y.loc[index], Yh.loc[index]
    ok = Y.notna().all(axis=1) & Yh.notna().all(axis=1)
    return Y[ok].sort_index(), Yh[ok].sort_index()


def reconcile_matrix(S, W):
    W_inv = np.linalg.pinv(W)
    StWinv = S.T @ W_inv
    return S @ np.linalg.pinv(StWinv @ S) @ StWinv


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    bottom = load_kept_bas()
    check_hierarchy_consistency(RESULTS_DIR, bottom)
    check_hierarchy_consistency(TRAIN_FORECAST_DIR, bottom)
    nodes = bottom + INTERCONNECTS + ["NATIONAL"]
    S = build_summing_matrix(bottom)
    print(f"Summing matrix S: {S.shape} ({len(bottom)} BAs + {len(INTERCONNECTS)} interconnects + national)")

    Y_tr, Yh_tr = load_matrix(TRAIN_FORECAST_DIR, nodes, "pred_train")
    print(f"W estimated on {len(Y_tr)} out-of-sample hours ({Y_tr.index[0]} to {Y_tr.index[-1]})")
    W = LedoitWolf().fit((Y_tr - Yh_tr).to_numpy()).covariance_
    P = reconcile_matrix(S, W)

    idx = build_eval_index(bottom)
    save_or_check_eval_index(idx, os.path.join(OUTPUT_DIR_ROOT, "eval_index.parquet"))
    Y_te, Yh_te = load_matrix(RESULTS_DIR, nodes, "y_pred", index=idx.index)
    rec = pd.DataFrame(Yh_te.to_numpy() @ P.T, index=Y_te.index, columns=nodes)
    print(f"Shared evaluation index: {len(idx)} hours")

    rows, frames = [], []
    for node in INTERCONNECTS + ["NATIONAL"]:
        d = pd.DataFrame({"y_true": Y_te[node], "y_pred": rec[node], "subperiod": idx})
        rows += summary_rows(d, node)
        frames.append(d.assign(node=node).reset_index())

    summary = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    print(summary.to_string(index=False))
    summary.to_csv(os.path.join(OUTPUT_DIR, "mint_summary.csv"), index=False)
    pd.concat(frames, ignore_index=True).to_parquet(os.path.join(OUTPUT_DIR, "node_forecasts.parquet"), index=False)
    rec.assign(subperiod=idx).reset_index().to_parquet(os.path.join(OUTPUT_DIR, "reconciled_forecasts.parquet"), index=False)
    print(f"\nSaved results to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
