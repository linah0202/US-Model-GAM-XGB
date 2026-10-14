"""
Rolling MinT (Minimum Trace) hierarchical reconciliation.

Hierarchy: kept BAs (bottom) -> Eastern / Western / ERCOT -> National. The BA set comes
from kept_bas.json, so the summing matrix S matches the top-level base forecasts.

W (Ledoit-Wolf shrinkage covariance of base-forecast errors) is re-estimated every
24 hours on a trailing 2,000-hour buffer of realized errors -- the same cadence and
buffer as the rolling lambda and the online CNN, so all adaptive methods are updated
on equal terms:
  - the buffer starts with the last 2,000 hours of OUT-OF-SAMPLE 2018-2019 errors
    (TRAIN_FORECAST_DIR, from train_oos_forecasts.py)
  - after each 24-hour batch is forecast, its realized errors enter the buffer and the
    oldest leave; W and the reconciliation matrix are re-estimated before the next batch
  - a batch is reconciled only with errors observed before it (no look-ahead)
Scored on the shared evaluation index.

Outputs (OUTPUT_DIR_ROOT/mint):
  mint_summary.csv, node_forecasts.parquet, reconciled_forecasts.parquet (all nodes)
"""
import os

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from config import (RESULTS_DIR, TRAIN_FORECAST_DIR, OUTPUT_DIR_ROOT, BA_TO_INTERCONNECT,
                    INTERCONNECTS, load_kept_bas, check_hierarchy_consistency,
                    build_eval_index, save_or_check_eval_index, summary_rows)

OUTPUT_DIR = os.path.join(OUTPUT_DIR_ROOT, "mint")
BATCH = 24      # re-estimation cadence (hours), same as lambda / online CNN
BUFFER = 2000   # trailing hours of errors used for W, same as lambda / online CNN


def build_summing_matrix(bottom_bas):
    m = len(bottom_bas)
    S = np.zeros((m + len(INTERCONNECTS) + 1, m))
    S[:m] = np.eye(m)
    for i, node in enumerate(INTERCONNECTS):
        for j, ba in enumerate(bottom_bas):
            S[m + i, j] = float(BA_TO_INTERCONNECT[ba] == node)
    S[-1, :] = 1.0
    return S


def load_matrix(directory, nodes, pred_col):
    """Aligned (T x n) actual and forecast matrices on hours where every node has both."""
    Y, Yh = {}, {}
    for node in nodes:
        df = pd.read_parquet(os.path.join(directory, f"{node}.parquet"),
                             columns=["datetime_utc", "y_true", pred_col]).set_index("datetime_utc")
        Y[node], Yh[node] = df["y_true"], df[pred_col]
    Y, Yh = pd.DataFrame(Y)[nodes], pd.DataFrame(Yh)[nodes]
    ok = Y.notna().all(axis=1) & Yh.notna().all(axis=1)
    return Y[ok].sort_index(), Yh[ok].sort_index()


def reconcile_matrix(S, W):
    W_inv = np.linalg.pinv(W)
    StWinv = S.T @ W_inv
    return S @ np.linalg.pinv(StWinv @ S) @ StWinv


def rolling_mint(S, err_init, Yh_test, Y_test):
    """Walk forward through the test period in BATCH-hour steps. Returns reconciled
    forecasts (same shape as Yh_test) and the shrinkage intensity used for each batch."""
    buf = err_init[-BUFFER:]
    Yh, Y = Yh_test.to_numpy(), Y_test.to_numpy()
    out, shrink = np.empty_like(Yh), np.empty(len(Yh))
    for i in range(0, len(Yh), BATCH):
        lw = LedoitWolf().fit(buf)
        P = reconcile_matrix(S, lw.covariance_)
        out[i:i + BATCH] = Yh[i:i + BATCH] @ P.T
        shrink[i:i + BATCH] = lw.shrinkage_
        buf = np.vstack([buf, Y[i:i + BATCH] - Yh[i:i + BATCH]])[-BUFFER:]  # realized after forecasting
    return out, shrink


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    bottom = load_kept_bas()
    check_hierarchy_consistency(RESULTS_DIR, bottom)
    check_hierarchy_consistency(TRAIN_FORECAST_DIR, bottom)
    nodes = bottom + INTERCONNECTS + ["NATIONAL"]
    S = build_summing_matrix(bottom)
    print(f"Summing matrix S: {S.shape} ({len(bottom)} BAs + {len(INTERCONNECTS)} interconnects + national)")

    Y_tr, Yh_tr = load_matrix(TRAIN_FORECAST_DIR, nodes, "pred_train")
    err_init = (Y_tr - Yh_tr).to_numpy()
    print(f"Initial W buffer: last {min(BUFFER, len(err_init))} of {len(err_init)} out-of-sample hours "
          f"(ending {Y_tr.index[-1]})")

    # walk forward over every test hour where all nodes have a forecast (the buffer needs
    # the hours before the evaluation window too); score only on the shared index
    Y_te, Yh_te = load_matrix(RESULTS_DIR, nodes, "y_pred")
    rec, shrink = rolling_mint(S, err_init, Yh_te, Y_te)
    rec = pd.DataFrame(rec, index=Y_te.index, columns=nodes)
    print(f"W re-estimated {int(np.ceil(len(Y_te) / BATCH))} times; "
          f"Ledoit-Wolf shrinkage mean {shrink.mean():.3f} (range {shrink.min():.3f}-{shrink.max():.3f})")

    idx = build_eval_index(bottom)
    save_or_check_eval_index(idx, os.path.join(OUTPUT_DIR_ROOT, "eval_index.parquet"))
    missing = idx.index.difference(rec.index)
    if len(missing):
        raise ValueError(f"{len(missing)} evaluation hours have no reconciled forecast (first: {missing[0]})")
    rec, Y_eval = rec.loc[idx.index], Y_te.loc[idx.index]
    print(f"Shared evaluation index: {len(idx)} hours")

    rows, frames = [], []
    for node in INTERCONNECTS + ["NATIONAL"]:
        d = pd.DataFrame({"y_true": Y_eval[node], "y_pred": rec[node], "subperiod": idx})
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
