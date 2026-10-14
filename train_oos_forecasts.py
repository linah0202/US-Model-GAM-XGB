"""
Out-of-sample training-period forecasts for MinT's W and the CNN's training data.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from pygam import LinearGAM, f, s, te
from scipy.optimize import minimize_scalar
from sklearn.model_selection import GridSearchCV

from config import BA_COL, TARGET, AGGREGATE_NODES, OOS_FOLDS, TRAIN_OOS_DIR, load_kept_bas
from data_prep import FEATURES, build_dataset

WINDOW, TUNE_WINDOW, BATCH_SIZE = 2000, 168, 24   # same as GAM_XGB.ipynb


def fit_gam(X, y):
    gam = LinearGAM(
        s(0, n_splines=24, basis="cp") + s(1, n_splines=7) + s(2, n_splines=10, basis="cp") +
        f(3) + s(4, n_splines=10) + s(5, n_splines=10) + s(6, n_splines=10) +
        s(7, n_splines=10) + s(8, n_splines=10) + te(0, 6, n_splines=[8, 8])
    )
    gam.gridsearch(X, y, lam=np.logspace(-3, 3, 11), progress=False)
    return gam


def fit_and_walk_forward(df_fit, df_fwd):
    """Identical procedure to the GAM and XGB cells of GAM_XGB.ipynb, with df_fit as the
    training data and df_fwd as the period forecast walk-forward."""
    t0 = time.perf_counter()
    X_fit = df_fit[FEATURES].astype(float)
    y_fit = df_fit[TARGET].to_numpy(dtype=float)
    X_fwd = df_fwd[FEATURES].astype(float)
    y_fwd = df_fwd[TARGET].to_numpy(dtype=float)
    gam = fit_gam(X_fit.to_numpy(), y_fit)
    gam_fit, gam_fwd = gam.predict(X_fit.to_numpy()), gam.predict(X_fwd.to_numpy())
    t_gam = time.perf_counter() - t0

    t0 = time.perf_counter()
    X_fit_xgb = X_fit.copy(); X_fit_xgb["gam_pred"] = gam_fit
    res_fit = (y_fit - gam_fit).astype(float)
    gs = GridSearchCV(xgb.XGBRegressor(objective="reg:squarederror", random_state=42),
                      param_grid={"n_estimators": [100, 200], "max_depth": [3, 4]},
                      cv=3, scoring="neg_mean_squared_error")
    gs.fit(X_fit_xgb, res_fit)
    model = gs.best_estimator_
    t_grid = time.perf_counter() - t0

    t0 = time.perf_counter()
    X_hist, res_hist = X_fit_xgb.copy(), list(res_fit)
    X_fwd_xgb = X_fwd.copy(); X_fwd_xgb["gam_pred"] = gam_fwd
    preds, lambdas, lam = [], [], 1.0
    for i in range(0, len(X_fwd_xgb), BATCH_SIZE):
        Xb = X_fwd_xgb.iloc[i:i + BATCH_SIZE]
        yb, gb = y_fwd[i:i + BATCH_SIZE], gam_fwd[i:i + BATCH_SIZE]
        preds.extend(gb + lam * model.predict(Xb))
        lambdas.extend([lam] * len(Xb))
        res_hist.extend((yb - gb).tolist())
        X_hist = pd.concat([X_hist, Xb]).iloc[-WINDOW:]
        res_hist = res_hist[-WINDOW:]
        model.fit(X_hist.iloc[:-TUNE_WINDOW], res_hist[:-TUNE_WINDOW])
        p_tune = model.predict(X_hist.iloc[-TUNE_WINDOW:])
        r_tune = np.array(res_hist[-TUNE_WINDOW:], dtype=float)
        lam = minimize_scalar(lambda l: np.mean((r_tune - l * p_tune) ** 2),
                              bounds=(0.0, 1.0), method="bounded").x
    t_walk = time.perf_counter() - t0

    out = df_fwd[["datetime_utc", BA_COL]].copy()
    out["y_true"], out["pred_train"] = y_fwd, np.asarray(preds, float)
    out["gam_pred"], out["lambda"] = gam_fwd, lambdas
    return out, {"gam_s": t_gam, "xgb_grid_s": t_grid, "walk_forward_s": t_walk,
                 "n_fit": len(df_fit), "n_forecast": len(df_fwd)}


def run_node(data, node):
    out_path = os.path.join(TRAIN_OOS_DIR, f"{node}.parquet")
    if os.path.exists(out_path):
        print(f"{node}: already done, skipping")
        return
    d = data[data[BA_COL] == node].dropna(subset=FEATURES + [TARGET])
    parts, timings = [], []
    for k, (fs, fe, ps, pe) in enumerate(OOS_FOLDS, start=1):
        df_fit = d[(d["datetime_utc"] >= fs) & (d["datetime_utc"] <= fe)].reset_index(drop=True)
        df_fwd = d[(d["datetime_utc"] >= ps) & (d["datetime_utc"] <= pe)].reset_index(drop=True)
        if len(df_fit) < 200 or len(df_fwd) == 0:
            print(f"{node} fold {k}: insufficient data (fit={len(df_fit)}, forecast={len(df_fwd)}), skipping fold")
            continue
        res, tm = fit_and_walk_forward(df_fit, df_fwd)
        res["fold"] = k
        parts.append(res)
        timings.append({"node": node, "fold": k, **tm})
        rmse = np.sqrt(np.mean((res["y_true"] - res["pred_train"]) ** 2))
        print(f"{node} fold {k}: RMSE%={100 * rmse / res['y_true'].mean():.2f}  "
              f"({sum(v for key, v in tm.items() if key.endswith('_s')):.0f} s)", flush=True)
    if not parts:
        return
    os.makedirs(os.path.join(TRAIN_OOS_DIR, "timings"), exist_ok=True)
    pd.DataFrame(timings).to_csv(os.path.join(TRAIN_OOS_DIR, "timings", f"{node}.csv"), index=False)
    pd.concat(parts, ignore_index=True).to_parquet(out_path, index=False)  


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--nchunks", type=int, default=1)
    ap.add_argument("--nodes", nargs="*", help="optional explicit node list")
    args = ap.parse_args()

    os.makedirs(TRAIN_OOS_DIR, exist_ok=True)
    nodes = args.nodes or (load_kept_bas() + AGGREGATE_NODES)
    nodes = nodes[args.chunk::args.nchunks]
    print(f"Running {len(nodes)} nodes (chunk {args.chunk}/{args.nchunks}): {nodes}", flush=True)
    data = build_dataset()
    for node in nodes:
        run_node(data, node)


if __name__ == "__main__":
    main()
