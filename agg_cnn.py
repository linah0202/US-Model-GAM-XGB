"""
Learned-weights aggregation via a 1D CNN.

REQUIRES:
  1. EASTERN/WESTERN/ERCOT/NATIONAL.parquet in RESULTS_DIR (test-period, from the
     GAM+XGB aggregation cell + rerun) -- true aggregate series to evaluate against,
     and each member BA's test-period y_pred as eval-time input.
  2. The matching *.parquet files in TRAIN_RESULTS_DIR (training-period GAM+XGB
     predictions, from the updated XGB cell in GAM+XGB.ipynb) -- used to TRAIN the CNN
     on genuinely held-out history (2016-2019) instead of carving a chunk out of the
     test period itself.

Evaluated over the FULL test period (2020-2022), broken down by subperiod (pre_shock /
acute_covid / recovery / stable) to compare adaptiveness to the COVID demand shock
against agg_simple_sum.py / agg_mint.py.

Note: the training-period input uses the initial (pre-walk-forward-tuned) GAM+XGB
prediction at lambda=1.0, the value the walk-forward loop itself starts from before
any tuning occurs, rather than the fully rolling-tuned prediction used at eval time --
same architecture as the test-period forecasts, but not identically tuned.

Needs PyTorch (CPU is fine for this size) -- see environment.yml.
"""
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

RESULTS_DIR = ""
TRAIN_RESULTS_DIR = ""
OUTPUT_DIR = ""
MIN_COVERAGE_FRAC = 0.95
SUBPERIODS = ["all", "pre_shock", "acute_covid", "recovery", "stable"]
WINDOW_LEN = 24
INTERNAL_VAL_FRAC = 0.15
HIDDEN_CHANNELS = 16
EPOCHS = 60
PATIENCE = 8
LR = 1e-3
SEED = 42

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


def filter_well_covered_bas():
    counts = {}
    for ba in sorted(BA_TO_INTERCONNECT):
        path = os.path.join(RESULTS_DIR, f"{ba}.parquet")
        counts[ba] = pd.read_parquet(path, columns=["datetime_utc"])["datetime_utc"].nunique()
    counts = pd.Series(counts)
    max_count = counts.max()
    keep = set(counts[counts >= MIN_COVERAGE_FRAC * max_count].index)
    dropped = counts[counts < MIN_COVERAGE_FRAC * max_count]
    if len(dropped):
        print(f"Excluding {len(dropped)} BAs with <{MIN_COVERAGE_FRAC:.0%} test coverage:")
        for ba, c in dropped.sort_values().items():
            print(f"  {ba}: {c} / {max_count} hours ({c / max_count:.1%})")
    return keep


class WeightCNN(nn.Module):
    def __init__(self, n_members, hidden=HIDDEN_CHANNELS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_members, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(hidden, n_members)

    def forward(self, x):
        h = self.net(x).squeeze(-1)
        scores = self.head(h)
        n_members = x.shape[1]
        # weights sum to n_members (avg=1) so uniform weights == simple-sum, see module docstring
        return torch.softmax(scores, dim=-1) * n_members


def load_ba_series(directory, bas, value_col):
    frames = {}
    for ba in bas:
        path = os.path.join(directory, f"{ba}.parquet")
        df = pd.read_parquet(path, columns=["datetime_utc", value_col]).set_index("datetime_utc")
        frames[ba] = df[value_col]
    combined = pd.DataFrame(frames)
    before = len(combined)
    combined = combined.dropna()
    dropped = before - len(combined)
    if dropped:
        print(f"  (dropped {dropped} hours with a gap in at least one member BA)")
    return combined


def load_target(directory, group, value_col, extra_cols=None):
    path = os.path.join(directory, f"{group}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Rerun GAM+XGB.ipynb (aggregation cell + updated GAM "
            f"cell) so {group}.parquet gets produced in both RESULTS_DIR and "
            f"TRAIN_RESULTS_DIR, then retry this script."
        )
    cols = ["datetime_utc", value_col] + (extra_cols or [])
    df = pd.read_parquet(path, columns=cols).set_index("datetime_utc")
    return df


def make_windows(pred_matrix, target, window_len):
    T, n_members = pred_matrix.shape
    n_samples = T - window_len + 1
    X = np.zeros((n_samples, n_members, window_len), dtype=np.float32)
    for i in range(n_samples):
        X[i] = pred_matrix[i:i + window_len].T
    y = target[window_len - 1:].astype(np.float32)
    current_preds = pred_matrix[window_len - 1:]
    return X, y, current_preds


def train_one_target(group, member_bas):
    print(f"\n=== {group} ({len(member_bas)} member BAs) ===")

    # training data: 2016-2019, GAM+XGB predictions (initial fit, lambda=1.0) 
    print("Loading training-period (GAM+XGB) predictions:")
    train_pred_df = load_ba_series(TRAIN_RESULTS_DIR, member_bas, "pred_train")
    train_target_df = load_target(TRAIN_RESULTS_DIR, group, "y_true")
    train_idx = train_pred_df.index.intersection(train_target_df.index).sort_values()
    train_pred_df = train_pred_df.loc[train_idx]
    train_target = train_target_df.loc[train_idx, "y_true"].to_numpy()

    X_all, y_all, cur_all = make_windows(train_pred_df.to_numpy(), train_target, WINDOW_LEN)
    n_val = int(len(X_all) * INTERNAL_VAL_FRAC)
    X_train, y_train, cur_train = X_all[:-n_val], y_all[:-n_val], cur_all[:-n_val]
    X_val, y_val, cur_val = X_all[-n_val:], y_all[-n_val:], cur_all[-n_val:]
    print(f"  {len(X_train)} training windows, {len(X_val)} internal-validation windows")

    # eval data: 2020-2022, actual GAM+XGB test predictions
    print("Loading test-period (GAM+XGB) predictions:")
    test_pred_df = load_ba_series(RESULTS_DIR, member_bas, "y_pred")
    test_target_df = load_target(RESULTS_DIR, group, "y_true", extra_cols=["subperiod"])
    test_idx = test_pred_df.index.intersection(test_target_df.index).sort_values()
    test_pred_df = test_pred_df.loc[test_idx]
    test_target = test_target_df.loc[test_idx, "y_true"].to_numpy()
    test_subperiod = test_target_df.loc[test_idx, "subperiod"]

    X_eval, y_eval, cur_eval = make_windows(test_pred_df.to_numpy(), test_target, WINDOW_LEN)
    dates_eval = test_idx[WINDOW_LEN - 1:]
    subperiod_eval = test_subperiod.iloc[WINDOW_LEN - 1:].to_numpy()
    print(f"  {len(X_eval)} evaluation windows")

    # normalize using ONLY the training scale (never test), see module docstring
    scale = np.abs(y_train).mean()
    X_train, X_val, X_eval_n = X_train / scale, X_val / scale, X_eval / scale
    y_train_n, y_val_n = y_train / scale, y_val / scale
    cur_train, cur_val, cur_eval_n = cur_train / scale, cur_val / scale, cur_eval / scale

    torch.manual_seed(SEED)
    model = WeightCNN(n_members=len(member_bas))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    def to_tensor(a):
        return torch.tensor(a, dtype=torch.float32)

    Xtr, ytr, curtr = to_tensor(X_train), to_tensor(y_train_n), to_tensor(cur_train)
    Xval, yval, curval = to_tensor(X_val), to_tensor(y_val_n), to_tensor(cur_val)

    best_val, best_state, patience_ctr = float("inf"), None, 0
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        w = model(Xtr)
        pred = (w * curtr).sum(dim=1)
        loss = nn.functional.mse_loss(pred, ytr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            w_val = model(Xval)
            pred_val = (w_val * curval).sum(dim=1)
            val_loss = nn.functional.mse_loss(pred_val, yval).item()

        if val_loss < best_val:
            best_val, best_state, patience_ctr = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"  early stop at epoch {epoch} (best val MSE, normalized units={best_val:.4f})")
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        w_eval = model(to_tensor(X_eval_n))
        pred_eval_norm = (w_eval * to_tensor(cur_eval_n)).sum(dim=1).numpy()
    pred_eval = pred_eval_norm * scale

    result_df = pd.DataFrame({
        "datetime_utc": dates_eval, "subperiod": subperiod_eval,
        "y_true": y_eval, "y_pred": pred_eval,
    })
    weights_df = pd.DataFrame(w_eval.numpy(), columns=member_bas)
    weights_df.insert(0, "datetime_utc", dates_eval)

    rows = []
    for sp in SUBPERIODS:
        d = result_df if sp == "all" else result_df[result_df["subperiod"] == sp]
        if d.empty:
            continue
        rmse = np.sqrt(np.mean((d["y_true"] - d["y_pred"]) ** 2))
        mae = np.mean(np.abs(d["y_true"] - d["y_pred"]))
        mean_demand = d["y_true"].mean()
        rows.append({
            "subperiod": sp, "level": "interconnect" if group != "NATIONAL" else "national",
            "group": group, "rmse": rmse, "mae": mae, "mean_demand": mean_demand,
            "rmse_pct": 100 * rmse / mean_demand, "mae_pct": 100 * mae / mean_demand,
            "n_hours": len(d),
        })
        if sp == "all":
            print(f"  RMSE={rmse:.2f} ({100*rmse/mean_demand:.2f}%)  MAE={mae:.2f} ({100*mae/mean_demand:.2f}%)")

    return rows, result_df, weights_df


def ercot_passthrough():
    """ERCOT has one member BA (ERCO); weight is trivially 1.0, no model to train."""
    df = pd.read_parquet(
        os.path.join(RESULTS_DIR, "ERCO.parquet"),
        columns=["datetime_utc", "y_true", "y_pred", "subperiod"],
    )
    print(f"\n=== ERCOT (1 member BA: ERCO, passthrough) ===")
    rows = []
    for sp in SUBPERIODS:
        d = df if sp == "all" else df[df["subperiod"] == sp]
        if d.empty:
            continue
        rmse = np.sqrt(np.mean((d["y_true"] - d["y_pred"]) ** 2))
        mae = np.mean(np.abs(d["y_true"] - d["y_pred"]))
        mean_demand = d["y_true"].mean()
        rows.append({
            "subperiod": sp, "level": "interconnect", "group": "ERCOT",
            "rmse": rmse, "mae": mae, "mean_demand": mean_demand,
            "rmse_pct": 100 * rmse / mean_demand, "mae_pct": 100 * mae / mean_demand,
            "n_hours": len(d),
        })
        if sp == "all":
            print(f"  RMSE={rmse:.2f} ({100*rmse/mean_demand:.2f}%)  MAE={mae:.2f} ({100*mae/mean_demand:.2f}%)")
    return rows, df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    well_covered = filter_well_covered_bas()
    print(f"\nKept {len(well_covered)} / {len(BA_TO_INTERCONNECT)} BAs")

    groups = {
        "EASTERN": sorted(ba for ba, g in BA_TO_INTERCONNECT.items() if g == "EASTERN" and ba in well_covered),
        "WESTERN": sorted(ba for ba, g in BA_TO_INTERCONNECT.items() if g == "WESTERN" and ba in well_covered),
        "NATIONAL": sorted(well_covered),
    }

    all_rows = []
    for group, member_bas in groups.items():
        rows, result_df, weights_df = train_one_target(group, member_bas)
        all_rows.extend(rows)
        result_df.to_parquet(os.path.join(OUTPUT_DIR, f"{group.lower()}_forecast.parquet"), index=False)
        weights_df.to_parquet(os.path.join(OUTPUT_DIR, f"{group.lower()}_weights.parquet"), index=False)

    ercot_rows, ercot_df = ercot_passthrough()
    all_rows.extend(ercot_rows)
    ercot_df.to_parquet(os.path.join(OUTPUT_DIR, "ercot_forecast.parquet"), index=False)

    summary = pd.DataFrame(all_rows)
    print()
    pd.set_option("display.width", 140)
    print(summary.to_string(index=False))
    summary.to_csv(os.path.join(OUTPUT_DIR, "cnn_summary.csv"), index=False)
    print(f"\nSaved results to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
