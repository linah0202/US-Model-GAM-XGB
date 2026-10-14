"""
Learned-weights aggregation via a 1D CNN.

"""
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from config import (RESULTS_DIR, TRAIN_FORECAST_DIR, OUTPUT_DIR_ROOT, load_kept_bas, members,
                    check_hierarchy_consistency, build_eval_index, save_or_check_eval_index,
                    summary_rows)


FT_STEPS = int(os.environ.get("FT_STEPS", 5))      
SEED = int(os.environ.get("SEED", 42))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR") or os.path.join(OUTPUT_DIR_ROOT, f"cnn_online_ft{FT_STEPS}_seed{SEED}")
WINDOW_LEN = 192
INTERNAL_VAL_FRAC = 0.15
HIDDEN_CHANNELS = 16
EPOCHS = 60
PATIENCE = 8
LR = 1e-3

# walk-forward (rolling) adaptation at test time
BATCH = 24        # same cadence as the λ loop
BUFFER = 2000     # trailing windows used for updates (same W as λ)
FT_LR = 1e-4      # lower than the offline 1e-3, so it fine-tunes rather than retrains





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
        # weights sum to n_members (avg=1) so uniform weights = simple-sum
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
            f"TRAIN_FORECAST_DIR, then retry this script."
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

def walk_forward_predict(model, X_eval, y_eval, cur_eval):
    """All inputs are normalized tensors; returns preds and weights in normalized units."""
    opt = torch.optim.Adam(model.parameters(), lr=FT_LR)
    preds, weights = [], []
    for i in range(0, len(X_eval), BATCH):
        sl = slice(i, i + BATCH)

        # 1) predict the next batch with current parameters (no targets used)
        model.eval()
        with torch.no_grad():
            w = model(X_eval[sl])
            preds.append((w * cur_eval[sl]).sum(dim=1))
            weights.append(w)

        # 2) targets for this batch are now realized, update on the trailing buffer
        lo = max(0, i + BATCH - BUFFER)
        Xb, yb, cb = X_eval[lo:i + BATCH], y_eval[lo:i + BATCH], cur_eval[lo:i + BATCH]
        model.train()
        for _ in range(FT_STEPS):
            opt.zero_grad()
            loss = nn.functional.mse_loss((model(Xb) * cb).sum(dim=1), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
    return torch.cat(preds).numpy(), torch.cat(weights).numpy()

def train_one_target(group, member_bas, idx):
    print(f"\n=== {group} ({len(member_bas)} member BAs) ===")

    # training data: out-of-sample 2018-2019 GAM+XGB forecasts (rolling lambda)
    print("Loading training-period (GAM+XGB) predictions:")
    train_pred_df = load_ba_series(TRAIN_FORECAST_DIR, member_bas, "pred_train")
    train_target_df = load_target(TRAIN_FORECAST_DIR, group, "y_true")
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

    # normalize using the training scale (never test)
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
    pred_eval_norm, w_eval = walk_forward_predict(
        model,
        to_tensor(X_eval_n),
        to_tensor(y_eval / scale),             
        to_tensor(cur_eval_n),
    )
    pred_eval = pred_eval_norm * scale

    result_df = pd.DataFrame({
        "datetime_utc": dates_eval, "subperiod": subperiod_eval,
        "y_true": y_eval, "y_pred": pred_eval,
    })
    weights_df = pd.DataFrame(w_eval, columns=member_bas)
    weights_df.insert(0, "datetime_utc", dates_eval)

    # restrict to the shared evaluation index (same hours as every other method)
    missing = idx.index.difference(pd.Index(result_df["datetime_utc"]))
    if len(missing):
        raise ValueError(f"{group}: {len(missing)} evaluation hours have no CNN forecast "
                         f"(first: {missing[0]}); increase EVAL_WARMUP_HOURS in config.py")
    keep = result_df["datetime_utc"].isin(idx.index).to_numpy()
    result_df = result_df[keep].reset_index(drop=True)
    result_df["subperiod"] = idx.loc[result_df["datetime_utc"]].to_numpy()
    weights_df = weights_df[keep].reset_index(drop=True)

    rows = summary_rows(result_df, group)
    r = rows[0]
    print(f"  RMSE={r['rmse']:.2f} ({r['rmse_pct']:.2f}%)  MAPE={r['mape']:.2f}%  on {r['n_hours']} hours")
    return rows, result_df, weights_df


def ercot_passthrough(idx):
    """ERCOT has one member BA (ERCO); weight is trivially 1.0, no model to train."""
    df = pd.read_parquet(os.path.join(RESULTS_DIR, "ERCO.parquet"),
                         columns=["datetime_utc", "y_true", "y_pred"]).set_index("datetime_utc")
    df = df.loc[idx.index].assign(subperiod=idx).reset_index()
    print(f"\n=== ERCOT (1 member BA: ERCO, passthrough) ===")
    return summary_rows(df, "ERCOT"), df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    kept = load_kept_bas()
    check_hierarchy_consistency(RESULTS_DIR, kept)
    check_hierarchy_consistency(TRAIN_FORECAST_DIR, kept)
    idx = build_eval_index(kept)
    save_or_check_eval_index(idx, os.path.join(OUTPUT_DIR_ROOT, "eval_index.parquet"))
    print(f"Shared evaluation index: {len(idx)} hours")
    groups = {node: members(node, kept) for node in ["EASTERN", "WESTERN", "NATIONAL"]}

    all_rows, frames = [], []
    for group, member_bas in groups.items():
        rows, result_df, weights_df = train_one_target(group, member_bas, idx)
        all_rows.extend(rows)
        frames.append(result_df.assign(node=group))
        result_df.to_parquet(os.path.join(OUTPUT_DIR, f"{group.lower()}_forecast.parquet"), index=False)
        weights_df.to_parquet(os.path.join(OUTPUT_DIR, f"{group.lower()}_weights.parquet"), index=False)

    ercot_rows, ercot_df = ercot_passthrough(idx)
    all_rows.extend(ercot_rows)
    frames.append(ercot_df.assign(node="ERCOT"))
    ercot_df.to_parquet(os.path.join(OUTPUT_DIR, "ercot_forecast.parquet"), index=False)

    summary = pd.DataFrame(all_rows)
    print()
    pd.set_option("display.width", 160)
    print(summary.to_string(index=False))
    summary.to_csv(os.path.join(OUTPUT_DIR, "cnn_summary.csv"), index=False)
    pd.concat(frames, ignore_index=True).to_parquet(os.path.join(OUTPUT_DIR, "node_forecasts.parquet"), index=False)
    print(f"\nSaved results to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
