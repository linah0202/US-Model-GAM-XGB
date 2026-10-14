"""
Bottom-up (simple sum) aggregation of the kept BA-level GAM+XGB forecasts.
"""
import os

import pandas as pd

from config import (RESULTS_DIR, OUTPUT_DIR_ROOT, BA_TO_INTERCONNECT, INTERCONNECTS,
                    load_kept_bas, build_eval_index, save_or_check_eval_index, summary_rows)

OUTPUT_DIR = os.path.join(OUTPUT_DIR_ROOT, "simple_sum")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    kept = load_kept_bas()
    idx = build_eval_index(kept)
    save_or_check_eval_index(idx, os.path.join(OUTPUT_DIR_ROOT, "eval_index.parquet"))
    print(f"Shared evaluation index: {len(idx)} hours ({idx.index[0]} to {idx.index[-1]})")

    preds, actuals = {}, {}
    for ba in kept:
        df = pd.read_parquet(os.path.join(RESULTS_DIR, f"{ba}.parquet"),
                             columns=["datetime_utc", "y_true", "y_pred"]).set_index("datetime_utc")
        preds[ba], actuals[ba] = df["y_pred"].loc[idx.index], df["y_true"].loc[idx.index]
    preds, actuals = pd.DataFrame(preds), pd.DataFrame(actuals)

    rows, frames = [], []
    for node in INTERCONNECTS + ["NATIONAL"]:
        cols = kept if node == "NATIONAL" else [b for b in kept if BA_TO_INTERCONNECT[b] == node]
        d = pd.DataFrame({"y_true": actuals[cols].sum(axis=1), "y_pred": preds[cols].sum(axis=1),
                          "subperiod": idx})
        rows += summary_rows(d, node)
        frames.append(d.assign(node=node).reset_index())

    summary = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    print(summary.to_string(index=False))
    summary.to_csv(os.path.join(OUTPUT_DIR, "simple_sum_summary.csv"), index=False)
    pd.concat(frames, ignore_index=True).to_parquet(os.path.join(OUTPUT_DIR, "node_forecasts.parquet"), index=False)
    print(f"\nSaved results to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
