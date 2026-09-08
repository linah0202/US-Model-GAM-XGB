
import glob
import os

import numpy as np
import pandas as pd

RESULTS_DIR = ""
OUTPUT_DIR = ""
MIN_COVERAGE_FRAC = 0.95  # BAs below this fraction of max observed test hours are excluded
SUBPERIODS = ["all", "pre_shock", "acute_covid", "recovery", "stable"]

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


def load_ba_results():
    """Load the 55 BA-level result parquets only. Ignores EASTERN/WESTERN/ERCOT/NATIONAL
    files if present in RESULTS_DIR -- simple-sum only ever uses bottom-level forecasts."""
    frames = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.parquet"))):
        ba = os.path.splitext(os.path.basename(path))[0]
        if ba not in BA_TO_INTERCONNECT:
            continue
        df = pd.read_parquet(
            path, columns=["datetime_utc", "balancing_authority_code_eia", "subperiod", "y_true", "y_pred"]
        )
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No BA-level result parquets found in {RESULTS_DIR}")
    return pd.concat(frames, ignore_index=True)


def filter_well_covered_bas(df):
    """Excludes BAs with severely truncated test windows (retired mid-period, large
    gaps, etc). Without this, a single short-lived BA (e.g. NSB has ~1 week of test
    data) bottlenecks the "timestamp present for every BA" intersection to near zero."""
    counts = df.groupby("balancing_authority_code_eia")["datetime_utc"].nunique()
    max_count = counts.max()
    keep = counts[counts >= MIN_COVERAGE_FRAC * max_count].index
    dropped = counts[counts < MIN_COVERAGE_FRAC * max_count]
    if len(dropped):
        print(f"Excluding {len(dropped)} BAs with <{MIN_COVERAGE_FRAC:.0%} test coverage "
              f"(would bottleneck the common-timestamp window):")
        for ba, c in dropped.sort_values().items():
            print(f"  {ba}: {c} / {max_count} hours ({c / max_count:.1%})")
    return df[df["balancing_authority_code_eia"].isin(keep)].copy(), sorted(keep)


def restrict_to_common_window(df, kept_bas):
    """Use only timestamps where ALL kept BAs have a forecast, across the full test
    period."""
    n_bas = len(kept_bas)
    counts = df.groupby("datetime_utc")["balancing_authority_code_eia"].nunique()
    full_coverage_times = counts[counts == n_bas].index
    return df[df["datetime_utc"].isin(full_coverage_times)].copy()


def summarize(df, group_col, label, subperiod):
    d = df if subperiod == "all" else df[df["subperiod"] == subperiod]
    if d.empty:
        return pd.DataFrame()
    agg = d.groupby(["datetime_utc", group_col]).agg(
        y_true=("y_true", "sum"), y_pred=("y_pred", "sum")
    ).reset_index()
    rows = []
    for g, sub in agg.groupby(group_col):
        rmse = np.sqrt(np.mean((sub["y_true"] - sub["y_pred"]) ** 2))
        mae = np.mean(np.abs(sub["y_true"] - sub["y_pred"]))
        mean_demand = sub["y_true"].mean()
        rows.append({
            "subperiod": subperiod, "level": label, "group": g,
            "rmse": rmse, "mae": mae, "mean_demand": mean_demand,
            "rmse_pct": 100 * rmse / mean_demand, "mae_pct": 100 * mae / mean_demand,
            "n_hours": len(sub),
        })
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_ba_results()
    df, kept_bas = filter_well_covered_bas(df)
    df["interconnect"] = df["balancing_authority_code_eia"].map(BA_TO_INTERCONNECT)
    df["national"] = "NATIONAL"

    common = restrict_to_common_window(df, kept_bas)
    print(f"\nKept {len(kept_bas)} / {len(BA_TO_INTERCONNECT)} BAs")
    print(f"Common-coverage timestamps (all kept BAs present): {common['datetime_utc'].nunique()}")
    print(common.groupby("subperiod")["datetime_utc"].nunique())

    all_summaries = []
    for subperiod in SUBPERIODS:
        s1 = summarize(common, "interconnect", "interconnect", subperiod)
        s2 = summarize(common, "national", "national", subperiod)
        all_summaries.append(s1)
        all_summaries.append(s2)

    summary = pd.concat(all_summaries, ignore_index=True)
    print()
    pd.set_option("display.width", 140)
    print(summary.to_string(index=False))

    summary.to_csv(os.path.join(OUTPUT_DIR, "simple_sum_summary.csv"), index=False)
    common.to_parquet(os.path.join(OUTPUT_DIR, "aligned_ba_forecasts.parquet"), index=False)
    print(f"\nSaved results to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
