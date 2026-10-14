
import json
import os

import holidays
import numpy as np
import pandas as pd

from config import (BA_COL, TARGET, AGGREGATE_NODES, TRAIN_START, TRAIN_END, TEMP_PATH,
                    KEPT_BAS_PATH, DATASET_CACHE, select_kept_bas, members, add_hourly_lags)

ALPHA_95, ALPHA_99 = 0.05, 0.01
FEATURES = ["hour", "dayofweek", "dayofyear", "is_holiday",
            "load_24", "load_168", "avg_temp_K", "temp_s95", "temp_s99"]
COLS = ["datetime_utc", BA_COL, TARGET, "avg_temp_K", "temp_s95", "temp_s99",
        "load_24", "load_168", "hour", "dayofweek", "dayofyear", "is_holiday"]
US_HOLIDAYS = holidays.US()


def load_temperature():
    temp_wide = pd.read_parquet(TEMP_PATH)
    df_temp = (temp_wide.reset_index().drop(columns="XTIME")
               .melt(id_vars="Time", var_name=BA_COL, value_name="avg_temp_K")
               .rename(columns={"Time": "datetime_utc"})
               .sort_values([BA_COL, "datetime_utc"]).reset_index(drop=True))
    g = df_temp.groupby(BA_COL)["avg_temp_K"]
    df_temp["temp_s95"] = g.transform(lambda x: x.ewm(alpha=ALPHA_95, adjust=False).mean())
    df_temp["temp_s99"] = g.transform(lambda x: x.ewm(alpha=ALPHA_99, adjust=False).mean())
    return df_temp


def load_eia930():
    import pyarrow.fs as pafs
    import pyarrow.parquet as pq
    s3 = pafs.S3FileSystem(anonymous=True, region="us-west-2")
    df_y = pq.read_table("pudl.catalyst.coop/nightly/core_eia930__hourly_operations.parquet",
                         filesystem=s3).to_pandas(types_mapper=pd.ArrowDtype)
    df_y["datetime_utc"] = df_y["datetime_utc"].astype("datetime64[ns]")
    return df_y


def add_calendar(df):
    df["hour"] = df["datetime_utc"].dt.hour
    df["dayofweek"] = df["datetime_utc"].dt.dayofweek
    df["dayofyear"] = df["datetime_utc"].dt.dayofyear
    df["is_holiday"] = df["datetime_utc"].dt.date.apply(lambda x: x in US_HOLIDAYS).astype(int)
    return df


def build_dataset(df_y=None, df_temp=None, use_cache=True):
    if use_cache and os.path.exists(DATASET_CACHE):
        return pd.read_parquet(DATASET_CACHE)
    df_temp = load_temperature() if df_temp is None else df_temp
    df_y = load_eia930() if df_y is None else df_y

    df = df_y.merge(df_temp, on=[BA_COL, "datetime_utc"], how="inner")
    df[BA_COL] = df[BA_COL].astype(str)
    df = df.sort_values([BA_COL, "datetime_utc"]).reset_index(drop=True)

    kept, _ = select_kept_bas(df)
    if os.path.exists(KEPT_BAS_PATH):  
        nb_kept = json.load(open(KEPT_BAS_PATH))["kept"]
        if kept != nb_kept:
            raise ValueError(f"Kept BAs differ from {KEPT_BAS_PATH}: "
                             f"{sorted(set(kept) ^ set(nb_kept))}")
    df = df[df[BA_COL].isin(kept)].reset_index(drop=True)
    df = add_calendar(add_hourly_lags(df))

    train_rows = (df["datetime_utc"] >= TRAIN_START) & (df["datetime_utc"] <= TRAIN_END)
    ba_weights = df.loc[train_rows].groupby(BA_COL)[TARGET].mean()
    dem_wide = df.pivot(index="datetime_utc", columns=BA_COL, values=TARGET)
    temp_wide = df_temp.pivot(index="datetime_utc", columns=BA_COL, values="avg_temp_K")

    frames = []
    for node in AGGREGATE_NODES:
        mem = members(node, kept)
        w = ba_weights.reindex(mem)
        w = w / w.sum()
        t = temp_wide[mem].dropna()
        agg_t = pd.DataFrame({"avg_temp_K": t.to_numpy() @ w.to_numpy()}, index=t.index)
        agg_t["temp_s95"] = agg_t["avg_temp_K"].ewm(alpha=ALPHA_95, adjust=False).mean()
        agg_t["temp_s99"] = agg_t["avg_temp_K"].ewm(alpha=ALPHA_99, adjust=False).mean()
        d = dem_wide.reindex(columns=mem)
        agg_d = d[d.notna().all(axis=1)].sum(axis=1).rename(TARGET)
        out = agg_t.join(agg_d, how="inner").reset_index()
        out[BA_COL] = node
        frames.append(out)
    df_agg = pd.concat(frames, ignore_index=True)
    df_agg = add_calendar(add_hourly_lags(df_agg))

    data = pd.concat([df[COLS], df_agg[COLS]], ignore_index=True)
    data[TARGET] = data[TARGET].astype(float)
    data = data.sort_values([BA_COL, "datetime_utc"]).reset_index(drop=True)
    if use_cache:
        os.makedirs(os.path.dirname(DATASET_CACHE) or ".", exist_ok=True)
        tmp = f"{DATASET_CACHE}.{os.getpid()}.tmp"   
        data.to_parquet(tmp, index=False)
        os.replace(tmp, DATASET_CACHE)
    return data
