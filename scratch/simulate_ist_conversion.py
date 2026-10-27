import gzip
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, ".")
from src.config import DELHI_STATIONS
from src.data_loader import compute_hourly_aqi_from_pollutants
from src.refresh_live import VAYUAYAN_STATIONS

raw_live_dir = Path("data/raw/live")
station_daily_list = []
station_meta = {}

for st_name, info in VAYUAYAN_STATIONS.items():
    raw_file_path = raw_live_dir / info["local_raw"]
    with gzip.open(raw_file_path, "rt") as f:
        df_raw = pd.read_csv(f)

    # 1. Convert timestamp from UTC to IST
    df_raw["Timestamp_UTC"] = pd.to_datetime(df_raw["timestamp"])
    df_raw["Timestamp"] = df_raw["Timestamp_UTC"] + pd.Timedelta(hours=5, minutes=30)

    min_ts_utc = df_raw["Timestamp_UTC"].min()
    max_ts_utc = df_raw["Timestamp_UTC"].max()
    min_ts_ist = df_raw["Timestamp"].min()
    max_ts_ist = df_raw["Timestamp"].max()

    station_meta[st_name] = {
        "site_id": info["site_id"],
        "station_name": info["station_name"],
        "url": "https://data.vayuayan.saketlab.org/data/" + info["remote_file"],
        "filename": info["local_raw"],
        "min_timestamp_utc": str(min_ts_utc),
        "max_timestamp_utc": str(max_ts_utc),
        "min_timestamp": str(min_ts_ist),
        "max_timestamp": str(max_ts_ist),
        "rows": len(df_raw),
    }

    # Clean pollutants
    df_raw["PM2.5"] = pd.to_numeric(df_raw["PM2.5_avg"].replace("-", np.nan), errors="coerce")
    df_raw["PM10"] = pd.to_numeric(df_raw["PM10_avg"].replace("-", np.nan), errors="coerce")
    df_raw["NO2"] = pd.to_numeric(df_raw["NO2_avg"].replace("-", np.nan), errors="coerce")
    df_raw["SO2"] = pd.to_numeric(df_raw["SO2_avg"].replace("-", np.nan), errors="coerce")
    df_raw["CO"] = pd.to_numeric(df_raw["CO_avg"].replace("-", np.nan), errors="coerce") / 1000.0
    df_raw["Ozone"] = pd.to_numeric(df_raw["OZONE_avg"].replace("-", np.nan), errors="coerce")

    # Aggregate to IST hourly means
    df_raw["Hour_TS"] = df_raw["Timestamp"].dt.floor("h")
    hourly = df_raw.groupby("Hour_TS")[["PM2.5", "PM10", "NO2", "SO2", "CO", "Ozone"]].mean().reset_index()

    # CPCB hourly AQI
    hourly["AQI"] = compute_hourly_aqi_from_pollutants(hourly)
    hourly["Date"] = pd.to_datetime(hourly["Hour_TS"].dt.date)

    # Daily mean requiring >= 18 valid hours
    grouped = hourly.groupby("Date")["AQI"]
    daily_count = grouped.count()
    daily_mean = grouped.mean()
    daily_valid = daily_mean.where(daily_count >= 18, np.nan).reset_index()
    daily_valid["valid_hours"] = daily_count.values
    daily_valid["Station"] = st_name

    station_daily_list.append(daily_valid)

df_all = pd.concat(station_daily_list, ignore_index=True)
print("=== 2026-10-25 Records (IST) ===")
print(df_all[df_all["Date"] == "2026-10-25"])

print("\n=== Valid 14-day Lookback Windows Preceding 2026-10-25 ===")
# Check lookback window [2026-10-12 to 2026-10-25]
lookback_dates = pd.date_range("2026-10-12", "2026-10-25", freq="D")
for st_name in list(DELHI_STATIONS) + ["City"]:
    if st_name == "City":
        df_target = df_all.groupby("Date")["AQI"].mean().reset_index()
    else:
        df_target = df_all[df_all["Station"] == st_name][["Date", "AQI"]].copy()
    
    sub = df_target[(df_target["Date"] >= "2026-10-12") & (df_target["Date"] <= "2026-10-25")].copy()
    sub_indexed = sub.set_index("Date").reindex(lookback_dates).reset_index()
    interp_aqi = sub_indexed["AQI"].interpolate(method="linear", limit=2)
    has_nans = interp_aqi.isna().any()
    valid_window = (len(sub_indexed) == 14) and (not has_nans)
    observed_days = sub_indexed["AQI"].notna().sum()
    print(f"Station {st_name:15s}: Observed days: {observed_days}/14 | Has valid window: {valid_window}")
