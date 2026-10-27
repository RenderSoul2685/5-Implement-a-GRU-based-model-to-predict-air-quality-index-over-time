import gzip
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, ".")
from src.data_loader import compute_hourly_aqi_from_pollutants

for st_name, lfile in [
    ("Anand_Vihar", "anand_vihar_live_site_301.csv.gz"),
    ("Bawana", "bawana_live_site_1560.csv.gz"),
    ("ITO", "ito_live_site_117.csv.gz"),
    ("Mandir_Marg", "mandir_marg_live_site_122.csv.gz"),
    ("Punjabi_Bagh", "punjabi_bagh_live_site_125.csv.gz"),
    ("RK_Puram", "rk_puram_live_site_124.csv.gz"),
]:
    print(f"\n==================== {st_name} ====================")
    with gzip.open(f"data/raw/live/{lfile}", "rt") as f:
        df = pd.read_csv(f)
    df["ts_utc"] = pd.to_datetime(df["timestamp"])
    df["ts_ist"] = df["ts_utc"] + pd.Timedelta(hours=5, minutes=30)
    
    # Pollutants
    for col, vcol in [("PM2.5", "PM2.5_avg"), ("PM10", "PM10_avg"), ("NO2", "NO2_avg"),
                      ("SO2", "SO2_avg"), ("CO", "CO_avg"), ("Ozone", "OZONE_avg")]:
        if vcol == "CO_avg":
            df[col] = pd.to_numeric(df[vcol].replace("-", np.nan), errors="coerce") / 1000.0
        else:
            df[col] = pd.to_numeric(df[vcol].replace("-", np.nan), errors="coerce")

    # Filter to last 14 days in IST: 2026-10-12 00:00:00 to 2026-10-25 23:59:59
    sub_14 = df[(df["ts_ist"] >= "2026-10-12 00:00:00") & (df["ts_ist"] <= "2026-10-25 23:59:59")].copy()
    print(f"Total raw records in last 14 days (IST): {len(sub_14)}")
    
    # 1. Negative readings
    pollutant_cols = ["PM2.5", "PM10", "NO2", "SO2", "CO", "Ozone"]
    neg_counts = {col: (sub_14[col] < 0).sum() for col in pollutant_cols}
    total_negs = sum(neg_counts.values())
    print(f"Negative readings: {total_negs} (by col: {neg_counts})")

    # 2. Runs of 6 or more identical consecutive readings
    # Check on raw readings per pollutant and on hourly AQI
    runs_6_raw = {}
    for col in pollutant_cols:
        s = sub_14[col].dropna()
        if len(s) > 0:
            # consecutive equal groups
            groups = (s != s.shift()).cumsum()
            counts = s.groupby(groups).transform("count")
            runs = (counts >= 6) & (counts != counts.shift()) # count runs
            # Number of runs of length >= 6
            run_lengths = s.groupby(groups).size()
            long_runs = run_lengths[run_lengths >= 6]
            runs_6_raw[col] = len(long_runs)
        else:
            runs_6_raw[col] = 0
    print(f"Runs of >= 6 identical consecutive raw readings: {runs_6_raw}")

    # Compute hourly in IST
    sub_14["Hour_TS"] = sub_14["ts_ist"].dt.floor("h")
    hourly = sub_14.groupby("Hour_TS")[pollutant_cols].mean().reset_index()
    hourly["AQI"] = compute_hourly_aqi_from_pollutants(hourly)
    hourly = hourly.sort_values("Hour_TS").reset_index(drop=True)
    
    # Consecutive identical in hourly AQI
    aqi_s = hourly["AQI"].dropna()
    if len(aqi_s) > 0:
        groups_aqi = (aqi_s != aqi_s.shift()).cumsum()
        run_lengths_aqi = aqi_s.groupby(groups_aqi).size()
        long_runs_aqi = run_lengths_aqi[run_lengths_aqi >= 6]
        print(f"Runs of >= 6 identical consecutive hourly AQI: {len(long_runs_aqi)}")
        if len(long_runs_aqi) > 0:
            print("   Details:", long_runs_aqi.to_dict())
    else:
        print("Runs of >= 6 identical consecutive hourly AQI: 0")

    # 3. Hourly changes above 300 AQI
    hourly["aqi_diff"] = hourly["AQI"].diff().abs()
    large_changes = hourly[hourly["aqi_diff"] > 300]
    print(f"Hourly changes > 300 AQI: {len(large_changes)}")
    if len(large_changes) > 0:
        for _, r in large_changes.iterrows():
            print(f"   At {r['Hour_TS']}: AQI diff = {r['aqi_diff']:.1f} (AQI: {r['AQI']})")
