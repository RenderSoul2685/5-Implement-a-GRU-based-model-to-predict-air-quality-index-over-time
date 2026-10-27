import gzip
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, ".")
from src.data_loader import compute_hourly_aqi_from_pollutants

def analyze_overlap(st_name, live_file, oc_file):
    with gzip.open(f"data/raw/live/{live_file}", "rt") as f:
        df_v = pd.read_csv(f)
    df_v["ts_raw"] = pd.to_datetime(df_v["timestamp"])
    for col, vcol in [("PM2.5", "PM2.5_avg"), ("PM10", "PM10_avg"), ("NO2", "NO2_avg"),
                      ("SO2", "SO2_avg"), ("CO", "CO_avg"), ("Ozone", "OZONE_avg")]:
        if vcol == "CO_avg":
            df_v[col] = pd.to_numeric(df_v[vcol].replace("-", np.nan), errors="coerce") / 1000.0
        else:
            df_v[col] = pd.to_numeric(df_v[vcol].replace("-", np.nan), errors="coerce")

    df_oc = pd.read_csv(f"data/raw/{oc_file}")
    df_oc["ts_raw"] = pd.to_datetime(df_oc["Timestamp"])
    df_oc["PM2.5"] = pd.to_numeric(df_oc["PM2.5 (ug/m3)"], errors="coerce")
    df_oc["PM10"] = pd.to_numeric(df_oc["PM10 (ug/m3)"], errors="coerce")
    df_oc["NO2"] = pd.to_numeric(df_oc["NO2 (ug/m3)"], errors="coerce")
    df_oc["SO2"] = pd.to_numeric(df_oc["SO2 (ug/m3)"], errors="coerce")
    df_oc["CO"] = pd.to_numeric(df_oc["CO (mg/m3)"], errors="coerce")
    df_oc["Ozone"] = pd.to_numeric(df_oc["Ozone (ug/m3)"], errors="coerce")

    # Filter OpenCity to 2025-10-01 .. 2025-12-31
    oc_win = df_oc[(df_oc["ts_raw"] >= "2025-10-01") & (df_oc["ts_raw"] <= "2025-12-31 23:59:59")].copy()
    oc_win["Hour_TS"] = oc_win["ts_raw"].dt.floor("h")
    oc_hourly = oc_win.groupby("Hour_TS")[["PM2.5", "PM10", "NO2", "SO2", "CO", "Ozone"]].mean().reset_index()
    oc_hourly["AQI"] = compute_hourly_aqi_from_pollutants(oc_hourly)
    oc_hourly = oc_hourly.set_index("Hour_TS")

    # Case A: Shift raw timestamp before floor('h')
    shifts = [step * 0.5 for step in range(-24, 25)]
    results = []
    for s in shifts:
        v_shifted = df_v.copy()
        v_shifted["ts_shifted"] = v_shifted["ts_raw"] + pd.Timedelta(hours=s)
        v_sub = v_shifted[(v_shifted["ts_shifted"] >= "2025-10-01") & (v_shifted["ts_shifted"] <= "2025-12-31 23:59:59")].copy()
        v_sub["Hour_TS"] = v_sub["ts_shifted"].dt.floor("h")
        v_hourly = v_sub.groupby("Hour_TS")[["PM2.5", "PM10", "NO2", "SO2", "CO", "Ozone"]].mean().reset_index()
        v_hourly["AQI"] = compute_hourly_aqi_from_pollutants(v_hourly)
        v_hourly = v_hourly.set_index("Hour_TS")

        aligned = pd.concat([
            oc_hourly[["PM2.5", "AQI"]].rename(columns={"PM2.5": "oc_pm25", "AQI": "oc_aqi"}),
            v_hourly[["PM2.5", "AQI"]].rename(columns={"PM2.5": "v_pm25", "AQI": "v_aqi"})
        ], axis=1, join="inner").dropna(subset=["oc_pm25", "v_pm25"])

        corr_pm25 = aligned["oc_pm25"].corr(aligned["v_pm25"])
        aligned_aqi = aligned.dropna(subset=["oc_aqi", "v_aqi"])
        corr_aqi = aligned_aqi["oc_aqi"].corr(aligned_aqi["v_aqi"])
        mean_diff_aqi = (aligned_aqi["v_aqi"] - aligned_aqi["oc_aqi"]).mean()
        mean_diff_pm25 = (aligned["v_pm25"] - aligned["oc_pm25"]).mean()

        results.append({
            "shift": s,
            "corr_pm25": corr_pm25,
            "corr_aqi": corr_aqi,
            "mean_diff_aqi": mean_diff_aqi,
            "mean_diff_pm25": mean_diff_pm25,
            "n": len(aligned),
            "n_aqi": len(aligned_aqi)
        })

    df_res = pd.DataFrame(results)
    best_idx = df_res["corr_pm25"].idxmax()
    best = df_res.loc[best_idx]
    zero = df_res[df_res["shift"] == 0.0].iloc[0]
    ist = df_res[df_res["shift"] == 5.5].iloc[0]
    return df_res, best, zero, ist

for st, lfile, ocfile in [
    ("Anand Vihar", "anand_vihar_live_site_301.csv.gz", "anand_vihar_dpcc_2024-25.csv"),
    ("Punjabi Bagh", "punjabi_bagh_live_site_125.csv.gz", "punjabi_bagh_dpcc_2024-25.csv")
]:
    df_res, best, zero, ist = analyze_overlap(st, lfile, ocfile)
    print(f"\n=== {st} ===")
    print(f"Highest PM2.5 corr shift: {best['shift']:+.1f}h | Corr: {best['corr_pm25']:.4f} | N: {int(best['n'])}")
    print(f"At best shift ({best['shift']:+.1f}h): AQI Corr: {best['corr_aqi']:.4f} | Mean AQI Diff (Vayuayan - OC): {best['mean_diff_aqi']:+.2f}")
    print(f"At shift 0.0h: PM2.5 Corr: {zero['corr_pm25']:.4f} | AQI Corr: {zero['corr_aqi']:.4f} | Mean AQI Diff: {zero['mean_diff_aqi']:+.2f}")
    print(f"At shift +5.5h (UTC->IST): PM2.5 Corr: {ist['corr_pm25']:.4f} | AQI Corr: {ist['corr_aqi']:.4f} | Mean AQI Diff: {ist['mean_diff_aqi']:+.2f}")
