import gzip
import pandas as pd
import numpy as np

# Test Anand Vihar with 24-hour rolling average of OpenCity
with gzip.open("data/raw/live/anand_vihar_live_site_301.csv.gz", "rt") as f:
    df_v = pd.read_csv(f)
df_v["ts_raw"] = pd.to_datetime(df_v["timestamp"])
df_v["PM2.5"] = pd.to_numeric(df_v["PM2.5_avg"].replace("-", np.nan), errors="coerce")

df_oc = pd.read_csv("data/raw/anand_vihar_dpcc_2024-25.csv")
df_oc["ts_raw"] = pd.to_datetime(df_oc["Timestamp"])
df_oc["PM2.5"] = pd.to_numeric(df_oc["PM2.5 (ug/m3)"], errors="coerce")

# OpenCity hourly PM2.5 across entire 2025 so rolling window is full
df_oc["Hour_TS"] = df_oc["ts_raw"].dt.floor("h")
oc_hourly = df_oc.groupby("Hour_TS")[["PM2.5"]].mean()
oc_rolling24 = oc_hourly["PM2.5"].rolling(24, min_periods=18).mean()

print("--- Aligning Vayuayan PM2.5_avg with OpenCity 24h Rolling PM2.5 ---")
shifts = [step * 0.5 for step in range(-24, 25)]
results_rolling = []
for s in shifts:
    v_shifted = df_v.copy()
    v_shifted["ts_shifted"] = v_shifted["ts_raw"] + pd.Timedelta(hours=s)
    v_sub = v_shifted[(v_shifted["ts_shifted"] >= "2025-10-01") & (v_shifted["ts_shifted"] <= "2025-12-31 23:59:59")].copy()
    v_sub["Hour_TS"] = v_sub["ts_shifted"].dt.floor("h")
    v_hourly = v_sub.groupby("Hour_TS")[["PM2.5"]].mean()
    
    aligned = pd.concat([oc_rolling24.rename("oc_roll24"), v_hourly["PM2.5"].rename("v_avg")], axis=1, join="inner").dropna()
    if len(aligned) > 10:
        corr = aligned["oc_roll24"].corr(aligned["v_avg"])
        results_rolling.append((s, corr, len(aligned)))

results_rolling.sort(key=lambda x: x[1], reverse=True)
print("Top 10 shifts aligning Vayuayan with OpenCity 24h Rolling:")
for r in results_rolling[:10]:
    print(f"Shift: {r[0]:+5.1f}h | Corr: {r[1]:.4f} | N: {r[2]}")

print("\nSpecific shifts with OpenCity 24h rolling:")
for s_target in [0.0, 5.5, -5.5]:
    match = [r for r in results_rolling if r[0] == s_target]
    if match:
        print(f"Shift {match[0][0]:+5.1f}h | Corr: {match[0][1]:.4f} | N: {match[0][2]}")
