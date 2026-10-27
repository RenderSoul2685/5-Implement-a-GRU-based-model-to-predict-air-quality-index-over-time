import gzip
import io
import pandas as pd
import numpy as np
from pathlib import Path
import sys

ROOT_DIR = Path('d:/DL_project/aqi-gru')
sys.path.insert(0, str(ROOT_DIR))
from src.data_loader import compute_hourly_aqi_from_pollutants
from src.refresh_live import convert_vayuayan_timestamps_to_ist, VAYUAYAN_STATIONS

# Load OpenCity daily data
df_opencity = pd.read_csv(ROOT_DIR / 'data/processed/delhi_stations_2023_2025.csv')
df_opencity['Date'] = pd.to_datetime(df_opencity['Date'])

stations = ['Anand_Vihar', 'Bawana', 'Mandir_Marg', 'Punjabi_Bagh', 'RK_Puram', 'ITO']

print("=== Vayuayan Dec 2025 raw hourly data extraction ===")
results = []
for st in stations:
    info = VAYUAYAN_STATIONS[st]
    raw_path = ROOT_DIR / "data/raw/live" / info["local_raw"]
    with gzip.open(raw_path, 'rb') as f:
        df_raw = pd.read_csv(f)
    
    df_raw = convert_vayuayan_timestamps_to_ist(df_raw)
    
    # Clean pollutants
    df_raw["PM2.5"] = pd.to_numeric(df_raw["PM2.5_avg"].replace("-", np.nan), errors="coerce")
    df_raw["PM10"] = pd.to_numeric(df_raw["PM10_avg"].replace("-", np.nan), errors="coerce")
    df_raw["NO2"] = pd.to_numeric(df_raw["NO2_avg"].replace("-", np.nan), errors="coerce")
    df_raw["SO2"] = pd.to_numeric(df_raw["SO2_avg"].replace("-", np.nan), errors="coerce")
    df_raw["CO"] = pd.to_numeric(df_raw["CO_avg"].replace("-", np.nan), errors="coerce") / 1000.0
    df_raw["Ozone"] = pd.to_numeric(df_raw["OZONE_avg"].replace("-", np.nan), errors="coerce")
    
    df_raw["Hour_TS"] = df_raw["Timestamp"].dt.floor("h")
    hourly = df_raw.groupby("Hour_TS")[["PM2.5", "PM10", "NO2", "SO2", "CO", "Ozone"]].mean().reset_index()
    hourly["AQI"] = compute_hourly_aqi_from_pollutants(hourly)
    hourly["Date"] = pd.to_datetime(hourly["Hour_TS"].dt.date)
    
    # Check overlap 2025-11-30 to 2025-12-31
    overlap_hourly = hourly[(hourly["Date"] >= "2025-11-30") & (hourly["Date"] <= "2025-12-31")]
    
    # 1. With >= 18 valid hours rule (pipeline standard)
    daily_18h = overlap_hourly.groupby("Date")["AQI"].agg(mean_aqi="mean", count="count").reset_index()
    daily_18h["valid_aqi"] = daily_18h["mean_aqi"].where(daily_18h["count"] >= 18, np.nan)
    
    # 2. Relaxed: mean of all available hours in Vayuayan
    daily_all = overlap_hourly.groupby("Date")["AQI"].agg(mean_aqi="mean", count="count").reset_index()
    
    # OpenCity daily
    oc_sub = df_opencity[(df_opencity["Station"] == st) & (df_opencity["Date"] >= "2025-11-30") & (df_opencity["Date"] <= "2025-12-31")][["Date", "AQI"]].rename(columns={"AQI": "AQI_OC"})
    
    # Merge with >= 18h
    m_18 = pd.merge(oc_sub, daily_18h.dropna(subset=["valid_aqi"]), on="Date")
    
    # Merge with all available hours
    m_all = pd.merge(oc_sub, daily_all, on="Date").dropna(subset=["AQI_OC", "mean_aqi"])
    
    bias_all = (m_all["mean_aqi"] - m_all["AQI_OC"]).mean()
    corr_all = m_all["mean_aqi"].corr(m_all["AQI_OC"])
    
    print(f"\nStation: {st}")
    print(f"  Strict >= 18 hours/day rule: N = {len(m_18)} valid days (max hours available in any Dec 2025 day was {daily_18h['count'].max()})")
    print(f"  Relaxed (mean of available {daily_all['count'].min()}-{daily_all['count'].max()} hours/day): N = {len(m_all)} days")
    print(f"    Bias (Vayuayan - OpenCity): {bias_all:+.2f}")
    print(f"    Pearson Correlation: {corr_all:.4f}")
    print(f"    OpenCity Mean AQI: {m_all['AQI_OC'].mean():.2f}")
    print(f"    Vayuayan Mean AQI: {m_all['mean_aqi'].mean():.2f}")
