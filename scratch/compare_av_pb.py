import gzip
import pandas as pd
import numpy as np
from pathlib import Path
import sys

ROOT_DIR = Path('d:/DL_project/aqi-gru')
sys.path.insert(0, str(ROOT_DIR))
from src.data_loader import compute_hourly_aqi_from_pollutants, compute_cpcb_subindex_vectorized
from src.config import CPCB_BREAKPOINTS
from src.refresh_live import convert_vayuayan_timestamps_to_ist, VAYUAYAN_STATIONS

def get_station_hourly(station_name):
    info = VAYUAYAN_STATIONS[station_name]
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
    
    # Sub-indices
    sub_pm25 = compute_cpcb_subindex_vectorized(hourly["PM2.5"], CPCB_BREAKPOINTS["PM2.5"])
    sub_pm10 = compute_cpcb_subindex_vectorized(hourly["PM10"], CPCB_BREAKPOINTS["PM10"])
    sub_no2 = compute_cpcb_subindex_vectorized(hourly["NO2"], CPCB_BREAKPOINTS["NO2"])
    sub_so2 = compute_cpcb_subindex_vectorized(hourly["SO2"], CPCB_BREAKPOINTS["SO2"])
    sub_co = compute_cpcb_subindex_vectorized(hourly["CO"], CPCB_BREAKPOINTS["CO"])
    sub_o3 = compute_cpcb_subindex_vectorized(hourly["Ozone"], CPCB_BREAKPOINTS["Ozone"])
    
    sub_df = pd.DataFrame({
        "PM2.5": sub_pm25, "PM10": sub_pm10, "NO2": sub_no2,
        "SO2": sub_so2, "CO": sub_co, "Ozone": sub_o3
    })
    
    hourly["AQI"] = compute_hourly_aqi_from_pollutants(hourly)
    all_na = sub_df.isna().all(axis=1)
    dominant = sub_df.fillna(-1).idxmax(axis=1)
    dominant[all_na] = "None"
    hourly["Dominant"] = dominant
    hourly["Date"] = hourly["Hour_TS"].dt.strftime("%Y-%m-%d")
    hourly["Hour"] = hourly["Hour_TS"].dt.hour
    hourly["Station"] = station_name
    return hourly

av = get_station_hourly("Anand_Vihar")
pb = get_station_hourly("Punjabi_Bagh")

for dt in ["2026-10-18", "2026-10-25"]:
    print(f"\n=======================================================")
    print(f"=== COMPARISON FOR DATE: {dt} (IST) ===")
    print(f"=======================================================")
    av_dt = av[av["Date"] == dt].copy().set_index("Hour")
    pb_dt = pb[pb["Date"] == dt].copy().set_index("Hour")
    
    hours = sorted(set(av_dt.index).union(set(pb_dt.index)))
    
    print(f"{'Hour':4s} | {'AV AQI':6s} {'AV Dom':6s} {'PM2.5':5s} {'PM10':5s} {'NO2':4s} {'SO2':4s} {'CO':4s} {'O3':4s} | {'PB AQI':6s} {'PB Dom':6s} {'PM2.5':5s} {'PM10':5s} {'NO2':4s} {'SO2':4s} {'CO':4s} {'O3':4s}")
    print("-" * 105)
    for h in hours:
        av_row = av_dt.loc[h] if h in av_dt.index else None
        pb_row = pb_dt.loc[h] if h in pb_dt.index else None
        
        av_str = f"{av_row['AQI']:6.1f} {av_row['Dominant']:6s} {av_row['PM2.5']:5.1f} {av_row['PM10']:5.1f} {av_row['NO2']:4.1f} {av_row['SO2']:4.1f} {av_row['CO']:4.1f} {av_row['Ozone']:4.1f}" if av_row is not None else " " * 44
        pb_str = f"{pb_row['AQI']:6.1f} {pb_row['Dominant']:6s} {pb_row['PM2.5']:5.1f} {pb_row['PM10']:5.1f} {pb_row['NO2']:4.1f} {pb_row['SO2']:4.1f} {pb_row['CO']:4.1f} {pb_row['Ozone']:4.1f}" if pb_row is not None else " " * 44
        print(f"{h:02d}:00 | {av_str} | {pb_str}")
        
    print("-" * 105)
    print(f"Daily Mean AQI -> Anand Vihar: {av_dt['AQI'].mean():.2f} (valid hours: {av_dt['AQI'].count()}) | Punjabi Bagh: {pb_dt['AQI'].mean():.2f} (valid hours: {pb_dt['AQI'].count()})")
