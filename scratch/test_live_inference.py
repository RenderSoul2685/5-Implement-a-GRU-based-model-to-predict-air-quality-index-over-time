import sys
sys.path.insert(0, ".")
import os
import urllib.request
import gzip
import io
import json
import torch
import numpy as np
import pandas as pd

from src.config import Config, DELHI_STATIONS
from src.data_loader import compute_hourly_aqi_from_pollutants
from src.preprocessing import prepare_daily_dataframe
from src.model import ResidualCityGRU
import joblib

# Load model bundle
model_dir = "models"
checkpoint = torch.load(os.path.join(model_dir, "best_model.pt"), map_location="cpu")
with open(os.path.join(model_dir, "model_config.json"), "r") as f:
    cfg = json.load(f)

model = ResidualCityGRU(
    input_dim=cfg["input_dim"],
    hidden_dim=cfg["hidden_dim"],
    num_layers=cfg["num_layers"],
    dropout=0.0,
    horizon=1,
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

feature_scaler = joblib.load(os.path.join(model_dir, "feature_scaler.joblib"))
target_scaler = joblib.load(os.path.join(model_dir, "target_scaler.joblib"))

# Fetch weather
weather_url = (
    "https://archive-api.open-meteo.com/v1/archive?"
    "latitude=28.6139&longitude=77.2090"
    "&start_date=2026-10-01&end_date=2026-10-25"
    "&daily=temperature_2m_mean,relative_humidity_2m_mean,wind_speed_10m_max,surface_pressure_mean"
    "&timezone=Asia%2FKolkata"
)
req = urllib.request.Request(weather_url, headers={"User-Agent": "AQI-GRU-Research/2.0"})
with urllib.request.urlopen(req) as resp:
    wdata = json.loads(resp.read().decode('utf-8'))

weather_df = pd.DataFrame({
    "Date": pd.to_datetime(wdata["daily"]["time"]),
    "temperature": wdata["daily"]["temperature_2m_mean"],
    "humidity": wdata["daily"]["relative_humidity_2m_mean"],
    "wind_speed": wdata["daily"]["wind_speed_10m_max"],
    "pressure": wdata["daily"]["surface_pressure_mean"],
})

stations = {
    'Anand_Vihar': 'delhi__delhi__anand_vihar_delhi_dpcc__site_301.csv.gz',
    'Bawana': 'delhi__delhi__bawana_delhi_dpcc__site_1560.csv.gz',
    'ITO': 'delhi__delhi__ito_delhi_cpcb__site_117.csv.gz',
    'Mandir_Marg': 'delhi__delhi__mandir_marg_delhi_dpcc__site_122.csv.gz',
    'Punjabi_Bagh': 'delhi__delhi__punjabi_bagh_delhi_dpcc__site_125.csv.gz',
    'RK_Puram': 'delhi__delhi__r_k_puram_delhi_dpcc__site_124.csv.gz',
}

base_url = "https://data.vayuayan.saketlab.org/data/"

for st_name, fname in stations.items():
    url = base_url + fname
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as res:
        data = res.read()
    with gzip.GzipFile(fileobj=io.BytesIO(data)) as gz:
        df = pd.read_csv(gz)
    
    df["Timestamp"] = pd.to_datetime(df["timestamp"])
    df["PM2.5"] = pd.to_numeric(df["PM2.5_avg"].replace("-", np.nan), errors="coerce")
    df["PM10"] = pd.to_numeric(df["PM10_avg"].replace("-", np.nan), errors="coerce")
    df["NO2"] = pd.to_numeric(df["NO2_avg"].replace("-", np.nan), errors="coerce")
    df["SO2"] = pd.to_numeric(df["SO2_avg"].replace("-", np.nan), errors="coerce")
    df["CO"] = pd.to_numeric(df["CO_avg"].replace("-", np.nan), errors="coerce") / 1000.0
    df["Ozone"] = pd.to_numeric(df["OZONE_avg"].replace("-", np.nan), errors="coerce")
    
    df["Hour_TS"] = df["Timestamp"].dt.floor("h")
    hourly = df.groupby("Hour_TS")[["PM2.5", "PM10", "NO2", "SO2", "CO", "Ozone"]].mean().reset_index()
    hourly["AQI"] = compute_hourly_aqi_from_pollutants(hourly)
    hourly["Date"] = pd.to_datetime(hourly["Hour_TS"].dt.date)
    
    grouped = hourly.groupby("Date")["AQI"]
    daily_count = grouped.count()
    daily_mean = grouped.mean()
    daily_valid = daily_mean.where(daily_count >= 18, np.nan).reset_index()
    
    # Merge with weather
    merged = pd.merge(daily_valid, weather_df, on="Date", how="right").sort_values("Date").reset_index(drop=True)
    merged["Station"] = st_name
    
    # Check 14 days ending 2026-10-25
    anchor_idx = merged[merged["Date"] == pd.to_datetime("2026-10-25")].index
    if len(anchor_idx) == 0:
        print(f"{st_name}: 2026-10-25 not found")
        continue
    idx = anchor_idx[0]
    window = merged.iloc[idx-13 : idx+1].copy()
    
    # Check max consecutive NaNs in AQI
    is_na = window["AQI"].isna()
    max_consec_na = is_na.astype(int).groupby((~is_na).cumsum()).sum().max()
    print(f"\n--- Station: {st_name} ---")
    print(f"Window length: {len(window)}, NaNs in AQI: {window['AQI'].isna().sum()}, max consecutive NaNs: {max_consec_na}")
    if max_consec_na > 2:
        print(f"Station {st_name} has > 2 consecutive NaNs in lookback window! Forecast cannot be computed.")
        continue
        
    # Interpolate up to 2 NaNs
    window["AQI_interp"] = window["AQI"].interpolate(method="linear", limit=2)
    # Add cyclical features
    dayofweek = window["Date"].dt.dayofweek
    window["dayofweek_sin"] = np.sin(2 * np.pi * dayofweek / 7)
    window["dayofweek_cos"] = np.cos(2 * np.pi * dayofweek / 7)
    month = window["Date"].dt.month
    window["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    window["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    
    feat_cols = ['AQI_interp', 'temperature', 'humidity', 'wind_speed', 'pressure', 'dayofweek_sin', 'dayofweek_cos', 'month_sin', 'month_cos']
    X_raw = window[feat_cols].values
    today_anchor_aqi = window["AQI_interp"].iloc[-1]
    
    X_scaled = feature_scaler.transform(X_raw)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        pred_scaled = model(X_tensor).numpy()
    pred_residual = target_scaler.inverse_transform(pred_scaled).item()
    pred_tomorrow_aqi = today_anchor_aqi + pred_residual
    print(f"Today's Anchor AQI ({window['Date'].iloc[-1].strftime('%Y-%m-%d')}): {today_anchor_aqi:.1f}")
    print(f"Tomorrow Predicted AQI: {pred_tomorrow_aqi:.1f} (delta: {pred_residual:+.1f})")
