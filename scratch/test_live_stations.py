import sys
sys.path.insert(0, ".")
from pathlib import Path
import pandas as pd
import numpy as np
from src.demo import get_model_bundle, get_cpcb_category
from src.preprocessing import prepare_daily_dataframe
import torch

bundle = get_model_bundle()
processed_dir = Path("data/processed")
df_st_live = pd.read_csv(processed_dir / "delhi_stations_live.csv")
df_st_live["Date"] = pd.to_datetime(df_st_live["Date"])
df_w_live = pd.read_csv(processed_dir / "delhi_weather_live.csv")
df_w_live["Date"] = pd.to_datetime(df_w_live["Date"])

model = bundle["model"]
feat_scaler = bundle["feat_scaler"]
target_scaler = bundle["target_scaler"]
config_info = bundle["config_info"]
feature_cols = config_info["feature_cols"]

today_dt = pd.to_datetime("2026-10-25")
lookback_start = today_dt - pd.Timedelta(days=13)
lookback_end = today_dt

print(f"Evaluating Lookback Window: {lookback_start.strftime('%Y-%m-%d')} to {lookback_end.strftime('%Y-%m-%d')}")

for station in ["Anand_Vihar", "Bawana", "ITO", "Mandir_Marg", "Punjabi_Bagh", "RK_Puram"]:
    sub = df_st_live[df_st_live["Station"] == station][["Date", "AQI"]].copy()
    df_prep = prepare_daily_dataframe(sub, df_w_live, interp_limit_days=2)
    window = df_prep[(df_prep["Date"] >= lookback_start) & (df_prep["Date"] <= lookback_end)]
    
    if len(window) != 14 or window["AQI_interp"].isna().any():
        print(f"Station {station}: INCOMPLETE (rows={len(window)}, NaNs={window['AQI_interp'].isna().sum() if len(window)==14 else 'N/A'})")
        continue
        
    X_input = window[feature_cols].values.astype(np.float32)
    today_aqi = float(window["AQI_interp"].iloc[-1])
    X_scaled = feat_scaler.transform(X_input).reshape(1, 14, -1).astype(np.float32)
    
    with torch.no_grad():
        pred_sc = model(torch.as_tensor(X_scaled)).cpu().numpy().ravel()[0]
    pred_delta = float(target_scaler.inverse_transform(np.array([[pred_sc]]))[0, 0])
    tomorrow_forecast = round(float(np.clip(today_aqi + pred_delta, 0.0, 500.0)), 1)
    
    print(f"Station {station}: Today AQI = {today_aqi:.1f} ({get_cpcb_category(today_aqi)}), Tomorrow Forecast = {tomorrow_forecast:.1f} ({get_cpcb_category(tomorrow_forecast)}) [delta: {pred_delta:+.1f}]")
