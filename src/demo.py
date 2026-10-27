"""Interactive Demo and Inference Engine for Next-Day AQI-GRU Forecasting.

Usage:
    python src/demo.py --station Anand_Vihar --date 2025-11-15

Task:
    From the previous 14 days of daily mean AQI (plus daily weather and calendar features),
    predicts tomorrow's daily mean AQI. Single output, horizon = 1 day.

Prints:
    - The last 14 days of historical daily observations with official CPCB categories.
    - Tomorrow's predicted daily mean AQI with its official CPCB health category.
    - The ground-truth actual observed AQI for tomorrow if available, and the prediction error.

Exposes `get_model_bundle()` and `compute_forecast()` for external and test reuse.
"""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np
import pandas as pd
import torch

from src.config import Config, DELHI_STATIONS
from src.model import AQIGRU
from src.preprocessing import prepare_daily_dataframe

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CPCB_CATEGORIES = [
    (0, 50, "Good", "#22c55e"),
    (50, 100, "Satisfactory", "#84cc16"),
    (100, 200, "Moderate", "#eab308"),
    (200, 300, "Poor", "#f97316"),
    (300, 400, "Very Poor", "#ef4444"),
    (400, 500, "Severe", "#881337"),
]

# Global singleton cache for CPU inference
_CACHED_BUNDLE: Optional[Dict] = None


def get_cpcb_category(aqi: Optional[float]) -> str:
    """Return the official CPCB health category for a given AQI score."""
    if aqi is None or (isinstance(aqi, (float, int)) and np.isnan(aqi)):
        return "N/A"
    val = float(aqi)
    if val <= 50.0:
        return "Good"
    elif val <= 100.0:
        return "Satisfactory"
    elif val <= 200.0:
        return "Moderate"
    elif val <= 300.0:
        return "Poor"
    elif val <= 400.0:
        return "Very Poor"
    else:
        return "Severe"


def get_aqi_reference_rows() -> List[Dict[str, str]]:
    """Return official CPCB AQI reference table rows dynamically mapped via get_cpcb_category.

    Guarantees that the reference table and badges can never disagree, without
    duplicating category thresholds.
    """
    ranges_and_meanings = [
        ("0-50", 25.0, "Little or no risk"),
        ("51-100", 75.0, "Minor discomfort for sensitive people"),
        ("101-200", 150.0, "Breathing discomfort for people with lung or heart conditions"),
        ("201-300", 250.0, "Discomfort on prolonged exposure"),
        ("301-400", 350.0, "Respiratory illness on prolonged exposure"),
        ("401-500", 450.0, "Affects healthy people and seriously affects those with existing illness"),
    ]
    return [
        {
            "range": r_str,
            "category": get_cpcb_category(test_val),
            "meaning": meaning,
        }
        for r_str, test_val, meaning in ranges_and_meanings
    ]


def get_model_bundle(repo_root: Optional[Path] = None) -> Dict:
    """Load and cache model weights, scalers, and daily datasets once on CPU."""
    global _CACHED_BUNDLE
    if _CACHED_BUNDLE is not None:
        return _CACHED_BUNDLE

    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent

    models_dir = repo_root / "models"
    processed_dir = repo_root / "data" / "processed"
    raw_dir = repo_root / "data" / "raw"

    config_path = models_dir / "model_config.json"
    weights_path = models_dir / "best_model.pt"
    feat_scaler_path = models_dir / "feature_scaler.joblib"
    target_scaler_path = models_dir / "target_scaler.joblib"

    if not (config_path.exists() and weights_path.exists()):
        raise FileNotFoundError(
            f"Trained model artifacts not found in {models_dir}. Please run `python src/train.py` first."
        )

    with open(config_path, "r") as f:
        config_info = json.load(f)

    feat_scaler = joblib.load(feat_scaler_path)
    target_scaler = joblib.load(target_scaler_path)

    arch = config_info["architecture"]
    model = AQIGRU(
        input_size=arch["input_size"],
        hidden_size=arch["hidden_size"],
        num_layers=arch["num_layers"],
        output_size=arch["output_size"],
        dropout=arch["dropout"],
    )
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    # Load authentic daily datasets (historical test split)
    daily_stations_path = processed_dir / "delhi_stations_2023_2025.csv"
    daily_city_path = processed_dir / "delhi_city_2023_2025.csv"
    daily_weather_path = raw_dir / "delhi_weather_2023_2025.csv"

    if not daily_stations_path.exists() or not daily_city_path.exists() or not daily_weather_path.exists():
        raise FileNotFoundError("Missing daily data files in data/processed or data/raw.")

    df_st = pd.read_csv(daily_stations_path)
    df_st["Date"] = pd.to_datetime(df_st["Date"])

    df_city = pd.read_csv(daily_city_path)
    df_city["Date"] = pd.to_datetime(df_city["Date"])

    df_w = pd.read_csv(daily_weather_path)
    df_w["Date"] = pd.to_datetime(df_w["Date"])

    # Load authentic live feeds if available
    live_stations_path = processed_dir / "delhi_stations_live.csv"
    live_city_path = processed_dir / "delhi_city_live.csv"
    live_weather_path = processed_dir / "delhi_weather_live.csv"
    live_meta_path = processed_dir / "live_meta.json"

    df_st_live = None
    df_city_live = None
    df_w_live = None
    live_meta = None

    if live_stations_path.exists() and live_weather_path.exists():
        try:
            df_st_live = pd.read_csv(live_stations_path)
            df_st_live["Date"] = pd.to_datetime(df_st_live["Date"])
            if live_city_path.exists():
                df_city_live = pd.read_csv(live_city_path)
                df_city_live["Date"] = pd.to_datetime(df_city_live["Date"])
            df_w_live = pd.read_csv(live_weather_path)
            df_w_live["Date"] = pd.to_datetime(df_w_live["Date"])
            if live_meta_path.exists():
                with open(live_meta_path, "r", encoding="utf-8") as f:
                    live_meta = json.load(f)
        except Exception as e:
            logger.warning("Could not load live cache: %s", e)

    _CACHED_BUNDLE = {
        "config_info": config_info,
        "feat_scaler": feat_scaler,
        "target_scaler": target_scaler,
        "model": model,
        "df_st": df_st,
        "df_city": df_city,
        "df_w": df_w,
        "df_st_live": df_st_live,
        "df_city_live": df_city_live,
        "df_w_live": df_w_live,
        "live_meta": live_meta,
        "repo_root": repo_root,
    }
    logger.info("Next-day AQI model bundle loaded and cached successfully on CPU.")
    return _CACHED_BUNDLE


def compute_forecast(station: str, date_str: str, bundle: Optional[Dict] = None) -> Dict:
    """Compute next-day AQI forecast from the preceding 14 days of observations.

    Args:
        station: Station name (e.g. "Anand_Vihar", "Bawana", or "City").
        date_str: Date string for today (t) in YYYY-MM-DD format (within 2025 test period).
        bundle: Optional preloaded model bundle.

    Returns:
        dict containing historical 14 days, tomorrow's forecast, actual (if available), and status.
    """
    if bundle is None:
        bundle = get_model_bundle()

    # 1. Normalize station name
    norm_station = None
    for s in DELHI_STATIONS:
        if s.lower() == station.lower().replace(" ", "_"):
            norm_station = s
            break
    if norm_station is None and station.lower() == "city":
        norm_station = "City"
    if norm_station is None:
        valid_options = list(DELHI_STATIONS) + ["City"]
        return {
            "success": False,
            "error": f"Invalid station '{station}'. Must be one of: {', '.join(valid_options)}.",
        }

    # 2. Validate date
    try:
        today_dt = pd.to_datetime(date_str)
    except Exception:
        return {"success": False, "error": f"Invalid date format '{date_str}'. Please use YYYY-MM-DD."}

    test_start = pd.to_datetime("2025-10-01")
    test_end = pd.to_datetime("2025-12-31")
    if not (test_start <= today_dt <= test_end):
        return {
            "success": False,
            "error": f"Date {date_str} is outside the 2025 test period (2025-10-01 to 2025-12-31).",
        }

    df_st = bundle["df_st"]
    df_city = bundle["df_city"]
    df_w = bundle["df_w"]
    config_info = bundle["config_info"]
    model = bundle["model"]
    feat_scaler = bundle["feat_scaler"]
    target_scaler = bundle["target_scaler"]

    # 3. Extract series for chosen station
    if norm_station == "City":
        df_target = df_city[["Date", "AQI"]].copy()
    else:
        df_target = df_st[df_st["Station"] == norm_station][["Date", "AQI"]].copy()

    # Prepare daily dataframe with <= 2 days interpolation and cyclical features
    df_prep = prepare_daily_dataframe(df_target, df_w, interp_limit_days=2)

    # 4. Extract 14-day lookback window: [today - 13 days, today]
    lookback_start = today_dt - pd.Timedelta(days=13)
    lookback_end = today_dt
    tomorrow_dt = today_dt + pd.Timedelta(days=1)

    window_df = df_prep[(df_prep["Date"] >= lookback_start) & (df_prep["Date"] <= lookback_end)]

    if len(window_df) != 14:
        return {
            "success": False,
            "error": f"Incomplete lookback: expected 14 consecutive daily records between {lookback_start.strftime('%Y-%m-%d')} and {lookback_end.strftime('%Y-%m-%d')}, found {len(window_df)}.",
        }

    if window_df["AQI_interp"].isna().any():
        return {
            "success": False,
            "error": f"Station {norm_station} has missing AQI data in the 14-day lookback preceding {date_str} that exceeds the 2-day interpolation threshold. Cannot generate forecast without complete inputs.",
        }

    feature_cols = config_info["feature_cols"]
    X_input = window_df[feature_cols].values.astype(np.float32)  # shape (14, D)
    today_aqi = float(window_df["AQI_interp"].iloc[-1])

    # Scale features and run inference
    X_scaled = feat_scaler.transform(X_input).reshape(1, 14, -1).astype(np.float32)

    with torch.no_grad():
        pred_sc = model(torch.as_tensor(X_scaled)).cpu().numpy().ravel()[0]

    # Unscale delta and add today's AQI
    pred_delta = float(target_scaler.inverse_transform(np.array([[pred_sc]]))[0, 0])
    tomorrow_forecast = round(float(np.clip(today_aqi + pred_delta, 0.0, 500.0)), 1)
    forecast_category = get_cpcb_category(tomorrow_forecast)

    # Ground-truth actual for tomorrow
    tomorrow_row = df_prep[df_prep["Date"] == tomorrow_dt]
    if len(tomorrow_row) > 0 and not np.isnan(tomorrow_row["AQI_observed"].values[0]):
        actual_val = round(float(tomorrow_row["AQI_observed"].values[0]), 1)
        actual_category = get_cpcb_category(actual_val)
        abs_error = round(abs(tomorrow_forecast - actual_val), 1)
    else:
        actual_val = None
        actual_category = "N/A"
        abs_error = None

    # Format the last 14 days table
    history_table = []
    for idx, row in window_df.iterrows():
        raw_aqi = row["AQI_observed"]
        interp_aqi = row["AQI_interp"]
        disp_aqi = round(float(interp_aqi), 1) if not np.isnan(interp_aqi) else None
        history_table.append({
            "date": row["Date"].strftime("%Y-%m-%d"),
            "aqi": disp_aqi,
            "observed_raw": not np.isnan(raw_aqi),
            "category": get_cpcb_category(disp_aqi),
        })

    return {
        "success": True,
        "error": None,
        "station": norm_station,
        "today_date": today_dt.strftime("%Y-%m-%d"),
        "tomorrow_date": tomorrow_dt.strftime("%Y-%m-%d"),
        "today_aqi": round(today_aqi, 1),
        "today_category": get_cpcb_category(today_aqi),
        "tomorrow_forecast": tomorrow_forecast,
        "forecast_category": forecast_category,
        "actual_aqi": actual_val,
        "actual_category": actual_category,
        "abs_error": abs_error,
        "history_table": history_table,
        "lookback_start": lookback_start.strftime("%Y-%m-%d"),
        "lookback_end": lookback_end.strftime("%Y-%m-%d"),
    }


def compute_live_forecast(
    station: str,
    bundle: Optional[Dict] = None,
    override_date: Optional[str] = None,
) -> Dict:
    """Compute next-day AQI forecast from live fresh CPCB observations and Open-Meteo weather.

    Args:
        station: Station name (e.g. "Anand_Vihar", "Bawana", or "City").
        bundle: Optional preloaded model bundle.
        override_date: Optional anchor date string (YYYY-MM-DD), default is latest available data date.

    Returns:
        dict containing historical 14 days, tomorrow's forecast, data freshness flags, and status.
    """
    if bundle is None:
        bundle = get_model_bundle()

    df_st_live = bundle.get("df_st_live")
    df_city_live = bundle.get("df_city_live")
    df_w_live = bundle.get("df_w_live")
    live_meta = bundle.get("live_meta") or {}

    if df_st_live is None or df_w_live is None:
        return {
            "success": False,
            "error": "Live telemetry cache not found in data/processed/. Please run `python src/refresh_live.py` first.",
        }

    # Normalize station name
    norm_station = None
    for s in DELHI_STATIONS:
        if s.lower() == station.lower().replace(" ", "_"):
            norm_station = s
            break
    if norm_station is None and station.lower() == "city":
        norm_station = "City"
    if norm_station is None:
        valid_options = list(DELHI_STATIONS) + ["City"]
        return {
            "success": False,
            "error": f"Invalid station '{station}'. Must be one of: {', '.join(valid_options)}.",
        }

    # Determine anchor date (today)
    if override_date:
        today_dt = pd.to_datetime(override_date)
    else:
        anchor_str = live_meta.get("latest_anchor_date")
        if anchor_str:
            today_dt = pd.to_datetime(anchor_str)
        else:
            today_dt = df_st_live["Date"].max()

    data_date = today_dt.strftime("%Y-%m-%d")

    # Stale data check: if older than 3 days from 2026-10-25
    ref_date = pd.to_datetime(live_meta.get("reference_date", "2026-10-25"))
    days_old = (ref_date - today_dt).days
    is_stale = days_old > 3
    stale_message = f"Not live. Latest available data is from {data_date}." if is_stale else None

    # Extract target series
    if norm_station == "City":
        if df_city_live is not None:
            df_target = df_city_live[["Date", "AQI"]].copy()
        else:
            df_target = df_st_live.groupby("Date")["AQI"].mean().reset_index()
    else:
        df_target = df_st_live[df_st_live["Station"] == norm_station][["Date", "AQI"]].copy()

    # Prepare daily dataframe with <= 2 days interpolation and cyclical features
    df_prep = prepare_daily_dataframe(df_target, df_w_live, interp_limit_days=2)

    lookback_start = today_dt - pd.Timedelta(days=13)
    lookback_end = today_dt
    tomorrow_dt = today_dt + pd.Timedelta(days=1)

    window_df = df_prep[(df_prep["Date"] >= lookback_start) & (df_prep["Date"] <= lookback_end)]

    if len(window_df) != 14:
        return {
            "success": False,
            "mode": "live",
            "station": norm_station,
            "data_date": data_date,
            "is_stale": is_stale,
            "stale_message": stale_message,
            "error": f"Incomplete lookback: expected 14 consecutive daily records for {norm_station} between {lookback_start.strftime('%Y-%m-%d')} and {lookback_end.strftime('%Y-%m-%d')}, found {len(window_df)}.",
        }

    if window_df["AQI_interp"].isna().any():
        return {
            "success": False,
            "mode": "live",
            "station": norm_station,
            "data_date": data_date,
            "is_stale": is_stale,
            "stale_message": stale_message,
            "error": f"Station {norm_station} has missing AQI data in the 14-day lookback preceding {data_date} that exceeds the 2-day interpolation threshold. Cannot generate forecast without complete inputs.",
        }

    config_info = bundle["config_info"]
    model = bundle["model"]
    feat_scaler = bundle["feat_scaler"]
    target_scaler = bundle["target_scaler"]

    feature_cols = config_info["feature_cols"]
    X_input = window_df[feature_cols].values.astype(np.float32)
    today_aqi = float(window_df["AQI_interp"].iloc[-1])

    X_scaled = feat_scaler.transform(X_input).reshape(1, 14, -1).astype(np.float32)
    with torch.no_grad():
        pred_sc = model(torch.as_tensor(X_scaled)).cpu().numpy().ravel()[0]

    pred_delta = float(target_scaler.inverse_transform(np.array([[pred_sc]]))[0, 0])
    tomorrow_forecast = round(float(np.clip(today_aqi + pred_delta, 0.0, 500.0)), 1)
    forecast_category = get_cpcb_category(tomorrow_forecast)

    history_table = []
    for idx, row in window_df.iterrows():
        raw_aqi = row["AQI_observed"]
        interp_aqi = row["AQI_interp"]
        disp_aqi = round(float(interp_aqi), 1) if not np.isnan(interp_aqi) else None
        history_table.append({
            "date": row["Date"].strftime("%Y-%m-%d"),
            "aqi": disp_aqi,
            "observed_raw": not np.isnan(raw_aqi),
            "category": get_cpcb_category(disp_aqi),
        })

    # Check valid hours for today's anchor date
    today_valid_hours = 24
    if df_st_live is not None and "valid_hours" in df_st_live.columns:
        if norm_station == "City":
            st_recs = df_st_live[df_st_live["Date"] == today_dt]
            if len(st_recs) > 0:
                today_valid_hours = int(round(st_recs["valid_hours"].mean()))
        else:
            st_recs = df_st_live[(df_st_live["Station"] == norm_station) & (df_st_live["Date"] == today_dt)]
            if len(st_recs) > 0:
                today_valid_hours = int(st_recs["valid_hours"].iloc[0])

    is_partial_day = today_valid_hours < 24
    partial_day_text = f"Today so far ({today_valid_hours} of 24 hours)" if is_partial_day else ""

    return {
        "success": True,
        "mode": "live",
        "error": None,
        "station": norm_station,
        "data_date": data_date,
        "is_stale": is_stale,
        "stale_message": stale_message,
        "today_date": today_dt.strftime("%Y-%m-%d"),
        "tomorrow_date": tomorrow_dt.strftime("%Y-%m-%d"),
        "today_aqi": round(today_aqi, 1),
        "today_category": get_cpcb_category(today_aqi),
        "tomorrow_forecast": tomorrow_forecast,
        "forecast_category": forecast_category,
        "actual_aqi": None,
        "actual_category": "N/A",
        "abs_error": None,
        "valid_hours": today_valid_hours,
        "is_partial_day": is_partial_day,
        "partial_day_text": partial_day_text,
        "history_table": history_table,
        "lookback_start": lookback_start.strftime("%Y-%m-%d"),
        "lookback_end": lookback_end.strftime("%Y-%m-%d"),
    }


def main():
    parser = argparse.ArgumentParser(description="Next-Day AQI Forecast Demo.")
    parser.add_argument("--mode", type=str, default="replay", choices=["replay", "live"], help="Forecast mode: 'replay' (2025 test period) or 'live' (latest data).")
    parser.add_argument("--station", type=str, default="Anand_Vihar", help="Station name (e.g. Anand_Vihar, Bawana, City).")
    parser.add_argument("--date", type=str, default="2025-11-15", help="Target evaluation date (YYYY-MM-DD) for replay mode.")
    args = parser.parse_args()

    if args.mode == "live":
        result = compute_live_forecast(args.station)
    else:
        result = compute_forecast(args.station, args.date)

    if not result["success"]:
        print(f"\n[ERROR] {result['error']}\n")
        sys.exit(1)

    print("\n" + "=" * 70)
    print(f"DELHI AIR QUALITY INDEX (AQI) NEXT-DAY FORECAST ({args.mode.upper()} MODE)")
    print(f"Station: {result['station']} | Target Evaluation Date: {result['today_date']}")
    if result.get("stale_message"):
        print(f"WARNING: {result['stale_message']}")
    print("=" * 70)

    print("\n--- Last 14 Days Historical Daily Mean AQI (Input Lookback Context) ---")
    print(f"{'Day':<6} {'Date':<14} {'Mean AQI':<12} {'CPCB Health Category':<20} {'Status':<12}")
    print("-" * 68)
    for i, h in enumerate(result["history_table"], start=1):
        status = "Observed" if h["observed_raw"] else "Interpolated"
        print(f"T-{14-i:<3} {h['date']:<14} {h['aqi']:<12.1f} {h['category']:<20} {status:<12}")

    print("-" * 68)
    print(f"Today's Anchor AQI ({result['today_date']}): {result['today_aqi']:.1f} ({result['today_category']})")
    print("\n--- Tomorrow's Next-Day Forecast (Lead Time: +1 Day) ---")
    print(f"Forecast Date    : {result['tomorrow_date']}")
    print(f"Predicted AQI    : {result['tomorrow_forecast']:.1f}")
    print(f"CPCB Category    : {result['forecast_category']}")

    if result.get("actual_aqi") is not None:
        print(f"Observed Actual  : {result['actual_aqi']:.1f} ({result['actual_category']})")
        print(f"Absolute Error   : {result['abs_error']:.1f} AQI points")
    else:
        print("Observed Actual  : Not Available (Live Forecast)")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
