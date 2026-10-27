"""Command-line inference script for AQI forecasting using saved GRU model and scaler."""

import argparse
import logging
from pathlib import Path
import sys
from typing import List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch

from src.config import Config, get_aqi_category
from src.data_loader import load_raw_data
from src.model import AQIGRU
from src.preprocessing import clean_and_prepare_timeseries

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_model_and_scalers(config: Config) -> tuple[AQIGRU, MinMaxScaler, MinMaxScaler]:
    """Load model checkpoint and pre-fitted feature and target scalers."""
    model_path = config.models_dir / "gru_model.pt"
    feat_scaler_path = config.models_dir / "feature_scaler.joblib"
    tgt_scaler_path = config.models_dir / "target_scaler.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}. Please run train.py first.")
    if not feat_scaler_path.exists() or not tgt_scaler_path.exists():
        raise FileNotFoundError("Saved scalers not found. Please run train.py first.")

    checkpoint = torch.load(model_path, map_location=config.device)
    cfg_dict = checkpoint["config"]

    model = AQIGRU(
        input_size=cfg_dict["input_size"],
        hidden_size=cfg_dict["hidden_size"],
        num_layers=cfg_dict["num_layers"],
        output_size=cfg_dict["output_size"],
        dropout=cfg_dict["dropout"],
    ).to(config.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    feature_scaler = joblib.load(feat_scaler_path)
    target_scaler = joblib.load(tgt_scaler_path)

    return model, feature_scaler, target_scaler


def forecast_next(
    model: AQIGRU,
    feature_scaler,
    target_scaler,
    latest_window_features: np.ndarray,
    steps: int = 1,
    device: str = "cpu",
) -> List[dict]:
    """Forecast future AQI given the latest historical window.

    Supports single-step prediction or multi-step autoregressive rollout.

    Args:
        model: Trained AQIGRU neural network.
        feature_scaler: Scaler for all input features.
        target_scaler: Scaler for target AQI.
        latest_window_features: Raw unscaled or scaled feature array of shape (lookback, num_features).
        steps: Number of future days to forecast (1 to 7).
        device: PyTorch device ('cpu' or 'cuda').

    Returns:
        List of dictionaries with forecasted day, numerical AQI, category, and health advisory.
    """
    model.eval()
    window = latest_window_features.copy()
    forecasts = []

    # Target is column index 6 (PM2.5, PM10, NO2, SO2, CO, O3, AQI)
    target_idx = 6

    for step_num in range(1, steps + 1):
        x_tensor = torch.as_tensor(window[np.newaxis, :, :], dtype=torch.float32).to(device)
        with torch.no_grad():
            pred_scaled = model(x_tensor).cpu().numpy()[0, 0]

        pred_unscaled = float(target_scaler.inverse_transform([[pred_scaled]])[0, 0])
        # Bound predictions to physical CPCB range [0, 500]
        pred_unscaled = max(0.0, min(500.0, round(pred_unscaled, 1)))

        category, advisory = get_aqi_category(pred_unscaled)
        forecasts.append(
            {
                "step": step_num,
                "predicted_aqi": pred_unscaled,
                "category": category,
                "advisory": advisory,
            }
        )

        # Autoregressive rollout: update latest window with prediction
        next_step_feat = window[-1].copy()
        next_step_feat[target_idx] = pred_scaled
        window = np.vstack([window[1:], next_step_feat])

    return forecasts


def run_prediction(
    city: str = "Delhi",
    steps: int = 1,
    custom_csv: Optional[str] = None,
    allow_synthetic: bool = False,
) -> None:
    """Run CLI inference using the latest available data."""
    config = Config(city=city, allow_synthetic=allow_synthetic)
    model, feature_scaler, target_scaler = load_model_and_scalers(config)

    # Load latest historical data
    if custom_csv and Path(custom_csv).exists():
        raw_df = pd.read_csv(custom_csv)
    else:
        raw_df = load_raw_data(city=city, allow_synthetic=allow_synthetic)

    processed_df = clean_and_prepare_timeseries(
        raw_df,
        pollutant_cols=config.pollutant_cols,
        target_col=config.target_col,
    )

    if len(processed_df) < config.lookback:
        raise ValueError(
            f"Insufficient data rows ({len(processed_df)}) to satisfy lookback window of {config.lookback}."
        )

    # Extract the most recent lookback days
    latest_df = processed_df.iloc[-config.lookback:]
    latest_date = latest_df.index.max().strftime("%Y-%m-%d")

    # Scale using the training feature scaler
    latest_features_scaled = feature_scaler.transform(latest_df[config.feature_cols].values)

    forecasts = forecast_next(
        model=model,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        latest_window_features=latest_features_scaled,
        steps=steps,
        device=config.device,
    )

    # Format output
    print("\n" + "=" * 70)
    print(f"             AIR QUALITY FORECAST REPORT: {city.upper()}             ")
    print("=" * 70)
    print(f" Historical Lookback Base Date: {latest_date}")
    print(f" Forecast Horizon:            {steps} Day(s) Ahead")
    print("-" * 70)

    for item in forecasts:
        print(f" Day +{item['step']}:")
        print(f"   Predicted AQI:   {item['predicted_aqi']:.1f}")
        print(f"   Category:        {item['category']}")
        print(f"   Health Advisory: {item['advisory']}")
        print("-" * 70)
    print("=" * 70 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Forecast AQI from latest historical window.")
    parser.add_argument("--city", type=str, default="Delhi", help="City name (default: Delhi)")
    parser.add_argument("--steps", type=int, default=1, help="Number of forecast steps ahead (1 to 7)")
    parser.add_argument("--csv", type=str, default=None, help="Optional path to custom input CSV file")
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        default=False,
        help="Allow synthetic dataset fallback",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_prediction(city=args.city, steps=args.steps, custom_csv=args.csv, allow_synthetic=args.allow_synthetic)

