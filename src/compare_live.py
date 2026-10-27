"""Compare saved live forecasts against observed daily actual means once the forecast date has passed.

Usage:
    python src/compare_live.py
    python src/compare_live.py --forecast-file outputs/live_forecasts/forecast_2026-10-25_23-17.json
    python src/compare_live.py --fetch
"""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.refresh_live import refresh_live_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def find_latest_forecast_file(forecast_dir: Path) -> Optional[Path]:
    """Find the most recent forecast JSON file in outputs/live_forecasts/."""
    files = sorted(forecast_dir.glob("forecast_*.json"))
    return files[-1] if files else None


def load_actual_daily_means(processed_dir: Path, fetch: bool = False) -> pd.DataFrame:
    """Load or refresh live station daily means from data/processed/delhi_stations_live.csv."""
    if fetch:
        logger.info("Fetching freshest telemetry feeds from Vayuayan and Open-Meteo...")
        refresh_live_data(repo_root=ROOT_DIR, skip_download=False)

    stations_csv = processed_dir / "delhi_stations_live.csv"
    if not stations_csv.exists():
        logger.warning("delhi_stations_live.csv not found at %s. Attempting to refresh...", stations_csv)
        refresh_live_data(repo_root=ROOT_DIR, skip_download=True)

    df_st = pd.read_csv(stations_csv)
    df_st["Date"] = pd.to_datetime(df_st["Date"])
    return df_st


def compare_forecasts(
    forecast_file: Optional[Path] = None,
    fetch: bool = False,
    repo_root: Optional[Path] = None,
) -> List[Dict]:
    """Compare predictions in forecast_file against actual observed daily means."""
    if repo_root is None:
        repo_root = ROOT_DIR

    forecast_dir = repo_root / "outputs" / "live_forecasts"
    processed_dir = repo_root / "data" / "processed"

    if forecast_file is None:
        forecast_file = find_latest_forecast_file(forecast_dir)

    if forecast_file is None or not forecast_file.exists():
        raise FileNotFoundError(
            f"No forecast files found in {forecast_dir}. Please save a forecast first."
        )

    with open(forecast_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Support list of records or dict container
    records = data if isinstance(data, list) else data.get("forecasts", [])
    if not records:
        print(f"No forecast records found in {forecast_file.name}")
        return []

    df_actuals = load_actual_daily_means(processed_dir, fetch=fetch)

    results = []
    print("\n" + "=" * 96)
    print(f"LIVE FORECAST VERIFICATION REPORT: {forecast_file.name}")
    print("=" * 96)
    header = f"{'Station':<16} | {'Anchor':<10} | {'Forecast':<10} | {'Pred AQI':>8} | {'Category':<11} | {'Actual AQI':>10} | {'Error':>8} | {'Status'}"
    print(header)
    print("-" * 96)

    for item in records:
        station = item["station"]
        anchor_date_str = item["anchor_date"]
        anchor_dt = pd.to_datetime(anchor_date_str)
        target_dt = anchor_dt + pd.Timedelta(days=1)
        target_date_str = target_dt.strftime("%Y-%m-%d")
        pred_aqi = float(item["predicted_aqi"])
        category = item.get("category", "N/A")

        # Query actual daily observation for station and target_date
        st_match = df_actuals[(df_actuals["Station"] == station) & (df_actuals["Date"] == target_dt)]

        if len(st_match) > 0 and pd.notna(st_match["AQI"].iloc[0]):
            actual_aqi = round(float(st_match["AQI"].iloc[0]), 1)
            valid_h = int(st_match["valid_hours"].iloc[0]) if "valid_hours" in st_match.columns else 24
            err = round(pred_aqi - actual_aqi, 1)
            abs_err = round(abs(err), 1)
            status = f"VERIFIED ({valid_h}h valid)"
            actual_str = f"{actual_aqi:.1f}"
            err_str = f"{err:+.1f}"
        else:
            actual_aqi = None
            err = None
            abs_err = None
            actual_str = "PENDING"
            err_str = "N/A"
            status = f"Awaiting actual for {target_date_str}"

        res_row = {
            "station": station,
            "anchor_date": anchor_date_str,
            "target_date": target_date_str,
            "predicted_aqi": pred_aqi,
            "category": category,
            "actual_aqi": actual_aqi,
            "error": err,
            "abs_error": abs_err,
            "status": status,
        }
        results.append(res_row)

        print(
            f"{station:<16} | {anchor_date_str:<10} | {target_date_str:<10} | "
            f"{pred_aqi:>8.1f} | {category:<11} | {actual_str:>10} | {err_str:>8} | {status}"
        )

    print("-" * 96)

    completed = [r for r in results if r["actual_aqi"] is not None]
    if completed:
        mae = float(np.mean([r["abs_error"] for r in completed]))
        rmse = float(np.sqrt(np.mean([r["error"] ** 2 for r in completed])))
        print(f"Completed Station Forecasts: {len(completed)}/{len(results)} | MAE: {mae:.2f} | RMSE: {rmse:.2f}")
    else:
        print(
            f"Note: Forecast date ({results[0]['target_date']}) has not fully concluded or actual daily telemetry is pending."
        )
    print("=" * 96 + "\n")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare saved live forecasts against observed daily actual means."
    )
    parser.add_argument(
        "--forecast-file",
        type=str,
        default=None,
        help="Path to specific forecast JSON file in outputs/live_forecasts/. Defaults to most recent.",
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Fetch fresh telemetry feeds from Vayuayan before running comparison.",
    )
    args = parser.parse_args()

    forecast_path = Path(args.forecast_file) if args.forecast_file else None
    compare_forecasts(forecast_file=forecast_path, fetch=args.fetch)


if __name__ == "__main__":
    main()
