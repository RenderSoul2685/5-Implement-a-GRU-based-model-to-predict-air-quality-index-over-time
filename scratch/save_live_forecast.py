import json
from pathlib import Path
import sys

sys.path.insert(0, ".")
from src.demo import compute_live_forecast, get_model_bundle
from src.config import DELHI_STATIONS

repo_root = Path(__file__).resolve().parent.parent
output_dir = repo_root / "outputs" / "live_forecasts"
output_dir.mkdir(parents=True, exist_ok=True)

bundle = get_model_bundle()
forecast_records = []

fetch_date = "2026-10-25"
fetch_time_str = "2026-10-25 23:17:20 IST"
file_time_tag = "23-17"

# Also support City if requested, but prompt says "for every station that has a valid window"
stations_to_check = list(DELHI_STATIONS)

for st in stations_to_check:
    res = compute_live_forecast(st, bundle=bundle)
    if res.get("success"):
        record = {
            "station": res["station"],
            "anchor_date": res["today_date"],
            "predicted_aqi": res["tomorrow_forecast"],
            "category": res["forecast_category"],
            "fetch_time": fetch_time_str,
        }
        forecast_records.append(record)
        print(f"Station {st:15s}: Valid window ({res['lookback_start']}..{res['lookback_end']}) -> Predicted: {res['tomorrow_forecast']} ({res['forecast_category']})")
    else:
        print(f"Station {st:15s}: Invalid/Incomplete window ({res.get('error')})")

target_filename = f"forecast_{fetch_date}_{file_time_tag}.json"
target_path = output_dir / target_filename

if target_path.exists():
    print(f"Warning: {target_path} already exists. NEVER overwriting older files.")
else:
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(forecast_records, f, indent=2)
    print(f"Saved {len(forecast_records)} forecasts to {target_path}")
