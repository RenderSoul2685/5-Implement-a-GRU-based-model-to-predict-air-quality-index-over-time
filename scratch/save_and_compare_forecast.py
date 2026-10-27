import json
from pathlib import Path
import sys

sys.path.insert(0, ".")
from src.demo import compute_live_forecast, get_model_bundle
from src.config import DELHI_STATIONS

output_dir = Path("outputs/live_forecasts")
output_dir.mkdir(parents=True, exist_ok=True)
bundle = get_model_bundle()

forecast_records = []
fetch_date = "2026-10-26"
fetch_time_str = "2026-10-26 11:25:00 IST"
file_time_tag = "11-25"

for st in DELHI_STATIONS:
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

target_path = output_dir / f"forecast_{fetch_date}_{file_time_tag}.json"
with open(target_path, "w", encoding="utf-8") as f:
    json.dump(forecast_records, f, indent=2)

print(f"Saved new forecast to {target_path}")

# Compare old vs new
old_path = output_dir / "forecast_2026-10-25_23-17.json"
with open(old_path) as f:
    old_data = json.load(f)

old_map = {r["station"]: (r["predicted_aqi"], r["category"]) for r in old_data}
new_map = {r["station"]: (r["predicted_aqi"], r["category"]) for r in forecast_records}

print("\n" + "=" * 65)
print(f"{'Station':<15} | {'Old Forecast':<14} | {'New Forecast':<14} | {'Difference':<10}")
print("=" * 65)
for st in old_map:
    old_val, old_cat = old_map[st]
    new_val, new_cat = new_map.get(st, (None, None))
    diff_str = f"{new_val - old_val:+6.1f}" if new_val is not None else "N/A"
    print(f"{st:<15} | {old_val:5.1f} ({old_cat}) | {new_val:5.1f} ({new_cat}) | {diff_str}")
print("=" * 65)
