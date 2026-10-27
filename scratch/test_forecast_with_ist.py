import sys
sys.path.insert(0, ".")
import pandas as pd
from scratch.simulate_ist_conversion import df_all
from src.demo import compute_live_forecast, get_model_bundle

bundle = get_model_bundle()
bundle["df_st_live"] = df_all
bundle["df_city_live"] = df_all.groupby("Date")["AQI"].mean().reset_index()

for st in ["Anand_Vihar", "Bawana", "ITO", "Mandir_Marg", "Punjabi_Bagh", "RK_Puram"]:
    res = compute_live_forecast(st, bundle=bundle)
    if res["success"]:
        print(f"{st:15s}: Today={res['today_aqi']} ({res['today_category']}) -> Forecast={res['tomorrow_forecast']} ({res['forecast_category']}) | Window: {res['lookback_start']} to {res['lookback_end']}")
    else:
        print(f"{st:15s}: FAILED: {res.get('error')}")
