import urllib.request
import json
import pandas as pd

# Open-Meteo archive API usually has a 2-5 day delay.
# Let's test what dates the archive API has for October 2026:
archive_url = (
    "https://archive-api.open-meteo.com/v1/archive?"
    "latitude=28.6139&longitude=77.2090"
    "&start_date=2026-10-01&end_date=2026-10-25"
    "&daily=temperature_2m_mean,relative_humidity_2m_mean,wind_speed_10m_max,surface_pressure_mean"
    "&timezone=Asia%2FKolkata"
)

print("Fetching from archive API:", archive_url)
try:
    req = urllib.request.Request(archive_url, headers={"User-Agent": "AQI-GRU-Research/2.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        arch_data = json.loads(resp.read().decode('utf-8'))
    df_arch = pd.DataFrame({
        "Date": arch_data["daily"]["time"],
        "temp": arch_data["daily"]["temperature_2m_mean"],
        "rh": arch_data["daily"]["relative_humidity_2m_mean"],
        "ws": arch_data["daily"]["wind_speed_10m_max"],
        "pres": arch_data["daily"]["surface_pressure_mean"],
    })
    print("Archive response dates:", df_arch["Date"].tolist())
    print("Archive nulls:\n", df_arch.isnull().sum())
    print("Archive tail:\n", df_arch.tail(5))
except Exception as e:
    print("Archive API failed:", e)

# Test forecast API with past_days
# https://api.open-meteo.com/v1/forecast?latitude=28.6139&longitude=77.2090&daily=temperature_2m_mean,relative_humidity_2m_mean,wind_speed_10m_max,surface_pressure_mean&past_days=14&forecast_days=1&timezone=Asia%2FKolkata
forecast_url = (
    "https://api.open-meteo.com/v1/forecast?"
    "latitude=28.6139&longitude=77.2090"
    "&daily=temperature_2m_mean,relative_humidity_2m_mean,wind_speed_10m_max,surface_pressure_mean"
    "&past_days=14&forecast_days=1"
    "&timezone=Asia%2FKolkata"
)

print("\nFetching from forecast API with past_days:", forecast_url)
try:
    req = urllib.request.Request(forecast_url, headers={"User-Agent": "AQI-GRU-Research/2.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        fc_data = json.loads(resp.read().decode('utf-8'))
    df_fc = pd.DataFrame({
        "Date": fc_data["daily"]["time"],
        "temp": fc_data["daily"]["temperature_2m_mean"],
        "rh": fc_data["daily"]["relative_humidity_2m_mean"],
        "ws": fc_data["daily"]["wind_speed_10m_max"],
        "pres": fc_data["daily"]["surface_pressure_mean"],
    })
    print("Forecast response dates:", df_fc["Date"].tolist())
    print("Forecast tail:\n", df_fc.tail(5))
except Exception as e:
    print("Forecast API failed:", e)
