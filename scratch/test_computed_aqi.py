import sys
sys.path.insert(0, ".")
import urllib.request
import gzip
import io
import pandas as pd
import numpy as np
from src.data_loader import compute_hourly_aqi_from_pollutants

url = "https://data.vayuayan.saketlab.org/data/delhi__delhi__anand_vihar_delhi_dpcc__site_301.csv.gz"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req) as res:
    data = res.read()

with gzip.GzipFile(fileobj=io.BytesIO(data)) as gz:
    df = pd.read_csv(gz)

df["Timestamp"] = pd.to_datetime(df["timestamp"])
df["PM2.5"] = pd.to_numeric(df["PM2.5_avg"].replace("-", np.nan), errors="coerce")
df["PM10"] = pd.to_numeric(df["PM10_avg"].replace("-", np.nan), errors="coerce")
df["NO2"] = pd.to_numeric(df["NO2_avg"].replace("-", np.nan), errors="coerce")
df["SO2"] = pd.to_numeric(df["SO2_avg"].replace("-", np.nan), errors="coerce")
df["CO"] = pd.to_numeric(df["CO_avg"].replace("-", np.nan), errors="coerce") / 1000.0  # ug/m3 -> mg/m3
df["Ozone"] = pd.to_numeric(df["OZONE_avg"].replace("-", np.nan), errors="coerce")

# Note: Vayuayan reports readings every 30 minutes (or multiple times per hour).
# Let's check the timestamps!
print("Timestamps frequency / diff:")
print(df["Timestamp"].tail(10))

# Resample to hourly if needed or check hour floor
df["Hour_TS"] = df["Timestamp"].dt.floor("h")
hourly = df.groupby("Hour_TS")[["PM2.5", "PM10", "NO2", "SO2", "CO", "Ozone"]].mean().reset_index()
hourly["AQI_computed"] = compute_hourly_aqi_from_pollutants(hourly)

print("\nComputed hourly AQI tail (latest):")
print(hourly[["Hour_TS", "PM2.5", "PM10", "AQI_computed"]].dropna().tail(10))

# Group by date to check valid hours (requiring >= 18 valid hours)
hourly["Date"] = hourly["Hour_TS"].dt.date
daily_stats = hourly.groupby("Date")["AQI_computed"].agg(["count", "mean"])
print("\nDaily stats for latest 15 days:")
print(daily_stats.tail(15))
