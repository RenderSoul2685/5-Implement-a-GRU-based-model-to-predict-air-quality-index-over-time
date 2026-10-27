import sys
sys.path.insert(0, ".")
import urllib.request
import gzip
import io
import pandas as pd
import numpy as np
from src.data_loader import compute_hourly_aqi_from_pollutants

stations = {
    'Anand_Vihar': 'delhi__delhi__anand_vihar_delhi_dpcc__site_301.csv.gz',
    'Bawana': 'delhi__delhi__bawana_delhi_dpcc__site_1560.csv.gz',
    'ITO': 'delhi__delhi__ito_delhi_cpcb__site_117.csv.gz',
    'Mandir_Marg': 'delhi__delhi__mandir_marg_delhi_dpcc__site_122.csv.gz',
    'Punjabi_Bagh': 'delhi__delhi__punjabi_bagh_delhi_dpcc__site_125.csv.gz',
    'RK_Puram': 'delhi__delhi__r_k_puram_delhi_dpcc__site_124.csv.gz',
}

base_url = "https://data.vayuayan.saketlab.org/data/"

station_daily = {}

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
    # Hourly average of pollutant concentrations
    hourly = df.groupby("Hour_TS")[["PM2.5", "PM10", "NO2", "SO2", "CO", "Ozone"]].mean().reset_index()
    # Compute hourly AQI
    hourly["AQI"] = compute_hourly_aqi_from_pollutants(hourly)
    hourly["Date"] = pd.to_datetime(hourly["Hour_TS"].dt.date)
    
    # Daily aggregation: require >= 18 valid hours
    grouped = hourly.groupby("Date")["AQI"]
    daily_count = grouped.count()
    daily_mean = grouped.mean()
    daily_valid = daily_mean.where(daily_count >= 18, np.nan)
    
    station_daily[st_name] = pd.DataFrame({
        "valid_hours": daily_count,
        "daily_aqi": daily_valid
    })

# Check last 10 days for each station
print("=== Latest Valid Hours Count per Day ===")
df_counts = pd.DataFrame({k: v["valid_hours"] for k, v in station_daily.items()})
print(df_counts.tail(10))

print("\n=== Latest Daily AQI (>= 18 valid hours) ===")
df_aqi = pd.DataFrame({k: v["daily_aqi"] for k, v in station_daily.items()})
print(df_aqi.tail(10))
