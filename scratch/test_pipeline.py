import sys
sys.path.insert(0, ".")
import os
import hashlib
import json
import urllib.request
import gzip
import io
import pandas as pd
import numpy as np
from src.data_loader import compute_hourly_aqi_from_pollutants
from src.config import DELHI_STATIONS

STATION_MAP = {
    'Anand_Vihar': {
        'site_id': 'site_301',
        'url': 'https://data.vayuayan.saketlab.org/data/delhi__delhi__anand_vihar_delhi_dpcc__site_301.csv.gz',
        'raw_filename': 'anand_vihar_live_site_301.csv.gz'
    },
    'Bawana': {
        'site_id': 'site_1560',
        'url': 'https://data.vayuayan.saketlab.org/data/delhi__delhi__bawana_delhi_dpcc__site_1560.csv.gz',
        'raw_filename': 'bawana_live_site_1560.csv.gz'
    },
    'ITO': {
        'site_id': 'site_117',
        'url': 'https://data.vayuayan.saketlab.org/data/delhi__delhi__ito_delhi_cpcb__site_117.csv.gz',
        'raw_filename': 'ito_live_site_117.csv.gz'
    },
    'Mandir_Marg': {
        'site_id': 'site_122',
        'url': 'https://data.vayuayan.saketlab.org/data/delhi__delhi__mandir_marg_delhi_dpcc__site_122.csv.gz',
        'raw_filename': 'mandir_marg_live_site_122.csv.gz'
    },
    'Punjabi_Bagh': {
        'site_id': 'site_125',
        'url': 'https://data.vayuayan.saketlab.org/data/delhi__delhi__punjabi_bagh_delhi_dpcc__site_125.csv.gz',
        'raw_filename': 'punjabi_bagh_live_site_125.csv.gz'
    },
    'RK_Puram': {
        'site_id': 'site_124',
        'url': 'https://data.vayuayan.saketlab.org/data/delhi__delhi__r_k_puram_delhi_dpcc__site_124.csv.gz',
        'raw_filename': 'rk_puram_live_site_124.csv.gz'
    }
}

os.makedirs("data/raw/live", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

station_dfs = []
station_meta = {}

for st_name, meta in STATION_MAP.items():
    print(f"Downloading {st_name} from {meta['url']}...")
    req = urllib.request.Request(meta['url'], headers={"User-Agent": "AQI-GRU-Live/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        content = resp.read()
    
    raw_path = os.path.join("data/raw/live", meta['raw_filename'])
    with open(raw_path, "wb") as f:
        f.write(content)
    
    sha256 = hashlib.sha256(content).hexdigest()
    
    with gzip.GzipFile(fileobj=io.BytesIO(content)) as gz:
        df_raw = pd.read_csv(gz)
        
    df_raw["Timestamp"] = pd.to_datetime(df_raw["timestamp"])
    max_ts = df_raw["Timestamp"].max()
    min_ts = df_raw["Timestamp"].min()
    
    station_meta[st_name] = {
        'site_id': meta['site_id'],
        'url': meta['url'],
        'sha256': sha256,
        'min_ts': str(min_ts),
        'max_ts': str(max_ts),
        'rows': len(df_raw)
    }
    
    df_raw["PM2.5"] = pd.to_numeric(df_raw["PM2.5_avg"].replace("-", np.nan), errors="coerce")
    df_raw["PM10"] = pd.to_numeric(df_raw["PM10_avg"].replace("-", np.nan), errors="coerce")
    df_raw["NO2"] = pd.to_numeric(df_raw["NO2_avg"].replace("-", np.nan), errors="coerce")
    df_raw["SO2"] = pd.to_numeric(df_raw["SO2_avg"].replace("-", np.nan), errors="coerce")
    df_raw["CO"] = pd.to_numeric(df_raw["CO_avg"].replace("-", np.nan), errors="coerce") / 1000.0
    df_raw["Ozone"] = pd.to_numeric(df_raw["OZONE_avg"].replace("-", np.nan), errors="coerce")
    
    df_raw["Hour_TS"] = df_raw["Timestamp"].dt.floor("h")
    hourly = df_raw.groupby("Hour_TS")[["PM2.5", "PM10", "NO2", "SO2", "CO", "Ozone"]].mean().reset_index()
    hourly["AQI"] = compute_hourly_aqi_from_pollutants(hourly)
    hourly["Date"] = pd.to_datetime(hourly["Hour_TS"].dt.date)
    
    # Daily aggregation: require >= 18 valid hours
    grouped = hourly.groupby("Date")["AQI"]
    daily_count = grouped.count()
    daily_mean = grouped.mean()
    daily_valid = daily_mean.where(daily_count >= 18, np.nan).reset_index()
    daily_valid["valid_hours"] = daily_count.values
    daily_valid["Station"] = st_name
    
    station_dfs.append(daily_valid)

df_all_stations = pd.concat(station_dfs, ignore_index=True)
print(f"Total station-days: {len(df_all_stations)}")
print("Station breakdown:")
for st in STATION_MAP:
    sub = df_all_stations[df_all_stations["Station"] == st]
    print(f"  {st}: {len(sub)} days, latest date: {sub['Date'].max().strftime('%Y-%m-%d')}, valid on latest: {not np.isnan(sub[sub['Date'] == sub['Date'].max()]['AQI'].values[0])}")

# City-level series: mean across active stations
df_city = df_all_stations.groupby("Date")["AQI"].mean().reset_index()
print(f"City days: {len(df_city)}, latest city date: {df_city['Date'].max().strftime('%Y-%m-%d')}")
