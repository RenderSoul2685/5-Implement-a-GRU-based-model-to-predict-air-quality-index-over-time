import pandas as pd
import numpy as np

# Load OpenCity daily data
df_opencity = pd.read_csv('data/processed/delhi_stations_2023_2025.csv')
df_opencity['Date'] = pd.to_datetime(df_opencity['Date'])

# Load Vayuayan live processed data
df_vayuayan = pd.read_csv('data/processed/delhi_stations_live.csv')
df_vayuayan['Date'] = pd.to_datetime(df_vayuayan['Date'])

stations = ['Anand_Vihar', 'Bawana', 'ITO', 'Mandir_Marg', 'Punjabi_Bagh', 'RK_Puram']

print('=== Daily-level overlap check (2025-11-30 to 2025-12-31) ===')
for st in stations:
    sub_oc = df_opencity[(df_opencity['Station'] == st) & (df_opencity['Date'] >= '2025-11-30') & (df_opencity['Date'] <= '2025-12-31')][['Date', 'AQI']].rename(columns={'AQI': 'AQI_OpenCity'})
    sub_vy = df_vayuayan[(df_vayuayan['Station'] == st) & (df_vayuayan['Date'] >= '2025-11-30') & (df_vayuayan['Date'] <= '2025-12-31')][['Date', 'AQI']].rename(columns={'AQI': 'AQI_Vayuayan'})
    
    merged = pd.merge(sub_oc, sub_vy, on='Date').dropna()
    n_days = len(merged)
    if n_days > 0:
        bias = (merged['AQI_Vayuayan'] - merged['AQI_OpenCity']).mean()
        corr = merged['AQI_Vayuayan'].corr(merged['AQI_OpenCity'])
        oc_mean = merged['AQI_OpenCity'].mean()
        vy_mean = merged['AQI_Vayuayan'].mean()
        print(f"{st:15s} | N = {n_days:2d} days | Bias (Vayuayan - OpenCity) = {bias:+7.2f} | Corr = {corr:.4f} | OpenCity mean = {oc_mean:.2f} | Vayuayan mean = {vy_mean:.2f}")
    else:
        print(f"{st:15s} | No overlapping days")
