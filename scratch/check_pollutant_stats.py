import urllib.request
import gzip
import io
import pandas as pd

url = "https://data.vayuayan.saketlab.org/data/delhi__delhi__anand_vihar_delhi_dpcc__site_301.csv.gz"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req) as res:
    data = res.read()

with gzip.GzipFile(fileobj=io.BytesIO(data)) as gz:
    df = pd.read_csv(gz)

print("Vayuayan summary stats:")
pollutants = ['PM2.5_avg', 'PM10_avg', 'NO2_avg', 'SO2_avg', 'CO_avg', 'OZONE_avg']
print(df[pollutants].describe())

# Check subindices already computed in vayuayan
sub_cols = [c for c in df.columns if 'subindex' in c]
print("\nSubindex cols in vayuayan:")
print(df[sub_cols].describe())
