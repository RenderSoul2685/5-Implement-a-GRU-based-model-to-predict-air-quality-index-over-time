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

print("Shape:", df.shape)
print("All columns:", df.columns.tolist())
print("\nTail 10 rows:")
print(df.tail(10))
