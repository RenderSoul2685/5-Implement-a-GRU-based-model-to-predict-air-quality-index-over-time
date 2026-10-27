import urllib.request
import gzip
import io
import pandas as pd
import json

base_url = "https://data.vayuayan.saketlab.org/data/"

# Test site_117 variations
variations = [
    "delhi__delhi__ito_delhi_cpcb__site_117.csv.gz",
    "delhi__delhi__ito__site_117.csv.gz",
    "delhi__delhi__ito_cpcb__site_117.csv.gz",
]

for v in variations:
    url = base_url + v
    try:
        r = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(r, timeout=5) as res:
            data = res.read()
        print(f"FOUND: {url} ({len(data)} bytes)")
        with gzip.GzipFile(fileobj=io.BytesIO(data)) as gz:
            df = pd.read_csv(gz)
        print("Columns:", df.columns.tolist()[:6])
        time_col = [c for c in df.columns if 'time' in c.lower() or 'date' in c.lower()][0]
        print(f"Min: {df[time_col].min()} | Max: {df[time_col].max()} | Rows: {len(df)}")
    except Exception as e:
        print(f"404 or error for {v}: {e}")

# Check stations.json on vayuayan to see all stations listed
try:
    r = urllib.request.Request("https://data.vayuayan.saketlab.org/stations.json", headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(r, timeout=10) as res:
        st_json = json.loads(res.read().decode('utf-8'))
    print(f"stations.json has {len(st_json)} entries")
    for s in st_json:
        if '117' in str(s) or 'ito' in str(s).lower():
            print("Match in stations.json:", s)
except Exception as e:
    print(f"stations.json error: {e}")
