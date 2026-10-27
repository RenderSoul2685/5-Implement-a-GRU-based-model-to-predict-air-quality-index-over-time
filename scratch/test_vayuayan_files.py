import urllib.request
import gzip
import io
import pandas as pd
import json

base_urls = [
    "https://saket-choudhary.me/vayuayan-archive/",
    "https://data.vayuayan.saketlab.org/",
    "https://saketkc.github.io/vayuayan-archive/"
]

def sanitize(s):
    import re
    s = s.lower()
    s = re.sub(r'[^a-z0-9]+', '_', s)
    s = re.sub(r'_+', '_', s)
    return s.strip('_')

# Fetch stations.geojson again to get exact names and IDs
req = urllib.request.Request("https://data.vayuayan.saketlab.org/stations.geojson", headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req) as resp:
    geo = json.loads(resp.read().decode('utf-8'))

stations_by_id = {f['properties']['station_id']: f['properties'] for f in geo['features']}

target_sids = {
    'Anand_Vihar': 'site_301',
    'Bawana': 'site_1560',
    'Mandir_Marg': 'site_122',
    'Punjabi_Bagh': 'site_125',
    'RK_Puram': 'site_124',
}

for station_name, sid in target_sids.items():
    props = stations_by_id.get(sid, {})
    st_name = props.get('station_name')
    city = props.get('city')
    state = props.get('state')
    fname = f"{sanitize(state)}__{sanitize(city)}__{sanitize(st_name)}__{sid}.csv.gz"
    print(f"\nStation: {station_name} -> {fname}")
    
    for base in base_urls:
        for folder in ['data/', '']:
            test_url = base + folder + fname
            try:
                r = urllib.request.Request(test_url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(r, timeout=5) as res:
                    data = res.read()
                print(f"  FOUND: {test_url} ({len(data)} bytes)")
                with gzip.GzipFile(fileobj=io.BytesIO(data)) as gz:
                    df = pd.read_csv(gz)
                print(f"  Columns: {df.columns.tolist()[:6]}")
                time_col = [c for c in df.columns if 'time' in c.lower() or 'date' in c.lower()][0]
                print(f"  Min date: {df[time_col].min()} | Max date: {df[time_col].max()} | Rows: {len(df)}")
                break
            except Exception as e:
                pass
        else:
            continue
        break
