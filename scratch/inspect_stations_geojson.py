import urllib.request
import json
import gzip
import io
import pandas as pd

req = urllib.request.Request(
    "https://data.vayuayan.saketlab.org/stations.geojson",
    headers={"User-Agent": "Mozilla/5.0"}
)
with urllib.request.urlopen(req, timeout=15) as resp:
    stations_geo = json.loads(resp.read().decode('utf-8'))

print("Sample feature properties:")
print(json.dumps(stations_geo['features'][0]['properties'], indent=2))

target_keywords = ['anand', 'bawana', 'ito', 'mandir', 'punjabi', 'puram']

matched = []
for f in stations_geo['features']:
    props = f['properties']
    city = str(props.get('city', '')).lower()
    state = str(props.get('state', '')).lower()
    name = str(props.get('name', '')).lower()
    if 'delhi' in city or 'delhi' in state or 'delhi' in name:
        for kw in target_keywords:
            if kw in name:
                matched.append((kw, props))

print(f"\nMatched {len(matched)} stations for targets:")
for kw, p in matched:
    print(f"Target '{kw}': Name='{p.get('name')}', ID='{p.get('site_id') or p.get('id')}', all_keys={list(p.keys())}")
    print(f"   Props: {p}")
