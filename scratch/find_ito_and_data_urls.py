import urllib.request
import json

req = urllib.request.Request(
    "https://data.vayuayan.saketlab.org/stations.geojson",
    headers={"User-Agent": "Mozilla/5.0"}
)
with urllib.request.urlopen(req, timeout=15) as resp:
    stations_geo = json.loads(resp.read().decode('utf-8'))

print("Searching for ITO across all 526 stations:")
for f in stations_geo['features']:
    props = f['properties']
    name = props.get('station_name', '')
    sid = props.get('station_id', '')
    if 'ito' in name.lower() or 'ito' in sid.lower():
        print(f"ITO match: ID={sid}, Name={name}, City={props.get('city')}, State={props.get('state')}, Latest={props.get('latest_timestamp')}")

print("\nChecking properties for site_301 (Anand Vihar):")
anand = next(f['properties'] for f in stations_geo['features'] if f['properties'].get('station_id') == 'site_301')
print(json.dumps(anand, indent=2))
