import urllib.request
import json

req = urllib.request.Request(
    "https://data.vayuayan.saketlab.org/stations.geojson",
    headers={"User-Agent": "Mozilla/5.0"}
)
with urllib.request.urlopen(req, timeout=15) as resp:
    stations_geo = json.loads(resp.read().decode('utf-8'))

print("Total stations:", len(stations_geo['features']))
states = set(f['properties'].get('state') for f in stations_geo['features'])
print("States:", sorted(str(s) for s in states))

delhi_stations = [f['properties'] for f in stations_geo['features'] if 'delhi' in str(f['properties'].get('state', '')).lower() or 'delhi' in str(f['properties'].get('city', '')).lower()]
print("Delhi stations count:", len(delhi_stations))
for s in delhi_stations:
    print(f"ID: {s.get('station_id')} | City: {s.get('city')} | Name: {s.get('station_name')} | Latest: {s.get('latest_timestamp')} | Live: {s.get('is_live')}")
