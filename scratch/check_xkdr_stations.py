import urllib.request
import json

base = "https://airquality.xkdr.org"

try:
    with urllib.request.urlopen(f"{base}/v1/meta", timeout=10) as r:
        meta = json.loads(r.read().decode('utf-8'))
    print("Meta:", json.dumps(meta, indent=2))
except Exception as e:
    print("Meta failed:", e)

try:
    with urllib.request.urlopen(f"{base}/v1/stations?city=Delhi", timeout=10) as r:
        st = json.loads(r.read().decode('utf-8'))
    print(f"Stations in Delhi: {len(st)}")
    for s in st:
        name = s.get('station_name') or s.get('name') or ''
        sid = s.get('station_id') or s.get('id') or ''
        if any(kw in name.lower() for kw in ['anand', 'bawana', 'ito', 'mandir', 'punjabi', 'puram']):
            print(f"Match: {sid} | {name} | from: {s.get('from_date')} | to: {s.get('to_date')}")
except Exception as e:
    print("Stations failed:", e)
