import urllib.request
import json
import gzip
import io
import pandas as pd
import datetime

print("=== Checking Source A: Vayuayan Archive ===")
try:
    req = urllib.request.Request(
        "https://data.vayuayan.saketlab.org/stations.geojson",
        headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        stations_geo = json.loads(resp.read().decode('utf-8'))
    print(f"Loaded stations.geojson with {len(stations_geo.get('features', []))} features")
    
    # Let's search for Delhi stations
    delhi_stations = []
    for f in stations_geo.get('features', []):
        props = f.get('properties', {})
        city = props.get('city', '').lower()
        state = props.get('state', '').lower()
        name = props.get('name', '').lower()
        if 'delhi' in city or 'delhi' in state or 'delhi' in name:
            delhi_stations.append(props)
            
    print(f"Found {len(delhi_stations)} Delhi stations in geojson")
    for s in delhi_stations:
        name = s.get('name', '')
        site_id = s.get('site_id', '') or s.get('id', '')
        slug = s.get('slug', '')
        data_url = s.get('data_url', '') or s.get('url', '')
        # check if it matches our 6 stations
        if any(target in name.lower() or target in str(slug).lower() for target in ['anand', 'bawana', 'ito', 'mandir', 'punjabi', 'puram']):
            print(f"Match: name='{name}', id='{site_id}', slug='{slug}', props={props}")
except Exception as e:
    print(f"Error fetching stations.geojson: {e}")

# Also check https://saketkc.github.io/vayuayan-archive/
print("\nTesting sample vayuayan station URL: delhi__anand_vihar__anand_vihar__site_244.csv.gz")
try:
    test_url = "https://saketkc.github.io/vayuayan-archive/data/delhi__anand_vihar__anand_vihar__site_244.csv.gz"
    req = urllib.request.Request(test_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        content = resp.read()
    print(f"Downloaded {len(content)} bytes from test_url")
    with gzip.GzipFile(fileobj=io.BytesIO(content)) as gz:
        df_test = pd.read_csv(gz)
    print("Columns:", df_test.columns.tolist()[:10])
    print("Head:\n", df_test.head(2))
    print("Tail:\n", df_test.tail(5))
    if 'timestamp' in df_test.columns or 'from_date' in df_test.columns or 'date' in df_test.columns:
        date_col = 'timestamp' if 'timestamp' in df_test.columns else ('from_date' if 'from_date' in df_test.columns else 'date')
        print("Latest date in Anand Vihar file:", df_test[date_col].max())
except Exception as e:
    print(f"Error checking vayuayan anand vihar: {e}")

print("\n=== Checking Source B: XKDR India Air Quality API ===")
try:
    xkdr_url = "https://airquality.xkdr.org/api"
    req = urllib.request.Request("https://airquality.xkdr.org/", headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        html = resp.read().decode('utf-8', errors='ignore')
    print("XKDR homepage length:", len(html))
    print("XKDR homepage snippet:", html[:300].replace('\n', ' '))
except Exception as e:
    print(f"Error fetching XKDR: {e}")
