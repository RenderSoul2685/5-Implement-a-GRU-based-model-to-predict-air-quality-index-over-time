import urllib.request
import json

req = urllib.request.Request("https://airquality.xkdr.org/v1/docs", headers={"User-Agent": "Mozilla/5.0"})
try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        text = resp.read().decode('utf-8')
    print("Length of /v1/docs:", len(text))
    print(text[:500])
except Exception as e:
    print("Error:", e)

# Also check openapi.json or swagger
for p in ['/openapi.json', '/v1/openapi.json', '/api/v1/stations']:
    try:
        r = urllib.request.Request("https://airquality.xkdr.org" + p, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(r, timeout=5) as resp:
            data = resp.read()
        print(f"Endpoint {p} exists! Size: {len(data)}")
    except Exception as e:
        print(f"Endpoint {p} failed: {e}")
