import urllib.request
import json

req = urllib.request.Request("https://airquality.xkdr.org/v1/openapi.json", headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req, timeout=10) as resp:
    spec = json.loads(resp.read().decode('utf-8'))

print("Paths in XKDR OpenAPI:")
for path, methods in spec.get('paths', {}).items():
    for method, info in methods.items():
        print(f"{method.upper()} {path} -> summary: {info.get('summary')}, params: {[p.get('name') for p in info.get('parameters', [])]}")
