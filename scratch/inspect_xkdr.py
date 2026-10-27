import re
import urllib.request

req = urllib.request.Request("https://airquality.xkdr.org/", headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req, timeout=10) as resp:
    html = resp.read().decode('utf-8', errors='ignore')

# Look for api endpoints or demo endpoints in xkdr html
links = set(re.findall(r'href=[\'"]([^\'"]+)[\'"]', html))
for l in sorted(links):
    if any(k in l.lower() for k in ['api', 'doc', 'swagger', 'station', 'data', 'v1', 'v2', 'endpoint']):
        print("Link:", l)

# Search text snippets
for m in re.finditer(r'(https?://[^\s"\'<>]+api[^\s"\'<>]*)', html):
    print("API URL:", m.group(1))

# Search for api keys or demo keys mentioned
for m in re.finditer(r'api[_-]?key', html, re.I):
    start = max(0, m.start() - 50)
    end = min(len(html), m.end() + 100)
    print("API key mention:", html[start:end].replace('\n', ' '))
