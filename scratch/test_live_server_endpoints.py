import urllib.request
import re

endpoints = [
    ("/", "Latest Data Mode (Default)"),
    ("/?station=Bawana", "Latest Data - Bawana"),
    ("/?station=ITO", "Latest Data - ITO (Incomplete Lookback Message)"),
    ("/?mode=replay&station=Anand_Vihar&date=2025-11-15", "Replay Mode - Anand Vihar 2025-11-15"),
    ("/?mode=live&station=Anand_Vihar&date=2026-10-18", "Stale Warning Test - 2026-10-18"),
    ("/about/", "About Page"),
]

for path, desc in endpoints:
    url = f"http://127.0.0.1:8000{path}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            status = resp.status
            html = resp.read().decode('utf-8')
        print(f"\n[PASS] {desc}: {url} -> HTTP {status} (length {len(html)})")
        
        # Check specific items
        if "Latest Data Mode" in desc:
            assert "Latest data:" in html, "Missing 'Latest data:' in HTML"
            assert "2026-10-25" in html, "Missing '2026-10-25' in HTML"
            assert "Tomorrow&#x27;s Predicted AQI" in html or "Tomorrow's Predicted AQI" in html
            print("  Verified: 'Latest data: 2026-10-25' and predicted AQI card present.")
        elif "ITO" in desc:
            assert "Unable to Compute Next-Day Forecast" in html
            print("  Verified: Incomplete lookback error alert shown cleanly for ITO.")
        elif "Replay Mode" in desc:
            assert "417.6" in html, "Missing 417.6 in HTML"
            assert "397.4" in html, "Missing 397.4 in HTML"
            assert "20.2" in html, "Missing 20.2 in HTML"
            print("  Verified: Replay forecast 417.6, actual 397.4, error 20.2 all present.")
        elif "Stale Warning" in desc:
            assert "Not live. Latest available data is from 2026-10-18." in html
            print("  Verified: Stale data warning banner cleanly displayed.")
        elif "About" in desc:
            assert "The GRU and Ridge regression are tied" in html
            print("  Verified: About page results table and tied statement present.")
    except Exception as e:
        print(f"\n[FAIL] {desc}: {url} -> {e}")
