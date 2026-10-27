import urllib.request

def check_url(url, expected_highlight=None):
    resp = urllib.request.urlopen(url)
    assert resp.status == 200, f"Status {resp.status} for {url}"
    html = resp.read().decode("utf-8")
    assert "How to read the AQI (CPCB categories)" in html, f"Table title missing in {url}"
    assert "AQI range" in html and "Category" in html and "Meaning" in html
    assert "Simplified summary of the CPCB (India) AQI categories." in html
    for r in ["0-50", "51-100", "101-200", "201-300", "301-400", "401-500"]:
        assert r in html, f"Range {r} missing in {url}"
    for c in ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]:
        assert c in html, f"Category {c} missing in {url}"
    
    if expected_highlight:
        assert "Tomorrow's forecast" in html, f"Expected highlight label missing in {url}"
        target_row = html.split(f'id="aqi-row-{expected_highlight}"')[1].split("</tr>")[0]
        assert "Tomorrow's forecast" in target_row, f"Highlight label not on expected row {expected_highlight} in {url}"
        print(f"{url}: OK! Highlighted '{expected_highlight}' row.")
    else:
        assert "Tomorrow's forecast" not in html, f"Unexpected highlight label in {url}"
        print(f"{url}: OK! No highlight (as expected).")

check_url("http://127.0.0.1:8000/?mode=live&station=Anand_Vihar", expected_highlight="poor")
check_url("http://127.0.0.1:8000/?mode=replay&station=Anand_Vihar&date=2025-11-15", expected_highlight="severe")
check_url("http://127.0.0.1:8000/about/", expected_highlight=None)
print("All HTTP live server checks passed successfully!")
