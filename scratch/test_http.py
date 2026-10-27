import urllib.request

# Test Live mode
with urllib.request.urlopen("http://127.0.0.1:8000/?mode=live&station=Anand_Vihar", timeout=5) as resp:
    html_live = resp.read().decode("utf-8")
    assert resp.status == 200
    assert "Today so far (23 of 24 hours)" in html_live
    print("HTTP Live Mode: 200 OK! Found 'Today so far (23 of 24 hours)' on page.")

# Test Replay mode
with urllib.request.urlopen("http://127.0.0.1:8000/?mode=replay&station=Anand_Vihar&date=2025-11-15", timeout=5) as resp:
    html_replay = resp.read().decode("utf-8")
    assert resp.status == 200
    assert "417.6" in html_replay
    print("HTTP Replay Mode: 200 OK! Found 417.6 replay forecast on page.")
