import os
import sys
from pathlib import Path
import django

sys.path.insert(0, "webapp")
sys.path.insert(0, ".")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "webapp.settings")
django.setup()
from django.test.utils import setup_test_environment
setup_test_environment()

from django.test import Client

client = Client()

# 1. Test live mode default
resp_live = client.get("/")
assert resp_live.status_code == 200
assert "Today so far (23 of 24 hours)" in resp_live.content.decode("utf-8")
print("Live mode default (Anand_Vihar) status 200 OK! Found partial day badge.")

# 2. Test live mode all stations
for st in ["Anand_Vihar", "Bawana", "Mandir_Marg", "Punjabi_Bagh", "RK_Puram", "ITO"]:
    resp = client.get("/", {"mode": "live", "station": st})
    assert resp.status_code == 200
    fd = resp.context["forecast_data"]
    if fd["success"]:
        print(f"Live mode {st:15s}: OK! Forecast = {fd['tomorrow_forecast']} ({fd['forecast_category']})")
    else:
        print(f"Live mode {st:15s}: Incomplete window handled gracefully ({fd['error'][:40]}...)")

# 3. Test replay mode
resp_replay = client.get("/", {"mode": "replay", "station": "Anand_Vihar", "date": "2025-11-15"})
assert resp_replay.status_code == 200
fd_rep = resp_replay.context["forecast_data"]
assert fd_rep["success"] is True
assert fd_rep["tomorrow_forecast"] == 417.6
print("Replay mode (2025-11-15 Anand_Vihar) status 200 OK! Forecast = 417.6 (Exact match).")
