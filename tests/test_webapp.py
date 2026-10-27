"""Unit tests for next-day Django web forecast view (live and replay modes)."""

import os
from pathlib import Path
import sys
import django
from django.test import Client

ROOT_DIR = Path(__file__).resolve().parent.parent
WEBAPP_DIR = ROOT_DIR / "webapp"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(WEBAPP_DIR) not in sys.path:
    sys.path.insert(0, str(WEBAPP_DIR))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "webapp.settings")
django.setup()

from django.test.utils import setup_test_environment
setup_test_environment()

from src.demo import compute_forecast, compute_live_forecast


def test_webapp_forecast_equals_demo_output():
    """Verify replay forecast for (Anand_Vihar, 2025-11-15) is still 417.6 and matches demo.py."""
    station = "Anand_Vihar"
    date_str = "2025-11-15"

    # Compute expected output directly from demo.py
    demo_result = compute_forecast(station, date_str)
    assert demo_result["success"] is True
    assert demo_result["tomorrow_forecast"] == 417.6
    assert demo_result["station"] == "Anand_Vihar"

    # Compute output via Django WebApp Client in replay mode
    client = Client()
    response = client.get("/", {"mode": "replay", "station": station, "date": date_str})

    assert response.status_code == 200
    assert "forecast_data" in response.context
    web_result = response.context["forecast_data"]
    assert web_result is not None
    assert web_result["success"] is True

    # Assert web output equals demo output and matches predicted AQI 417.6
    assert web_result["tomorrow_forecast"] == 417.6
    assert web_result["tomorrow_forecast"] == demo_result["tomorrow_forecast"]
    assert web_result["today_aqi"] == demo_result["today_aqi"]
    assert web_result["actual_aqi"] == demo_result["actual_aqi"]
    assert web_result["forecast_category"] == demo_result["forecast_category"]
    assert web_result["actual_category"] == demo_result["actual_category"]
    assert web_result["abs_error"] == demo_result["abs_error"]
    assert len(web_result["history_table"]) == len(demo_result["history_table"])


def test_webapp_latest_data_page_loads():
    """Verify that the latest-data page loads with status 200 and defaults to live mode."""
    client = Client()
    response = client.get("/")  # default mode is latest data

    assert response.status_code == 200
    assert response.context["mode"] == "live"
    assert response.context["data_date"] == "2026-10-25"
    assert response.context["is_stale"] is False

    forecast_data = response.context["forecast_data"]
    assert forecast_data is not None
    assert forecast_data["success"] is True
    assert forecast_data["mode"] == "live"
    assert isinstance(forecast_data["tomorrow_forecast"], float)
    # Tomorrow actual should not exist for live forecasts
    assert forecast_data["actual_aqi"] is None

    # Check HTML content contains live header notice and partial day badge
    content = response.content.decode("utf-8")
    assert "Latest data:" in content
    assert "2026-10-25" in content
    assert "Tomorrow&#x27;s Predicted AQI" in content or "Tomorrow's Predicted AQI" in content
    assert "Today so far (23 of 24 hours)" in content


def test_webapp_stale_data_warning_when_date_old():
    """Verify that the stale-data warning appears when the data date is older than 3 days."""
    client = Client()
    old_date = "2026-10-18"  # 7 days old relative to 2026-10-25 reference
    response = client.get("/", {"mode": "live", "station": "Anand_Vihar", "date": old_date})

    assert response.status_code == 200
    assert response.context["mode"] == "live"
    assert response.context["is_stale"] is True
    assert response.context["stale_message"] == f"Not live. Latest available data is from {old_date}."

    content = response.content.decode("utf-8")
    expected_warning = f"Not live. Latest available data is from {old_date}."
    assert expected_warning in content
    assert "Data Freshness Notice:" in content


def test_webapp_live_disclaimer_note_rendered():
    """Verify that the live page renders the 24-hour rolling averages disclaimer note."""
    client = Client()
    response = client.get("/", {"mode": "live", "station": "Anand_Vihar"})

    assert response.status_code == 200
    content = response.content.decode("utf-8")
    expected_note = "Live inputs come from a CPCB feed of 24-hour rolling averages, unlike the training data; forecasts are indicative only."
    assert expected_note in content
    assert 'id="live-inputs-note"' in content


def test_webapp_aqi_reference_table_categories_and_ranges():
    """Verify that both the main forecast page and about page contain all 6 CPCB categories, ranges, and meanings."""
    client = Client()

    for path, params in [("/", {"mode": "live", "station": "Anand_Vihar"}), ("/about/", {})]:
        response = client.get(path, params)
        assert response.status_code == 200
        content = response.content.decode("utf-8")

        # Title and column headers
        assert "How to read the AQI (CPCB categories)" in content
        assert "AQI range" in content
        assert "Category" in content
        assert "Meaning" in content

        # All 6 rows
        expected_rows = [
            ("0-50", "Good", "Little or no risk"),
            ("51-100", "Satisfactory", "Minor discomfort for sensitive people"),
            ("101-200", "Moderate", "Breathing discomfort for people with lung or heart conditions"),
            ("201-300", "Poor", "Discomfort on prolonged exposure"),
            ("301-400", "Very Poor", "Respiratory illness on prolonged exposure"),
            ("401-500", "Severe", "Affects healthy people and seriously affects those with existing illness"),
        ]

        for range_str, cat_name, meaning_str in expected_rows:
            assert range_str in content
            assert cat_name in content
            assert meaning_str in content

        # Footnote line
        expected_footnote = (
            "Simplified summary of the CPCB (India) AQI categories. "
            "The AQI is the highest sub-index among PM2.5, PM10, NO2, SO2, CO and ozone."
        )
        assert expected_footnote in content


def test_webapp_replay_severe_highlight_and_value():
    """Verify that for (Anand_Vihar, 2025-11-15), tomorrow's forecast remains 417.6 and Severe is highlighted."""
    client = Client()
    response = client.get("/", {"mode": "replay", "station": "Anand_Vihar", "date": "2025-11-15"})

    assert response.status_code == 200
    web_result = response.context["forecast_data"]
    assert web_result["success"] is True
    # Forecast value must remain exactly 417.6
    assert web_result["tomorrow_forecast"] == 417.6
    assert web_result["forecast_category"] == "Severe"

    content = response.content.decode("utf-8")
    assert "Tomorrow's forecast" in content

    # Severe row must have highlight class and label
    severe_row_html = content.split('id="aqi-row-severe"')[1].split('</tr>')[0]
    assert 'class="aqi-ref-highlight"' in content
    assert "Tomorrow's forecast" in severe_row_html

    # Other rows must NOT have Tomorrow's forecast label
    for other_slug in ["good", "satisfactory", "moderate", "poor", "very-poor"]:
        other_row_html = content.split(f'id="aqi-row-{other_slug}"')[1].split('</tr>')[0]
        assert "Tomorrow's forecast" not in other_row_html


def test_webapp_about_page_no_highlight():
    """Verify that the about page shows the AQI reference table without highlighting any row."""
    client = Client()
    response = client.get("/about/")

    assert response.status_code == 200
    content = response.content.decode("utf-8")
    assert "How to read the AQI (CPCB categories)" in content
    assert "Tomorrow's forecast" not in content
    assert 'class="aqi-ref-highlight"' not in content


