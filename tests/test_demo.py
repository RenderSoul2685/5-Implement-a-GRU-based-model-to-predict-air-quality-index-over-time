"""Unit tests for demo script functions, CPCB category mappings, and metric calculations."""

import numpy as np
import pytest

from src.demo import get_cpcb_category
from src.train import compute_metrics_masked


def test_get_cpcb_category():
    """Verify that AQI values map strictly to official CPCB health categories."""
    assert get_cpcb_category(0.0) == "Good"
    assert get_cpcb_category(45.0) == "Good"
    assert get_cpcb_category(50.0) == "Good"
    assert get_cpcb_category(51.0) == "Satisfactory"
    assert get_cpcb_category(100.0) == "Satisfactory"
    assert get_cpcb_category(101.0) == "Moderate"
    assert get_cpcb_category(200.0) == "Moderate"
    assert get_cpcb_category(201.0) == "Poor"
    assert get_cpcb_category(300.0) == "Poor"
    assert get_cpcb_category(301.0) == "Very Poor"
    assert get_cpcb_category(400.0) == "Very Poor"
    assert get_cpcb_category(401.0) == "Severe"
    assert get_cpcb_category(485.0) == "Severe"
    assert get_cpcb_category(float("nan")) == "N/A"


def test_compute_metrics_masked():
    """Verify that masked evaluation accurately calculates RMSE and MAE ignoring NaNs and False masks."""
    y_true = np.array([[100.0, 200.0], [300.0, 400.0]])
    y_pred = np.array([[110.0, 190.0], [320.0, 500.0]])
    mask = np.array([[True, True], [True, False]])  # exclude (1, 1)

    # Valid diffs: +10, -10, +20
    # squared diffs: 100, 100, 400 -> mean = 200 -> sqrt = 14.142
    # abs diffs: 10, 10, 20 -> mean = 13.333
    metrics = compute_metrics_masked(y_true, y_pred, mask)
    np.testing.assert_allclose(metrics["rmse"], np.sqrt(200.0), rtol=1e-4)
    np.testing.assert_allclose(metrics["mae"], 40.0 / 3.0, rtol=1e-4)


def test_compute_forecast_anand_vihar():
    """Verify demo compute_forecast generates valid next-day predictions."""
    from src.demo import compute_forecast
    res = compute_forecast("Anand_Vihar", "2025-11-15")
    assert res["success"] is True
    assert res["station"] == "Anand_Vihar"
    assert res["today_date"] == "2025-11-15"
    assert res["tomorrow_date"] == "2025-11-16"
    assert len(res["history_table"]) == 14
    assert 0.0 <= res["tomorrow_forecast"] <= 500.0
    assert res["forecast_category"] in ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]


def test_compute_live_forecast_partial_day():
    """Verify live forecast returns valid 23-hour partial day metadata for 2026-10-25."""
    from src.demo import compute_live_forecast
    res = compute_live_forecast("Anand_Vihar")
    assert res["success"] is True
    assert res["mode"] == "live"
    assert res["today_date"] == "2026-10-25"
    assert res["valid_hours"] == 23
    assert res["is_partial_day"] is True
    assert res["partial_day_text"] == "Today so far (23 of 24 hours)"
    assert res["tomorrow_forecast"] == 240.4


def test_compare_live_tool():
    """Verify compare_live tool executes cleanly on saved live forecast."""
    from src.compare_live import compare_forecasts
    results = compare_forecasts()
    assert len(results) == 5
    station_names = [r["station"] for r in results]
    assert "Anand_Vihar" in station_names
    assert "Punjabi_Bagh" in station_names


