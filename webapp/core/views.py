"""Views for AQI-GRU Next-Day Forecasting Demonstration Web Application."""

import json
from pathlib import Path
from django.shortcuts import render
from src.config import DELHI_STATIONS
from src.demo import compute_forecast, compute_live_forecast, get_aqi_reference_rows

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def forecast_view(request):
    """Render interactive next-day AQI forecast page supporting live and replay modes."""
    mode = request.GET.get("mode", "live").strip().lower()
    if mode not in ["live", "replay"]:
        mode = "live"

    station = request.GET.get("station", "Anand_Vihar").strip()

    available_stations = [
        "Anand_Vihar",
        "Bawana",
        "ITO",
        "Mandir_Marg",
        "Punjabi_Bagh",
        "RK_Puram",
    ]
    min_date = "2025-10-01"
    max_date = "2025-12-30"

    context = {
        "mode": mode,
        "available_stations": available_stations,
        "selected_station": station,
        "selected_date": None,
        "data_date": "2026-10-25",
        "min_date": min_date,
        "max_date": max_date,
        "notice": "",
        "is_stale": False,
        "stale_message": None,
        "forecast_data": None,
        "chart_data_json": None,
        "error_message": None,
        "aqi_reference_rows": get_aqi_reference_rows(),
        "forecast_category": None,
    }

    # Validate station input
    if station not in available_stations:
        context["error_message"] = (
            f"Invalid station '{station}'. Please select one of the 6 monitored stations: {', '.join(available_stations)}."
        )
        return render(request, "forecast.html", context)

    if mode == "replay":
        date_str = request.GET.get("date", "2025-11-15").strip()
        context["selected_date"] = date_str
        context["notice"] = (
            "Historical replay: dates are from the 2025 test period, not live. "
            "The city-level model is applied to each station's daily readings. "
            "The forecast is for the day after the selected date."
        )

        try:
            import pandas as pd
            parsed_dt = pd.to_datetime(date_str)
            formatted_date = parsed_dt.strftime("%Y-%m-%d")
            if formatted_date < min_date or formatted_date > max_date:
                context["error_message"] = (
                    f"Date '{date_str}' is outside the allowable forecast-from test period ({min_date} to {max_date})."
                )
                return render(request, "forecast.html", context)
        except Exception:
            context["error_message"] = (
                f"Invalid date format '{date_str}'. Please select a valid date between {min_date} and {max_date}."
            )
            return render(request, "forecast.html", context)

        result = compute_forecast(station, date_str)
    else:
        # Live mode
        date_override = request.GET.get("date")
        if date_override:
            date_override = date_override.strip()
            context["selected_date"] = date_override

        result = compute_live_forecast(station, override_date=date_override)
        context["data_date"] = result.get("data_date", "2026-10-25")
        context["is_stale"] = result.get("is_stale", False)
        context["stale_message"] = result.get("stale_message")

        if result.get("is_stale"):
            context["notice"] = f"Not live. Latest available data is from {context['data_date']}."
        else:
            context["notice"] = f"Latest data: {context['data_date']}"

    if not result.get("success"):
        context["error_message"] = result.get("error", "An error occurred while computing the forecast.")
        context["forecast_data"] = result
        return render(request, "forecast.html", context)

    context["forecast_data"] = result
    context["forecast_category"] = result.get("forecast_category")

    # Prepare JSON payload for Chart.js
    labels = [h["date"] for h in result["history_table"]] + [f"{result['tomorrow_date']} (Tomorrow)"]
    hist_aqi = [h["aqi"] for h in result["history_table"]] + [None]
    forecast_line = [None] * 13 + [result["today_aqi"], result["tomorrow_forecast"]]

    if mode == "replay":
        actual_point = [None] * 14 + [result.get("actual_aqi")]
    else:
        actual_point = [None] * 15

    chart_payload = {
        "labels": labels,
        "historical": hist_aqi,
        "forecast_line": forecast_line,
        "actual_point": actual_point,
        "today_date": result["today_date"],
        "tomorrow_date": result["tomorrow_date"],
        "tomorrow_forecast": result["tomorrow_forecast"],
        "actual_aqi": result.get("actual_aqi"),
        "mode": mode,
    }
    context["chart_data_json"] = json.dumps(chart_payload)

    return render(request, "forecast.html", context)


def about_view(request):
    """Render methodology, data provenance, results table, and limitations page."""
    results_path = REPO_ROOT / "outputs" / "results_table.md"
    results_headers = []
    results_rows = []

    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str.startswith("|") and not line_str.startswith("| :---"):
                    cells = [c.strip().replace("**", "") for c in line_str.split("|")[1:-1]]
                    if not results_headers:
                        results_headers = cells
                    else:
                        results_rows.append(cells)

    context = {
        "results_headers": results_headers,
        "results_rows": results_rows,
        "tied_statement": (
            "The GRU and Ridge regression are tied (difference far smaller than the seed spread); "
            "both beat persistence by about 8%. The test set has only 92 days."
        ),
        "aqi_reference_rows": get_aqi_reference_rows(),
        "forecast_category": None,
    }
    return render(request, "about.html", context)
