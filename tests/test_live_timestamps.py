"""Unit tests verifying that live telemetry timestamps are converted to IST before hourly aggregation."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.refresh_live import convert_vayuayan_timestamps_to_ist

ROOT_DIR = Path(__file__).resolve().parent.parent


def test_vayuayan_timestamp_conversion_to_ist():
    """Verify that raw Vayuayan UTC timestamps are shifted by +5:30 into Indian Standard Time (IST)."""
    raw_df = pd.DataFrame(
        {
            "timestamp": [
                "2026-10-25T17:20:50.519450",
                "2026-10-25T00:00:00.000000",
            ],
            "PM2.5_avg": [150.0, 120.0],
        }
    )

    converted_df = convert_vayuayan_timestamps_to_ist(raw_df)

    assert "Timestamp_UTC" in converted_df.columns
    assert "Timestamp" in converted_df.columns

    # 17:20:50 UTC must map to 22:50:50 IST
    t0_ist = converted_df["Timestamp"].iloc[0]
    assert t0_ist.hour == 22, f"Expected IST hour 22, got {t0_ist.hour} (failed to convert UTC to IST)"
    assert t0_ist.minute == 50

    # 00:00:00 UTC must map to 05:30:00 IST
    t1_ist = converted_df["Timestamp"].iloc[1]
    assert t1_ist.hour == 5, f"Expected IST hour 5, got {t1_ist.hour}"
    assert t1_ist.minute == 30

    # Ensure difference is strictly +5:30
    diff = converted_df["Timestamp"] - converted_df["Timestamp_UTC"]
    for d in diff:
        assert d == pd.Timedelta(hours=5, minutes=30)


def test_failure_when_timestamps_are_not_ist():
    """Verify that validation fails if timestamps are kept in UTC instead of IST."""
    raw_utc_str = "2026-10-25T17:20:50.519450"
    raw_df = pd.DataFrame({"timestamp": [raw_utc_str], "PM2.5_avg": [150.0]})

    # Case 1: Unconverted / naive UTC timestamp
    unconverted_df = raw_df.copy()
    unconverted_df["Timestamp"] = pd.to_datetime(unconverted_df["timestamp"])
    with pytest.raises(AssertionError, match="Timestamp is not IST"):
        # Simulated test validator enforcing IST hour
        sample_hour = unconverted_df["Timestamp"].iloc[0].hour
        assert sample_hour == 22, f"Timestamp is not IST: hour is {sample_hour} (expected 22 for 17:20 UTC)"

    # Case 2: Properly converted
    converted_df = convert_vayuayan_timestamps_to_ist(raw_df)
    sample_hour_converted = converted_df["Timestamp"].iloc[0].hour
    assert sample_hour_converted == 22, "Converted timestamp must be in IST"


def test_processed_live_stations_has_23_ist_hours_today():
    """Verify that processed delhi_stations_live.csv reflects IST accounting (23 hours for today, not 18)."""
    live_stations_path = ROOT_DIR / "data" / "processed" / "delhi_stations_live.csv"
    assert live_stations_path.exists(), "Processed live stations file does not exist"

    df_st = pd.read_csv(live_stations_path)
    today_records = df_st[df_st["Date"] == "2026-10-25"]
    assert len(today_records) > 0, "No records for 2026-10-25 in live stations"

    # In IST, hours 00:00 to 22:00 yield 23 valid hours
    # If left in UTC, valid hours would erroneously be 18 (hours 00:00 to 17:00 UTC)
    for _, row in today_records.iterrows():
        valid_h = row["valid_hours"]
        assert valid_h == 23, (
            f"Station {row['Station']} has {valid_h} valid hours for 2026-10-25. "
            f"Expected 23 hours in IST (18 hours indicates uncorrected UTC aggregation)."
        )
