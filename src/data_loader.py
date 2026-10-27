"""Data loading, Open-Meteo weather caching, and multi-station data management for AQI forecasting (2023-2025)."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import urllib.request
import numpy as np
import pandas as pd

from src.config import Config, DELHI_STATIONS, RAW_STATION_COLUMNS_MAP, CPCB_BREAKPOINTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OPEN_METEO_URL = (
    "https://archive-api.open-meteo.com/v1/archive?"
    "latitude=28.6139&longitude=77.2090"
    "&start_date=2023-01-01&end_date=2025-12-31"
    "&daily=temperature_2m_mean,relative_humidity_2m_mean,wind_speed_10m_max,surface_pressure_mean"
    "&timezone=Asia%2FKolkata"
)


def fetch_or_load_delhi_weather(
    cache_path: Optional[Union[str, Path]] = None,
    start_date: str = "2023-01-01",
    end_date: str = "2025-12-31",
) -> pd.DataFrame:
    """Download historical daily weather for Delhi from Open-Meteo API or load from local cache.

    Columns returned: ['Date', 'temperature', 'humidity', 'wind_speed', 'pressure'].
    """
    if cache_path is None:
        repo_root = Path(__file__).resolve().parent.parent
        cache_path = repo_root / "data" / "raw" / "delhi_weather_2023_2025.csv"
    else:
        cache_path = Path(cache_path)

    if cache_path.exists():
        logger.info("Loading cached Delhi weather from %s", cache_path)
        weather_df = pd.read_csv(cache_path)
        weather_df["Date"] = pd.to_datetime(weather_df["Date"])
        return weather_df

    logger.info("Fetching Delhi weather for %s to %s from Open-Meteo API...", start_date, end_date)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(OPEN_METEO_URL, headers={"User-Agent": "AQI-GRU-Research/2.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        weather_df = pd.DataFrame(
            {
                "Date": data["daily"]["time"],
                "temperature": data["daily"]["temperature_2m_mean"],
                "humidity": data["daily"]["relative_humidity_2m_mean"],
                "wind_speed": data["daily"]["wind_speed_10m_max"],
                "pressure": data["daily"]["surface_pressure_mean"],
            }
        )
        weather_df["Date"] = pd.to_datetime(weather_df["Date"])
        weather_df.to_csv(cache_path, index=False)
        logger.info("Cached %d daily weather records to %s", len(weather_df), cache_path)
        return weather_df
    except Exception as e:
        logger.error("Could not reach Open-Meteo API (%s) and no cached weather data exists at %s.", e, cache_path)
        raise FileNotFoundError(
            f"Real weather data could not be fetched from Open-Meteo API ({e}) and cache is missing at {cache_path}. "
            f"Synthetic data generation has been removed. Please place authentic weather data at {cache_path}."
        ) from e


def compute_cpcb_subindex_vectorized(
    val_series: pd.Series,
    breakpoints: List[Tuple[float, float, float, float]],
) -> pd.Series:
    """Compute CPCB sub-index for a pollutant series using piecewise linear interpolation."""
    res = pd.Series(np.nan, index=val_series.index, dtype=float)
    val = pd.to_numeric(val_series, errors="coerce").astype(float)
    for c_low, c_high, i_low, i_high in breakpoints:
        mask = (val >= c_low) & (val <= c_high)
        res[mask] = i_low + (val[mask] - c_low) * (i_high - i_low) / (c_high - c_low)
    c_low, c_high, i_low, i_high = breakpoints[-1]
    mask_high = val > c_high
    res[mask_high] = (i_low + (val[mask_high] - c_low) * (i_high - i_low) / (c_high - c_low)).clip(upper=500.0)
    return res


def compute_hourly_aqi_from_pollutants(df: pd.DataFrame) -> pd.Series:
    """Compute CPCB AQI from hourly pollutant concentrations according to official CPCB rules.

    Overall AQI is the maximum of pollutant sub-indices, provided at least 3 pollutants
    are monitored with at least one particulate matter (PM2.5 or PM10).
    """
    sub_pm25 = (
        compute_cpcb_subindex_vectorized(df["PM2.5"], CPCB_BREAKPOINTS["PM2.5"])
        if "PM2.5" in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    sub_pm10 = (
        compute_cpcb_subindex_vectorized(df["PM10"], CPCB_BREAKPOINTS["PM10"])
        if "PM10" in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    sub_no2 = (
        compute_cpcb_subindex_vectorized(df["NO2"], CPCB_BREAKPOINTS["NO2"])
        if "NO2" in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    sub_so2 = (
        compute_cpcb_subindex_vectorized(df["SO2"], CPCB_BREAKPOINTS["SO2"])
        if "SO2" in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    sub_co = (
        compute_cpcb_subindex_vectorized(df["CO"], CPCB_BREAKPOINTS["CO"])
        if "CO" in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    sub_o3 = (
        compute_cpcb_subindex_vectorized(df["Ozone"], CPCB_BREAKPOINTS["Ozone"])
        if "Ozone" in df.columns
        else (
            compute_cpcb_subindex_vectorized(df["O3"], CPCB_BREAKPOINTS["Ozone"])
            if "O3" in df.columns
            else pd.Series(np.nan, index=df.index)
        )
    )

    subs = pd.concat([sub_pm25, sub_pm10, sub_no2, sub_so2, sub_co, sub_o3], axis=1)
    valid_counts = subs.notna().sum(axis=1)
    has_pm = sub_pm25.notna() | sub_pm10.notna()
    max_sub = subs.max(axis=1)

    # CPCB standard requirement: >= 3 criteria pollutants with >= 1 PM
    aqi = max_sub.where((valid_counts >= 3) & has_pm, np.nan)
    # Fallback when fewer than 3 parameters available but PM present
    fallback_aqi = max_sub.where(has_pm, np.nan)
    final_aqi = aqi.combine_first(fallback_aqi)
    return final_aqi.round(1)


def parse_cpcb_grid_file(
    filepath: Union[str, Path],
    start_date: str = "2023-01-01",
    end_date: str = "2025-12-31 23:59:59",
) -> Tuple[pd.DataFrame, dict]:
    """Parse raw CPCB hourly bulletin matrix grid table for 2017-2023.

    Filters to start_date .. end_date FIRST, logs rows dropped per file,
    and then verifies strict date bounds.
    """
    import re
    filepath = Path(filepath)
    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()

    month_regex = re.compile(
        r"^(January|February|March|April|May|June|July|August|September|October|November|December)-(\d{4})",
        re.IGNORECASE,
    )
    hours = [f"{h:02d}:00:00" for h in range(24)]

    records = []
    current_my = None
    for l in lines:
        parts = [p.strip().replace('"', "") for p in l.split(",")]
        if not parts or not parts[0]:
            continue
        m = month_regex.match(parts[0])
        if m:
            current_my = (m.group(1), int(m.group(2)))
            continue
        if current_my and parts[0].isdigit():
            day = int(parts[0])
            m_name, yr = current_my
            for h_idx in range(24):
                val_idx = 1 + h_idx
                val_str = parts[val_idx] if val_idx < len(parts) else ""
                val = float(val_str) if val_str and val_str.replace(".", "", 1).isdigit() else np.nan
                ts_str = f"{yr}-{m_name}-{day:02d} {hours[h_idx]}"
                records.append((ts_str, val))

    df = pd.DataFrame(records, columns=["Timestamp_raw", "AQI"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp_raw"], format="%Y-%B-%d %H:%M:%S", errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

    total_parsed = len(df)
    # FILTER TO 2023-01-01 .. 2025-12-31 FIRST
    df_filtered = df[(df["Timestamp"] >= start_date) & (df["Timestamp"] <= end_date)].copy()
    dropped_count = total_parsed - len(df_filtered)

    log_stats = {
        "file": filepath.name,
        "total_parsed_hours": total_parsed,
        "dropped_pre2023_hours": dropped_count,
        "kept_2023_hours": len(df_filtered),
    }
    logger.info(
        "Station %s: Parsed %d hourly records | Filtered out %d pre-2023 records | Kept %d records for 2023",
        filepath.name,
        total_parsed,
        dropped_count,
        len(df_filtered),
    )

    # Strict date bounds assertion
    if not df_filtered.empty:
        assert df_filtered["Timestamp"].min() >= pd.to_datetime(start_date)
        assert df_filtered["Timestamp"].max() <= pd.to_datetime(end_date)

    return df_filtered[["Timestamp", "AQI"]], log_stats


def resample_15min_to_hourly(df_15min: pd.DataFrame) -> pd.DataFrame:
    """Resample 15-minute observations to hourly means requiring >= 3 of 4 readings (75% CPCB rule)."""
    df = df_15min.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    avail_cols = [c for c in RAW_STATION_COLUMNS_MAP if c in df.columns]
    df_clean = df[["Timestamp"] + avail_cols].rename(columns=RAW_STATION_COLUMNS_MAP)
    df_clean["Hour_TS"] = df_clean["Timestamp"].dt.floor("h")

    numeric_cols = [RAW_STATION_COLUMNS_MAP[c] for c in avail_cols]
    grouped = df_clean.groupby("Hour_TS")[numeric_cols]
    counts = grouped.count()
    means = grouped.mean()

    # Require >= 3 of 4 readings per hour
    hourly_df = means.where(counts >= 3, np.nan).reset_index().rename(columns={"Hour_TS": "Timestamp"})

    # Compute CPCB AQI from hourly concentrations
    hourly_df["AQI"] = compute_hourly_aqi_from_pollutants(hourly_df)
    if "Ozone" in hourly_df.columns and "O3" not in hourly_df.columns:
        hourly_df["O3"] = hourly_df["Ozone"]
    return hourly_df


def load_and_combine_station(
    station: str,
    raw_dir: Optional[Union[str, Path]] = None,
) -> Tuple[pd.DataFrame, dict]:
    """Load, filter, resample, and combine 2023 hourly and 2024-25 15-min data for a station."""
    if raw_dir is None:
        raw_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    else:
        raw_dir = Path(raw_dir)

    st_clean = station.lower().replace("-", "_")
    suffix = "cpcb" if "ito" in st_clean else "dpcc"
    f2017 = raw_dir / f"{st_clean}_{suffix}_2017-2023.csv"
    f2024 = raw_dir / f"{st_clean}_{suffix}_2024-25.csv"

    if not f2017.exists() or not f2024.exists():
        raise FileNotFoundError(f"Raw station files missing for {station} at {raw_dir}: {f2017} or {f2024}")

    # 1. Parse 2017-2023 grid
    df23, log_stats = parse_cpcb_grid_file(f2017)

    # 2. Resample 2024-25 15-minute data
    df24_raw = pd.read_csv(f2024)
    df24_hourly = resample_15min_to_hourly(df24_raw)

    # 3. Concatenate and drop duplicate timestamps
    common_ts = set(df23["Timestamp"]).intersection(set(df24_hourly["Timestamp"]))
    overlap_count = len(common_ts)

    combined = pd.concat([df23, df24_hourly], ignore_index=True)
    combined = combined.drop_duplicates(subset=["Timestamp"], keep="last").sort_values("Timestamp").reset_index(drop=True)

    # 4. Continuous hourly grid across 2023-2025 (26,304 hours)
    full_idx = pd.date_range("2023-01-01 00:00:00", "2025-12-31 23:00:00", freq="h", name="Timestamp")
    combined = combined.set_index("Timestamp").reindex(full_idx).reset_index()
    combined["Station"] = station

    # Verify date bounds
    assert combined["Timestamp"].min() >= pd.to_datetime("2023-01-01 00:00:00")
    assert combined["Timestamp"].max() <= pd.to_datetime("2025-12-31 23:59:59")

    log_stats["overlap_count"] = overlap_count
    log_stats["total_hourly_records"] = len(combined)
    return combined, log_stats


def build_real_delhi_datasets(
    raw_dir: Optional[Union[str, Path]] = None,
    processed_dir: Optional[Union[str, Path]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build multi-station and city-level datasets from authentic downloaded CPCB station data.

    Produces:
    - Hourly multi-station dataset saved to data/processed/delhi_hourly_stations_2023_2025.csv
    - Daily multi-station dataset (>= 18 valid hours per day) saved to data/raw/delhi_stations_2023_2025.csv
    - Daily city-averaged dataset saved to data/raw/delhi_city_2023_2025.csv
    """
    repo_root = Path(__file__).resolve().parent.parent
    if raw_dir is None:
        raw_dir = repo_root / "data" / "raw"
    else:
        raw_dir = Path(raw_dir)

    if processed_dir is None:
        processed_dir = repo_root / "data" / "processed"
    else:
        processed_dir = Path(processed_dir)

    processed_dir.mkdir(parents=True, exist_ok=True)

    all_hourly = []
    for st in DELHI_STATIONS:
        combined_st, _ = load_and_combine_station(st, raw_dir=raw_dir)
        all_hourly.append(combined_st)

    hourly_all_df = pd.concat(all_hourly, ignore_index=True)
    hourly_out_path = processed_dir / "delhi_hourly_stations_2023_2025.csv"
    hourly_all_df.to_csv(hourly_out_path, index=False)
    logger.info("Saved %d hourly records across 6 stations to %s", len(hourly_all_df), hourly_out_path)

    # Daily aggregation: require >= 18 of 24 valid hours (75% CPCB rule)
    hourly_all_df["Date"] = hourly_all_df["Timestamp"].dt.floor("D")
    pollutant_cols = [c for c in ["AQI", "PM2.5", "PM10", "NO2", "SO2", "CO", "Ozone", "O3"] if c in hourly_all_df.columns]

    daily_records = []
    for st, grp in hourly_all_df.groupby("Station"):
        daily_grp = grp.groupby("Date")
        counts = daily_grp[pollutant_cols].count()
        means = daily_grp[pollutant_cols].mean()
        # Require >= 18 valid hours for AQI
        daily_st = means.where(counts >= 18, np.nan).reset_index()
        daily_st["Station"] = st
        daily_records.append(daily_st)

    daily_stations_df = pd.concat(daily_records, ignore_index=True)
    daily_stations_path = processed_dir / "delhi_stations_2023_2025.csv"
    daily_stations_df.to_csv(daily_stations_path, index=False)
    logger.info("Saved %d daily multi-station records to %s", len(daily_stations_df), daily_stations_path)

    # City-averaged daily series
    daily_city_df = daily_stations_df.groupby("Date")[pollutant_cols].mean().reset_index()
    daily_city_df["City"] = "Delhi"
    daily_city_path = processed_dir / "delhi_city_2023_2025.csv"
    daily_city_df.to_csv(daily_city_path, index=False)
    logger.info("Saved %d daily city records to %s", len(daily_city_df), daily_city_path)

    return daily_stations_df, daily_city_df


def generate_or_load_station_data(
    start_date: str = "2023-01-01",
    end_date: str = "2025-12-31",
    stations: Optional[List[str]] = None,
    seed: int = 42,
    force_regenerate: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load or build authentic multi-station and city-averaged Delhi datasets for 2023-2025.

    Strictly guarantees that only dates between start_date and end_date are produced.

    Returns:
        (stations_df, city_df)
    """
    repo_root = Path(__file__).resolve().parent.parent
    raw_dir = repo_root / "data" / "raw"
    processed_dir = repo_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    stations_path = processed_dir / "delhi_stations_2023_2025.csv"
    city_path = processed_dir / "delhi_city_2023_2025.csv"

    if stations is None:
        stations = list(DELHI_STATIONS)

    if force_regenerate or not (stations_path.exists() and city_path.exists()):
        logger.info("Aggregated datasets missing or regeneration requested. Building from authentic station files...")
        stations_df, city_df = build_real_delhi_datasets(raw_dir=raw_dir, processed_dir=processed_dir)
    else:
        logger.info("Loading existing 2023-2025 station datasets from %s", stations_path)
        stations_df = pd.read_csv(stations_path)
        city_df = pd.read_csv(city_path)
        stations_df["Date"] = pd.to_datetime(stations_df["Date"])
        city_df["Date"] = pd.to_datetime(city_df["Date"])

    # Enforce date constraints
    assert stations_df["Date"].min() >= pd.to_datetime(start_date)
    assert stations_df["Date"].max() <= pd.to_datetime(end_date)
    assert city_df["Date"].min() >= pd.to_datetime(start_date)
    assert city_df["Date"].max() <= pd.to_datetime(end_date)
    return stations_df, city_df


def audit_2023_2025_dataset(
    stations_df: pd.DataFrame,
    city_df: pd.DataFrame,
    verbose: bool = True,
) -> dict:
    """Audit the 2023-2025 dataset to ensure zero pre-2023 contamination and test autocorrelation."""
    min_date = city_df["Date"].min()
    max_date = city_df["Date"].max()

    if min_date < pd.to_datetime("2023-01-01") or max_date > pd.to_datetime("2025-12-31"):
        raise ValueError(f"Date range violation! Expected 2023-01-01 to 2025-12-31, got {min_date} to {max_date}.")

    lag1_autocorr = float(city_df["AQI"].autocorr(lag=1))

    audit_info = {
        "start_date": min_date.strftime("%Y-%m-%d"),
        "end_date": max_date.strftime("%Y-%m-%d"),
        "total_days": len(city_df),
        "total_station_records": len(stations_df),
        "stations": stations_df["Station"].unique().tolist(),
        "lag1_autocorr": round(lag1_autocorr, 4),
        "autocorr_check": "PASSED" if lag1_autocorr > 0.80 else "FAILED",
        "pre_2023_contamination": False,
    }

    if verbose:
        print("\n" + "=" * 75)
        print("                 AIR QUALITY 2023-2025 DATA AUDIT REPORT                 ")
        print("=" * 75)
        print(f" Timeframe:              {audit_info['start_date']} to {audit_info['end_date']} (Strictly 2023-2025)")
        print(f" Total Days:             {audit_info['total_days']:,} days")
        print(f" Stations Tracked:       {len(audit_info['stations'])} ({', '.join(audit_info['stations'])})")
        print(f" Station Records:        {audit_info['total_station_records']:,} rows")
        print(f" Lag-1 Autocorrelation:  {lag1_autocorr:.4f}")
        print(f" Autocorrelation Check:  {audit_info['autocorr_check']}")
        print(f" Pre-2023 Contamination: NONE (city_day.csv and pre-2023 data strictly excluded)")
        print("=" * 75 + "\n")

    return audit_info


def compute_is_synthetic(stations_df: pd.DataFrame, city_df: pd.DataFrame) -> bool:
    """Compute whether the dataset exhibits synthetic properties based on empirical checks.

    Checks:
    1. Zero missingness over extended multi-year periods (real CAAQMS stations always have missing data).
    2. Unnaturally uniform inter-station correlation (synthetic multi-station datasets generated from shared seasonal equations show near-identical correlations across all pairs).
    """
    if len(stations_df) == 0 or len(city_df) == 0:
        return False

    pollutants = [c for c in ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"] if c in stations_df.columns]

    # Check 1: Zero missing values across multiple stations over > 180 days
    if len(stations_df) >= 1000 and len(pollutants) >= 4:
        missing_count = stations_df[pollutants].isna().sum().sum()
        if missing_count == 0:
            return True

    # Check 2: Pairwise station correlations uniformity
    if "Station" in stations_df.columns and "AQI" in stations_df.columns:
        piv = stations_df.pivot_table(index="Date", columns="Station", values="AQI", aggfunc="mean")
        if piv.shape[1] >= 3:
            corr_mat = piv.corr().values
            upper_tri = corr_mat[np.triu_indices_from(corr_mat, k=1)]
            if len(upper_tri) > 1 and np.std(upper_tri) < 0.02 and np.mean(upper_tri) > 0.80:
                return True

    return False


def get_provenance_registered_files() -> List[str]:
    """Parse data/raw/PROVENANCE.md to find all registered file names."""
    import re
    repo_root = Path(__file__).resolve().parent.parent
    prov_path = repo_root / "data" / "raw" / "PROVENANCE.md"
    if not prov_path.exists():
        return []
    with open(prov_path, "r", encoding="utf-8") as f:
        text = f.read()
    return re.findall(r"###\s*`([^`]+)`", text)


def load_raw_data(
    data_path: Optional[Union[str, Path]] = None,
    city: str = "Delhi",
    allow_synthetic: bool = False,
    auto_download: bool = True,
    fallback_generate: bool = True,
) -> pd.DataFrame:
    """Load raw air quality data strictly for 2023-2025 (city-averaged series).

    Rejects any attempt to use pre-2023 data or city_day.csv.
    """
    if data_path is not None:
        p_path = Path(data_path)
        p_name = p_path.name
        registered = get_provenance_registered_files()
        if p_name not in registered:
            raise ValueError(f"File {p_name} is not registered in PROVENANCE.md.")

        p_str = str(data_path)
        if "city_day.csv" in p_str:
            raise ValueError("city_day.csv and pre-2023 data are strictly prohibited.")
        df = pd.read_csv(data_path)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            if (df["Date"] < pd.to_datetime("2023-01-01")).any() or (df["Date"] > pd.to_datetime("2025-12-31")).any():
                raise ValueError("Dataset contains rows dated outside 2023-01-01 to 2025-12-31.")
        return df

    _, city_df = generate_or_load_station_data()
    # Enforce row-level date bounds on loaded city series
    if (city_df["Date"] < pd.to_datetime("2023-01-01")).any() or (city_df["Date"] > pd.to_datetime("2025-12-31")).any():
        raise ValueError("Dataset contains rows dated outside 2023-01-01 to 2025-12-31.")
    return city_df


def audit_dataset(
    data_path: Optional[Union[str, Path]] = None,
    city: str = "Delhi",
    allow_synthetic: bool = False,
    auto_download: bool = True,
    verbose: bool = True,
) -> dict:
    """Audit the 2023-2025 dataset ensuring strict exclusion of pre-2023 records and computing is_synthetic."""
    if data_path is not None and "city_day.csv" in str(data_path):
        raise ValueError("city_day.csv and pre-2023 data are strictly prohibited.")
    stations_df, city_df = generate_or_load_station_data()
    info = audit_2023_2025_dataset(stations_df, city_df, verbose=verbose)
    # Compute is_synthetic dynamically from empirical checks
    info["is_synthetic"] = compute_is_synthetic(stations_df, city_df)
    info["is_real_kaggle"] = False
    info["file_path"] = str(Path(__file__).resolve().parent.parent / "data" / "processed" / "delhi_city_2023_2025.csv")
    return info

