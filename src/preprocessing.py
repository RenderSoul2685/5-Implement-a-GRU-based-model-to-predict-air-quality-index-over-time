"""Data preprocessing, date-based chronological splitting, feature engineering, and window generation."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
from torch.utils.data import DataLoader, Dataset

from src.config import Config, DELHI_STATIONS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def prepare_timeseries(
    df: pd.DataFrame,
    pollutant_cols: List[str],
    target_col: str = "AQI",
    weather_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Parse dates, sort chronologically, interpolate any missing values, and add cyclical calendar features.

    Strictly operates within 2023-2025 timeframe.
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)
    df = df.set_index("Date")

    # Continuous daily indexing
    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
    df = df.reindex(full_idx)
    df.index.name = "Date"

    all_numeric = list(dict.fromkeys(pollutant_cols + [target_col] + (weather_cols or [])))
    for col in all_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            raise KeyError(f"Required column '{col}' missing from DataFrame.")

    # Time-based interpolation
    df[all_numeric] = df[all_numeric].interpolate(method="time").bfill().ffill()

    # Cyclical temporal features
    months = df.index.month.values
    days_of_week = df.index.dayofweek.values

    df["month_sin"] = np.sin(2.0 * np.pi * months / 12.0)
    df["month_cos"] = np.cos(2.0 * np.pi * months / 12.0)
    df["dayofweek_sin"] = np.sin(2.0 * np.pi * days_of_week / 7.0)
    df["dayofweek_cos"] = np.cos(2.0 * np.pi * days_of_week / 7.0)

    # Fallback to zero if any column was entirely NaN
    if df[all_numeric].isna().any().any():
        df[all_numeric] = df[all_numeric].fillna(0.0)

    return df


# Compatibility alias
clean_and_prepare_timeseries = prepare_timeseries


def split_chronological(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronologically split DataFrame into train, val, and test splits."""
    min_date = df.index.min() if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df["Date"]).min()
    max_date = df.index.max() if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df["Date"]).max()
    if min_date >= pd.to_datetime("2023-01-01") and max_date >= pd.to_datetime("2025-10-01"):
        return split_by_date(df)
    total_len = len(df)
    train_end = int(total_len * train_ratio)
    val_end = int(total_len * (train_ratio + val_ratio))
    return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()


def split_by_date(
    df: pd.DataFrame,
    train_end: str = "2025-06-30",
    val_start: str = "2025-07-01",
    val_end: str = "2025-09-30",
    test_start: str = "2025-10-01",
    test_end: str = "2025-12-31",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into Train, Validation, and Test splits based on strict date cutoffs.

    Train: 2023-01-01 to 2025-06-30
    Val:   2025-07-01 to 2025-09-30
    Test:  2025-10-01 to 2025-12-31
    """
    df_reset = df.reset_index() if isinstance(df.index, pd.DatetimeIndex) else df.copy()
    dates = pd.to_datetime(df_reset["Date"])

    train_mask = dates <= pd.to_datetime(train_end)
    val_mask = (dates >= pd.to_datetime(val_start)) & (dates <= pd.to_datetime(val_end))
    test_mask = (dates >= pd.to_datetime(test_start)) & (dates <= pd.to_datetime(test_end))

    train_df = df_reset[train_mask].set_index("Date") if "Date" in df_reset.columns else df_reset[train_mask]
    val_df = df_reset[val_mask].set_index("Date") if "Date" in df_reset.columns else df_reset[val_mask]
    test_df = df_reset[test_mask].set_index("Date") if "Date" in df_reset.columns else df_reset[test_mask]

    logger.info(
        "Date-based Split -> Train: %d rows (to %s) | Val: %d rows (%s to %s) | Test: %d rows (%s to %s)",
        len(train_df),
        train_end,
        len(val_df),
        val_start,
        val_end,
        len(test_df),
        test_start,
        test_end,
    )
    return train_df, val_df, test_df


def fit_and_scale_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "AQI",
    save_dir: Optional[Union[str, Path]] = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    MinMaxScaler,
    MinMaxScaler,
]:
    """Fit MinMaxScaler on training split ONLY to prevent leakage, then transform all splits."""
    feature_scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    target_scaler = MinMaxScaler(feature_range=(0.0, 1.0))

    X_train_scaled = feature_scaler.fit_transform(train_df[feature_cols].values)
    y_train_scaled = target_scaler.fit_transform(train_df[[target_col]].values)

    X_val_scaled = feature_scaler.transform(val_df[feature_cols].values)
    y_val_scaled = target_scaler.transform(val_df[[target_col]].values)

    X_test_scaled = feature_scaler.transform(test_df[feature_cols].values)
    y_test_scaled = target_scaler.transform(test_df[[target_col]].values)

    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(feature_scaler, save_path / "feature_scaler.joblib")
        joblib.dump(target_scaler, save_path / "target_scaler.joblib")
        logger.info("Saved fitted scalers to %s", save_path)

    return (
        X_train_scaled,
        y_train_scaled,
        X_val_scaled,
        y_val_scaled,
        X_test_scaled,
        y_test_scaled,
        feature_scaler,
        target_scaler,
    )


def create_sliding_windows(
    data_features: np.ndarray,
    data_targets: np.ndarray,
    lookback: int = 14,
    horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate sliding window sequences and multi-step targets.

    X[i] = data_features[i : i + lookback]
    y[i] = data_targets[i + lookback : i + lookback + horizon]
    """
    if len(data_targets.shape) == 1:
        data_targets = data_targets.reshape(-1, 1)

    n_samples = len(data_features) - lookback - horizon + 1
    if n_samples <= 0:
        raise ValueError(
            f"Not enough rows ({len(data_features)}) for lookback={lookback} and horizon={horizon}."
        )

    num_features = data_features.shape[1]
    X = np.empty((n_samples, lookback, num_features), dtype=np.float32)
    y = np.empty((n_samples, horizon), dtype=np.float32)

    for i in range(n_samples):
        X[i] = data_features[i : i + lookback]
        y[i] = data_targets[i + lookback : i + lookback + horizon].ravel()

    return X, y


def create_residual_targets(
    X_windows: np.ndarray,
    y_windows: np.ndarray,
    target_feature_idx: int = 6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute residual targets: delta = y[t+h] - AQI[t].

    Args:
        X_windows: Array of shape (N, lookback, num_features).
        y_windows: Array of shape (N, horizon).
        target_feature_idx: Index of AQI column inside feature set.

    Returns:
        (y_residual, y_last_step)
    """
    y_last = X_windows[:, -1, target_feature_idx : target_feature_idx + 1]  # shape (N, 1)
    if y_windows.ndim == 1:
        y_windows = y_windows[:, np.newaxis]
    y_residual = y_windows - y_last
    return y_residual, y_last


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset wrapping sliding window tensors, with optional station IDs."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        station_ids: Optional[np.ndarray] = None,
    ):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        if station_ids is not None:
            self.station_ids = torch.as_tensor(station_ids, dtype=torch.long)
        else:
            self.station_ids = None

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        if self.station_ids is not None:
            return self.X[idx], self.station_ids[idx], self.y[idx]
        return self.X[idx], self.y[idx]


def get_data_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 32,
    station_ids_train: Optional[np.ndarray] = None,
    station_ids_val: Optional[np.ndarray] = None,
    station_ids_test: Optional[np.ndarray] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyTorch DataLoaders."""
    train_dataset = TimeSeriesDataset(X_train, y_train, station_ids_train)
    val_dataset = TimeSeriesDataset(X_val, y_val, station_ids_val)
    test_dataset = TimeSeriesDataset(X_test, y_test, station_ids_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def create_expanding_walk_forward_folds(
    df: pd.DataFrame,
    n_folds: int = 4,
    min_train_days: int = 400,
    val_days: int = 92,
) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Generate 4-fold expanding-window rolling-origin walk-forward CV folds across 2023-2025.

    Guarantees strict temporal order: Train < Val < Test.
    """
    total_len = len(df)
    remaining_days = total_len - min_train_days - val_days
    test_days = remaining_days // n_folds

    folds = []
    for k in range(n_folds):
        train_end_idx = min_train_days + k * test_days
        val_end_idx = train_end_idx + val_days
        test_end_idx = val_end_idx + test_days if k < n_folds - 1 else total_len

        train_fold = df.iloc[:train_end_idx].copy()
        val_fold = df.iloc[train_end_idx:val_end_idx].copy()
        test_fold = df.iloc[val_end_idx:test_end_idx].copy()

        folds.append((train_fold, val_fold, test_fold))
        logger.info(
            "Expanding Walk-Forward Fold %d/%d -> Train: %d days | Val: %d days | Test: %d days",
            k + 1,
            n_folds,
            len(train_fold),
            len(val_fold),
            len(test_fold),
        )

    return folds


create_walk_forward_folds = create_expanding_walk_forward_folds


def prepare_hourly_city_dataframe(
    df_hourly_stations: pd.DataFrame,
    df_hourly_weather: pd.DataFrame,
    min_stations_required: int = 4,
    interp_limit_hours: int = 6,
) -> pd.DataFrame:
    """Prepare city-level hourly DataFrame with weather, calendar features, and <= 6h interpolation.

    Per CPCB / project specifications:
    - City series is arithmetic mean across stations, valid only when >= 4 of 6 stations report.
    - Interpolates missing AQI gaps up to 6 hours; longer gaps remain NaN.
    - Tracks raw observed mask so evaluation strictly evaluates only on genuinely observed targets.
    """
    piv = df_hourly_stations.pivot(index="Timestamp", columns="Station", values="AQI")
    valid_counts = piv.notna().sum(axis=1)
    city_raw = piv.mean(axis=1).where(valid_counts >= min_stations_required, np.nan)

    df_city = pd.DataFrame(
        {
            "Timestamp": pd.to_datetime(piv.index.values),
            "AQI": city_raw.values,
            "observed_raw": city_raw.notna().values,
        }
    )

    # Merge weather
    df_weather_clean = df_hourly_weather.copy()
    df_weather_clean["Timestamp"] = pd.to_datetime(df_weather_clean["Timestamp"])
    df_city = df_city.merge(df_weather_clean, on="Timestamp", how="left")

    # Cyclical calendar features
    hours = df_city["Timestamp"].dt.hour.values
    dows = df_city["Timestamp"].dt.dayofweek.values
    months = df_city["Timestamp"].dt.month.values

    df_city["hour_sin"] = np.sin(2.0 * np.pi * hours / 24.0)
    df_city["hour_cos"] = np.cos(2.0 * np.pi * hours / 24.0)
    df_city["dayofweek_sin"] = np.sin(2.0 * np.pi * dows / 7.0)
    df_city["dayofweek_cos"] = np.cos(2.0 * np.pi * dows / 7.0)
    df_city["month_sin"] = np.sin(2.0 * np.pi * months / 12.0)
    df_city["month_cos"] = np.cos(2.0 * np.pi * months / 12.0)

    # Time-weighted interpolation up to 6 hours
    df_city = df_city.set_index("Timestamp")
    df_city["AQI_interp"] = df_city["AQI"].interpolate(method="time", limit=interp_limit_hours)
    # Ensure weather has no isolated missing hours
    weather_cols = ["temperature", "humidity", "wind_speed", "wind_direction", "pressure"]
    avail_w = [c for c in weather_cols if c in df_city.columns]
    if avail_w:
        df_city[avail_w] = df_city[avail_w].interpolate(method="time").bfill().ffill()
    df_city = df_city.reset_index()

    return df_city


def prepare_hourly_station_dataframe(
    df_hourly_stations: pd.DataFrame,
    station: str,
    df_hourly_weather: pd.DataFrame,
    interp_limit_hours: int = 6,
) -> pd.DataFrame:
    """Prepare a single station's hourly DataFrame with weather, calendar features, and <= 6h interpolation."""
    df_st = df_hourly_stations[df_hourly_stations["Station"] == station].copy()
    df_st["Timestamp"] = pd.to_datetime(df_st["Timestamp"])
    df_st = df_st.sort_values("Timestamp").reset_index(drop=True)

    df_st["observed_raw"] = df_st["AQI"].notna().values

    # Merge weather
    df_weather_clean = df_hourly_weather.copy()
    df_weather_clean["Timestamp"] = pd.to_datetime(df_weather_clean["Timestamp"])
    df_st = df_st.merge(df_weather_clean, on="Timestamp", how="left")

    hours = df_st["Timestamp"].dt.hour.values
    dows = df_st["Timestamp"].dt.dayofweek.values
    months = df_st["Timestamp"].dt.month.values

    df_st["hour_sin"] = np.sin(2.0 * np.pi * hours / 24.0)
    df_st["hour_cos"] = np.cos(2.0 * np.pi * hours / 24.0)
    df_st["dayofweek_sin"] = np.sin(2.0 * np.pi * dows / 7.0)
    df_st["dayofweek_cos"] = np.cos(2.0 * np.pi * dows / 7.0)
    df_st["month_sin"] = np.sin(2.0 * np.pi * months / 12.0)
    df_st["month_cos"] = np.cos(2.0 * np.pi * months / 12.0)

    df_st = df_st.set_index("Timestamp")
    df_st["AQI_interp"] = df_st["AQI"].interpolate(method="time", limit=interp_limit_hours)
    weather_cols = ["temperature", "humidity", "wind_speed", "wind_direction", "pressure"]
    avail_w = [c for c in weather_cols if c in df_st.columns]
    if avail_w:
        df_st[avail_w] = df_st[avail_w].interpolate(method="time").bfill().ffill()
    df_st = df_st.reset_index()

    return df_st


def create_hourly_windows(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "AQI_interp",
    raw_col: str = "AQI",
    lookback: int = 48,
    horizon: int = 24,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate 48-hour lookback and 24-hour horizon sequences.

    Drops any window where input features or target window contain NaN values.
    Returns:
        (X, y, y_observed_mask, last_known_aqi, forecast_start_timestamps)
    """
    vals = df[feature_cols].values.astype(np.float32)
    target_vals = df[target_col].values.astype(np.float32)
    raw_vals = df[raw_col].values.astype(np.float32)
    ts_vals = pd.to_datetime(df["Timestamp"]).values

    total_len = len(df)
    n_possible = total_len - lookback - horizon + 1
    if n_possible <= 0:
        raise ValueError(f"Not enough rows ({total_len}) for lookback={lookback} and horizon={horizon}")

    X_list, y_list, y_mask_list, last_list, ts_list = [], [], [], [], []

    for i in range(n_possible):
        x_win = vals[i : i + lookback]
        y_win = target_vals[i + lookback : i + lookback + horizon]
        raw_win = raw_vals[i + lookback : i + lookback + horizon]

        # Drop any window with missing inputs or missing targets
        if np.isnan(x_win).any() or np.isnan(y_win).any():
            continue

        last_known = x_win[-1, 0]  # AQI is the first feature in feature_cols
        X_list.append(x_win)
        y_list.append(y_win)
        y_mask_list.append(~np.isnan(raw_win))  # True strictly on observed points
        last_list.append(last_known)
        ts_list.append(ts_vals[i + lookback])

    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_list, dtype=np.float32),
        np.array(y_mask_list, dtype=bool),
        np.array(last_list, dtype=np.float32),
        np.array(ts_list),
    )


def split_hourly_windows_by_date(
    X: np.ndarray,
    y: np.ndarray,
    y_mask: np.ndarray,
    last_known: np.ndarray,
    forecast_start_ts: np.ndarray,
    train_end: str = "2025-06-30 23:00:00",
    val_start: str = "2025-07-01 00:00:00",
    val_end: str = "2025-09-30 23:00:00",
    test_start: str = "2025-10-01 00:00:00",
    test_end: str = "2025-12-31 23:00:00",
    horizon_hours: int = 24,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Chronologically partition hourly sliding windows into Train, Validation, and Test sets.

    Ensures zero target leakage across splits:
    - Train: forecast start + horizon <= train_end
    - Val:   forecast start >= val_start and forecast start + horizon <= val_end
    - Test:  forecast start >= test_start and forecast start + horizon <= test_end
    """
    ts_series = pd.to_datetime(forecast_start_ts)
    target_end_ts = ts_series + pd.to_timedelta(horizon_hours - 1, unit="h")

    t_end = pd.to_datetime(train_end)
    v_start = pd.to_datetime(val_start)
    v_end = pd.to_datetime(val_end)
    te_start = pd.to_datetime(test_start)
    te_end = pd.to_datetime(test_end)

    tr_idx = np.where(target_end_ts <= t_end)[0]
    val_idx = np.where((ts_series >= v_start) & (target_end_ts <= v_end))[0]
    te_idx = np.where((ts_series >= te_start) & (target_end_ts <= te_end))[0]

    splits = {
        "train": {
            "X": X[tr_idx],
            "y": y[tr_idx],
            "y_mask": y_mask[tr_idx],
            "last_known": last_known[tr_idx],
            "ts": forecast_start_ts[tr_idx],
        },
        "val": {
            "X": X[val_idx],
            "y": y[val_idx],
            "y_mask": y_mask[val_idx],
            "last_known": last_known[val_idx],
            "ts": forecast_start_ts[val_idx],
        },
        "test": {
            "X": X[te_idx],
            "y": y[te_idx],
            "y_mask": y_mask[te_idx],
            "last_known": last_known[te_idx],
            "ts": forecast_start_ts[te_idx],
        },
    }

    logger.info(
        "Hourly Window Splits -> Train: %d windows | Val: %d windows | Test: %d windows",
        len(tr_idx),
        len(val_idx),
        len(te_idx),
    )
    return splits


def prepare_daily_dataframe(
    df_daily: pd.DataFrame,
    df_weather: pd.DataFrame,
    interp_limit_days: int = 2,
    start_date: Optional[Union[str, pd.Timestamp]] = None,
    end_date: Optional[Union[str, pd.Timestamp]] = None,
) -> pd.DataFrame:
    """Prepare a daily AQI DataFrame with weather, cyclical calendar features, and <= 2-day interpolation.

    Preserves raw observed target mask so scoring evaluates strictly on observed days.
    """
    df = df_daily.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)

    # Reindex to full continuous daily calendar
    if start_date is not None and end_date is not None:
        full_idx = pd.date_range(start_date, end_date, freq="D")
    elif not df.empty and df["Date"].max() > pd.to_datetime("2025-12-31"):
        full_idx = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")
    else:
        full_idx = pd.date_range("2023-01-01", "2025-12-31", freq="D")
    df = df.set_index("Date").reindex(full_idx)
    df.index.name = "Date"
    df = df.reset_index()

    # Track raw observed ground truth
    df["AQI_observed"] = df["AQI"].copy()
    df["is_observed"] = df["AQI"].notna().values

    # Merge daily weather
    df_w = df_weather.copy()
    df_w["Date"] = pd.to_datetime(df_w["Date"])
    weather_cols = ["temperature", "humidity", "wind_speed", "pressure"]
    df = df.merge(df_w[["Date"] + [c for c in weather_cols if c in df_w.columns]], on="Date", how="left")

    # Cyclical calendar features
    dows = df["Date"].dt.dayofweek.values
    months = df["Date"].dt.month.values

    df["dayofweek_sin"] = np.sin(2.0 * np.pi * dows / 7.0)
    df["dayofweek_cos"] = np.cos(2.0 * np.pi * dows / 7.0)
    df["month_sin"] = np.sin(2.0 * np.pi * months / 12.0)
    df["month_cos"] = np.cos(2.0 * np.pi * months / 12.0)

    # Linear time-based interpolation up to interp_limit_days (2 days)
    df = df.set_index("Date")
    df["AQI_interp"] = df["AQI"].interpolate(method="time", limit=interp_limit_days)

    # Weather interpolation if any missing
    for c in weather_cols:
        if c in df.columns:
            df[c] = df[c].interpolate(method="time").bfill().ffill()

    df = df.reset_index()
    return df


def create_daily_windows(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "AQI_interp",
    raw_col: str = "AQI_observed",
    lookback: int = 14,
    horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate 14-day lookback and 1-day horizon next-day sequences.

    Drops any window where input features contain NaN (gaps > 2 days) or target is NaN.
    Returns:
        (X, y, y_observed_mask, last_known_aqi, target_dates)
    """
    vals = df[feature_cols].values.astype(np.float32)
    target_vals = df[target_col].values.astype(np.float32)
    raw_vals = df[raw_col].values.astype(np.float32)
    date_vals = pd.to_datetime(df["Date"]).values

    total_len = len(df)
    n_possible = total_len - lookback - horizon + 1
    if n_possible <= 0:
        raise ValueError(f"Not enough rows ({total_len}) for lookback={lookback} and horizon={horizon}")

    X_list, y_list, y_mask_list, last_list, dt_list = [], [], [], [], []

    for i in range(n_possible):
        x_win = vals[i : i + lookback]
        y_win = target_vals[i + lookback : i + lookback + horizon]
        raw_win = raw_vals[i + lookback : i + lookback + horizon]

        if np.isnan(x_win).any() or np.isnan(y_win).any():
            continue

        last_known = x_win[-1, 0]  # AQI_interp is the first feature
        X_list.append(x_win)
        y_list.append(y_win if horizon > 1 else y_win[0])
        y_mask_list.append(~np.isnan(raw_win) if horizon > 1 else ~np.isnan(raw_win[0]))
        last_list.append(last_known)
        dt_list.append(date_vals[i + lookback])

    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_list, dtype=np.float32),
        np.array(y_mask_list, dtype=bool),
        np.array(last_list, dtype=np.float32),
        np.array(dt_list),
    )


def split_daily_windows_by_date(
    X: np.ndarray,
    y: np.ndarray,
    y_mask: np.ndarray,
    last_known: np.ndarray,
    target_dates: np.ndarray,
    train_end: str = "2025-06-30",
    val_start: str = "2025-07-01",
    val_end: str = "2025-09-30",
    test_start: str = "2025-10-01",
    test_end: str = "2025-12-31",
) -> Dict[str, Dict[str, np.ndarray]]:
    """Chronologically partition daily sliding windows into Train, Validation, and Test sets by target date."""
    target_dt = pd.to_datetime(target_dates)
    t_end = pd.to_datetime(train_end)
    v_start = pd.to_datetime(val_start)
    v_end = pd.to_datetime(val_end)
    te_start = pd.to_datetime(test_start)
    te_end = pd.to_datetime(test_end)

    tr_idx = np.where(target_dt <= t_end)[0]
    val_idx = np.where((target_dt >= v_start) & (target_dt <= v_end))[0]
    te_idx = np.where((target_dt >= te_start) & (target_dt <= te_end))[0]

    splits = {
        "train": {
            "X": X[tr_idx],
            "y": y[tr_idx],
            "y_mask": y_mask[tr_idx],
            "last_known": last_known[tr_idx],
            "target_dates": target_dates[tr_idx],
        },
        "val": {
            "X": X[val_idx],
            "y": y[val_idx],
            "y_mask": y_mask[val_idx],
            "last_known": last_known[val_idx],
            "target_dates": target_dates[val_idx],
        },
        "test": {
            "X": X[te_idx],
            "y": y[te_idx],
            "y_mask": y_mask[te_idx],
            "last_known": last_known[te_idx],
            "target_dates": target_dates[te_idx],
        },
    }
    logger.info(
        "Daily Window Splits -> Train: %d windows | Val: %d windows | Test: %d windows",
        len(tr_idx),
        len(val_idx),
        len(te_idx),
    )
    return splits



