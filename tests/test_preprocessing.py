"""Unit tests for time-series preprocessing, windowing, date-based splitting, and leak-free scaling."""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler

from src.preprocessing import (
    clean_and_prepare_timeseries,
    create_expanding_walk_forward_folds,
    create_residual_targets,
    create_sliding_windows,
    fit_and_scale_splits,
    prepare_timeseries,
    split_by_date,
    split_chronological,
)


@pytest.fixture
def sample_raw_data() -> pd.DataFrame:
    """Fixture providing raw 2023-2025 air quality data with intentional missing values."""
    dates = pd.date_range("2023-01-01", "2023-06-30", freq="D")
    rng = np.random.RandomState(123)
    df = pd.DataFrame(
        {
            "City": "Delhi",
            "Date": dates,
            "PM2.5": rng.uniform(20, 200, len(dates)),
            "PM10": rng.uniform(50, 300, len(dates)),
            "NO2": rng.uniform(10, 80, len(dates)),
            "SO2": rng.uniform(5, 30, len(dates)),
            "CO": rng.uniform(0.5, 3.0, len(dates)),
            "O3": rng.uniform(10, 60, len(dates)),
            "AQI": rng.uniform(50, 350, len(dates)),
        }
    )
    # Inject 10% intentional NaNs to test interpolation
    nan_mask = rng.rand(len(df)) < 0.10
    df.loc[nan_mask, "PM2.5"] = np.nan
    return df


def test_clean_and_prepare_no_nans(sample_raw_data):
    """Verify that cleaning eliminates all NaNs via interpolation and fill."""
    pollutants = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
    cleaned_df = prepare_timeseries(sample_raw_data, pollutant_cols=pollutants, target_col="AQI")

    assert not cleaned_df.isna().any().any(), "Cleaned dataset should not contain any NaN values."
    assert cleaned_df.index.is_monotonic_increasing, "Cleaned dataset index must be strictly monotonic increasing."


def test_cyclical_features_bounds(sample_raw_data):
    """Verify that sine and cosine temporal features remain bounded in [-1, 1]."""
    pollutants = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
    cleaned_df = prepare_timeseries(sample_raw_data, pollutant_cols=pollutants, target_col="AQI")

    for col in ["month_sin", "month_cos", "dayofweek_sin", "dayofweek_cos"]:
        assert col in cleaned_df.columns, f"Missing cyclical feature: {col}"
        assert cleaned_df[col].min() >= -1.0 - 1e-5, f"{col} has values below -1"
        assert cleaned_df[col].max() <= 1.0 + 1e-5, f"{col} has values above 1"


def test_split_by_date_preserves_strict_cutoffs():
    """Verify date-based split adheres strictly to train, val, and test cutoff boundaries."""
    dates = pd.date_range("2023-01-01", "2025-12-31", freq="D")
    df = pd.DataFrame(
        {"Date": dates, "AQI": np.arange(len(dates), dtype=np.float32)}
    ).set_index("Date")

    train_df, val_df, test_df = split_by_date(
        df,
        train_end="2025-06-30",
        val_start="2025-07-01",
        val_end="2025-09-30",
        test_start="2025-10-01",
        test_end="2025-12-31",
    )

    assert len(train_df) == 912  # 2023-01-01 to 2025-06-30 (365 + 366 + 181 = 912)
    assert len(val_df) == 92    # 2025-07-01 to 2025-09-30 (31 + 31 + 30 = 92)
    assert len(test_df) == 92   # 2025-10-01 to 2025-12-31 (31 + 30 + 31 = 92)
    assert train_df.index.max() == pd.Timestamp("2025-06-30")
    assert val_df.index.min() == pd.Timestamp("2025-07-01")
    assert val_df.index.max() == pd.Timestamp("2025-09-30")
    assert test_df.index.min() == pd.Timestamp("2025-10-01")
    assert test_df.index.max() == pd.Timestamp("2025-12-31")


def test_sliding_window_shapes():
    """Verify output shapes of sliding window sequence generator for horizon=1 and horizon=7."""
    n_samples = 100
    n_features = 11
    data_features = np.random.randn(n_samples, n_features).astype(np.float32)
    data_targets = np.random.randn(n_samples, 1).astype(np.float32)

    # Single-step: lookback 14, horizon 1
    X_single, y_single = create_sliding_windows(data_features, data_targets, lookback=14, horizon=1)
    expected_samples = 100 - 14 - 1 + 1  # 86
    assert X_single.shape == (expected_samples, 14, n_features)
    assert y_single.shape == (expected_samples, 1)

    # Multi-step: lookback 14, horizon 7
    X_multi, y_multi = create_sliding_windows(data_features, data_targets, lookback=14, horizon=7)
    expected_multi = 100 - 14 - 7 + 1  # 80
    assert X_multi.shape == (expected_multi, 14, n_features)
    assert y_multi.shape == (expected_multi, 7)


def test_create_residual_targets():
    """Verify residual target computation: delta = y[t+h] - AQI[t]."""
    N = 10
    lookback = 14
    num_features = 11
    target_idx = 6

    X = np.random.randn(N, lookback, num_features).astype(np.float32)
    y = np.random.randn(N, 1).astype(np.float32)

    y_res, y_last = create_residual_targets(X, y, target_feature_idx=target_idx)

    expected_last = X[:, -1, target_idx : target_idx + 1]
    np.testing.assert_allclose(y_last, expected_last)
    expected_res = y - expected_last
    np.testing.assert_allclose(y_res, expected_res)


def test_expanding_walk_forward_cv_folds():
    """Verify 4-fold expanding-window cross-validation preserves strict chronological order."""
    dates = pd.date_range("2023-01-01", "2025-12-31", freq="D")
    df = pd.DataFrame({"AQI": np.random.randn(len(dates))}, index=dates)

    folds = create_expanding_walk_forward_folds(df, n_folds=4, min_train_days=400, val_days=92)
    assert len(folds) == 4

    prev_train_len = 0
    for k, (train_f, val_f, test_f) in enumerate(folds):
        assert train_f.index.max() < val_f.index.min(), f"Fold {k}: Train must precede Val"
        assert val_f.index.max() < test_f.index.min(), f"Fold {k}: Val must precede Test"
        # Expanding window check
        assert len(train_f) > prev_train_len, f"Fold {k}: Train window must expand"
        prev_train_len = len(train_f)
        assert len(val_f) == 92
        assert len(test_f) > 0


def test_no_data_leakage_in_scaling(sample_raw_data):
    """Verify that scalers are fitted solely on training split without leaking val or test stats."""
    pollutants = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
    cleaned_df = prepare_timeseries(sample_raw_data, pollutant_cols=pollutants, target_col="AQI")
    feature_cols = pollutants + ["AQI", "month_sin", "month_cos", "dayofweek_sin", "dayofweek_cos"]

    train_df, val_df, test_df = split_chronological(cleaned_df, 0.7, 0.15, 0.15)

    (
        X_train_scaled,
        y_train_scaled,
        X_val_scaled,
        y_val_scaled,
        X_test_scaled,
        y_test_scaled,
        feat_scaler,
        tgt_scaler,
    ) = fit_and_scale_splits(train_df, val_df, test_df, feature_cols=feature_cols, target_col="AQI")

    # Scaler min and max must match train_df exactly
    independent_scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    independent_scaler.fit(train_df[feature_cols].values)

    np.testing.assert_allclose(feat_scaler.data_min_, independent_scaler.data_min_)
    np.testing.assert_allclose(feat_scaler.data_max_, independent_scaler.data_max_)
    assert X_train_scaled.min() >= -1e-6
    assert X_train_scaled.max() <= 1.0 + 1e-6
