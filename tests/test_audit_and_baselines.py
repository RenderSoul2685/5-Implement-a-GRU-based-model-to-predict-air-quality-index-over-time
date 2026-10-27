"""Unit tests for baselines, ablations, quantile outputs, and 2023-2025 data audit."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.data_loader import audit_2023_2025_dataset, audit_dataset, load_raw_data
from src.model import (
    AQIAttentionGRU,
    AQIBiGRU,
    AQIGRU,
    AQILSTM,
    AQIQuantileGRU,
    AQIStationEmbeddingGRU,
    LinearBaseline,
    MovingAverageBaseline,
    PersistenceBaseline,
    PinballLoss,
    SeasonalNaiveBaseline,
    compute_interval_coverage,
    mc_dropout_predict,
)
from src.preprocessing import create_expanding_walk_forward_folds


def test_moving_average_baseline():
    """Verify 7-day MovingAverageBaseline computes mean of last 7 target observations."""
    batch_size = 6
    seq_len = 14
    num_features = 8
    target_idx = 3
    horizon = 4

    X = np.random.randn(batch_size, seq_len, num_features).astype(np.float32)
    expected_ma = np.mean(X[:, -7:, target_idx], axis=1)

    model = MovingAverageBaseline(target_feature_idx=target_idx, window_size=7, horizon=horizon)
    preds = model.predict(X)

    assert preds.shape == (batch_size, horizon)
    for h in range(horizon):
        np.testing.assert_allclose(preds[:, h], expected_ma, rtol=1e-5)


def test_seasonal_naive_baseline_cycle():
    """Verify SeasonalNaiveBaseline correctly uses cycle lag (7 days prior)."""
    batch_size = 4
    seq_len = 14
    num_features = 5
    target_idx = 2
    horizon = 3

    X = np.random.randn(batch_size, seq_len, num_features).astype(np.float32)
    expected_cycle = X[:, -7, target_idx]

    model = SeasonalNaiveBaseline(target_feature_idx=target_idx, horizon=horizon, cycle_len=7)
    preds = model.predict(X)

    assert preds.shape == (batch_size, horizon)
    for h in range(horizon):
        np.testing.assert_allclose(preds[:, h], expected_cycle)


def test_seasonal_naive_baseline_annual_lookup():
    """Verify SeasonalNaiveBaseline performs annual history lookup when history provided."""
    target_history = np.arange(1000, dtype=np.float32)
    model = SeasonalNaiveBaseline(target_feature_idx=0, horizon=2, cycle_len=7, annual_lag=365)
    model.fit(target_history)

    X_dummy = np.zeros((3, 14, 2), dtype=np.float32)
    test_indices = np.array([500, 600, 700])
    preds = model.predict(X_dummy, test_start_indices=test_indices)

    assert preds.shape == (3, 2)
    assert preds[0, 0] == 135.0
    assert preds[0, 1] == 136.0
    assert preds[1, 0] == 235.0
    assert preds[2, 0] == 335.0


def test_bigru_forward_shape():
    """Verify Bidirectional GRU output shape."""
    model = AQIBiGRU(input_size=10, hidden_size=32, num_layers=2, output_size=7)
    x = torch.randn(4, 14, 10)
    out = model(x)
    assert out.shape == (4, 7)
    assert not torch.isnan(out).any()


def test_attention_gru_forward_and_weights():
    """Verify Attention GRU output shape and softmax weights sum to 1."""
    model = AQIAttentionGRU(input_size=8, hidden_size=32, num_layers=2, output_size=1)
    x = torch.randn(5, 14, 8)
    out, weights = model(x, return_attention=True)

    assert out.shape == (5, 1)
    assert weights.shape == (5, 14, 1)
    sum_weights = torch.sum(weights, dim=1)
    np.testing.assert_allclose(sum_weights.detach().numpy(), np.ones((5, 1)), atol=1e-5)


def test_quantile_gru_and_pinball_loss():
    """Verify AQIQuantileGRU produces 3 quantiles and PinballLoss calculates loss correctly."""
    quantiles = (0.10, 0.50, 0.90)
    model = AQIQuantileGRU(input_size=6, hidden_size=32, num_layers=2, output_size=1, quantiles=quantiles)
    x = torch.randn(4, 14, 6)
    out = model(x)

    assert out.shape == (4, 1, 3)

    criterion = PinballLoss(quantiles=quantiles)
    target = torch.randn(4, 1)
    loss = criterion(out, target)

    assert loss.item() >= 0.0
    assert not torch.isnan(loss)


def test_mc_dropout_predict_and_coverage():
    """Verify MC-dropout outputs quantiles and computes interval coverage."""
    model = AQIGRU(input_size=5, hidden_size=16, num_layers=2, output_size=1, dropout=0.3)
    X = np.random.randn(20, 14, 5).astype(np.float32)

    q10, q50, q90, mean_p = mc_dropout_predict(model, X, n_passes=15, device="cpu")

    assert q10.shape == (20, 1)
    assert q50.shape == (20, 1)
    assert q90.shape == (20, 1)

    y_true = q50.copy()
    cov = compute_interval_coverage(y_true, q10, q90)
    assert cov == 100.0


def test_walk_forward_folds_temporality():
    """Verify expanding walk-forward folds maintain strict chronological precedence (Train < Val < Test)."""
    dates = pd.date_range("2023-01-01", "2025-12-31", freq="D")
    df = pd.DataFrame({"AQI": np.random.randn(len(dates))}, index=dates)

    folds = create_expanding_walk_forward_folds(df, n_folds=4, min_train_days=400, val_days=92)
    assert len(folds) == 4

    prev_train_len = 0
    for k, (train_f, val_f, test_f) in enumerate(folds):
        assert train_f.index.max() < val_f.index.min(), f"Fold {k}: Train must precede Val"
        assert val_f.index.max() < test_f.index.min(), f"Fold {k}: Val must precede Test"
        assert len(train_f) >= prev_train_len
        prev_train_len = len(train_f)


def test_audit_2023_2025_rejects_pre2023_and_city_day():
    """Verify audit and loading reject city_day.csv and pre-2023 data."""
    # 1. Rejecting city_day.csv path
    with pytest.raises(ValueError, match="city_day.csv and pre-2023 data are strictly prohibited"):
        load_raw_data(data_path="data/raw/city_day.csv")

    with pytest.raises(ValueError, match="city_day.csv and pre-2023 data are strictly prohibited"):
        audit_dataset(data_path="data/raw/city_day.csv")

    # 2. Rejecting dates outside 2023-2025
    bad_df = pd.DataFrame({
        "Station": ["ITO"] * 10,
        "Date": pd.date_range("2022-01-01", periods=10, freq="D"),
        "AQI": [150.0] * 10,
    })
    bad_city_df = pd.DataFrame({
        "Date": pd.date_range("2022-01-01", periods=10, freq="D"),
        "AQI": [150.0] * 10,
    })
    with pytest.raises(ValueError, match="Date range violation"):
        audit_2023_2025_dataset(bad_df, bad_city_df, verbose=False)


def test_loader_rejects_unregistered_provenance_files():
    """Verify that loader rejects any data path not registered in PROVENANCE.md."""
    with pytest.raises(ValueError, match="not registered in PROVENANCE.md"):
        load_raw_data(data_path="data/raw/unregistered_mystery_station.csv")
