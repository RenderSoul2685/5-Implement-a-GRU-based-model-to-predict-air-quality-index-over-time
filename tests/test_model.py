"""Unit tests for GRU neural network architecture, baselines, and gradient flow."""

import numpy as np
import pytest
import torch

from src.model import AQIGRU, AQILSTM, LinearBaseline, PersistenceBaseline


def test_gru_forward_shape_single_step():
    """Verify GRU forward pass tensor shape for horizon=1."""
    batch_size = 8
    seq_len = 14
    input_size = 11
    horizon = 1

    model = AQIGRU(input_size=input_size, hidden_size=32, num_layers=2, output_size=horizon)
    x = torch.randn(batch_size, seq_len, input_size)
    out = model(x)

    assert out.shape == (batch_size, horizon)
    assert not torch.isnan(out).any(), "Output should not contain NaN values."


def test_gru_forward_shape_multi_step():
    """Verify GRU forward pass tensor shape for horizon=7."""
    batch_size = 4
    seq_len = 14
    input_size = 11
    horizon = 7

    model = AQIGRU(input_size=input_size, hidden_size=64, num_layers=2, output_size=horizon)
    x = torch.randn(batch_size, seq_len, input_size)
    out = model(x)

    assert out.shape == (batch_size, horizon)


def test_gru_gradient_flow():
    """Verify that gradients propagate cleanly without vanishing/exploding to NaNs."""
    batch_size = 4
    seq_len = 14
    input_size = 11
    horizon = 1

    model = AQIGRU(input_size=input_size, hidden_size=32, num_layers=2, output_size=horizon)
    x = torch.randn(batch_size, seq_len, input_size, requires_grad=True)
    target = torch.randn(batch_size, horizon)

    criterion = torch.nn.HuberLoss()
    out = model(x)
    loss = criterion(out, target)
    loss.backward()

    # Check gradients on all model parameters
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient."
        assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradients."


def test_persistence_baseline():
    """Verify naive persistence baseline repeats last step target across horizon."""
    batch_size = 5
    seq_len = 10
    num_features = 4
    target_idx = 2
    horizon = 3

    X = np.random.randn(batch_size, seq_len, num_features)
    # The last known target values in the input windows
    last_known_target = X[:, -1, target_idx]

    persistence = PersistenceBaseline(target_feature_idx=target_idx, horizon=horizon)
    preds = persistence.predict(X)

    assert preds.shape == (batch_size, horizon)
    for h in range(horizon):
        np.testing.assert_allclose(preds[:, h], last_known_target)


def test_linear_baseline():
    """Verify LinearBaseline fits and predicts correct shape."""
    n_samples = 20
    seq_len = 14
    num_features = 6
    horizon = 2

    X = np.random.randn(n_samples, seq_len, num_features).astype(np.float32)
    y = np.random.randn(n_samples, horizon).astype(np.float32)

    linear = LinearBaseline(horizon=horizon)
    linear.fit(X, y)
    preds = linear.predict(X)

    assert preds.shape == (n_samples, horizon)
    assert not np.isnan(preds).any()


def test_station_embedding_gru_forward_and_gradients():
    """Verify AQIStationEmbeddingGRU concatenates station embeddings and propagates gradients."""
    from src.model import AQIStationEmbeddingGRU

    batch_size = 4
    seq_len = 14
    input_size = 11
    embedding_dim = 8
    num_stations = 6
    horizon = 1

    model = AQIStationEmbeddingGRU(
        input_size=input_size,
        hidden_size=32,
        num_layers=2,
        output_size=horizon,
        num_stations=num_stations,
        station_embedding_dim=embedding_dim,
    )

    x = torch.randn(batch_size, seq_len, input_size, requires_grad=True)
    station_ids = torch.tensor([0, 1, 3, 5], dtype=torch.long)
    target = torch.randn(batch_size, horizon)

    criterion = torch.nn.HuberLoss()
    out = model(x, station_ids)

    assert out.shape == (batch_size, horizon)
    assert not torch.isnan(out).any()

    loss = criterion(out, target)
    loss.backward()

    assert model.station_embedding.weight.grad is not None
    assert not torch.isnan(model.station_embedding.weight.grad).any()
    for name, param in model.gru.named_parameters():
        assert param.grad is not None


def test_ridge_residual_and_station_one_hot():
    """Verify LinearBaseline supports residual forecasting and station one-hot encoding."""
    n_samples = 30
    seq_len = 14
    num_features = 8
    target_idx = 3
    num_stations = 4

    X = np.random.randn(n_samples, seq_len, num_features).astype(np.float32)
    y = np.random.randn(n_samples, 1).astype(np.float32)
    station_ids = np.random.randint(0, num_stations, size=n_samples)
    one_hot = np.eye(num_stations)[station_ids]

    ridge_res = LinearBaseline(horizon=1, residual=True, target_feature_idx=target_idx)
    ridge_res.fit(X, y, station_one_hot=one_hot)
    preds = ridge_res.predict(X, station_one_hot=one_hot)

    assert preds.shape == (n_samples, 1)
    assert not np.isnan(preds).any()


def test_same_hour_yesterday_baseline():
    """Verify SameHourYesterdayBaseline forecasts step (t+h) using reading at (t+h-24)."""
    from src.model import SameHourYesterdayBaseline

    n_samples = 5
    seq_len = 48
    num_features = 4
    target_idx = 0
    horizon = 24

    X = np.random.randn(n_samples, seq_len, num_features).astype(np.float32)

    model = SameHourYesterdayBaseline(target_feature_idx=target_idx, horizon=horizon)
    preds = model.predict(X)

    assert preds.shape == (n_samples, horizon)
    for h in range(1, horizon + 1):
        expected = X[:, seq_len - 1 + (h - 24), target_idx]
        np.testing.assert_allclose(preds[:, h - 1], expected)


