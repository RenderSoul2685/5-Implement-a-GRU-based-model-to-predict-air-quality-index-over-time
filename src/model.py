"""GRU and baseline neural network architectures for AQI time series forecasting."""

import logging
from typing import Optional, Tuple
import numpy as np
from sklearn.linear_model import Ridge
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class AQIGRU(nn.Module):
    """Stacked Gated Recurrent Unit (GRU) model with linear prediction head.

    Args:
        input_size: Number of input features per time step.
        hidden_size: Hidden dimension of the GRU layers.
        num_layers: Number of stacked GRU layers.
        output_size: Forecast horizon (number of future time steps to predict).
        dropout: Dropout rate between GRU layers.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout_rate = dropout

        # Multi-layer GRU
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Fully connected projection head with layer normalization and dropout
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(
        self,
        x: torch.Tensor,
        h_0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size).
            h_0: Optional initial hidden state.

        Returns:
            Output prediction tensor of shape (batch_size, output_size).
        """
        # gru_out shape: (batch_size, seq_len, hidden_size)
        # h_n shape: (num_layers, batch_size, hidden_size)
        gru_out, _ = self.gru(x, h_0)

        # Take last time-step representation
        last_step = gru_out[:, -1, :]  # shape: (batch_size, hidden_size)

        out = self.head(last_step)  # shape: (batch_size, output_size)
        return out


class AQIStationEmbeddingGRU(nn.Module):
    """Stacked Gated Recurrent Unit (GRU) with learned station embedding for multi-station forecasting.

    Concatenates a learned station embedding vector to the input features at every time-step,
    enabling a single GRU to leverage pooled observations across all monitoring stations while
    maintaining station-specific micro-environment calibration.

    Args:
        input_size: Number of continuous input features per time step.
        num_stations: Total number of monitoring stations.
        station_embedding_dim: Embedding vector dimension.
        hidden_size: Hidden dimension of the GRU layers.
        num_layers: Number of stacked GRU layers.
        output_size: Forecast horizon.
        dropout: Dropout rate between GRU layers.
    """

    def __init__(
        self,
        input_size: int,
        num_stations: int = 6,
        station_embedding_dim: int = 16,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_size = input_size
        self.num_stations = num_stations
        self.station_embedding_dim = station_embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout_rate = dropout

        self.station_embedding = nn.Embedding(num_stations, station_embedding_dim)

        self.gru = nn.GRU(
            input_size=input_size + station_embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size + station_embedding_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(
        self,
        x: torch.Tensor,
        station_ids: Optional[torch.Tensor] = None,
        h_0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size).
            station_ids: LongTensor of shape (batch_size,) indicating station identity.
                         If None, defaults to station index 0.
            h_0: Optional initial hidden state.

        Returns:
            Output prediction tensor of shape (batch_size, output_size).
        """
        batch_size, seq_len, _ = x.shape
        if station_ids is None:
            station_ids = torch.zeros(batch_size, dtype=torch.long, device=x.device)
        elif station_ids.ndim == 0:
            station_ids = station_ids.unsqueeze(0).expand(batch_size)

        emb = self.station_embedding(station_ids)  # (batch_size, station_embedding_dim)
        emb_seq = emb.unsqueeze(1).expand(-1, seq_len, -1)  # (batch_size, seq_len, station_embedding_dim)

        x_cat = torch.cat([x, emb_seq], dim=-1)
        gru_out, _ = self.gru(x_cat, h_0)
        last_step = gru_out[:, -1, :]  # (batch_size, hidden_size)

        head_in = torch.cat([last_step, emb], dim=-1)
        return self.head(head_in)


class AQILSTM(nn.Module):
    """Stacked Long Short-Term Memory (LSTM) baseline architecture.

    Args:
        input_size: Number of input features per time step.
        hidden_size: Hidden dimension of LSTM layers.
        num_layers: Number of stacked LSTM layers.
        output_size: Forecast horizon.
        dropout: Dropout rate between LSTM layers.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size).

        Returns:
            Prediction tensor of shape (batch_size, output_size).
        """
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]
        return self.fc(last_step)


class PersistenceBaseline:
    """Naive persistence forecast baseline: future AQI equals latest known AQI.

    Args:
        target_feature_idx: Index of the target column inside the input window features.
        horizon: Number of future steps to predict.
    """

    def __init__(self, target_feature_idx: int = 0, horizon: int = 1):
        self.target_feature_idx = target_feature_idx
        self.horizon = horizon

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Forecast tomorrow = today across the entire forecast horizon.

        Args:
            X: Input array of shape (N, seq_len, num_features).

        Returns:
            Predictions of shape (N, horizon).
        """
        # Latest observed target value in each window
        last_val = X[:, -1, self.target_feature_idx]  # shape: (N,)
        # Broadcast across forecast horizon
        return np.repeat(last_val[:, np.newaxis], self.horizon, axis=1)


class MovingAverageBaseline:
    """7-day (or K-day) rolling moving average baseline calculated over observed input window.

    Args:
        target_feature_idx: Index of the target column inside the input window features.
        window_size: Number of past time steps to average (default: 7).
        horizon: Forecast horizon.
    """

    def __init__(self, target_feature_idx: int = 0, window_size: int = 7, horizon: int = 1):
        self.target_feature_idx = target_feature_idx
        self.window_size = window_size
        self.horizon = horizon

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Forecast future steps as the average of the last K observed values in each window.

        Args:
            X: Input array of shape (N, seq_len, num_features).

        Returns:
            Predictions of shape (N, horizon).
        """
        k = min(self.window_size, X.shape[1])
        # Last k observed values
        last_k = X[:, -k:, self.target_feature_idx]
        ma = np.mean(last_k, axis=1)  # shape: (N,)
        return np.repeat(ma[:, np.newaxis], self.horizon, axis=1)


class SeasonalNaiveBaseline:
    """Seasonal-naive baseline: same day last year (365d) or same day of previous cycle (7d).

    Args:
        target_feature_idx: Index of the target column inside the input window features.
        horizon: Forecast horizon.
        cycle_len: Short-term cycle lag (default: 7 for weekly pattern).
        annual_lag: Long-term seasonal lag (default: 365 for annual pattern).
    """

    def __init__(
        self,
        target_feature_idx: int = 0,
        horizon: int = 1,
        cycle_len: int = 7,
        annual_lag: int = 365,
    ):
        self.target_feature_idx = target_feature_idx
        self.horizon = horizon
        self.cycle_len = cycle_len
        self.annual_lag = annual_lag
        self.history_targets: Optional[np.ndarray] = None

    def fit(self, history_targets: np.ndarray) -> "SeasonalNaiveBaseline":
        """Store chronological target history for exact annual calendar lookup."""
        self.history_targets = np.asarray(history_targets).ravel()
        return self

    def predict(self, X: np.ndarray, test_start_indices: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict future values using seasonal lag.

        Args:
            X: Input array of shape (N, seq_len, num_features).
            test_start_indices: Optional array of global time indices for each window's target start.

        Returns:
            Predictions of shape (N, horizon).
        """
        N = X.shape[0]
        preds = np.empty((N, self.horizon), dtype=np.float32)

        # 1. Exact 365-day annual lookup if full historical array and indices are available
        if self.history_targets is not None and test_start_indices is not None and len(test_start_indices) == N:
            for i in range(N):
                base_t = test_start_indices[i]
                for h in range(self.horizon):
                    target_t = base_t + h
                    lag_365 = target_t - self.annual_lag
                    if 0 <= lag_365 < len(self.history_targets):
                        preds[i, h] = self.history_targets[lag_365]
                    else:
                        lag_7 = target_t - self.cycle_len
                        if 0 <= lag_7 < len(self.history_targets):
                            preds[i, h] = self.history_targets[lag_7]
                        else:
                            preds[i, h] = X[i, -1, self.target_feature_idx]
            return preds

        # 2. In-window cycle fallback (same day of previous weekly cycle, e.g. 7 days prior in lookback window)
        cycle_step = min(self.cycle_len, X.shape[1])
        cycle_val = X[:, -cycle_step, self.target_feature_idx]
        return np.repeat(cycle_val[:, np.newaxis], self.horizon, axis=1)


class SameHourYesterdayBaseline:
    """Same-hour-yesterday baseline: predicts the observation from 24 hours prior.

    For horizon h in 1..24, step (t + h) is predicted as AQI at (t + h - 24).
    Within a 48-hour lookback window [t-47, ..., t], the value at (t + h - 24)
    is located at index (-24 + h - 1), which corresponds to the last 24 steps of the lookback.
    """

    def __init__(self, target_feature_idx: int = 0, horizon: int = 24):
        self.target_feature_idx = target_feature_idx
        self.horizon = horizon

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Forecast future steps using the reading from 24 hours prior.

        Args:
            X: Input array of shape (N, seq_len, num_features).

        Returns:
            Predictions of shape (N, horizon).
        """
        N, seq_len, _ = X.shape
        preds = np.empty((N, self.horizon), dtype=np.float32)
        for h in range(1, self.horizon + 1):
            idx = (seq_len - 1) + (h - 24)
            preds[:, h - 1] = X[:, idx, self.target_feature_idx]
        return preds


class LinearBaseline:
    """Linear regression (Ridge) baseline trained on flattened historical windows.

    To ensure a strictly fair comparison with the GRU, receives the exact same lag window
    across all input features (AQI history, calendar features, and weather),
    optionally concatenated with one-hot station indicators in multi-station experiments,
    and supports residual-target forecasting (predicting delta over latest observed AQI).

    Args:
        horizon: Forecast horizon.
        alpha: Ridge regularization strength.
        residual: If True, fits on (y[t+h] - AQI[t]) and predicts AQI[t] + delta.
        target_feature_idx: Index of the target column inside input features.
    """

    def __init__(
        self,
        horizon: int = 24,
        alpha: float = 1.0,
        residual: bool = False,
        target_feature_idx: int = 0,
    ):
        self.horizon = horizon
        self.alpha = alpha
        self.residual = residual
        self.target_feature_idx = target_feature_idx
        self.model = Ridge(alpha=alpha)

    def _extract_features(
        self, X: np.ndarray, station_one_hot: Optional[np.ndarray] = None
    ) -> np.ndarray:
        n_samples, seq_len, n_feat = X.shape
        X_flat = X.reshape(n_samples, seq_len * n_feat)
        if station_one_hot is not None:
            if len(station_one_hot) != n_samples:
                raise ValueError("station_one_hot length must match X sample count.")
            return np.hstack([X_flat, station_one_hot])
        return X_flat

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        station_one_hot: Optional[np.ndarray] = None,
    ) -> "LinearBaseline":
        """Fit Ridge on flattened window features (and optional one-hot station vector)."""
        X_feat = self._extract_features(X, station_one_hot)
        if self.residual:
            y_last = X[:, -1, self.target_feature_idx : self.target_feature_idx + 1]
            if y.ndim == 1:
                y = y[:, np.newaxis]
            y_target = y - y_last
        else:
            y_target = y
        self.model.fit(X_feat, y_target)
        return self

    def predict(
        self,
        X: np.ndarray,
        station_one_hot: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predict target using flattened sequence features."""
        X_feat = self._extract_features(X, station_one_hot)
        pred = self.model.predict(X_feat)
        if pred.ndim == 1:
            pred = pred[:, np.newaxis]
        if self.residual:
            y_last = X[:, -1, self.target_feature_idx : self.target_feature_idx + 1]
            pred = y_last + pred
        return pred


class AQIBiGRU(nn.Module):
    """Bidirectional Stacked GRU architecture (Ablation model).

    Args:
        input_size: Number of input features per time step.
        hidden_size: Hidden dimension of each GRU direction.
        num_layers: Number of stacked GRU layers.
        output_size: Forecast horizon.
        dropout: Dropout rate between layers.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout_rate = dropout

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_out, _ = self.gru(x)  # shape: (batch_size, seq_len, 2 * hidden_size)
        last_step = gru_out[:, -1, :]
        return self.head(last_step)


class AQIAttentionGRU(nn.Module):
    """Stacked GRU with temporal additive attention mechanism (Ablation model).

    Computes dynamically weighted representation over all input sequence time-steps.

    Args:
        input_size: Number of input features.
        hidden_size: Hidden dimension of GRU.
        num_layers: Number of GRU layers.
        output_size: Forecast horizon.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout_rate = dropout

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        gru_out, _ = self.gru(x)  # shape: (batch_size, seq_len, hidden_size)
        scores = self.attention(gru_out)  # shape: (batch_size, seq_len, 1)
        weights = torch.softmax(scores, dim=1)  # shape: (batch_size, seq_len, 1)
        context = torch.sum(gru_out * weights, dim=1)  # shape: (batch_size, hidden_size)
        out = self.head(context)

        if return_attention:
            return out, weights
        return out


class PinballLoss(nn.Module):
    """Pinball (Quantile) loss for training quantile regression models.

    Args:
        quantiles: Tuple of target quantiles (e.g. 0.10, 0.50, 0.90).
    """

    def __init__(self, quantiles: Tuple[float, ...] = (0.10, 0.50, 0.90)):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute mean pinball loss across all samples and quantiles.

        Args:
            preds: Tensor of shape (batch_size, horizon, num_quantiles).
            target: Tensor of shape (batch_size, horizon) or (batch_size, horizon, 1).
        """
        if target.ndim == 2:
            target = target.unsqueeze(-1)  # (batch_size, horizon, 1)

        total_loss = 0.0
        for i, q in enumerate(self.quantiles):
            pred_q = preds[..., i:i + 1]
            diff = target - pred_q
            loss_q = torch.max(q * diff, (q - 1.0) * diff)
            total_loss += torch.mean(loss_q)

        return total_loss / len(self.quantiles)


class AQIQuantileGRU(nn.Module):
    """Stacked GRU outputting multiple quantile estimates (10th, 50th, 90th percentile).

    Args:
        input_size: Number of input features.
        hidden_size: Hidden dimension.
        num_layers: Number of GRU layers.
        output_size: Forecast horizon.
        quantiles: Tuple of quantiles to predict (default: 0.10, 0.50, 0.90).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        quantiles: Tuple[float, ...] = (0.10, 0.50, 0.90),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size * self.num_quantiles),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_out, _ = self.gru(x)
        last_step = gru_out[:, -1, :]
        out = self.head(last_step)
        # Reshape to (batch_size, output_size, num_quantiles)
        return out.view(-1, self.output_size, self.num_quantiles)


def mc_dropout_predict(
    model: nn.Module,
    X: np.ndarray | torch.Tensor,
    n_passes: int = 50,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Execute Monte Carlo Dropout inference with N stochastic forward passes.

    Args:
        model: PyTorch model with Dropout layers.
        X: Input array or tensor of shape (batch_size, seq_len, num_features).
        n_passes: Number of forward passes (default: 50).
        device: Hardware device.

    Returns:
        Tuple of (q10, q50, q90, mean_pred) each of shape (batch_size, horizon).
    """
    model.train()  # Keep dropout enabled
    if isinstance(X, np.ndarray):
        X_tensor = torch.as_tensor(X, dtype=torch.float32)
    else:
        X_tensor = X

    X_tensor = X_tensor.to(device)
    passes = []
    with torch.no_grad():
        for _ in range(n_passes):
            out = model(X_tensor)
            if out.ndim == 3:  # If quantile model output, take median slice
                out = out[..., 1]
            passes.append(out.cpu().numpy())

    passes_arr = np.array(passes)  # shape: (n_passes, batch_size, horizon)
    q10 = np.percentile(passes_arr, 10, axis=0)
    q50 = np.percentile(passes_arr, 50, axis=0)
    q90 = np.percentile(passes_arr, 90, axis=0)
    mean_pred = np.mean(passes_arr, axis=0)

    model.eval()  # Reset to eval mode
    return q10, q50, q90, mean_pred


def compute_interval_coverage(
    y_true: np.ndarray,
    lower_pred: np.ndarray,
    upper_pred: np.ndarray,
) -> float:
    """Compute empirical coverage percentage of ground truth inside prediction interval.

    Args:
        y_true: Ground truth target array.
        lower_pred: 10th percentile predictions.
        upper_pred: 90th percentile predictions.

    Returns:
        Coverage percentage in [0.0, 100.0].
    """
    y_flat = y_true.ravel()
    l_flat = lower_pred.ravel()
    u_flat = upper_pred.ravel()
    inside = (y_flat >= l_flat) & (y_flat <= u_flat)
    return float(np.mean(inside) * 100.0)

