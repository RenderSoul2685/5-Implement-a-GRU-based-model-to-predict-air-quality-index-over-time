"""Comprehensive Hourly Benchmark Pipeline for AQI-GRU (2023-2025).

Rules & Requirements strictly enforced:
- 100% REAL data (2023-01-01 to 2025-12-31); zero synthetic, simulated, or randomized data.
- Feature set: AQI history, calendar features (hour, dayofweek, month as sin/cos), hourly weather.
- NO pollutant columns used.
- Lookback 48 hours, predict horizons 1..24 jointly.
- Evaluation strictly masks unobserved target positions (only genuinely observed points evaluated).
- Chronological split: Train to 2025-06-30, Val 2025-07-01..2025-09-30, Test 2025-10-01..2025-12-31.
- Scalers fitted strictly on training split only.
- 5 seeds: [42, 123, 456, 789, 1011], mean +/- std.
- 4-fold expanding-window walk-forward CV over train+validation.
- Moving-block bootstrap (7-day blocks, 1,000 resamples) for paired test-RMSE difference (95% CI).
- Multi-station vs Ridge Multi vs City GRU per station comparison.
- Quantile GRU 80% interval coverage ($q_{10}$ to $q_{90}$).
- Generates 4 presentation plots.
"""

import copy
import json
import logging
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.config import Config, DELHI_STATIONS
from src.model import (
    AQIBiGRU,
    AQIGRU,
    AQILSTM,
    AQIQuantileGRU,
    AQIStationEmbeddingGRU,
    LinearBaseline,
    PersistenceBaseline,
    PinballLoss,
    SameHourYesterdayBaseline,
)
from src.preprocessing import (
    create_hourly_windows,
    prepare_hourly_city_dataframe,
    prepare_hourly_station_dataframe,
    split_hourly_windows_by_date,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("hourly_benchmark")

SEEDS = [42, 123, 456, 789, 1011]
REPORT_HORIZONS = [1, 3, 6, 12, 24]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics_masked(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, float]:
    """Compute RMSE, MAE, R2, MAPE strictly over valid masked points."""
    valid = mask & ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not np.any(valid):
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "mape": np.nan}

    yt = y_true[valid]
    yp = y_pred[valid]

    diff = yt - yp
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))

    # R2
    ss_res = np.sum(diff ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-8 else 0.0

    # MAPE
    non_zero = np.abs(yt) > 1.0
    if np.any(non_zero):
        mape = float(np.mean(np.abs(diff[non_zero] / yt[non_zero])) * 100.0)
    else:
        mape = np.nan

    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}


def train_torch_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 35,
    patience: int = 7,
    lr: float = 0.001,
    is_quantile: bool = False,
    quantiles: Tuple[float, ...] = (0.10, 0.50, 0.90),
    device: str = "cpu",
) -> Tuple[nn.Module, dict]:
    """Train PyTorch model with early stopping and learning rate scheduling."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5
    )
    criterion = PinballLoss(quantiles=quantiles) if is_quantile else nn.HuberLoss(delta=1.0)

    best_val_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    patience_counter = 0

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch in train_loader:
            if len(batch) == 3:  # (x, station_ids, y)
                x_b, s_b, y_b = batch
                x_b, s_b, y_b = x_b.to(device), s_b.to(device), y_b.to(device)
                pred = model(x_b, s_b)
            else:
                x_b, y_b = batch
                x_b, y_b = x_b.to(device), y_b.to(device)
                pred = model(x_b)

            loss = criterion(pred, y_b)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item() * len(y_b)
            train_count += len(y_b)

        train_loss = train_loss_sum / train_count

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    x_b, s_b, y_b = batch
                    x_b, s_b, y_b = x_b.to(device), s_b.to(device), y_b.to(device)
                    pred = model(x_b, s_b)
                else:
                    x_b, y_b = batch
                    x_b, y_b = x_b.to(device), y_b.to(device)
                    pred = model(x_b)

                loss = criterion(pred, y_b)
                val_loss_sum += loss.item() * len(y_b)
                val_count += len(y_b)

        val_loss = val_loss_sum / val_count
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_weights)
    stop_epoch = len(history["train_loss"])
    train_meta = {
        "best_epoch": best_epoch,
        "stop_epoch": stop_epoch,
        "best_val_loss": best_val_loss,
        "history": history,
    }
    return model, train_meta


def moving_block_bootstrap(
    err1: np.ndarray,
    err2: np.ndarray,
    block_size: int = 168,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Moving-block bootstrap of paired RMSE difference: RMSE(model1) - RMSE(model2).

    Blocks of block_size consecutive errors are sampled with replacement.
    Returns:
        (diff_mean, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(seed)
    N = len(err1)
    if N < block_size:
        block_size = max(1, N // 4)

    n_blocks = int(math.ceil(N / block_size))
    max_start = N - block_size + 1

    diffs = []
    for _ in range(n_boot):
        starts = rng.randint(0, max_start, size=n_blocks)
        indices = np.concatenate([np.arange(s, s + block_size) for s in starts])[:N]
        e1_boot = err1[indices]
        e2_boot = err2[indices]
        rmse1 = np.sqrt(np.mean(e1_boot ** 2))
        rmse2 = np.sqrt(np.mean(e2_boot ** 2))
        diffs.append(rmse1 - rmse2)

    diffs = np.array(diffs)
    ci_lower = float(np.percentile(diffs, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(diffs, 100.0 * (1.0 - alpha / 2.0)))
    diff_mean = float(np.mean(diffs))
    return diff_mean, ci_lower, ci_upper


def run_benchmark():
    logger.info("Starting AQI-GRU Hourly Benchmark Rebuild (2023-2025)...")
    cfg = Config()

    raw_dir = repo_root / "data" / "raw"
    processed_dir = repo_root / "data" / "processed"
    outputs_dir = repo_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    hourly_stations_path = processed_dir / "delhi_hourly_stations_2023_2025.csv"
    hourly_weather_path = raw_dir / "delhi_weather_hourly_2023_2025.csv"

    if not hourly_stations_path.exists():
        raise FileNotFoundError(f"Missing {hourly_stations_path}. Run build_real_delhi_datasets first.")
    if not hourly_weather_path.exists():
        raise FileNotFoundError(f"Missing {hourly_weather_path}. Run weather fetch step first.")

    df_stations = pd.read_csv(hourly_stations_path)
    df_weather = pd.read_csv(hourly_weather_path)

    # 2. Build City-level dataset
    df_city = prepare_hourly_city_dataframe(df_stations, df_weather, min_stations_required=4, interp_limit_hours=6)

    # Feature definitions (NO pollutants per A1)
    # AQI_interp is feature index 0
    feat_cols = [
        "AQI_interp",
        "hour_sin",
        "hour_cos",
        "dayofweek_sin",
        "dayofweek_cos",
        "month_sin",
        "month_cos",
        "temperature",
        "humidity",
        "wind_speed",
        "wind_direction",
        "pressure",
    ]
    num_features = len(feat_cols)
    lookback = 48
    horizon = 24

    logger.info("Feature set (%d features): %s", num_features, feat_cols)

    # 3. Create City Windows
    X_city, y_city, y_mask_city, last_city, ts_city = create_hourly_windows(
        df_city,
        feature_cols=feat_cols,
        target_col="AQI_interp",
        raw_col="AQI",
        lookback=lookback,
        horizon=horizon,
    )
    splits_city = split_hourly_windows_by_date(
        X_city,
        y_city,
        y_mask_city,
        last_city,
        ts_city,
        train_end=cfg.train_end_date + " 23:00:00",
        val_start=cfg.val_start_date + " 00:00:00",
        val_end=cfg.val_end_date + " 23:00:00",
        test_start=cfg.test_start_date + " 00:00:00",
        test_end=cfg.test_end_date + " 23:00:00",
        horizon_hours=horizon,
    )

    X_train_c, y_train_c, mask_train_c, last_train_c = (
        splits_city["train"]["X"],
        splits_city["train"]["y"],
        splits_city["train"]["y_mask"],
        splits_city["train"]["last_known"],
    )
    X_val_c, y_val_c, mask_val_c, last_val_c = (
        splits_city["val"]["X"],
        splits_city["val"]["y"],
        splits_city["val"]["y_mask"],
        splits_city["val"]["last_known"],
    )
    X_test_c, y_test_c, mask_test_c, last_test_c = (
        splits_city["test"]["X"],
        splits_city["test"]["y"],
        splits_city["test"]["y_mask"],
        splits_city["test"]["last_known"],
    )

    # 4. Prepare Multi-Station Windows
    station_dfs = {}
    station_windows = {}
    X_train_multi_list = []
    y_train_multi_list = []
    s_train_multi_list = []
    last_train_multi_list = []

    X_val_multi_list = []
    y_val_multi_list = []
    s_val_multi_list = []
    last_val_multi_list = []

    for s_idx, st in enumerate(DELHI_STATIONS):
        df_st = prepare_hourly_station_dataframe(df_stations, st, df_weather, interp_limit_hours=6)
        station_dfs[st] = df_st
        X_st, y_st, mask_st, last_st, ts_st = create_hourly_windows(
            df_st,
            feature_cols=feat_cols,
            target_col="AQI_interp",
            raw_col="AQI",
            lookback=lookback,
            horizon=horizon,
        )
        splits_st = split_hourly_windows_by_date(
            X_st,
            y_st,
            mask_st,
            last_st,
            ts_st,
            train_end=cfg.train_end_date + " 23:00:00",
            val_start=cfg.val_start_date + " 00:00:00",
            val_end=cfg.val_end_date + " 23:00:00",
            test_start=cfg.test_start_date + " 00:00:00",
            test_end=cfg.test_end_date + " 23:00:00",
            horizon_hours=horizon,
        )
        station_windows[st] = splits_st

        # Add to pooled training
        n_tr = len(splits_st["train"]["X"])
        X_train_multi_list.append(splits_st["train"]["X"])
        y_train_multi_list.append(splits_st["train"]["y"])
        s_train_multi_list.append(np.full((n_tr,), s_idx, dtype=np.int64))
        last_train_multi_list.append(splits_st["train"]["last_known"])

        # Add to pooled validation
        n_val = len(splits_st["val"]["X"])
        X_val_multi_list.append(splits_st["val"]["X"])
        y_val_multi_list.append(splits_st["val"]["y"])
        s_val_multi_list.append(np.full((n_val,), s_idx, dtype=np.int64))
        last_val_multi_list.append(splits_st["val"]["last_known"])

    X_train_multi = np.concatenate(X_train_multi_list, axis=0)
    y_train_multi = np.concatenate(y_train_multi_list, axis=0)
    s_train_multi = np.concatenate(s_train_multi_list, axis=0)
    last_train_multi = np.concatenate(last_train_multi_list, axis=0)

    X_val_multi = np.concatenate(X_val_multi_list, axis=0)
    y_val_multi = np.concatenate(y_val_multi_list, axis=0)
    s_val_multi = np.concatenate(s_val_multi_list, axis=0)
    last_val_multi = np.concatenate(last_val_multi_list, axis=0)

    logger.info("Pooled Multi-Station Windows -> Train: %d | Val: %d", len(X_train_multi), len(X_val_multi))

    # 5. Fit Scalers STRICTLY on Training Split Only
    feat_scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    # Reshape (N, 48, D) -> (N*48, D) to fit scaler
    feat_scaler.fit(X_train_c.reshape(-1, num_features))

    def scale_X(X_arr):
        N, seq_l, D = X_arr.shape
        scaled = feat_scaler.transform(X_arr.reshape(-1, D))
        return scaled.reshape(N, seq_l, D).astype(np.float32)

    X_tr_c_sc = scale_X(X_train_c)
    X_val_c_sc = scale_X(X_val_c)
    X_te_c_sc = scale_X(X_test_c)

    X_tr_m_sc = scale_X(X_train_multi)
    X_val_m_sc = scale_X(X_val_multi)

    # Scalers for Targets:
    # (a) Residual deltas: delta = y - last_known
    delta_train_c = y_train_c - last_train_c[:, np.newaxis]
    delta_val_c = y_val_c - last_val_c[:, np.newaxis]
    delta_test_c = y_test_c - last_test_c[:, np.newaxis]

    delta_scaler_c = StandardScaler()
    delta_scaler_c.fit(delta_train_c)

    delta_tr_c_sc = delta_scaler_c.transform(delta_train_c).astype(np.float32)
    delta_val_c_sc = delta_scaler_c.transform(delta_val_c).astype(np.float32)

    # (b) Direct target scaling:
    direct_scaler_c = StandardScaler()
    direct_scaler_c.fit(y_train_c)
    direct_tr_c_sc = direct_scaler_c.transform(y_train_c).astype(np.float32)
    direct_val_c_sc = direct_scaler_c.transform(y_val_c).astype(np.float32)

    # (c) Multi-station residual target scaler
    delta_train_m = y_train_multi - last_train_multi[:, np.newaxis]
    delta_val_m = y_val_multi - last_val_multi[:, np.newaxis]
    delta_scaler_m = StandardScaler()
    delta_scaler_m.fit(delta_train_m)
    delta_tr_m_sc = delta_scaler_m.transform(delta_train_m).astype(np.float32)
    delta_val_m_sc = delta_scaler_m.transform(delta_val_m).astype(np.float32)

    # DataLoaders (City)
    batch_size = 128
    loader_tr_res = DataLoader(TensorDataset(torch.as_tensor(X_tr_c_sc), torch.as_tensor(delta_tr_c_sc)), batch_size=batch_size, shuffle=True)
    loader_val_res = DataLoader(TensorDataset(torch.as_tensor(X_val_c_sc), torch.as_tensor(delta_val_c_sc)), batch_size=batch_size, shuffle=False)

    loader_tr_dir = DataLoader(TensorDataset(torch.as_tensor(X_tr_c_sc), torch.as_tensor(direct_tr_c_sc)), batch_size=batch_size, shuffle=True)
    loader_val_dir = DataLoader(TensorDataset(torch.as_tensor(X_val_c_sc), torch.as_tensor(direct_val_c_sc)), batch_size=batch_size, shuffle=False)

    loader_tr_multi = DataLoader(TensorDataset(torch.as_tensor(X_tr_m_sc), torch.as_tensor(s_train_multi), torch.as_tensor(delta_tr_m_sc)), batch_size=batch_size, shuffle=True)
    loader_val_multi = DataLoader(TensorDataset(torch.as_tensor(X_val_m_sc), torch.as_tensor(s_val_multi), torch.as_tensor(delta_val_m_sc)), batch_size=batch_size, shuffle=False)

    # 6. Run Baselines
    logger.info("Computing Baselines on Test Set...")
    # Persistence
    pers_model = PersistenceBaseline(target_feature_idx=0, horizon=horizon)
    y_pred_pers = pers_model.predict(X_test_c)

    # Same-Hour-Yesterday
    shy_model = SameHourYesterdayBaseline(target_feature_idx=0, horizon=horizon)
    y_pred_shy = shy_model.predict(X_test_c)

    # Ridge City Direct
    ridge_dir = LinearBaseline(horizon=horizon, alpha=1.0, residual=False, target_feature_idx=0)
    ridge_dir.fit(X_tr_c_sc, y_train_c)
    y_pred_ridge_dir = ridge_dir.predict(X_te_c_sc)

    # Ridge City Residual
    ridge_res = LinearBaseline(horizon=horizon, alpha=1.0, residual=True, target_feature_idx=0)
    ridge_res.fit(X_tr_c_sc, y_train_c)
    y_pred_ridge_res = ridge_res.predict(X_te_c_sc)

    # Multi-Station Ridge
    s_tr_one_hot = np.eye(len(DELHI_STATIONS))[s_train_multi]
    ridge_multi_res = LinearBaseline(horizon=horizon, alpha=1.0, residual=True, target_feature_idx=0)
    ridge_multi_res.fit(X_tr_m_sc, y_train_multi, station_one_hot=s_tr_one_hot)

    ridge_multi_dir = LinearBaseline(horizon=horizon, alpha=1.0, residual=False, target_feature_idx=0)
    ridge_multi_dir.fit(X_tr_m_sc, y_train_multi, station_one_hot=s_tr_one_hot)

    # For city evaluation of Multi-Station Ridge: average across all 6 station one-hots
    preds_rm_c_list = []
    for s_idx in range(len(DELHI_STATIONS)):
        s_one_hot_te = np.zeros((len(X_te_c_sc), len(DELHI_STATIONS)))
        s_one_hot_te[:, s_idx] = 1.0
        preds_rm_c_list.append(ridge_multi_res.predict(X_te_c_sc, station_one_hot=s_one_hot_te))
    y_pred_ridge_multi_res_city = np.mean(preds_rm_c_list, axis=0)

    preds_rm_dir_c_list = []
    for s_idx in range(len(DELHI_STATIONS)):
        s_one_hot_te = np.zeros((len(X_te_c_sc), len(DELHI_STATIONS)))
        s_one_hot_te[:, s_idx] = 1.0
        preds_rm_dir_c_list.append(ridge_multi_dir.predict(X_te_c_sc, station_one_hot=s_one_hot_te))
    y_pred_ridge_multi_dir_city = np.mean(preds_rm_dir_c_list, axis=0)

    # Baseline metrics
    baselines = {
        "Persistence": y_pred_pers,
        "Same-Hour-Yesterday": y_pred_shy,
        "Ridge City (Direct)": y_pred_ridge_dir,
        "Ridge City (Residual)": y_pred_ridge_res,
        "Ridge Multi (Direct)": y_pred_ridge_multi_dir_city,
        "Ridge Multi (Residual)": y_pred_ridge_multi_res_city,
    }
    baseline_metrics = {}
    for name, pred in baselines.items():
        # Overall
        m_all = compute_metrics_masked(y_test_c, pred, mask_test_c)
        m_per_h = {}
        for h in REPORT_HORIZONS:
            h_idx = h - 1
            m_h = compute_metrics_masked(y_test_c[:, h_idx : h_idx + 1], pred[:, h_idx : h_idx + 1], mask_test_c[:, h_idx : h_idx + 1])
            m_per_h[h] = m_h
        baseline_metrics[name] = {"overall": m_all, "horizons": m_per_h}
        logger.info("Baseline %-24s -> RMSE: %.2f | MAE: %.2f | R2: %.4f", name, m_all["rmse"], m_all["mae"], m_all["r2"])

    # 7. Multi-Seed Model Training Loop (5 Seeds)
    models_to_train = [
        "City GRU (Residual)",
        "City GRU (Direct)",
        "Multi-Station GRU (Residual)",
        "LSTM (Residual)",
        "BiGRU (Residual)",
        "Quantile GRU (Residual)",
    ]

    seed_results = {m: [] for m in models_to_train}
    model_epoch_logs = {m: [] for m in models_to_train}
    best_models = {}
    best_preds = {}

    # Store loss curves for plotting (seed 42)
    history_curves = {}

    for seed in SEEDS:
        logger.info("\n========== RUNNING SEED %d ==========", seed)
        set_seed(seed)

        # 1. City GRU (Residual)
        gru_res = AQIGRU(input_size=num_features, hidden_size=64, num_layers=2, output_size=horizon, dropout=0.2)
        gru_res, meta_gru_res = train_torch_model(gru_res, loader_tr_res, loader_val_res, epochs=35, patience=7, lr=0.001)
        model_epoch_logs["City GRU (Residual)"].append((meta_gru_res["best_epoch"], meta_gru_res["stop_epoch"]))
        if seed == 42:
            history_curves["City GRU (Residual)"] = meta_gru_res["history"]

        # Inference City GRU (Residual)
        gru_res.eval()
        with torch.no_grad():
            pred_delta_sc = gru_res(torch.as_tensor(X_te_c_sc)).cpu().numpy()
        pred_delta = delta_scaler_c.inverse_transform(pred_delta_sc)
        pred_gru_res = np.clip(last_test_c[:, np.newaxis] + pred_delta, 0.0, 500.0)
        seed_results["City GRU (Residual)"].append(pred_gru_res)
        if seed == 42:
            best_models["City GRU (Residual)"] = gru_res
            best_preds["City GRU (Residual)"] = pred_gru_res

        # 2. City GRU (Direct)
        gru_dir = AQIGRU(input_size=num_features, hidden_size=64, num_layers=2, output_size=horizon, dropout=0.2)
        gru_dir, meta_gru_dir = train_torch_model(gru_dir, loader_tr_dir, loader_val_dir, epochs=35, patience=7, lr=0.001)
        model_epoch_logs["City GRU (Direct)"].append((meta_gru_dir["best_epoch"], meta_gru_dir["stop_epoch"]))

        gru_dir.eval()
        with torch.no_grad():
            pred_dir_sc = gru_dir(torch.as_tensor(X_te_c_sc)).cpu().numpy()
        pred_gru_dir = np.clip(direct_scaler_c.inverse_transform(pred_dir_sc), 0.0, 500.0)
        seed_results["City GRU (Direct)"].append(pred_gru_dir)
        if seed == 42:
            best_models["City GRU (Direct)"] = gru_dir
            best_preds["City GRU (Direct)"] = pred_gru_dir

        # 3. Multi-Station GRU (Residual)
        gru_multi = AQIStationEmbeddingGRU(
            input_size=num_features,
            num_stations=len(DELHI_STATIONS),
            station_embedding_dim=16,
            hidden_size=64,
            num_layers=2,
            output_size=horizon,
            dropout=0.2,
        )
        gru_multi, meta_multi = train_torch_model(gru_multi, loader_tr_multi, loader_val_multi, epochs=30, patience=6, lr=0.001)
        model_epoch_logs["Multi-Station GRU (Residual)"].append((meta_multi["best_epoch"], meta_multi["stop_epoch"]))

        # City test inference: average over all 6 station embeddings
        gru_multi.eval()
        multi_city_preds = []
        with torch.no_grad():
            for s_idx in range(len(DELHI_STATIONS)):
                s_tensor = torch.full((len(X_te_c_sc),), s_idx, dtype=torch.long)
                delta_sc_st = gru_multi(torch.as_tensor(X_te_c_sc), s_tensor).cpu().numpy()
                multi_city_preds.append(delta_sc_st)
        avg_delta_sc = np.mean(multi_city_preds, axis=0)
        pred_delta_multi = delta_scaler_m.inverse_transform(avg_delta_sc)
        pred_multi_city = np.clip(last_test_c[:, np.newaxis] + pred_delta_multi, 0.0, 500.0)
        seed_results["Multi-Station GRU (Residual)"].append(pred_multi_city)
        if seed == 42:
            best_models["Multi-Station GRU (Residual)"] = gru_multi
            best_preds["Multi-Station GRU (Residual)"] = pred_multi_city

        # 4. LSTM (Residual)
        lstm = AQILSTM(input_size=num_features, hidden_size=64, num_layers=2, output_size=horizon, dropout=0.2)
        lstm, meta_lstm = train_torch_model(lstm, loader_tr_res, loader_val_res, epochs=35, patience=7, lr=0.001)
        model_epoch_logs["LSTM (Residual)"].append((meta_lstm["best_epoch"], meta_lstm["stop_epoch"]))

        lstm.eval()
        with torch.no_grad():
            pred_lstm_sc = lstm(torch.as_tensor(X_te_c_sc)).cpu().numpy()
        pred_delta_lstm = delta_scaler_c.inverse_transform(pred_lstm_sc)
        pred_lstm = np.clip(last_test_c[:, np.newaxis] + pred_delta_lstm, 0.0, 500.0)
        seed_results["LSTM (Residual)"].append(pred_lstm)
        if seed == 42:
            best_preds["LSTM (Residual)"] = pred_lstm

        # 5. BiGRU (Residual)
        bigru = AQIBiGRU(input_size=num_features, hidden_size=64, num_layers=2, output_size=horizon, dropout=0.2)
        bigru, meta_bigru = train_torch_model(bigru, loader_tr_res, loader_val_res, epochs=35, patience=7, lr=0.001)
        model_epoch_logs["BiGRU (Residual)"].append((meta_bigru["best_epoch"], meta_bigru["stop_epoch"]))

        bigru.eval()
        with torch.no_grad():
            pred_bigru_sc = bigru(torch.as_tensor(X_te_c_sc)).cpu().numpy()
        pred_delta_bigru = delta_scaler_c.inverse_transform(pred_bigru_sc)
        pred_bigru = np.clip(last_test_c[:, np.newaxis] + pred_delta_bigru, 0.0, 500.0)
        seed_results["BiGRU (Residual)"].append(pred_bigru)
        if seed == 42:
            best_preds["BiGRU (Residual)"] = pred_bigru

        # 6. Quantile GRU (Residual, 10/50/90)
        q_gru = AQIQuantileGRU(
            input_size=num_features,
            hidden_size=64,
            num_layers=2,
            output_size=horizon,
            quantiles=(0.10, 0.50, 0.90),
            dropout=0.2,
        )
        q_gru, meta_q = train_torch_model(q_gru, loader_tr_res, loader_val_res, epochs=35, patience=7, lr=0.001, is_quantile=True)
        model_epoch_logs["Quantile GRU (Residual)"].append((meta_q["best_epoch"], meta_q["stop_epoch"]))

        q_gru.eval()
        with torch.no_grad():
            q_out = q_gru(torch.as_tensor(X_te_c_sc)).cpu().numpy()  # (N, 24, 3)

        # Unscale each quantile
        q10_raw = np.clip(last_test_c[:, np.newaxis] + delta_scaler_c.inverse_transform(q_out[..., 0]), 0.0, 500.0)
        q50_raw = np.clip(last_test_c[:, np.newaxis] + delta_scaler_c.inverse_transform(q_out[..., 1]), 0.0, 500.0)
        q90_raw = np.clip(last_test_c[:, np.newaxis] + delta_scaler_c.inverse_transform(q_out[..., 2]), 0.0, 500.0)

        seed_results["Quantile GRU (Residual)"].append(q50_raw)
        if seed == 42:
            best_preds["Quantile GRU (Residual)"] = q50_raw
            best_preds["Quantile GRU (q10)"] = q10_raw
            best_preds["Quantile GRU (q90)"] = q90_raw

    # 8. Compute Multi-Seed Statistics (Mean +/- Std)
    benchmark_table = {}
    for m in models_to_train:
        runs = seed_results[m]  # List of 5 prediction arrays
        rmse_runs = []
        mae_runs = []
        r2_runs = []
        mape_runs = []

        h_runs = {h: {"rmse": [], "mae": []} for h in REPORT_HORIZONS}

        for pred in runs:
            m_all = compute_metrics_masked(y_test_c, pred, mask_test_c)
            rmse_runs.append(m_all["rmse"])
            mae_runs.append(m_all["mae"])
            r2_runs.append(m_all["r2"])
            mape_runs.append(m_all["mape"])

            for h in REPORT_HORIZONS:
                h_idx = h - 1
                mh = compute_metrics_masked(y_test_c[:, h_idx : h_idx + 1], pred[:, h_idx : h_idx + 1], mask_test_c[:, h_idx : h_idx + 1])
                h_runs[h]["rmse"].append(mh["rmse"])
                h_runs[h]["mae"].append(mh["mae"])

        benchmark_table[m] = {
            "rmse_mean": float(np.mean(rmse_runs)),
            "rmse_std": float(np.std(rmse_runs)),
            "mae_mean": float(np.mean(mae_runs)),
            "mae_std": float(np.std(mae_runs)),
            "r2_mean": float(np.mean(r2_runs)),
            "r2_std": float(np.std(r2_runs)),
            "mape_mean": float(np.mean(mape_runs)),
            "mape_std": float(np.std(mape_runs)),
            "horizons": {
                h: {
                    "rmse_mean": float(np.mean(h_runs[h]["rmse"])),
                    "rmse_std": float(np.std(h_runs[h]["rmse"])),
                    "mae_mean": float(np.mean(h_runs[h]["mae"])),
                    "mae_std": float(np.std(h_runs[h]["mae"])),
                }
                for h in REPORT_HORIZONS
            },
        }

    # 9. Quantile Interval Coverage (80% nominal: q10 to q90)
    q10 = best_preds["Quantile GRU (q10)"]
    q90 = best_preds["Quantile GRU (q90)"]
    coverage_results = {}
    for h in REPORT_HORIZONS:
        h_idx = h - 1
        valid = mask_test_c[:, h_idx] & ~np.isnan(y_test_c[:, h_idx])
        yt = y_test_c[valid, h_idx]
        lo = q10[valid, h_idx]
        hi = q90[valid, h_idx]
        inside = (yt >= lo) & (yt <= hi)
        cov = float(np.mean(inside) * 100.0)
        coverage_results[h] = cov
        logger.info("Quantile GRU 80%% Coverage at h=%-2d: %.2f%%", h, cov)

    # 10. Significance Testing (Moving-Block Bootstrap: Best GRU vs Ridge City Residual)
    best_gru_pred = best_preds["City GRU (Residual)"]
    ridge_pred = y_pred_ridge_res

    # At h = 1
    err_ridge_h1 = (ridge_pred[:, 0] - y_test_c[:, 0])[mask_test_c[:, 0]]
    err_gru_h1 = (best_gru_pred[:, 0] - y_test_c[:, 0])[mask_test_c[:, 0]]
    diff_mean_h1, ci_lo_h1, ci_hi_h1 = moving_block_bootstrap(err_ridge_h1, err_gru_h1, block_size=168, n_boot=1000, seed=42)

    # At h = 24
    err_ridge_h24 = (ridge_pred[:, 23] - y_test_c[:, 23])[mask_test_c[:, 23]]
    err_gru_h24 = (best_gru_pred[:, 23] - y_test_c[:, 23])[mask_test_c[:, 23]]
    diff_mean_h24, ci_lo_h24, ci_hi_h24 = moving_block_bootstrap(err_ridge_h24, err_gru_h24, block_size=168, n_boot=1000, seed=42)

    sig_h1 = "Statistically Significant Winner" if (ci_lo_h1 > 0 or ci_hi_h1 < 0) else "statistically tied"
    sig_h24 = "Statistically Significant Winner" if (ci_lo_h24 > 0 or ci_hi_h24 < 0) else "statistically tied"

    logger.info("Bootstrap h=1  (Ridge - GRU): Mean=%.2f, 95%% CI=[%.2f, %.2f] -> %s", diff_mean_h1, ci_lo_h1, ci_hi_h1, sig_h1)
    logger.info("Bootstrap h=24 (Ridge - GRU): Mean=%.2f, 95%% CI=[%.2f, %.2f] -> %s", diff_mean_h24, ci_lo_h24, ci_hi_h24, sig_h24)

    # 11. "More Data" Evaluation: Multi-Station GRU vs Ridge Multi vs City GRU per Station
    multi_station_eval = {}
    best_multi_model = best_models["Multi-Station GRU (Residual)"]
    best_multi_model.eval()

    best_city_model = best_models["City GRU (Residual)"]
    best_city_model.eval()

    for s_idx, st in enumerate(DELHI_STATIONS):
        st_test_split = station_windows[st]["test"]
        X_te_st = st_test_split["X"]
        y_te_st = st_test_split["y"]
        mask_te_st = st_test_split["y_mask"]
        last_te_st = st_test_split["last_known"]

        X_te_st_sc = scale_X(X_te_st)

        # (a) Multi-Station GRU with station embedding s_idx
        with torch.no_grad():
            s_tensor = torch.full((len(X_te_st_sc),), s_idx, dtype=torch.long)
            p_m_delta_sc = best_multi_model(torch.as_tensor(X_te_st_sc), s_tensor).cpu().numpy()
        p_m_delta = delta_scaler_m.inverse_transform(p_m_delta_sc)
        p_multi_st = np.clip(last_te_st[:, np.newaxis] + p_m_delta, 0.0, 500.0)
        m_multi_st = compute_metrics_masked(y_te_st, p_multi_st, mask_te_st)

        # (b) Ridge Multi with one-hot station s_idx
        s_oh = np.zeros((len(X_te_st_sc), len(DELHI_STATIONS)))
        s_oh[:, s_idx] = 1.0
        p_ridge_multi_st = ridge_multi_res.predict(X_te_st_sc, station_one_hot=s_oh)
        m_ridge_multi_st = compute_metrics_masked(y_te_st, p_ridge_multi_st, mask_te_st)

        # (c) City GRU evaluated on station test set
        with torch.no_grad():
            p_c_delta_sc = best_city_model(torch.as_tensor(X_te_st_sc)).cpu().numpy()
        p_c_delta = delta_scaler_c.inverse_transform(p_c_delta_sc)
        p_city_on_st = np.clip(last_te_st[:, np.newaxis] + p_c_delta, 0.0, 500.0)
        m_city_on_st = compute_metrics_masked(y_te_st, p_city_on_st, mask_te_st)

        multi_station_eval[st] = {
            "Multi_GRU_RMSE": m_multi_st["rmse"],
            "Ridge_Multi_RMSE": m_ridge_multi_st["rmse"],
            "City_GRU_RMSE": m_city_on_st["rmse"],
        }
        logger.info(
            "Station %-15s -> Multi GRU: %.2f | Ridge Multi: %.2f | City GRU: %.2f",
            st,
            m_multi_st["rmse"],
            m_ridge_multi_st["rmse"],
            m_city_on_st["rmse"],
        )

    # 12. 4-Fold Expanding-Window Walk-Forward Cross-Validation (Train+Val)
    logger.info("\nRunning 4-Fold Expanding-Window Walk-Forward CV over Train+Validation...")
    # Concatenate train+val sequences
    X_tv = np.concatenate([X_train_c, X_val_c], axis=0)
    y_tv = np.concatenate([y_train_c, y_val_c], axis=0)
    mask_tv = np.concatenate([mask_train_c, mask_val_c], axis=0)
    last_tv = np.concatenate([last_train_c, last_val_c], axis=0)

    total_tv = len(X_tv)
    min_train_win = 8000
    step_win = (total_tv - min_train_win) // 4
    cv_results = []

    for fold in range(4):
        tr_end = min_train_win + fold * step_win
        val_start = tr_end
        val_end = tr_end + step_win if fold < 3 else total_tv

        X_fold_tr = X_tv[:tr_end]
        y_fold_tr = y_tv[:tr_end]
        last_fold_tr = last_tv[:tr_end]

        X_fold_val = X_tv[val_start:val_end]
        y_fold_val = y_tv[val_start:val_end]
        mask_fold_val = mask_tv[val_start:val_end]
        last_fold_val = last_tv[val_start:val_end]

        # Scaler
        f_sc = MinMaxScaler().fit(X_fold_tr.reshape(-1, num_features))
        X_f_tr_sc = f_sc.transform(X_fold_tr.reshape(-1, num_features)).reshape(X_fold_tr.shape)
        X_f_val_sc = f_sc.transform(X_fold_val.reshape(-1, num_features)).reshape(X_fold_val.shape)

        d_sc = StandardScaler().fit(y_fold_tr - last_fold_tr[:, np.newaxis])
        delta_f_tr_sc = d_sc.transform(y_fold_tr - last_fold_tr[:, np.newaxis])
        delta_f_val_sc = d_sc.transform(y_fold_val - last_fold_val[:, np.newaxis])

        # Train Ridge
        ridge_fold = LinearBaseline(horizon=horizon, alpha=1.0, residual=True, target_feature_idx=0)
        ridge_fold.fit(X_f_tr_sc, y_fold_tr)
        pred_ridge_fold = ridge_fold.predict(X_f_val_sc)
        m_rf = compute_metrics_masked(y_fold_val, pred_ridge_fold, mask_fold_val)

        # Train Best GRU (City GRU Residual)
        ld_tr = DataLoader(TensorDataset(torch.as_tensor(X_f_tr_sc), torch.as_tensor(delta_f_tr_sc)), batch_size=128, shuffle=True)
        ld_val = DataLoader(TensorDataset(torch.as_tensor(X_f_val_sc), torch.as_tensor(delta_f_val_sc)), batch_size=128, shuffle=False)
        gru_f = AQIGRU(input_size=num_features, hidden_size=64, num_layers=2, output_size=horizon)
        gru_f, _ = train_torch_model(gru_f, ld_tr, ld_val, epochs=20, patience=5)

        gru_f.eval()
        with torch.no_grad():
            p_gru_f_sc = gru_f(torch.as_tensor(X_f_val_sc)).cpu().numpy()
        pred_gru_f = last_fold_val[:, np.newaxis] + d_sc.inverse_transform(p_gru_f_sc)
        m_gf = compute_metrics_masked(y_fold_val, pred_gru_f, mask_fold_val)

        cv_results.append({
            "Fold": fold + 1,
            "Train_Windows": tr_end,
            "Val_Windows": val_end - val_start,
            "GRU_RMSE": m_gf["rmse"],
            "Ridge_RMSE": m_rf["rmse"],
        })
        logger.info("Fold %d -> GRU RMSE: %.2f | Ridge RMSE: %.2f", fold + 1, m_gf["rmse"], m_rf["rmse"])

    # 13. Generate Plots
    logger.info("\nGenerating Visualizations...")

    # Plot 1: Actual vs Predicted on Test Period with Rolling Mean
    plt.figure(figsize=(12, 5), dpi=300)
    test_dates = pd.to_datetime(splits_city["test"]["ts"])
    y_test_h1 = y_test_c[:, 0]
    pred_h1 = best_preds["City GRU (Residual)"][:, 0]
    
    # 24h rolling mean
    s_actual = pd.Series(y_test_h1, index=test_dates).rolling(24, min_periods=1).mean()
    s_pred = pd.Series(pred_h1, index=test_dates).rolling(24, min_periods=1).mean()

    plt.plot(test_dates, y_test_h1, label="Actual AQI (Hourly)", color="#94a3b8", alpha=0.35, linewidth=0.8)
    plt.plot(test_dates, s_actual, label="Actual (24h Rolling Mean)", color="#0f172a", linewidth=1.8)
    plt.plot(test_dates, s_pred, label="City GRU Residual Forecast (24h Rolling Mean)", color="#2563eb", linewidth=1.8, linestyle="--")

    plt.title("Delhi City AQI: Actual vs Predicted (Test Period: 2025-10-01 to 2025-12-31)", fontsize=13, fontweight="bold")
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Air Quality Index (AQI)", fontsize=11)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=14))
    plt.grid(True, alpha=0.3, linestyle=":")
    plt.legend(frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(outputs_dir / "test_actual_vs_predicted.png")
    plt.close()

    # Plot 2: Training & Validation Loss Curves
    plt.figure(figsize=(9, 4.5), dpi=300)
    hist = history_curves["City GRU (Residual)"]
    epochs_range = np.arange(1, len(hist["train_loss"]) + 1)
    plt.plot(epochs_range, hist["train_loss"], label="Training Huber Loss", color="#0284c7", linewidth=2.0)
    plt.plot(epochs_range, hist["val_loss"], label="Validation Huber Loss", color="#ea580c", linewidth=2.0)
    best_ep = model_epoch_logs["City GRU (Residual)"][0][0]
    plt.axvline(best_ep, color="#16a34a", linestyle=":", label=f"Best Early-Stop Epoch ({best_ep})")

    plt.title("City GRU (Residual): Training Loss Curves (Data Range: 2023-01-01 to 2025-09-30)", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Huber Loss (delta=1.0)", fontsize=11)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.grid(True, alpha=0.3, linestyle=":")
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(outputs_dir / "training_loss_curves.png")
    plt.close()

    # Plot 3: Error vs Forecast Horizon (h = 1..24)
    plt.figure(figsize=(9, 4.5), dpi=300)
    horizons_all = np.arange(1, 25)
    rmse_gru_h = [compute_metrics_masked(y_test_c[:, i:i+1], best_preds["City GRU (Residual)"][:, i:i+1], mask_test_c[:, i:i+1])["rmse"] for i in range(24)]
    rmse_ridge_h = [compute_metrics_masked(y_test_c[:, i:i+1], y_pred_ridge_res[:, i:i+1], mask_test_c[:, i:i+1])["rmse"] for i in range(24)]
    rmse_shy_h = [compute_metrics_masked(y_test_c[:, i:i+1], y_pred_shy[:, i:i+1], mask_test_c[:, i:i+1])["rmse"] for i in range(24)]
    rmse_pers_h = [compute_metrics_masked(y_test_c[:, i:i+1], y_pred_pers[:, i:i+1], mask_test_c[:, i:i+1])["rmse"] for i in range(24)]

    plt.plot(horizons_all, rmse_gru_h, marker="o", label="City GRU (Residual)", color="#2563eb", linewidth=2.0)
    plt.plot(horizons_all, rmse_ridge_h, marker="s", label="Ridge City (Residual)", color="#059669", linewidth=1.8, linestyle="--")
    plt.plot(horizons_all, rmse_shy_h, marker="^", label="Same-Hour-Yesterday", color="#d97706", linewidth=1.5, linestyle=":")
    plt.plot(horizons_all, rmse_pers_h, marker="x", label="Persistence", color="#dc2626", linewidth=1.5, linestyle="-.")

    plt.title("Forecast RMSE vs Lead Time (Test Period: 2025-10-01 to 2025-12-31)", fontsize=12, fontweight="bold")
    plt.xlabel("Lead Time Horizon (Hours Ahead)", fontsize=11)
    plt.ylabel("Test RMSE (AQI units)", fontsize=11)
    plt.xticks(np.arange(1, 25, 2))
    plt.grid(True, alpha=0.3, linestyle=":")
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(outputs_dir / "error_vs_horizon.png")
    plt.close()

    # Plot 4: Forecast with 80% Prediction Intervals (Quantile GRU)
    plt.figure(figsize=(12, 5), dpi=300)
    # Zoom in on a representative 14-day winter smog surge: Dec 01 to Dec 15, 2025
    zoom_mask = (test_dates >= "2025-12-01") & (test_dates <= "2025-12-15")
    z_dates = test_dates[zoom_mask]
    z_yt = y_test_h1[zoom_mask]
    z_q10 = q10[zoom_mask, 0]
    z_q50 = best_preds["Quantile GRU (Residual)"][zoom_mask, 0]
    z_q90 = q90[zoom_mask, 0]

    plt.plot(z_dates, z_yt, label="Actual Observed AQI", color="#0f172a", linewidth=2.0)
    plt.plot(z_dates, z_q50, label="Quantile GRU Median Forecast (h=1)", color="#7c3aed", linewidth=1.8, linestyle="--")
    plt.fill_between(z_dates, z_q10, z_q90, color="#a78bfa", alpha=0.35, label="80% Prediction Interval [q10, q90]")

    plt.title("Quantile GRU Probabilistic Forecast: Winter Smog Episode (2025-12-01 to 2025-12-15)", fontsize=12, fontweight="bold")
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Air Quality Index (AQI)", fontsize=11)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.grid(True, alpha=0.3, linestyle=":")
    plt.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    plt.savefig(outputs_dir / "quantile_forecast_intervals.png")
    plt.close()

    # 14. Save Benchmark Markdown Artifact & Summary Dictionary
    summary_output = {
        "benchmark_table": benchmark_table,
        "baseline_metrics": baseline_metrics,
        "model_epoch_logs": model_epoch_logs,
        "bootstrap": {
            "h1": {"diff_mean": diff_mean_h1, "ci_lower": ci_lo_h1, "ci_upper": ci_hi_h1, "winner": sig_h1},
            "h24": {"diff_mean": diff_mean_h24, "ci_lower": ci_lo_h24, "ci_upper": ci_hi_h24, "winner": sig_h24},
        },
        "coverage_80pct": coverage_results,
        "multi_station_eval": multi_station_eval,
        "cv_results": cv_results,
    }

    with open(outputs_dir / "hourly_benchmark_summary.json", "w") as f:
        json.dump(summary_output, f, indent=2)

    logger.info("Hourly Benchmark Completed Successfully! Results saved to %s", outputs_dir)
    return summary_output


if __name__ == "__main__":
    run_benchmark()
