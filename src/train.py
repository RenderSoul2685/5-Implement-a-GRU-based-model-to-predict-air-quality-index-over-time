"""Training pipeline for AQI-GRU Next-Day Forecasting (2023-2025).

Task: from previous 14 days of daily mean AQI, daily weather, and cyclical calendar features,
predict tomorrow's daily mean AQI (single output, horizon = 1 day).

Trains City GRU in residual mode (tomorrow minus today) across random seeds (42, 123, 456),
with early stopping on validation loss up to 60 epochs.
Saves the best model and scalers to models/ for downstream evaluation and demo.
"""

import argparse
import copy
import json
import logging
from pathlib import Path
import random
import sys
import time
from typing import Dict, List, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import Config, DELHI_STATIONS
from src.model import AQIGRU
from src.preprocessing import (
    create_daily_windows,
    prepare_daily_dataframe,
    split_daily_windows_by_date,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute RMSE, MAE, and R2 on valid target arrays."""
    valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not np.any(valid):
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}
    yt = y_true[valid]
    yp = y_pred[valid]
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mae = float(mean_absolute_error(yt, yp))
    r2 = float(r2_score(yt, yp))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def compute_metrics_masked(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Compatibility function for tests: compute metrics on masked points."""
    valid = mask & ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not np.any(valid):
        return {"rmse": float("nan"), "mae": float("nan")}
    diff = y_true[valid] - y_pred[valid]
    return {
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "mae": float(np.mean(np.abs(diff))),
    }


def train_single_seed(
    seed: int,
    X_train_scaled: np.ndarray,
    y_delta_train_scaled: np.ndarray,
    X_val_scaled: np.ndarray,
    y_delta_val_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    last_known_val: np.ndarray,
    y_val_true: np.ndarray,
    val_obs_mask: np.ndarray,
    last_known_test: np.ndarray,
    y_test_true: np.ndarray,
    test_obs_mask: np.ndarray,
    target_scaler: StandardScaler,
    epochs: int = 60,
    patience: int = 12,
    batch_size: int = 32,
    lr: float = 0.001,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    device: str = "cpu",
) -> Dict:
    """Train City GRU for a single seed with early stopping on validation loss."""
    set_seed(seed)
    input_size = X_train_scaled.shape[2]

    model = AQIGRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=1,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-5
    )
    criterion = nn.HuberLoss(delta=1.0)

    train_ds = TensorDataset(
        torch.as_tensor(X_train_scaled, dtype=torch.float32),
        torch.as_tensor(y_delta_train_scaled, dtype=torch.float32).unsqueeze(-1),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_x = torch.as_tensor(X_val_scaled, dtype=torch.float32).to(device)
    val_y = torch.as_tensor(y_delta_val_scaled, dtype=torch.float32).unsqueeze(-1).to(device)

    best_val_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    patience_cnt = 0

    train_losses, val_losses = [], []
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        n_batches = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss_sum += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss_sum / max(n_batches, 1)

        # Validation loss
        model.eval()
        with torch.no_grad():
            v_pred = model(val_x)
            v_loss = criterion(v_pred, val_y).item()

        scheduler.step(v_loss)
        train_losses.append(avg_train_loss)
        val_losses.append(v_loss)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                logger.info("Seed %d: Early stopping at epoch %d (best epoch %d, val loss %.4f)",
                            seed, epoch, best_epoch, best_val_loss)
                break

    train_time = time.time() - t_start

    # Evaluate best weights on Val and Test
    model.load_state_dict(best_weights)
    model.eval()

    with torch.no_grad():
        val_sc = model(val_x).cpu().numpy().ravel()
        test_x = torch.as_tensor(X_test_scaled, dtype=torch.float32).to(device)
        test_sc = model(test_x).cpu().numpy().ravel()

    # Unscale delta and add today's AQI
    val_pred_delta = target_scaler.inverse_transform(val_sc.reshape(-1, 1)).ravel()
    val_pred_aqi = np.clip(last_known_val + val_pred_delta, 0.0, 500.0)

    test_pred_delta = target_scaler.inverse_transform(test_sc.reshape(-1, 1)).ravel()
    test_pred_aqi = np.clip(last_known_test + test_pred_delta, 0.0, 500.0)

    # Compute metrics on observed target days only
    val_metrics = compute_metrics(y_val_true[val_obs_mask], val_pred_aqi[val_obs_mask])
    test_metrics = compute_metrics(y_test_true[test_obs_mask], test_pred_aqi[test_obs_mask])

    logger.info(
        "Seed %d Result -> Val RMSE: %.2f (best epoch %d) | Test RMSE: %.2f, MAE: %.2f, R2: %.4f",
        seed, val_metrics["rmse"], best_epoch, test_metrics["rmse"], test_metrics["mae"], test_metrics["r2"]
    )

    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "test_predictions": test_pred_aqi,
        "weights": best_weights,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_time": train_time,
    }


def main():
    parser = argparse.ArgumentParser(description="Train City GRU for next-day AQI forecasting.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456], help="Random seeds to evaluate.")
    parser.add_argument("--epochs", type=int, default=60, help="Max epochs.")
    parser.add_argument("--patience", type=int, default=12, help="Early stopping patience.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--hidden-size", type=int, default=64, help="GRU hidden units.")
    parser.add_argument("--num-layers", type=int, default=2, help="GRU layers.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate.")
    args = parser.parse_args()

    cfg = Config(
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        seeds=args.seeds,
    )

    repo_root = Path(__file__).resolve().parent.parent
    processed_dir = repo_root / "data" / "processed"
    raw_dir = repo_root / "data" / "raw"
    models_dir = repo_root / "models"
    outputs_dir = repo_root / "outputs"
    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    city_path = processed_dir / "delhi_city_2023_2025.csv"
    weather_path = raw_dir / "delhi_weather_2023_2025.csv"

    if not city_path.exists() or not weather_path.exists():
        raise FileNotFoundError(f"Missing required data files: {city_path} or {weather_path}")

    logger.info("Loading daily city and weather series...")
    df_city = pd.read_csv(city_path)
    df_weather = pd.read_csv(weather_path)

    # 1. Prepare daily series with <= 2 days interpolation and calendar features
    df_prepared = prepare_daily_dataframe(df_city, df_weather, interp_limit_days=cfg.interp_limit_days)

    feature_cols = cfg.feature_cols
    logger.info("Features (%d): %s", len(feature_cols), feature_cols)

    # 2. Generate 14-day sliding windows
    X_all, y_all, y_obs_all, last_known_all, target_dates = create_daily_windows(
        df_prepared,
        feature_cols=feature_cols,
        target_col="AQI_interp",
        raw_col="AQI_observed",
        lookback=cfg.lookback,
        horizon=cfg.horizon,
    )

    # 3. Chronological split by target date
    splits = split_daily_windows_by_date(
        X_all, y_all, y_obs_all, last_known_all, target_dates,
        train_end=cfg.train_end_date,
        val_start=cfg.val_start_date,
        val_end=cfg.val_end_date,
        test_start=cfg.test_start_date,
        test_end=cfg.test_end_date,
    )

    X_train = splits["train"]["X"]
    y_train = splits["train"]["y"]
    last_known_train = splits["train"]["last_known"]

    X_val = splits["val"]["X"]
    y_val = splits["val"]["y"]
    y_val_obs = splits["val"]["y_mask"]
    last_known_val = splits["val"]["last_known"]

    X_test = splits["test"]["X"]
    y_test = splits["test"]["y"]
    y_test_obs = splits["test"]["y_mask"]
    last_known_test = splits["test"]["last_known"]
    test_dates = splits["test"]["target_dates"]

    # 4. Residual targets: delta = tomorrow - today
    y_delta_train = y_train - last_known_train
    y_delta_val = y_val - last_known_val

    # 5. Fit scalers strictly on train set
    feat_scaler = StandardScaler()
    feat_scaler.fit(X_train.reshape(-1, len(feature_cols)))

    X_train_scaled = feat_scaler.transform(X_train.reshape(-1, len(feature_cols))).reshape(X_train.shape)
    X_val_scaled = feat_scaler.transform(X_val.reshape(-1, len(feature_cols))).reshape(X_val.shape)
    X_test_scaled = feat_scaler.transform(X_test.reshape(-1, len(feature_cols))).reshape(X_test.shape)

    target_scaler = StandardScaler()
    y_delta_train_scaled = target_scaler.fit_transform(y_delta_train.reshape(-1, 1)).ravel()
    y_delta_val_scaled = target_scaler.transform(y_delta_val.reshape(-1, 1)).ravel()

    # 6. Train across seeds
    seed_results = []
    best_overall_seed = None
    best_val_rmse = float("inf")

    for seed in cfg.seeds:
        res = train_single_seed(
            seed=seed,
            X_train_scaled=X_train_scaled,
            y_delta_train_scaled=y_delta_train_scaled,
            X_val_scaled=X_val_scaled,
            y_delta_val_scaled=y_delta_val_scaled,
            X_test_scaled=X_test_scaled,
            last_known_val=last_known_val,
            y_val_true=y_val,
            val_obs_mask=y_val_obs,
            last_known_test=last_known_test,
            y_test_true=y_test,
            test_obs_mask=y_test_obs,
            target_scaler=target_scaler,
            epochs=cfg.epochs,
            patience=cfg.patience,
            batch_size=cfg.batch_size,
            lr=cfg.learning_rate,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        )
        seed_results.append(res)
        if res["val_metrics"]["rmse"] < best_val_rmse:
            best_val_rmse = res["val_metrics"]["rmse"]
            best_overall_seed = res

    # 7. Save winning model artifacts
    logger.info("Selected winning seed: %d (Val RMSE: %.2f)", best_overall_seed["seed"], best_val_rmse)
    torch.save(best_overall_seed["weights"], models_dir / "best_model.pt")
    joblib.dump(feat_scaler, models_dir / "feature_scaler.joblib")
    joblib.dump(target_scaler, models_dir / "target_scaler.joblib")

    test_rmses = [r["test_metrics"]["rmse"] for r in seed_results]
    test_maes = [r["test_metrics"]["mae"] for r in seed_results]
    test_r2s = [r["test_metrics"]["r2"] for r in seed_results]

    config_info = {
        "model_type": "AQIGRU_Daily",
        "task": "next_day_forecast",
        "lookback": cfg.lookback,
        "horizon": cfg.horizon,
        "mode": "residual",
        "feature_cols": feature_cols,
        "selected_seed": best_overall_seed["seed"],
        "architecture": {
            "input_size": len(feature_cols),
            "hidden_size": cfg.hidden_size,
            "num_layers": cfg.num_layers,
            "output_size": 1,
            "dropout": cfg.dropout,
        },
        "performance_3seeds": {
            "test_rmse_mean": float(np.mean(test_rmses)),
            "test_rmse_std": float(np.std(test_rmses)),
            "test_mae_mean": float(np.mean(test_maes)),
            "test_mae_std": float(np.std(test_maes)),
            "test_r2_mean": float(np.mean(test_r2s)),
            "test_r2_std": float(np.std(test_r2s)),
        },
    }

    with open(models_dir / "model_config.json", "w") as f:
        json.dump(config_info, f, indent=2)

    # 8. Save training loss curve plot
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2563eb", "#059669", "#d97706"]
    for i, res in enumerate(seed_results):
        c = colors[i % len(colors)]
        ax.plot(res["train_losses"], label=f"Seed {res['seed']} Train Loss", color=c, linestyle="--", alpha=0.7)
        ax.plot(res["val_losses"], label=f"Seed {res['seed']} Val Loss", color=c, linewidth=2.0)
    ax.set_title("City GRU (Residual) Next-Day Training & Validation Loss (Smooth L1)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(outputs_dir / "loss_curve.png", dpi=300)
    plt.savefig(outputs_dir / "training_loss_curves.png", dpi=300)
    plt.close()
    logger.info("Saved loss curve to %s", outputs_dir / "loss_curve.png")


if __name__ == "__main__":
    main()
