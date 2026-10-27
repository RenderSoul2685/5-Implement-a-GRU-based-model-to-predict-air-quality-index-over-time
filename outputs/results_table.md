# AQI Next-Day Forecasting Benchmark Results (Test Period: 2025-10-01 to 2025-12-31)

Held-out test set evaluation on authentic Delhi daily series (92 target days scored on observed ground truth).

| Model | Test RMSE (mean ± std) | Test MAE (mean ± std) | Test R² (mean ± std) | Conclusion |
| :--- | :---: | :---: | :---: | :---: |
| **City GRU (Residual, 3 seeds)** | **40.42 ± 0.47** | **31.42 ± 0.32** | **0.8014 ± 0.0046** | **Tied with Ridge** |
| **Ridge Regression (14d lag)** | 40.46 | 31.81 | 0.8011 | **Tied with GRU** |
| **Persistence (tomorrow = today)** | 44.08 | 33.67 | 0.7639 | Baseline |
| **7-Day Moving Average** | 54.37 | 43.43 | 0.6408 | Baseline |

### Benchmark Takeaways:
- **Statistical Parity**: The GRU and Ridge differ by 0.04 RMSE points, which is less than the seed standard deviation of 0.47 RMSE points, so the two are treated as **tied**. Both significantly outperform the 7-day moving average and persistence baselines.
- **Horizon & Features**: From the previous 14 days of daily mean AQI, daily weather (temperature, humidity, wind speed, pressure), and cyclical day-of-week/month features, models predict tomorrow's daily mean AQI (horizon = 1 day).
- **Target Parameterization**: The City GRU operates on residual deltas (tomorrow minus today) added back to today's AQI before computing metrics.
