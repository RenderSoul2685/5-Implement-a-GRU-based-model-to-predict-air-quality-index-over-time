# AQI-GRU: Next-Day Air Quality Index Forecasting (Delhi 2023–2025)

An end-to-end deep learning system predicting tomorrow's daily mean Air Quality Index (AQI) across Delhi using 14 days of historical monitoring station telemetry, cyclical calendar encodings, and meteorological reanalysis.

---

## 🎯 1. Project Goal
The primary objective of `aqi-gru` is to forecast next-day urban air quality across the National Capital Territory (NCT) of Delhi. Given the previous 14 days of daily mean AQI, daily weather (temperature, humidity, wind speed, pressure), and day-of-week/month cyclical features, the model predicts tomorrow's daily mean AQI ($horizon = 1$ day), evaluated rigorously against linear (Ridge) and non-parametric (Persistence, 7-Day Moving Average) baselines.

---

## 📊 2. Authentic Data & Provenance
* **Timeframe**: Strictly **2023-01-01 to 2025-12-31** (1,096 consecutive days; 100% authentic data, zero synthetic or simulated records).
* **Monitoring Stations**: 6 continuous CAAQMS stations from the Central Pollution Control Board (CPCB) / DPCC network downloaded via OpenCity:
  1. `Anand_Vihar` (Eastern border transit hub / heavy interstate bus traffic)
  2. `Bawana` (North-western industrial corridor)
  3. `ITO` (Central high-density arterial intersection)
  4. `Mandir_Marg` (Central institutional background site)
  5. `Punjabi_Bagh` (Western commercial/residential corridor)
  6. `RK_Puram` (Southern urban residential sector)
* **Daily Summary & Missing Data Rule**:
  - A day counts only if it has at least **18 valid hours** (75% CPCB completeness criterion); days with $< 18$ valid hours are treated as missing (`NaN`).
  - **Partial Day Bias**: For ongoing or live days with fewer than 24 valid hours (e.g., 'Today so far (23 of 24 hours)'), computing the arithmetic mean over a partial day can bias the daily mean. Delhi air quality experiences distinct diurnal fluctuations (nighttime stagnation peaks vs. daytime boundary layer venting); an incomplete day reflects only the hours observed so far rather than a complete 24-hour cycle.
  - Gaps of **up to 2 days only** are linearly interpolated (`limit = 2`). Gaps exceeding 2 days remain `NaN` and any input window spanning such gaps is discarded.
  - Evaluation and scoring are performed **strictly on observed target days**.
* **City-Level Series**: Daily arithmetic mean computed across active monitoring stations.
* **Feature Set (9 Features)**:
  - `AQI_interp`: Past daily mean AQI history (linear interpolation $\le 2$ days).
  - Daily Weather: Open-Meteo Delhi reanalysis (`temperature`, `humidity`, `wind_speed`, `pressure`).
  - Cyclical Calendar: `dayofweek_sin`, `dayofweek_cos`, `month_sin`, `month_cos`.
* **Chronological Splits (Leakage-Free)**:
  - **Train**: `2023-01-01` to `2025-06-30` (898 daily sliding windows) — Scalers fit strictly on this split.
  - **Validation**: `2025-07-01` to `2025-09-30` (92 daily sliding windows; monsoon low-pollution regime).
  - **Test**: `2025-10-01` to `2025-12-31` (92 daily sliding windows; severe winter smog regime).

---

## 🧠 3. Model Architecture & Selection
* **Input Window**: 14-day historical lookback ($lookback = 14$, 9 features).
* **Architecture**: 2-layer Gated Recurrent Unit (hidden size: 64, dropout: 0.20) with a linear projection head outputting a single prediction ($horizon = 1$).
* **Target Formulation**: Residual delta ($\Delta \text{AQI}_{t+1} = \text{AQI}_{t+1} - \text{AQI}_t$) added back to today's anchor AQI before metric computation.
* **Training Protocol**: Adam optimizer, Smooth L1 (Huber) loss, batch size 32, early stopping on validation loss (max 60 epochs, patience 12) across random seeds: 42, 123, 456.
* **Selected Winning Model**: Seed 42 (Validation RMSE: **18.64**).

---

## 📈 4. Test Split Results

Evaluated on the held-out winter test split (`2025-10-01` to `2025-12-31`; 92 observed target days):

| Model | Test RMSE (mean ± std) | Test MAE (mean ± std) | Test R² (mean ± std) | Conclusion |
| :--- | :---: | :---: | :---: | :---: |
| **City GRU (Residual, 3 seeds)** | **40.42 ± 0.47** | **31.42 ± 0.32** | **0.8014 ± 0.0046** | **Tied with Ridge** |
| **Ridge Regression (14d lag)** | 40.46 | 31.81 | 0.8011 | **Tied with GRU** |
| **Persistence (tomorrow = today)** | 44.08 | 33.67 | 0.7639 | Baseline |
| **7-Day Moving Average** | 54.37 | 43.43 | 0.6408 | Baseline |

### Key Benchmark Findings:
- **Tied Performance**: The GRU and Ridge regression differ by **0.04 RMSE points**, which is substantially less than the seed-to-seed standard deviation of **0.47 RMSE points**. Consequently, the two models are treated as **tied**.
- **Baselines Comparison**: Both City GRU and Ridge substantially outperform the 7-day moving average (54.37 RMSE) and persistence (44.08 RMSE), demonstrating strong predictive signal in multi-day lags and weather co-variates.
- **Diagnostic Plots in `outputs/`**:
  - `outputs/loss_curve.png`: Convergence curves for training and validation loss across epochs.
  - `outputs/test_actual_vs_predicted.png`: Time series overlay of observed actuals vs. model forecasts.
  - `outputs/scatter_predicted_vs_actual.png`: Scatter plot with 1:1 parity line and $R^2 = 0.8014$.

---

## 🚀 5. How to Run

### Installation
```bash
git clone https://github.com/RenderSoul2685/5-Implement-a-GRU-based-model-to-predict-air-quality-index-over-time.git
cd 5-Implement-a-GRU-based-model-to-predict-air-quality-index-over-time
pip install -r requirements.txt
```

### 1. Train Models
Trains the City GRU in residual mode across seeds 42, 123, and 456 with early stopping on validation loss (max 60 epochs), saving the winning weights and scalers to `models/`:
```bash
python src/train.py --seeds 42 123 456 --epochs 60 --patience 12
```

### 2. Evaluate on Test Split
Evaluates the trained GRU across seeds against Ridge, Persistence, and 7-Day Moving Average baselines, producing `outputs/results_table.md` and comparison figures:
```bash
python src/evaluate.py
```

### 3. Interactive Next-Day Forecast Demo
Forecasts tomorrow's AQI from the previous 14 days for any station and date, printing the 14-day history, tomorrow's forecast, CPCB health classification, and ground-truth comparison:
```bash
python src/demo.py --station Anand_Vihar --date 2025-11-15
```

### 4. Interactive Next-Day Forecast Website
Refresh the local telemetry cache from authentic CPCB/Open-Meteo feeds:
```bash
python src/refresh_live.py
```
Launch the minimal Django demonstration server:
```bash
python webapp/manage.py runserver 8000
```
Then open `http://127.0.0.1:8000` in your web browser. The application offers two interactive modes:
- **Latest data (default)**: Evaluates fresh real-world CPCB telemetry (ending on `2026-10-25`) to forecast tomorrow's AQI.
- **Replay (2025 test set)**: Replays historical forecast anchors (`2025-10-01` to `2025-12-30`) against held-out observed ground truth.
- Both modes and the About page include an official CPCB AQI reference table highlighting tomorrow's predicted category.

### 5. Run Unit Tests
```bash
pytest
```

---

## ⚠️ 6. Honest Limitations
1. **Station Sample Size (6 Stations)**: Delhi NCT operates over 40 CAAQMS monitoring stations. While the 6 stations capture primary industrial, traffic, and residential corridors, outer peripheral rural regimes are not fully represented.
2. **Missing Precursors in 2023**: Because 2023 raw data from CPCB provided published aggregate AQI only, chemical precursor concentrations ($\text{PM}_{2.5}$, $\text{NO}_2$, $\text{SO}_2$) could not be included as model inputs without discarding all of 2023.
3. **Daily Aggregation Filter**: Requiring $\ge 18$ valid hours per day drops days with prolonged telemetry outages.
4. **Seasonal Split Regime Disparity**: The validation split is restricted to monsoon months (July–Sept; lowest annual pollution), whereas the test split covers winter inversion (Oct–Dec; severe smog episodes).
5. **Feed Freshness Dependency**: Live forecasts depend on the freshness of the CPCB-derived feed.
6. **Partial Day Bias**: When live feeds have fewer than 24 hours for the current observation anchor (e.g., 23 of 24 hours), the partial day can bias the daily mean because diurnal emissions and atmospheric mixing height vary dramatically between day and night.
7. **24-Hour Rolling Average Live Inputs**: Live inputs come from a CPCB feed of 24-hour rolling averages, unlike the training data; forecasts are indicative only.

