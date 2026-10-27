# Air Quality Dataset Information

This project predicts the **Air Quality Index (AQI)** using historical pollutant and atmospheric observations.

## Primary Dataset
The standard dataset used for real-world benchmarking is the public **"Air Quality Data in India (2015-2020)"** dataset hosted on Kaggle:
- **Source**: [Kaggle - Air Quality Data in India](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)
- **Target File**: `city_day.csv`

### How to use the Kaggle dataset:
1. Download `city_day.csv` from Kaggle.
2. Place `city_day.csv` directly into either:
   - `data/raw/city_day.csv` or
   - `data/city_day.csv`
3. Run the preprocessing / training pipeline:
   ```bash
   python src/train.py --city Delhi
   ```

## Automatic Fallback (Realistic Synthetic Dataset)
If `city_day.csv` is not present, the data loader automatically generates a realistic multi-year synthetic dataset for the configured city (e.g., Delhi, 2015-2020 daily records).

The synthetic generator models:
- **Long-term trend**: Gradual industrial growth and policy intervention shifts.
- **Annual seasonality**: Severe winter inversion peaks (November to January) typical of northern India, and monsoon clearing (July to September).
- **Weekly cycles**: Weekday commuter spikes vs. weekend dips.
- **Physically correlated pollutants**:
  - $PM_{2.5}$ & $PM_{10}$ (particulate matter)
  - $NO_2$ & $CO$ (vehicular emissions)
  - $SO_2$ (industrial emissions)
  - $O_3$ (photochemical ozone)
- **Injected missing values**: 5-10% realistic missing segments to exercise time-based interpolation and imputation.
- **CPCB-calibrated AQI calculation**: Computed as the sub-index maximum across critical pollutants as defined by the Central Pollution Control Board (CPCB) of India.

This ensures the training, validation, evaluation, and test suites are 100% runnable out of the box.
