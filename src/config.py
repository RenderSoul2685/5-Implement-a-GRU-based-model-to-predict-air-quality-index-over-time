"""Configuration settings and hyperparameters for AQI GRU forecasting."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple
import torch


DELHI_STATIONS: List[str] = [
    "Anand_Vihar",
    "Punjabi_Bagh",
    "RK_Puram",
    "ITO",
    "Mandir_Marg",
    "Bawana",
]

# Raw station column mappings from CPCB / OpenCity headers to standardized identifiers
RAW_STATION_COLUMNS_MAP = {
    "PM2.5 (ug/m3)": "PM2.5",
    "PM10 (ug/m3)": "PM10",
    "NO (ug/m3)": "NO",
    "NO2 (ug/m3)": "NO2",
    "NOx (ppb)": "NOx",
    "NH3 (ug/m3)": "NH3",
    "SO2 (ug/m3)": "SO2",
    "CO (mg/m3)": "CO",
    "Ozone (ug/m3)": "Ozone",
    "AT (degC)": "AT",
    "RH (%)": "RH",
    "WS (m/s)": "WS",
    "WD (deg)": "WD",
    "BP (mmHg)": "BP",
}

# CPCB Sub-Index Breakpoint Tables (C_low, C_high, I_low, I_high)
CPCB_BREAKPOINTS = {
    "PM2.5": [
        (0.0, 30.0, 0.0, 50.0),
        (30.0, 60.0, 50.0, 100.0),
        (60.0, 90.0, 100.0, 200.0),
        (90.0, 120.0, 200.0, 300.0),
        (120.0, 250.0, 300.0, 400.0),
        (250.0, 500.0, 400.0, 500.0),
    ],
    "PM10": [
        (0.0, 50.0, 0.0, 50.0),
        (50.0, 100.0, 50.0, 100.0),
        (100.0, 250.0, 100.0, 200.0),
        (250.0, 350.0, 200.0, 300.0),
        (350.0, 430.0, 300.0, 400.0),
        (430.0, 600.0, 400.0, 500.0),
    ],
    "NO2": [
        (0.0, 40.0, 0.0, 50.0),
        (40.0, 80.0, 50.0, 100.0),
        (80.0, 180.0, 100.0, 200.0),
        (180.0, 280.0, 200.0, 300.0),
        (280.0, 400.0, 300.0, 400.0),
        (400.0, 600.0, 400.0, 500.0),
    ],
    "SO2": [
        (0.0, 40.0, 0.0, 50.0),
        (40.0, 80.0, 50.0, 100.0),
        (80.0, 380.0, 100.0, 200.0),
        (380.0, 800.0, 200.0, 300.0),
        (800.0, 1600.0, 300.0, 400.0),
        (1600.0, 2000.0, 400.0, 500.0),
    ],
    "CO": [
        (0.0, 1.0, 0.0, 50.0),
        (1.0, 2.0, 50.0, 100.0),
        (2.0, 10.0, 100.0, 200.0),
        (10.0, 17.0, 200.0, 300.0),
        (17.0, 34.0, 300.0, 400.0),
        (34.0, 50.0, 400.0, 500.0),
    ],
    "Ozone": [
        (0.0, 50.0, 0.0, 50.0),
        (50.0, 100.0, 50.0, 100.0),
        (100.0, 168.0, 100.0, 200.0),
        (168.0, 208.0, 200.0, 300.0),
        (208.0, 748.0, 300.0, 400.0),
        (748.0, 1000.0, 400.0, 500.0),
    ],
}


@dataclass
class Config:
    """Project configuration containing paths, hyperparameters, and constants."""

    # Project directories (relative to project root)
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    data_raw_dir: Path = field(init=False)
    data_processed_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    outputs_dir: Path = field(init=False)

    # City and dataset selection
    city: str = "Delhi"
    allow_synthetic: bool = False
    is_synthetic: bool = False
    target_col: str = "AQI"
    pollutant_cols: List[str] = field(default_factory=list)  # Pollutants not used
    temporal_cols: List[str] = field(
        default_factory=lambda: [
            "dayofweek_sin",
            "dayofweek_cos",
            "month_sin",
            "month_cos",
        ]
    )
    weather_cols: List[str] = field(
        default_factory=lambda: [
            "temperature",
            "humidity",
            "wind_speed",
            "pressure",
        ]
    )
    with_weather: bool = True

    # Strictly enforced date range: 2023-01-01 to 2025-12-31
    start_date: str = "2023-01-01"
    end_date: str = "2025-12-31"
    train_end_date: str = "2025-06-30"
    val_start_date: str = "2025-07-01"
    val_end_date: str = "2025-09-30"
    test_start_date: str = "2025-10-01"
    test_end_date: str = "2025-12-31"

    # Multi-station list
    stations: List[str] = field(default_factory=lambda: list(DELHI_STATIONS))
    station_embedding_dim: int = 16

    # Daily sequence windowing parameters
    lookback: int = 14  # previous 14 days used as context
    horizon: int = 1    # tomorrow's daily mean AQI (1 day ahead)
    interp_limit_days: int = 2  # interpolate gaps of up to 2 days only

    # Model architecture hyperparameters
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    mode: str = "residual"  # tomorrow minus today

    # Training hyperparameters
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    epochs: int = 60
    patience: int = 12  # Early stopping patience on validation loss
    grad_clip: float = 1.0
    random_seed: int = 42
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])

    # Hardware device
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )

    def __post_init__(self):
        """Construct dependent directory paths and ensure they exist."""
        self.data_raw_dir = self.base_dir / "data" / "raw"
        self.data_processed_dir = self.base_dir / "data" / "processed"
        self.models_dir = self.base_dir / "models"
        self.outputs_dir = self.base_dir / "outputs"

        for directory in [
            self.data_raw_dir,
            self.data_processed_dir,
            self.models_dir,
            self.outputs_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def feature_cols(self) -> List[str]:
        """All features fed into the model (AQI_interp + weather + temporal)."""
        target_name = "AQI_interp" if self.target_col == "AQI" else self.target_col
        cols = [target_name]
        if self.with_weather:
            cols += self.weather_cols
        cols += self.temporal_cols
        return cols

    @property
    def num_features(self) -> int:
        """Total number of input features per time step."""
        return len(self.feature_cols)

    @property
    def num_stations(self) -> int:
        """Total number of monitoring stations."""
        return len(self.stations)


# Indian Central Pollution Control Board (CPCB) AQI Categories
AQI_CATEGORIES: List[Tuple[float, float, str, str]] = [
    (0.0, 50.0, "Good", "Minimal impact on health."),
    (51.0, 100.0, "Satisfactory", "Minor breathing discomfort to sensitive people."),
    (101.0, 200.0, "Moderate", "Breathing discomfort to people with lungs, asthma and heart diseases."),
    (201.0, 300.0, "Poor", "Breathing discomfort to most people on prolonged exposure."),
    (301.0, 400.0, "Very Poor", "Respiratory illness on prolonged exposure."),
    (401.0, float("inf"), "Severe", "Affects healthy people and seriously impacts those with existing diseases."),
]


def get_aqi_category(aqi_value: float) -> Tuple[str, str]:
    """Return the CPCB AQI Category and health advisory for a given numerical AQI.

    Args:
        aqi_value: Numerical Air Quality Index value.

    Returns:
        Tuple of (Category Name, Health Advisory Description).
    """
    for low, high, category, advisory in AQI_CATEGORIES:
        if low <= aqi_value <= high:
            return category, advisory
    return "Severe", "Emergency health warnings of serious injury."
