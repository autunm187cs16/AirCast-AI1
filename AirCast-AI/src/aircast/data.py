"""Synthetic air-quality dataset generation for AirCast AI.

The dataset produced here is SIMULATED, not measured. It exists so the
demonstration is reproducible: the same random seed always yields the same
records. Do not present these values as real monitoring-station data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_RECORDS = 5000

FEATURES = ["temperature", "humidity", "wind_speed", "pressure", "hour", "month"]
TARGET = "pm25"
UNITS = {
    "temperature": "°C",
    "humidity": "%",
    "wind_speed": "m/s",
    "pressure": "hPa",
    "hour": "0-23",
    "month": "1-12",
    "pm25": "µg/m³",
}


def generate_dataset(n_records: int = N_RECORDS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Return a reproducible synthetic PM2.5 dataset.

    Parameters
    ----------
    n_records : number of rows to simulate.
    seed : fixed random seed so the demonstration can be reproduced.
    """
    rng = np.random.default_rng(seed)

    hour = rng.integers(0, 24, n_records)
    month = rng.integers(1, 13, n_records)

    # Seasonal temperature curve: warmest around June-July.
    seasonal = 6.0 * np.sin((month - 3) / 12.0 * 2.0 * np.pi)
    diurnal = 4.0 * np.sin((hour - 6) / 24.0 * 2.0 * np.pi)
    temperature = 24.0 + seasonal + diurnal + rng.normal(0.0, 2.5, n_records)

    humidity = np.clip(58.0 - 0.8 * (temperature - 24.0) + rng.normal(0.0, 11.0, n_records), 12.0, 98.0)
    wind_speed = np.clip(rng.gamma(2.0, 1.1, n_records), 0.1, 12.0)
    pressure = 1009.0 + rng.normal(0.0, 6.5, n_records)

    # Traffic-like rush hours raise concentrations.
    traffic = (
        6.0 * np.exp(-((hour - 8.0) ** 2) / 6.0)
        + 7.0 * np.exp(-((hour - 19.0) ** 2) / 8.0)
        + 2.0
    )
    # Winter inversion conditions trap particulates near the surface.
    winter = 26.0 * np.cos((month - 1) / 12.0 * 2.0 * np.pi)
    # Wind disperses pollution; humidity and pressure support accumulation.
    wind_effect = -6.0 * wind_speed
    humidity_effect = 0.72 * (humidity - 50.0)
    pressure_effect = 0.35 * (pressure - 1009.0)

    noise = rng.normal(0.0, 6.5, n_records)
    pm25 = 48.0 + traffic + winter + wind_effect + humidity_effect + pressure_effect + noise
    pm25 = np.clip(pm25, 2.0, 300.0)

    return pd.DataFrame(
        {
            "temperature": np.round(temperature, 1),
            "humidity": np.round(humidity, 1),
            "wind_speed": np.round(wind_speed, 2),
            "pressure": np.round(pressure, 1),
            "hour": hour.astype(int),
            "month": month.astype(int),
            "pm25": np.round(pm25, 1),
        }
    )


def risk_category(pm25: float) -> tuple[str, str]:
    """Map a PM2.5 value to an ILLUSTRATIVE band and a colour.

    These bands are for demonstration only; they are not official air-quality
    guidance and must not be used for health decisions.
    """
    if pm25 <= 30:
        return "Good", "#2e7d32"
    if pm25 <= 60:
        return "Satisfactory", "#7cb342"
    if pm25 <= 90:
        return "Moderate", "#fbc02d"
    if pm25 <= 120:
        return "Poor", "#f57c00"
    if pm25 <= 250:
        return "Very Poor", "#d32f2f"
    return "Severe", "#7b1fa2"
