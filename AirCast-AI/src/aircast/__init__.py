"""AirCast AI — intelligent air quality prediction and pollution risk analysis."""

from .data import (
    FEATURES,
    N_RECORDS,
    RANDOM_SEED,
    TARGET,
    UNITS,
    generate_dataset,
    risk_category,
)
from .models import (
    TrainingReport,
    feature_importance,
    predict_pm25,
    train_and_evaluate,
)

__all__ = [
    "FEATURES",
    "N_RECORDS",
    "RANDOM_SEED",
    "TARGET",
    "UNITS",
    "TrainingReport",
    "feature_importance",
    "generate_dataset",
    "predict_pm25",
    "risk_category",
    "train_and_evaluate",
]

__version__ = "1.0.0"
