"""Model training, evaluation and selection for AirCast AI."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from .data import FEATURES, RANDOM_SEED, TARGET

TEST_SIZE = 0.2


@dataclass
class ModelResult:
    """Container for one trained model and its test-set metrics."""

    name: str
    model: object
    mae: float
    rmse: float
    r2: float
    y_test: np.ndarray
    y_pred: np.ndarray
    residuals: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.residuals = self.y_test - self.y_pred


@dataclass
class TrainingReport:
    """Everything the dashboard needs after a training run."""

    results: dict[str, ModelResult]
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    best_model_name: str

    @property
    def best(self) -> ModelResult:
        return self.results[self.best_model_name]

    def metrics_table(self) -> pd.DataFrame:
        rows = [
            {"Model": r.name, "MAE": r.mae, "RMSE": r.rmse, "R²": r.r2}
            for r in self.results.values()
        ]
        return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)


def build_models() -> dict[str, object]:
    """Return fresh, untrained estimators keyed by display name."""
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1, min_samples_leaf=2
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=250, learning_rate=0.05, max_depth=3, random_state=RANDOM_SEED
        ),
    }


def train_and_evaluate(df: pd.DataFrame, test_size: float = TEST_SIZE) -> TrainingReport:
    """Split the data, fit all three regressors and score them on the test set.

    The model with the lowest test-set RMSE is reported as the selected model.
    """
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED
    )

    results: dict[str, ModelResult] = {}
    for name, estimator in build_models().items():
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        results[name] = ModelResult(
            name=name,
            model=estimator,
            mae=float(mean_absolute_error(y_test, y_pred)),
            rmse=float(np.sqrt(mean_squared_error(y_test, y_pred))),
            r2=float(r2_score(y_test, y_pred)),
            y_test=y_test.to_numpy(),
            y_pred=np.asarray(y_pred),
        )

    best_name = min(results, key=lambda key: results[key].rmse)
    return TrainingReport(
        results=results,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        best_model_name=best_name,
    )


def feature_importance(report: TrainingReport) -> pd.DataFrame | None:
    """Return sorted feature importances when the selected model exposes them."""
    model = report.best.model
    if not hasattr(model, "feature_importances_"):
        return None
    return (
        pd.DataFrame({"Feature": FEATURES, "Importance": model.feature_importances_})
        .sort_values("Importance", ascending=True)
        .reset_index(drop=True)
    )


def predict_pm25(report: TrainingReport, values: dict[str, float], model_name: str | None = None) -> float:
    """Predict PM2.5 for a single set of environmental conditions."""
    name = model_name or report.best_model_name
    row = pd.DataFrame([{feature: values[feature] for feature in FEATURES}])
    prediction = float(report.results[name].model.predict(row)[0])
    return max(prediction, 0.0)
