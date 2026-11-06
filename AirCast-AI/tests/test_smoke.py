"""End-to-end smoke test for the AirCast AI pipeline.

Run with:  python tests/test_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aircast import (  # noqa: E402
    FEATURES,
    TARGET,
    feature_importance,
    generate_dataset,
    predict_pm25,
    risk_category,
    train_and_evaluate,
)


def main() -> None:
    df = generate_dataset()

    assert df.shape == (5000, 7), df.shape
    assert list(df.columns) == FEATURES + [TARGET]
    assert df[TARGET].between(2, 300).all()
    assert not df.isna().any().any()

    # The same seed must reproduce identical records.
    assert df.equals(generate_dataset()), "dataset is not reproducible"

    report = train_and_evaluate(df)
    assert set(report.results) == {
        "Linear Regression",
        "Random Forest",
        "Gradient Boosting",
    }

    for name, result in report.results.items():
        assert result.mae > 0 and result.rmse > 0, name
        assert result.rmse >= result.mae - 1e-9, name
        assert result.r2 > 0.3, f"{name} R² unexpectedly low: {result.r2}"

    best = report.best
    assert best.rmse == min(r.rmse for r in report.results.values())

    importance = feature_importance(report)
    assert importance is not None
    assert abs(importance["Importance"].sum() - 1.0) < 1e-6

    prediction = predict_pm25(
        report,
        {
            "temperature": 26.0,
            "humidity": 70.0,
            "wind_speed": 1.2,
            "pressure": 1012.0,
            "hour": 9,
            "month": 1,
        },
    )
    assert 0 <= prediction <= 300, prediction

    label, colour = risk_category(prediction)
    assert label and colour.startswith("#")

    print(f"records           : {len(df):,}")
    print(f"selected model    : {best.name}")
    print(f"MAE / RMSE / R²   : {best.mae:.2f} / {best.rmse:.2f} / {best.r2:.3f}")
    print(f"sample prediction : {prediction:.1f} µg/m³ ({label})")
    print("all checks passed ✅")


if __name__ == "__main__":
    main()
