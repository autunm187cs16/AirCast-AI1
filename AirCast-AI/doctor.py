"""Environment and installation check for AirCast AI.

Run from the project root:  python doctor.py
"""

from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MIN_PY = (3, 10)
REQUIRED = {
    "streamlit": "1.32",
    "pandas": "2.0",
    "numpy": "1.24",
    "sklearn": "1.3",
    "plotly": "5.18",
}

problems: list[str] = []

print("=" * 62)
print("AirCast AI — environment check")
print("=" * 62)
print(f"Python           : {platform.python_version()}  ({sys.executable})")
print(f"Platform         : {platform.system()} {platform.release()}")
print(f"Working folder   : {Path.cwd()}")
print(f"Project folder   : {ROOT}")

if sys.version_info < MIN_PY:
    problems.append(
        f"Python {platform.python_version()} is too old — install Python 3.10 or newer."
    )

print("-" * 62)
for module, minimum in REQUIRED.items():
    try:
        mod = importlib.import_module(module)
        version = getattr(mod, "__version__", "unknown")
        print(f"{module:<17}: {version}")
    except Exception as exc:  # noqa: BLE001
        print(f"{module:<17}: MISSING ({exc})")
        problems.append(f"{module} is not installed — run: pip install -r requirements.txt")

print("-" * 62)
structure = {
    "app.py": ROOT / "app.py",
    "src/aircast/__init__.py": ROOT / "src" / "aircast" / "__init__.py",
    "requirements.txt": ROOT / "requirements.txt",
}
for label, path in structure.items():
    print(f"{label:<25}: {'OK' if path.exists() else 'MISSING'}")
    if not path.exists():
        problems.append(f"{label} is missing — re-extract the project ZIP and keep the folder structure.")

print("-" * 62)
try:
    sys.path.insert(0, str(ROOT / "src"))
    from aircast import generate_dataset, train_and_evaluate  # noqa: E402

    df = generate_dataset()
    report = train_and_evaluate(df)
    print(f"Pipeline         : OK ({len(df):,} records)")
    print(f"Selected model   : {report.best_model_name}")
    print(f"R² / RMSE        : {report.best.r2:.3f} / {report.best.rmse:.2f}")
except Exception as exc:  # noqa: BLE001
    print(f"Pipeline         : FAILED ({type(exc).__name__}: {exc})")
    problems.append(f"The model pipeline failed to run: {type(exc).__name__}: {exc}")

if sys.version_info >= (3, 14):
    print("-" * 62)
    print(
        "NOTE: Python 3.14 is very new. The packages above ship 3.14 wheels, but"
        "\n      if any install fails to build, install Python 3.12 or 3.13 and"
        "\n      re-create the virtual environment with that interpreter."
    )

print("=" * 62)
if problems:
    print("Issues found:")
    for index, problem in enumerate(problems, 1):
        print(f"  {index}. {problem}")
    sys.exit(1)

print("Everything looks good — start the app with:")
print("    python -m streamlit run app.py")
