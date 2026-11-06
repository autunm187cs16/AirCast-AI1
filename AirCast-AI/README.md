# 🌤️ AirCast AI

**Intelligent air quality prediction and pollution risk analysis using machine learning.**

AirCast AI is an educational Machine Learning application that estimates fine particulate
matter (**PM2.5**) from six environmental variables and compares three supervised regression
algorithms through an interactive Streamlit dashboard.

> **Dataset disclosure:** the application generates a reproducible **synthetic** dataset of
> 5,000 records. These are simulated values, **not** measurements from air-quality monitoring
> stations. The model is an academic demonstration and is not validated for public-health,
> safety or official environmental decisions.

---

## Table of contents

- [Overview](#overview)
- [Features](#features)
- [Project structure](#project-structure)
- [Quickstart](#quickstart)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Screenshots](#screenshots)
- [Limitations](#limitations)
- [Future enhancements](#future-enhancements)
- [Viva quick answers](#viva-quick-answers)
- [License](#license)

---

## Overview

Air pollution is a major environmental concern, and PM2.5 (particles with an aerodynamic
diameter of 2.5 µm or less) is one of its most closely tracked indicators. AirCast AI
demonstrates how environmental conditions can be used to estimate PM2.5 concentration with
supervised regression, and how model quality can be compared using standard error metrics.

The project covers a complete introductory ML workflow:

```
Synthetic dataset → feature/target selection → 80/20 split → model training →
evaluation → best-model selection → user input → PM2.5 estimate → dashboard
```

## Features

| Module | What it does |
| --- | --- |
| **Dashboard** | Project indicators, dataset size, selected model and PM2.5 distribution |
| **Dataset Exploration** | Sample records, descriptive statistics and a correlation heatmap |
| **Model Comparison** | MAE / RMSE / R² table, metric charts, actual-vs-predicted plots, residuals and feature importance |
| **Prediction** | Form for six environmental inputs, estimated PM2.5, gauge and an illustrative risk band |
| **About Project** | Objective, algorithms, workflow, technologies and dataset limitations |

- Reproducible synthetic data (fixed random seed `42`)
- Three models trained side by side: Linear Regression, Random Forest, Gradient Boosting
- The model with the **lowest test-set RMSE** is selected automatically and served to the user
- Interactive Plotly visualisations
- Cached training (`st.cache_resource`) so page navigation stays fast

## Project structure

```
AirCast-AI/
├── app.py                     # Streamlit entry point and all five pages
├── doctor.py                  # environment + installation self-check
├── run.bat / run.sh           # one-click launchers (Windows / macOS-Linux)
├── push_to_github.bat         # one-click GitHub publishing (Windows)
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── .streamlit/
│   └── config.toml            # light theme, teal accent
├── src/
│   └── aircast/
│       ├── __init__.py        # public package API
│       ├── data.py            # synthetic dataset generator + risk bands
│       ├── models.py          # training, evaluation, selection, prediction
│       └── viz.py             # Plotly chart builders
└── tests/
    └── test_smoke.py          # end-to-end pipeline smoke test
```

## Quickstart

Requires **Python 3.10+**.

```bash
# 1. create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 2. install dependencies
pip install -r requirements.txt

# 3. run the app
python -m streamlit run app.py
```

Open the local address printed in the terminal, normally <http://localhost:8501>.

Optional checks:

```bash
python doctor.py                           # python version, packages, structure, pipeline
python tests/test_smoke.py                 # pipeline runs and metrics are sane
python -m streamlit run app.py --server.headless true
```

Or just double-click **`run.bat`** (Windows) / run **`./run.sh`** (macOS, Linux) — they create the
virtual environment, install dependencies and start the app in one step.

### Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `NameError` on startup | Usually an incomplete copy: `app.py` was moved away from the `src/` folder, or an old saved version of the file is being run. Keep the whole `AirCast-AI` folder together, then run `python doctor.py` to confirm. |
| `ModuleNotFoundError: No module named 'aircast'` | The `src/` folder is missing or `app.py` is outside the project root. Re-extract the ZIP and run from the folder that contains `app.py` **and** `src/`. |
| `ModuleNotFoundError: No module named 'streamlit'` (or pandas/sklearn/plotly) | The virtual environment is not active, or dependencies are not installed: `pip install -r requirements.txt`. |
| `TypeError: plotly_chart() got an unexpected keyword argument` | Very old Streamlit. Upgrade: `pip install --upgrade streamlit`. |
| Port 8501 already in use | `python -m streamlit run app.py --server.port 8502`. |
| Blank page / spinner forever | First run trains the models (~3 s). If it persists, delete the `__pycache__` folders and restart. |

## Dataset

| Column | Role | Unit / Values |
| --- | --- | --- |
| `temperature` | Input feature | °C |
| `humidity` | Input feature | % |
| `wind_speed` | Input feature | m/s |
| `pressure` | Input feature | hPa |
| `hour` | Input feature | 0 – 23 |
| `month` | Input feature | 1 – 12 |
| `pm25` | Target | µg/m³ |

**Generation model** (`src/aircast/data.py`, seed `42`, 5,000 rows) — PM2.5 is assembled from:

- a **traffic profile** with morning (≈08:00) and evening (≈19:00) peaks,
- a **seasonal term** raising concentrations in winter months,
- **wind dispersion** (negative effect),
- **humidity and pressure** terms (accumulation effects),
- Gaussian noise, then clipped to `2 – 300 µg/m³`.

The simulated correlations are therefore known by construction: wind speed and month are the
strongest drivers, which is reflected in the feature-importance chart.

## Methodology

1. **Features / target** — six input columns, PM2.5 as the continuous target.
2. **Split** — `train_test_split(test_size=0.2, random_state=42)`.
3. **Models**
   - *Linear Regression* — interpretable baseline, linear relationship only.
   - *Random Forest* — 200 trees, bagging, captures non-linear interactions.
   - *Gradient Boosting* — 250 shallow trees, learning rate 0.05, sequential error correction.
4. **Metrics**
   - **MAE** — average absolute error (µg/m³).
   - **RMSE** — square root of mean squared error, penalises large errors.
   - **R²** — proportion of variance in the target explained by the model.
5. **Selection** — lowest test-set RMSE becomes the prediction model.

## Results

Typical output on the default seed (5,000 records, 1,000 test rows):

| Model | MAE (µg/m³) | RMSE (µg/m³) | R² |
| --- | ---: | ---: | ---: |
| **Gradient Boosting** ✅ | **5.48** | **6.84** | **0.919** |
| Random Forest | 5.92 | 7.39 | 0.905 |
| Linear Regression | 12.07 | 14.85 | 0.618 |

Gradient Boosting is selected. The gap between the tree ensembles and the linear baseline
confirms that the simulated relationships are non-linear. These numbers describe performance on
**generated demonstration data only** and do not establish real-world predictive accuracy.

## Screenshots

Add your own captures after the first run, for example:

```
docs/screenshots/dashboard.png
docs/screenshots/dataset.png
docs/screenshots/models.png
docs/screenshots/prediction.png
```

```markdown
![Dashboard](docs/screenshots/dashboard.png)
```

## Limitations

- The dataset is **synthetic**, created by a mathematical simulation.
- No external validation against monitoring-station observations.
- The random train/test split is appropriate for this demonstration but is **not** a
  time-series forecasting validation design.
- The displayed risk bands are **illustrative** and are not regulatory air-quality categories.
- The application does not provide live or location-specific air-quality information.

## Future enhancements

- Replace the simulation with a documented public dataset (with a clear source and licence).
- Use chronological train/validation/test splits for forecasting tasks.
- Validate on independent monitoring-station observations.
- Add time-series forecasting and location-aware analysis.
- Connect environmental sensors or a verified air-quality API.
- Deploy the application and add user access controls.

## Viva quick answers

| Question | Answer |
| --- | --- |
| What is Machine Learning? | A field of AI where systems learn patterns from data to make predictions or decisions. |
| What is PM2.5? | Fine particulate matter with an aerodynamic diameter of 2.5 µm or less. |
| Type of ML problem? | Supervised regression — the target is continuous. |
| Target variable? | PM2.5 concentration (µg/m³). |
| Why Linear Regression? | A simple, interpretable baseline for linear relationships. |
| Why Random Forest? | It averages many decision trees and models non-linear patterns. |
| What is Gradient Boosting? | An ensemble that adds trees sequentially, each correcting prior errors. |
| What is MAE? | Average absolute difference between actual and predicted values. |
| What is RMSE? | Square root of the mean squared error; penalises large errors more. |
| What is R²? | The fraction of target variance explained by the model. |
| Why Streamlit? | It turns Python scripts into interactive web apps with minimal code. |
| Is the dataset real? | No — 5,000 synthetic records generated with a fixed seed. |
| Can it be used for health decisions? | No — it is not validated on real observations. |

## Publish to GitHub

**Option A — one click (Windows):** double-click **`push_to_github.bat`**. It initialises the
repository, commits everything (`.venv/` is ignored) and pushes to the URL you paste in.

**Option B — manual commands:**

```bash
git init
git branch -M main
git add .
git commit -m "AirCast AI - PM2.5 prediction with machine learning"
git remote add origin https://github.com/USERNAME/AirCast-AI.git
git push -u origin main
```

Create the repository first at <https://github.com/new> — name it **AirCast-AI**, set it to
**Public**, and leave *Add a README / .gitignore / licence* **unticked** (the project already has
them; an initialised repo causes a `failed to push some refs` error).

Before submitting, confirm:

- the repository opens in an **incognito** window (proves it is public),
- `.venv/`, `__pycache__/` and `*.pyc` are **not** in the file list,
- the README renders with the synthetic-data disclosure visible.

## License

Released under the MIT License — see [LICENSE](LICENSE).

---

**Author:** _Your Name_ · **Register Number:** _XXXXXXX_ · **Department:** M.Sc. Computer Science
· **Academic Year:** 2026–2027