"""AirCast AI — Streamlit dashboard.

Run with:  python -m streamlit run app.py

Educational demonstration only. The dataset is synthetic and the risk bands are
illustrative; nothing here is validated for health or official decisions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

# --- friendly startup diagnostics ----------------------------------------- #
if not (SRC / "aircast" / "__init__.py").exists():
    st.error(
        "**The `src/aircast` package was not found.**\n\n"
        f"Expected it at: `{SRC / 'aircast'}`\n\n"
        "Run the app from the project root (the folder that contains `app.py` "
        "**and** `src/`):\n\n"
        "```\npython -m streamlit run app.py\n```"
    )
    st.stop()

sys.path.insert(0, str(SRC))

try:
    from aircast import (  # noqa: E402
        FEATURES,
        N_RECORDS,
        RANDOM_SEED,
        TARGET,
        UNITS,
        feature_importance,
        generate_dataset,
        predict_pm25,
        risk_category,
        train_and_evaluate,
    )
    from aircast import viz  # noqa: E402
except ImportError as exc:  # pragma: no cover - startup guard
    st.error(
        f"**Could not import the project code or a dependency.**\n\n`{exc}`\n\n"
        "Install the requirements, then restart:\n\n"
        "```\npip install -r requirements.txt\n```"
    )
    st.stop()

st.set_page_config(
    page_title="AirCast AI",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    return generate_dataset(N_RECORDS, RANDOM_SEED)


@st.cache_resource(show_spinner="Training models…")
def load_report():
    return train_and_evaluate(load_data())


def show_chart(fig: go.Figure) -> None:
    """Render a Plotly chart across the available width on any Streamlit version."""
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:  # older Streamlit releases
        st.plotly_chart(fig, use_container_width=True)


def show_table(frame: pd.DataFrame, height: int | None = None, **kwargs) -> None:
    try:
        st.dataframe(frame, height=height, width="stretch", **kwargs)
    except TypeError:
        st.dataframe(frame, height=height, use_container_width=True, **kwargs)


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #

st.markdown(
    """
    <style>
      .block-container { padding-top: 2.2rem; max-width: 1250px; }
      .hero {
          background: linear-gradient(120deg, #0f766e 0%, #0ea5e9 55%, #6366f1 100%);
          border-radius: 18px; padding: 26px 30px; color: #ffffff;
          box-shadow: 0 12px 30px rgba(15, 118, 110, .18);
      }
      .hero h1 { margin: 0 0 6px 0; font-size: 2.0rem; letter-spacing: -.5px; }
      .hero p { margin: 0; opacity: .93; font-size: .98rem; }
      .pill {
          display: inline-block; margin-top: 14px; padding: 5px 13px;
          background: rgba(255,255,255,.18); border-radius: 999px;
          font-size: .78rem; letter-spacing: .4px; text-transform: uppercase;
      }
      .card {
          border: 1px solid #e2e8f0; border-radius: 14px; padding: 18px 20px;
          background: #ffffff; height: 100%;
      }
      .card h4 { margin: 0 0 8px 0; font-size: .82rem; text-transform: uppercase;
                 letter-spacing: .6px; color: #64748b; }
      .card .value { font-size: 1.7rem; font-weight: 700; color: #0f172a; }
      .card .note { font-size: .82rem; color: #64748b; margin-top: 4px; }
      .risk {
          border-radius: 16px; padding: 22px 26px; color: #ffffff;
          box-shadow: 0 12px 26px rgba(15, 23, 42, .16);
      }
      .risk .label { font-size: .8rem; text-transform: uppercase; letter-spacing: 1px; opacity: .9; }
      .risk .big { font-size: 2.6rem; font-weight: 800; line-height: 1.15; }
      section[data-testid="stSidebar"] .stRadio label { font-size: .94rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.markdown("### 🌤️ AirCast AI")
    st.caption("PM2.5 estimation with supervised regression")
    page = st.radio(
        "Navigate",
        [
            "Dashboard",
            "Dataset Exploration",
            "Model Comparison",
            "Prediction",
            "About Project",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**Active model**")
    report = load_report()
    st.success(report.best_model_name)
    st.caption(f"Selected by lowest test-set RMSE: {report.best.rmse:.2f} µg/m³")
    st.divider()
    st.caption(
        "Demonstration only — synthetic data, not validated for health or "
        "official environmental decisions."
    )

df = load_data()
importance = feature_importance(report)
best = report.best

# --------------------------------------------------------------------------- #
# Dashboard
# --------------------------------------------------------------------------- #

if page == "Dashboard":
    st.markdown(
        """
        <div class="hero">
          <h1>AirCast AI</h1>
          <p>Intelligent air quality prediction and pollution risk analysis using
          supervised machine learning.</p>
          <span class="pill">M.Sc. Computer Science · Machine Learning Project</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="card"><h4>Dataset records</h4>'
            f'<div class="value">{len(df):,}</div>'
            f'<div class="note">Synthetic, seed {RANDOM_SEED}</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="card"><h4>Selected model</h4>'
            f'<div class="value" style="font-size:1.25rem">{best.name}</div>'
            f'<div class="note">Lowest test-set RMSE</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="card"><h4>R² score</h4>'
            f'<div class="value">{best.r2:.3f}</div>'
            f'<div class="note">Variance explained on test split</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f'<div class="card"><h4>MAE</h4>'
            f'<div class="value">{best.mae:.2f}</div>'
            f'<div class="note">µg/m³ average absolute error</div></div>',
            unsafe_allow_html=True,
        )

    st.write("")
    left, right = st.columns([1.35, 1])
    with left:
        show_chart(viz.pm25_distribution(df))
    with right:
        st.markdown("#### Model metrics")
        metrics = report.metrics_table().rename(
            columns={"MAE": "MAE (µg/m³)", "RMSE": "RMSE (µg/m³)"}
        )
        show_table(
            metrics.style.format({"MAE (µg/m³)": "{:.2f}", "RMSE (µg/m³)": "{:.2f}", "R²": "{:.3f}"}),
            height=210,
        )
        st.info(
            "The application trains Linear Regression, Random Forest and "
            "Gradient Boosting, then serves predictions from the model with the "
            "lowest RMSE."
        )

    st.markdown("#### How it works")
    steps = [
        ("1 · Generate", "5,000 synthetic environmental records with a fixed seed"),
        ("2 · Prepare", "Select six input features and the PM2.5 target"),
        ("3 · Split", "80% training / 20% test, randomly shuffled"),
        ("4 · Train", "Fit three supervised regression algorithms"),
        ("5 · Evaluate", "Compare MAE, RMSE and R² on the held-out test set"),
        ("6 · Predict", "Serve the best model through the prediction form"),
    ]
    for col, (title, text) in zip(st.columns(3), steps[:3]):
        col.markdown(
            f'<div class="card"><h4>{title}</h4><div class="note">{text}</div></div>',
            unsafe_allow_html=True,
        )
    st.write("")
    for col, (title, text) in zip(st.columns(3), steps[3:]):
        col.markdown(
            f'<div class="card"><h4>{title}</h4><div class="note">{text}</div></div>',
            unsafe_allow_html=True,
        )

# --------------------------------------------------------------------------- #
# Dataset Exploration
# --------------------------------------------------------------------------- #

elif page == "Dataset Exploration":
    st.title("Dataset Exploration")
    st.caption(
        "Synthetic records generated by the application. These are simulated "
        "values, not measurements from monitoring stations."
    )

    tab_sample, tab_stats, tab_corr = st.tabs(
        ["Sample records", "Descriptive statistics", "Correlation"]
    )

    with tab_sample:
        col_a, col_b = st.columns([1, 1])
        with col_a:
            n_rows = st.slider("Rows to display", 5, 50, 10, step=5)
        with col_b:
            sort_feature = st.selectbox("Sort by", [TARGET, *FEATURES], index=0)
        show_table(
            df.sort_values(sort_feature, ascending=False).head(n_rows),
            height=430,
        )
        st.caption(
            f"Showing {n_rows} of {len(df):,} rows · units: "
            + ", ".join(f"{key} ({value})" for key, value in UNITS.items())
        )

    with tab_stats:
        stats = df.describe().T.rename(columns={"50%": "median"}).round(2)
        stats.index.name = "Column"
        show_table(stats, height=300)
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Mean PM2.5", f"{df[TARGET].mean():.1f} µg/m³")
        col_b.metric("Maximum PM2.5", f"{df[TARGET].max():.1f} µg/m³")
        col_c.metric("Standard deviation", f"{df[TARGET].std():.1f} µg/m³")
        show_chart(viz.pm25_distribution(df))

    with tab_corr:
        show_chart(viz.correlation_heatmap(df))
        st.caption(
            "Correlation describes linear association inside the simulated "
            "dataset only. Wind speed and humidity are the strongest drivers "
            "by construction."
        )

# --------------------------------------------------------------------------- #
# Model Comparison
# --------------------------------------------------------------------------- #

elif page == "Model Comparison":
    st.title("Model Comparison")

    metrics = report.metrics_table()
    cols = st.columns(3)
    for col, (_, row) in zip(cols, metrics.iterrows()):
        with col:
            is_best = row["Model"] == report.best_model_name
            st.markdown(
                f'<div class="card"><h4>{row["Model"]}</h4>'
                f'<div class="value">{row["R²"]:.3f} <span style="font-size:.9rem;'
                f'color:#64748b">R²</span></div>'
                f'<div class="note">MAE {row["MAE"]:.2f} · RMSE {row["RMSE"]:.2f}'
                + ("<br><strong>Selected model</strong>" if is_best else "")
                + "</div></div>",
                unsafe_allow_html=True,
            )

    st.write("")
    show_table(
        metrics.style.format(
            {"MAE": "{:.3f}", "RMSE": "{:.3f}", "R²": "{:.4f}"}
        ).highlight_min(subset=["MAE", "RMSE"], color="#d1fae5")
        .highlight_max(subset=["R²"], color="#d1fae5"),
        height=190,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        show_chart(viz.metric_comparison(report, "RMSE"))
    with col_b:
        show_chart(viz.metric_comparison(report, "R²"))

    st.divider()
    selected = st.selectbox(
        "Inspect a model", list(report.results), index=list(report.results).index(best.name)
    )
    col_a, col_b = st.columns(2)
    with col_a:
        show_chart(viz.actual_vs_predicted(report, selected))
    with col_b:
        show_chart(viz.residuals_histogram(report, selected))

    if importance is not None:
        st.divider()
        col_a, col_b = st.columns([1.2, 1])
        with col_a:
            show_chart(viz.feature_importance_chart(importance))
        with col_b:
            st.markdown("#### Feature importance")
            show_table(
                importance.sort_values("Importance", ascending=False).style.format(
                    {"Importance": "{:.4f}"}
                ),
                height=330,
            )
            st.caption(
                f"Importances come from the selected model "
                f"({best.name}); the linear baseline does not expose them."
            )

# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #

elif page == "Prediction":
    st.title("PM2.5 Prediction")
    st.caption(
        "Enter environmental conditions to obtain an estimated PM2.5 "
        "concentration from the selected model."
    )

    with st.form("prediction_form"):
        st.markdown("##### Environmental inputs")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            temperature = st.slider(
                "Temperature (°C)", -5.0, 50.0, float(df["temperature"].mean()), 0.5
            )
            hour = st.slider("Hour of day", 0, 23, 9)
        with col_b:
            humidity = st.slider(
                "Humidity (%)", 5.0, 100.0, float(df["humidity"].mean()), 1.0
            )
            month = st.slider("Month", 1, 12, 1)
        with col_c:
            wind_speed = st.slider(
                "Wind speed (m/s)", 0.0, 15.0, float(df["wind_speed"].mean()), 0.1
            )
            pressure = st.slider(
                "Pressure (hPa)", 960.0, 1045.0, float(df["pressure"].mean()), 0.5
            )

        model_choice = st.selectbox(
            "Model used for prediction",
            list(report.results),
            index=list(report.results).index(best.name),
            help="The dashboard defaults to the model with the lowest test-set RMSE.",
        )
        submitted = st.form_submit_button("Estimate PM2.5", type="primary")

    if submitted:
        values = {
            "temperature": temperature,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "pressure": pressure,
            "hour": hour,
            "month": month,
        }
        estimate = predict_pm25(report, values, model_choice)
        label, colour = risk_category(estimate)

        col_a, col_b = st.columns([1, 1.25])
        with col_a:
            st.markdown(
                f'<div class="risk" style="background:{colour}">'
                f'<div class="label">Estimated PM2.5</div>'
                f'<div class="big">{estimate:.1f} µg/m³</div>'
                f'<div style="margin-top:6px;font-size:.95rem">'
                f'Illustrative band: <strong>{label}</strong></div></div>',
                unsafe_allow_html=True,
            )
            st.caption(
                f"Produced by {model_choice}. Bands are illustrative only and "
                "are not official air-quality guidance."
            )
        with col_b:
            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=estimate,
                    number={"suffix": " µg/m³"},
                    gauge={
                        "axis": {"range": [0, 320]},
                        "bar": {"color": colour},
                        "steps": [
                            {"range": [0, 30], "color": "#dcfce7"},
                            {"range": [30, 60], "color": "#ecfccb"},
                            {"range": [60, 90], "color": "#fef9c3"},
                            {"range": [90, 120], "color": "#ffedd5"},
                            {"range": [120, 250], "color": "#fee2e2"},
                            {"range": [250, 320], "color": "#f3e8ff"},
                        ],
                    },
                )
            )
            gauge.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=30, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            show_chart(gauge)

        st.markdown("##### Illustrative reference bands")
        bands = pd.DataFrame(
            [
                ("Good", "≤ 30", "#2e7d32"),
                ("Satisfactory", "31 – 60", "#7cb342"),
                ("Moderate", "61 – 90", "#fbc02d"),
                ("Poor", "91 – 120", "#f57c00"),
                ("Very Poor", "121 – 250", "#d32f2f"),
                ("Severe", "> 250", "#7b1fa2"),
            ],
            columns=["Band", "PM2.5 (µg/m³)", "Colour"],
        )
        show_table(bands, height=250, hide_index=True)
        st.warning(
            "This output is a classroom demonstration on synthetic data. Do not "
            "use it for health, safety or official environmental decisions."
        )
    else:
        st.info("Set the conditions above, then choose **Estimate PM2.5**.")

# --------------------------------------------------------------------------- #
# About Project
# --------------------------------------------------------------------------- #

else:
    st.title("About the Project")

    st.markdown(
        """
**AirCast AI** is an educational machine-learning application that estimates
fine particulate matter (PM2.5) from six environmental variables. It trains and
compares three supervised regression algorithms and presents the results in an
interactive dashboard.
"""
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Objective")
        st.markdown(
            "- Demonstrate an end-to-end ML workflow\n"
            "- Prepare and inspect environmental features\n"
            "- Train three regression algorithms\n"
            "- Compare models with MAE, RMSE and R²\n"
            "- Provide an interface for sample predictions"
        )
        st.markdown("#### Inputs and output")
        show_table(
            pd.DataFrame(
                {
                    "Variable": FEATURES + [TARGET],
                    "Role": ["Input"] * len(FEATURES) + ["Target"],
                    "Unit": [UNITS[key] for key in FEATURES + [TARGET]],
                }
            ),
            height=290,
        )
    with col_b:
        st.markdown("#### Technologies")
        st.markdown(
            "- **Python** — implementation language\n"
            "- **Pandas / NumPy** — data handling and simulation\n"
            "- **Scikit-learn** — models, split and metrics\n"
            "- **Plotly** — interactive charts\n"
            "- **Streamlit** — dashboard and forms\n"
            "- **Git / GitHub** — version control and hosting"
        )
        st.markdown("#### Workflow")
        st.markdown(
            "Synthetic dataset → feature/target selection → 80/20 split → model "
            "training → evaluation → best-model selection → user input → PM2.5 "
            "estimate → dashboard"
        )

    st.divider()
    st.markdown("#### Limitations")
    st.markdown(
        "- The dataset is **synthetic**, produced by a mathematical simulation.\n"
        "- No external validation against monitoring-station observations.\n"
        "- The random train/test split is not a time-series forecasting design.\n"
        "- Displayed risk bands are illustrative, not regulatory.\n"
        "- The application does not provide live or location-specific readings."
    )
    st.markdown("#### Future enhancements")
    st.markdown(
        "- Use a documented public air-quality dataset with a clear licence.\n"
        "- Adopt chronological train/validation/test splits.\n"
        "- Add time-series forecasting and location-aware analysis.\n"
        "- Connect environmental sensors or verified data APIs.\n"
        "- Deploy the application with access controls."
    )
    st.divider()
    st.caption(
        f"AirCast AI v1.0.0 · synthetic dataset, seed {RANDOM_SEED}, "
        f"{len(df):,} records · built with Streamlit and scikit-learn."
    )
