"""Plotly chart builders for the AirCast AI dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .data import FEATURES, TARGET
from .models import TrainingReport

ACCENT = "#0f766e"
WARM = "#f97316"


def _base_layout(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
    )
    return fig


def pm25_distribution(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df, x=TARGET, nbins=45, color_discrete_sequence=[ACCENT], opacity=0.85
    )
    fig.update_traces(marker_line_width=0)
    fig.update_layout(
        title="PM2.5 distribution (synthetic records)",
        xaxis_title="PM2.5 (µg/m³)",
        yaxis_title="Number of records",
        bargap=0.03,
    )
    return _base_layout(fig)


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    corr = df[FEATURES + [TARGET]].corr(numeric_only=True)
    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
            text=corr.round(2).values,
            texttemplate="%{text}",
            colorbar=dict(title="r"),
        )
    )
    fig.update_layout(title="Feature correlation with PM2.5")
    return _base_layout(fig, height=480)


def actual_vs_predicted(report: TrainingReport, model_name: str) -> go.Figure:
    result = report.results[model_name]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=result.y_test,
            y=result.y_pred,
            mode="markers",
            marker=dict(size=5, color=ACCENT, opacity=0.55),
            name="Test records",
        )
    )
    low = float(min(result.y_test.min(), result.y_pred.min()))
    high = float(max(result.y_test.max(), result.y_pred.max()))
    fig.add_trace(
        go.Scatter(
            x=[low, high],
            y=[low, high],
            mode="lines",
            line=dict(color=WARM, dash="dash", width=2),
            name="Perfect prediction",
        )
    )
    fig.update_layout(
        title=f"Actual vs predicted — {model_name}",
        xaxis_title="Actual PM2.5 (µg/m³)",
        yaxis_title="Predicted PM2.5 (µg/m³)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return _base_layout(fig, height=440)


def residuals_histogram(report: TrainingReport, model_name: str) -> go.Figure:
    result = report.results[model_name]
    fig = px.histogram(
        x=result.residuals, nbins=45, color_discrete_sequence=[WARM], opacity=0.85
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#334155")
    fig.update_layout(
        title=f"Residual distribution — {model_name}",
        xaxis_title="Actual − predicted (µg/m³)",
        yaxis_title="Number of records",
        bargap=0.03,
    )
    return _base_layout(fig)


def metric_comparison(report: TrainingReport, metric: str) -> go.Figure:
    table = report.metrics_table()
    fig = px.bar(
        table,
        x="Model",
        y=metric,
        color="Model",
        text=table[metric].round(3),
        color_discrete_sequence=[ACCENT, WARM, "#6366f1"],
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(title=f"{metric} by model (lower is better)" if metric != "R²" else "R² by model (higher is better)", showlegend=False)
    return _base_layout(fig)


def feature_importance_chart(importance: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        importance,
        x="Importance",
        y="Feature",
        orientation="h",
        color_discrete_sequence=[ACCENT],
        text=importance["Importance"].round(3),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(title="Feature importance (selected model)", showlegend=False)
    return _base_layout(fig, height=400)
