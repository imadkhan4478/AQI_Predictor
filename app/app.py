"""Streamlit dashboard: current AQI + 3-day forecast, with hazardous-AQI alerts."""

import os
import sys
from pathlib import Path

# `streamlit run app/app.py` puts this file's own folder on sys.path, not the
# repo root, so sibling packages (app.load_model, feature_pipeline, ...)
# wouldn't otherwise resolve. Same fix as the EDA notebook's kernel-cwd issue.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from app.load_model import load_latest_model
from feature_pipeline.aqi import _CATEGORIES, aqi_category
from feature_pipeline.hopsworks_store import read_features_df
from training_pipeline.data import FEATURE_COLUMNS, add_time_features

load_dotenv()

HORIZONS_HOURS = [24, 48, 72]
SEVERITY_RANK = {None: 0, "warning": 1, "serious": 2, "critical": 3}
STATUS_MESSAGE = {
    "warning": "Sensitive groups (children, elderly, respiratory/heart conditions) should limit prolonged outdoor exertion.",
    "serious": "Everyone may begin to experience health effects. Limit prolonged outdoor exertion.",
    "critical": "Health alert: everyone may experience serious health effects. Avoid outdoor activity.",
}

CARD_CSS = """
<style>
.aqi-hero {
    border-radius: 12px;
    padding: 1.5rem 1.75rem;
    color: white;
    text-shadow: 0 1px 2px rgba(0,0,0,0.25);
}
.aqi-hero .value { font-size: 3.2rem; font-weight: 700; line-height: 1; }
.aqi-hero .category { font-size: 1.2rem; font-weight: 600; margin-top: 0.25rem; }
.legend-chip {
    display: inline-flex; align-items: center; gap: 0.4rem;
    padding: 0.15rem 0.6rem; border-radius: 999px;
    font-size: 0.78rem; color: white; margin: 0.15rem;
}
</style>
"""


@st.cache_data(ttl=600)
def load_latest_row(city_name):
    df = read_features_df()
    df = add_time_features(df)
    city_rows = df[df["city_name"] == city_name].sort_values("timestamp")
    return city_rows.iloc[-1]


@st.cache_resource
def load_models():
    models = {}
    for horizon_hours in HORIZONS_HOURS:
        model, version, metrics = load_latest_model(f"aqi_random_forest_{horizon_hours}h")
        models[horizon_hours] = {"model": model, "version": version, "metrics": metrics}
    return models


def render_hero(aqi_value, category_name, color):
    st.markdown(
        f"""<div class="aqi-hero" style="background:{color};">
                <div class="value">{aqi_value}</div>
                <div class="category">{category_name}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def render_legend():
    chips = "".join(
        f'<span class="legend-chip" style="background:{color};">{name}</span>'
        for _, _, name, _, color in _CATEGORIES
    )
    st.markdown(chips, unsafe_allow_html=True)


def plot_forecast(forecast):
    labels = ["Now" if h == 0 else f"+{h}h" for h in forecast]
    values = list(forecast.values())
    categories = [aqi_category(v) for v in values]
    colors = [color for _, _, color in categories]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=values,
            textposition="outside",
            customdata=[name for name, _, _ in categories],
            hovertemplate="<b>%{x}</b><br>AQI: %{y}<br>%{customdata}<extra></extra>",
        )
    )
    fig.update_layout(
        title="3-Day AQI Forecast",
        yaxis_title="AQI",
        showlegend=False,
        margin=dict(t=50, b=10, l=10, r=10),
        height=380,
    )
    return fig


def main():
    city_name = os.environ["CITY_NAME"]
    st.set_page_config(page_title=f"AQI Predictor — {city_name}", page_icon="🌫️", layout="wide")
    st.markdown(CARD_CSS, unsafe_allow_html=True)

    st.title(f"🌫️ AQI Predictor — {city_name}")
    st.caption("3-day Air Quality Index forecast · Random Forest, retrained daily · data refreshed hourly")

    with st.spinner("Loading latest data and models..."):
        latest_row = load_latest_row(city_name)
        models = load_models()

    X_live = latest_row[FEATURE_COLUMNS].to_frame().T

    forecast = {0: int(latest_row["aqi"])}
    for horizon_hours in HORIZONS_HOURS:
        prediction = models[horizon_hours]["model"].predict(X_live)[0]
        forecast[horizon_hours] = round(prediction)

    current_category, _, current_color = aqi_category(forecast[0])

    col1, col2 = st.columns([1, 2], gap="large")
    with col1:
        render_hero(forecast[0], current_category, current_color)
    with col2:
        st.write("")
        st.caption(f"As of **{latest_row['timestamp']} UTC**")
        st.write(f"Dominant pollutant: **{latest_row['dominant_pollutant']}**")
        render_legend()

    worst_horizon, worst_severity = max(
        ((h, aqi_category(v)[1]) for h, v in forecast.items()),
        key=lambda item: SEVERITY_RANK[item[1]],
    )
    if worst_severity:
        when = "now" if worst_horizon == 0 else f"in {worst_horizon}h"
        category_name, _, _ = aqi_category(forecast[worst_horizon])
        alert_fn = st.error if worst_severity == "critical" else st.warning
        alert_fn(f"**{category_name} air quality expected {when}** (AQI {forecast[worst_horizon]}). {STATUS_MESSAGE[worst_severity]}")

    st.plotly_chart(plot_forecast(forecast), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        with st.expander("Why does the model predict this? (SHAP feature importance)"):
            st.image("reports/shap_feature_importance.png")
            st.caption(
                "Feature importance for the Random Forest model, from the 4-model comparison "
                "(Ridge/RandomForest/NeuralNet/ARIMA) documented in training_pipeline/."
            )
    with col_b:
        with st.expander("Model details"):
            for horizon_hours in HORIZONS_HOURS:
                info = models[horizon_hours]
                st.write(f"**{horizon_hours}h model** — v{info['version']}")
                st.json(info["metrics"])


if __name__ == "__main__":
    main()
