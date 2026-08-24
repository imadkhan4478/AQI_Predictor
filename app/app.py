"""Streamlit dashboard: current AQI + 3-day forecast, with hazardous-AQI alerts.

This file is presentation only. Every number it shows comes from
serving/forecast.py -- either called in-process, or fetched from the FastAPI
service when AQI_API_URL is set. Deliberately: the dashboard used to assemble
features and apply the model itself, drifted out of step with the training
pipeline, and served a stale model while looking healthy.
"""

import os
import sys
from pathlib import Path

# `streamlit run app/app.py` puts this file's own folder on sys.path, not the
# repo root, so sibling packages (serving, feature_pipeline, ...) wouldn't
# otherwise resolve. Same fix as the EDA notebook's kernel-cwd issue.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv

from feature_pipeline.aqi import _CATEGORIES
from serving.forecast import (
    HORIZONS_HOURS,
    ForecastUnavailable,
    build_forecast,
    load_feature_row,
    load_forecast_models,
    model_details,
)

load_dotenv()

# Set AQI_API_URL to read from the FastAPI service instead of loading the models
# into the Streamlit process. Both paths return the same payload -- the service
# calls the same serving/forecast.py functions -- so this is a deployment
# choice, not a behavioural one: one process per concern when the API is running,
# a single self-contained process when it is not.
API_URL = os.environ.get("AQI_API_URL", "").rstrip("/")
API_TIMEOUT_SECONDS = 30

SHAP_PLOT_PATH = "reports/shap_feature_importance.png"

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


@st.cache_resource
def cached_models():
    """Cached as a resource, not data: model objects are not serialisable and
    should be loaded once per process, not once per session."""
    return load_forecast_models(HORIZONS_HOURS)


@st.cache_data(ttl=600)
def fetch_payload(city_name):
    """The forecast payload, from the API if one is configured, otherwise
    computed here. TTL matches the hourly cadence of the feature pipeline."""
    if API_URL:
        response = requests.get(f"{API_URL}/forecast", timeout=API_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()
    return build_forecast(city_name, cached_models(), load_feature_row(city_name))


@st.cache_data(ttl=600)
def fetch_model_details():
    if API_URL:
        response = requests.get(f"{API_URL}/models", timeout=API_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()
    return model_details(cached_models())


def render_hero(reading):
    st.markdown(
        f"""<div class="aqi-hero" style="background:{reading['color']};">
                <div class="value">{reading['aqi']}</div>
                <div class="category">{reading['category']}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def render_legend():
    chips = "".join(
        f'<span class="legend-chip" style="background:{color};">{name}</span>'
        for _, _, name, _, color in _CATEGORIES
    )
    st.markdown(chips, unsafe_allow_html=True)


def plot_forecast(payload):
    points = [{"horizon_hours": 0, **payload["current"]}] + payload["forecast"]
    labels = ["Now" if p["horizon_hours"] == 0 else f"+{p['horizon_hours']}h" for p in points]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=[p["aqi"] for p in points],
            marker_color=[p["color"] for p in points],
            text=[p["aqi"] for p in points],
            textposition="outside",
            customdata=[p["category"] for p in points],
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
    st.caption(
        "3-day Air Quality Index forecast · gradient boosting blended with a persistence "
        "baseline, retrained daily · data refreshed hourly"
        + (f" · served by the API at {API_URL}" if API_URL else "")
    )

    try:
        with st.spinner("Loading latest data and models..."):
            payload = fetch_payload(city_name)
    except ForecastUnavailable as error:
        st.warning(f"No forecast available yet: {error}")
        return
    except requests.RequestException as error:
        st.error(f"Could not reach the forecast API at {API_URL}: {error}")
        return
    except ConnectionError as error:
        st.error(f"Could not reach the model registry: {error}")
        return

    if payload["unavailable_horizons"]:
        # Better a visibly missing horizon than a plausible-looking number
        # produced with a guessed blend weight.
        st.error(
            "No blend weight is registered for the "
            + ", ".join(f"{h}h" for h in payload["unavailable_horizons"])
            + " model, so its forecast is withheld. Re-run the training pipeline."
        )

    col1, col2 = st.columns([1, 2], gap="large")
    with col1:
        render_hero(payload["current"])
    with col2:
        st.write("")
        st.caption(f"As of **{payload['observed_at']}**")
        if payload["current"]["dominant_pollutant"]:
            st.write(f"Dominant pollutant: **{payload['current']['dominant_pollutant']}**")
        render_legend()

    if payload["alert"]:
        alert_fn = st.error if payload["alert"]["severity"] == "critical" else st.warning
        alert_fn(f"**{payload['alert']['message']}**")

    st.plotly_chart(plot_forecast(payload), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        with st.expander("Why does the model predict this? (SHAP feature importance)"):
            if os.path.exists(SHAP_PLOT_PATH):
                st.image(SHAP_PLOT_PATH)
                st.caption(
                    "Which features drive the predicted *change* in AQI. Regenerated by "
                    "the explainability workflow."
                )
            else:
                st.info(
                    f"No SHAP plot at {SHAP_PLOT_PATH} yet — run "
                    "`python -m training_pipeline.explain`."
                )
    with col_b:
        with st.expander("Model details"):
            st.caption(
                "Each model predicts the change in AQI; the forecast shown is "
                "`current AQI + blend weight × predicted change`. A blend weight of 0 "
                "would be the naive persistence baseline."
            )
            for info in fetch_model_details():
                weight = info["blend_weight"]
                weight_text = "not registered" if weight is None else f"{weight:.2f}"
                st.write(
                    f"**{info['horizon_hours']}h model** — v{info['version']} · "
                    f"blend weight {weight_text}"
                )
                st.json(info["metrics"])


if __name__ == "__main__":
    main()
