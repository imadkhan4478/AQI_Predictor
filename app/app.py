"""Streamlit dashboard: current AQI + 3-day forecast, with hazardous-AQI alerts.

Presentation only -- every number comes from serving/forecast.py, either
in-process or over HTTP from the FastAPI service when AQI_API_URL is set.
"""

import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

# `streamlit run app/app.py` puts this file's folder on sys.path, not the repo
# root, so sibling packages would not otherwise resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv

from feature_pipeline.aqi import _CATEGORIES
from monitoring import report, start_monitoring
from serving.forecast import (
    HORIZONS_HOURS,
    ForecastUnavailable,
    build_forecast,
    load_feature_row,
    load_forecast_models,
    model_details,
)

load_dotenv()

# Streamlit Community Cloud has no .env -- it supplies configuration through
# st.secrets. Copy those into the environment so the serving core, which is not
# Streamlit-aware, reads configuration the same way in both places.
try:
    for _key, _value in st.secrets.items():
        if isinstance(_value, str):
            os.environ.setdefault(_key, _value)
except Exception:  # no secrets configured -- normal when running locally from .env
    pass

start_monitoring("dashboard")

# Set AQI_API_URL to read from the FastAPI service instead of loading models into
# the Streamlit process. Both paths return the same payload, so this is a
# deployment choice rather than a behavioural one.
API_URL = os.environ.get("AQI_API_URL", "").rstrip("/")
API_TIMEOUT_SECONDS = 30

SHAP_PLOT_PATH = "reports/shap_feature_importance.png"

# The page is read by people in the city it forecasts, so timestamps are shown in
# local time. UTC stays the storage and API format.
DISPLAY_TIMEZONE = os.environ.get("CITY_TZ", "Asia/Karachi")

# An hourly pipeline that is two hours behind is late; a day behind is broken.
# Stated on the page because the claim "refreshed hourly" was false for eleven
# days and nothing on the page contradicted it.
STALE_AFTER_HOURS = 2
BROKEN_AFTER_HOURS = 24

CARD_CSS = """
<style>
/* Text colour is set per card from the background's luminance, not here. */
.aqi-hero {
    border-radius: 12px;
    padding: 1.5rem 1.75rem;
}
.aqi-hero .value { font-size: 3.2rem; font-weight: 700; line-height: 1; }
.aqi-hero .category { font-size: 1.2rem; font-weight: 600; margin-top: 0.25rem; }
.legend-chip {
    display: inline-flex; align-items: center; gap: 0.4rem;
    padding: 0.15rem 0.6rem; border-radius: 999px;
    font-size: 0.78rem; margin: 0.15rem;
}
</style>
"""


@st.cache_resource
def cached_models():
    """Cached as a resource, not data: model objects are not serialisable and
    belong to the process, not the session."""
    return load_forecast_models(HORIZONS_HOURS)


@st.cache_data(ttl=600)
def fetch_payload(city_name):
    """Forecast payload, from the API if configured. TTL matches the hourly
    cadence of the feature pipeline."""
    if API_URL:
        response = requests.get(f"{API_URL}/forecast", timeout=API_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()
    return build_forecast(city_name, cached_models(), load_feature_row(city_name))


@st.cache_resource
def last_good_payload():
    """Holder for the most recent payload that loaded, shared across sessions.

    Hopsworks' Query Service is the single point of failure on the read path and
    it goes down for stretches. A reader is better served by the last reading with
    its age stated plainly than by an error, so long as the age is impossible to
    miss -- which is what render_freshness guarantees.
    """
    return {}


def fetch_payload_or_last_good(city_name):
    """(payload, live) -- `live` is False when serving a retained payload.

    The broad except is deliberate: the failure modes here are a remote service's,
    not this code's, and no exception from them should reach the page as a
    traceback. The detail goes to the log instead.
    """
    try:
        payload = fetch_payload(city_name)
    except Exception as error:
        retained = last_good_payload().get("payload")
        traceback.print_exc()
        report(error, city=city_name, served_retained_payload=retained is not None)
        if retained is None:
            raise
        return retained, False
    last_good_payload()["payload"] = payload
    return payload, True


@st.cache_data(ttl=600)
def fetch_model_details():
    if API_URL:
        response = requests.get(f"{API_URL}/models", timeout=API_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()
    return model_details(cached_models())


def readable_text_color(background_hex):
    """Black or white against this background, whichever has more contrast.

    Every category card used to carry white text, which fails WCAG on the yellow
    and orange bands -- the two most common readings in Lahore outside winter.
    Crossover is at relative luminance 0.179, where the ratio against black and
    against white are equal.
    """
    channels = [int(background_hex.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    luminance = 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]
    return "#111111" if luminance > 0.179 else "#ffffff"


def observation_age(payload):
    """(hours since the observation, local-time string, timezone abbreviation)."""
    observed = datetime.fromisoformat(payload["observed_at"])
    age_hours = (datetime.now(timezone.utc) - observed).total_seconds() / 3600
    local = observed.astimezone(ZoneInfo(DISPLAY_TIMEZONE))
    return age_hours, local.strftime("%d %B %Y, %H:%M"), local.tzname()


def relative_age(age_hours):
    if age_hours < 1.5:
        return "less than an hour ago"
    if age_hours < 48:
        return f"{round(age_hours)} hours ago"
    return f"{round(age_hours / 24)} days ago"


def render_freshness(payload):
    """State how old the reading is, and say so loudly when it is too old.

    The dashboard once served an eleven-day-old reading under a caption promising
    hourly refresh. A claim a page makes about itself has to be falsifiable on
    that page.
    """
    age_hours, local_text, tz_label = observation_age(payload)
    # "complete", not "latest": serving uses the newest row that has a full set of
    # lag features, which is not the newest row when there is a gap behind it.
    line = f"Latest complete observation **{local_text} {tz_label}** — {relative_age(age_hours)}"

    if age_hours >= BROKEN_AFTER_HOURS:
        st.error(
            f"{line}. Either the hourly pipeline is not reaching the offline store or "
            "there is a gap in the recent history, so every number below describes that "
            "observation rather than now."
        )
    elif age_hours >= STALE_AFTER_HOURS:
        st.warning(f"{line}. Later than an hourly pipeline should be.")
    else:
        st.success(line)
    return age_hours


REPORT_URL = (
    "https://github.com/imadkhan4478/AQI_Predictor/blob/main/docs/"
    "AQI_Predictor_Technical_Report.pdf"
)


def beats_baseline(metrics):
    """Which metrics the blend actually wins on, computed rather than asserted.

    R2 rewards being higher, MAE and RMSE reward being lower. Comparing each one
    separately matters here: the blend wins clearly on R2 and RMSE while being
    level with persistence on MAE at the shorter horizons, because its advantage
    is in large errors rather than typical ones. Stating "beats the baseline"
    without saying on what would be the kind of claim this project exists to avoid.
    """
    wins = []
    if _better(metrics.get("R2"), metrics.get("persistence_R2"), higher_is_better=True):
        wins.append("R2")
    if _better(metrics.get("MAE"), metrics.get("persistence_MAE"), higher_is_better=False):
        wins.append("MAE")
    if _better(metrics.get("RMSE"), metrics.get("persistence_RMSE"), higher_is_better=False):
        wins.append("RMSE")
    return wins


def _better(model_value, baseline_value, higher_is_better):
    if model_value is None or baseline_value is None:
        return False
    return model_value > baseline_value if higher_is_better else model_value < baseline_value


def _number(value, places=3):
    return "—" if value is None else f"{value:.{places}f}"


def evaluation_rows(details):
    """One row per horizon: the blend beside the baseline it has to beat."""
    rows = []
    for info in details:
        metrics = info["metrics"] or {}
        wins = beats_baseline(metrics)
        rows.append(
            {
                "Horizon": f"{info['horizon_hours']}h",
                "R2": _number(metrics.get("R2")),
                "R2 baseline": _number(metrics.get("persistence_R2")),
                "MAE": _number(metrics.get("MAE"), 1),
                "MAE baseline": _number(metrics.get("persistence_MAE"), 1),
                "RMSE": _number(metrics.get("RMSE"), 1),
                "RMSE baseline": _number(metrics.get("persistence_RMSE"), 1),
                "Beats baseline on": ", ".join(wins) if wins else "nothing",
                "Blend weight": _number(info["blend_weight"], 2),
                "Version": info["version"],
            }
        )
    return rows


def render_evaluation(details):
    """The held-out comparison, on the page rather than only in the report.

    This is the project's strongest claim and it lived in a PDF. A reader cannot
    judge a forecast of 117 without knowing what the naive alternative scores, so
    the baseline sits in the same table as the model at the same size.
    """
    st.subheader("Model and evaluation")
    st.dataframe(evaluation_rows(details), hide_index=True, use_container_width=True)
    st.caption(
        "Walk-forward evaluation: retrain at successive monthly origins and score only "
        "the following month, never a single frozen split. **Baseline** is persistence — "
        "tomorrow equals today — which for hourly AQI is a strong competitor, not a "
        "straw man. Each forecast is `current AQI + blend weight x predicted change`, so "
        "a weight of 0 would be the baseline exactly and the weight is the share of the "
        f"forecast that is model rather than persistence. Full method in the "
        f"[technical report]({REPORT_URL})."
    )


def render_hero(reading):
    st.markdown(
        f"""<div class="aqi-hero" style="background:{reading['color']};
                    color:{readable_text_color(reading['color'])};">
                <div class="value">{reading['aqi']}</div>
                <div class="category">{reading['category']}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def render_legend():
    chips = "".join(
        f'<span class="legend-chip" style="background:{color};'
        f'color:{readable_text_color(color)};">{name}</span>'
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
        "baseline, retrained daily · feature pipeline runs hourly"
        + (f" · served by the API at {API_URL}" if API_URL else "")
    )

    try:
        with st.spinner("Loading latest data and models..."):
            payload, live = fetch_payload_or_last_good(city_name)
    except ForecastUnavailable as error:
        st.warning(f"No forecast available yet: {error}")
        return
    except requests.RequestException:
        st.error(
            f"The forecast API at {API_URL} is not responding. Nothing is cached yet, "
            "so there is nothing to show. Try again shortly."
        )
        return
    except Exception:
        # Whatever broke is on the far side of a network call. A traceback on the
        # page tells a reader nothing and looks like a broken project.
        traceback.print_exc()
        st.error(
            "The feature store is not answering reads right now, and no earlier "
            "reading is cached in this process. Details are in the app log. "
            "This clears on its own -- the hourly pipeline keeps collecting either way."
        )
        return

    if not live:
        st.info(
            "Live read failed, so this is the last reading that loaded successfully. "
            "Its age is stated below."
        )

    render_freshness(payload)

    if payload["unavailable_horizons"]:
        # A visibly missing horizon beats a plausible number from a guessed weight.
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
        _, local_text, tz_label = observation_age(payload)
        st.caption(f"Observed **{local_text} {tz_label}** · stored as {payload['observed_at']}")
        if payload["current"]["dominant_pollutant"]:
            st.write(f"Dominant pollutant: **{payload['current']['dominant_pollutant']}**")
        render_legend()

    if payload["alert"]:
        alert_fn = st.error if payload["alert"]["severity"] == "critical" else st.warning
        alert_fn(f"**{payload['alert']['message']}**")

    st.plotly_chart(plot_forecast(payload), use_container_width=True)

    details = fetch_model_details()
    render_evaluation(details)

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
        with st.expander("Raw registry metrics"):
            for info in details:
                st.write(f"**{info['model_name']}** — version {info['version']}")
                st.json(info["metrics"])


if __name__ == "__main__":
    main()
