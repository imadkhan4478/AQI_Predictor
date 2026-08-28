"""Streamlit dashboard: current AQI + 3-day forecast, with hazardous-AQI alerts.

Presentation only -- every number comes from serving/forecast.py, either
in-process or over HTTP from the FastAPI service when AQI_API_URL is set.
"""

import json
import os
import re
import sys
import traceback
from datetime import datetime, timedelta, timezone
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
    load_forecast_models,
    load_observation,
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
SHAP_RANKING_PATH = "reports/shap_feature_importance.json"

# Enough to show the shape of the ranking without turning the panel into a wall.
SHAP_FEATURES_SHOWN = 15

# The page is read by people in the city it forecasts, so timestamps are shown in
# local time. UTC stays the storage and API format.
DISPLAY_TIMEZONE = os.environ.get("CITY_TZ", "Asia/Karachi")

# An hourly pipeline that is two hours behind is late; a day behind is broken.
# Stated on the page because the claim "refreshed hourly" was false for eleven
# days and nothing on the page contradicted it.
STALE_AFTER_HOURS = 2
BROKEN_AFTER_HOURS = 24

# Lahore exceeds 300 in winter, but a fixed 0-500 axis flattens every summer
# reading into the bottom fifth of the chart. 300 keeps the categories a reader
# actually sees legible; the y-range is stated rather than auto-scaled so a
# 6-point move never looks like a cliff.
AXIS_CEILING = 300

CARD_CSS = """
<style>
/* A reading card: neutral surface, category colour carried by a left rule and by
   the number itself. Four large saturated fills read as a warning banner rather
   than as data. */
.reading {
    border: 1px solid rgba(0,0,0,0.10);
    border-left: 6px solid var(--accent);
    border-radius: 8px;
    padding: 0.85rem 1.1rem 0.95rem;
    background: #ffffff;
    box-shadow: 0 1px 2px rgba(16,24,40,0.04);
    height: 100%;
}
.reading .when {
    font-size: 0.70rem; letter-spacing: 0.04em; text-transform: uppercase;
    color: #667085; font-weight: 700;
}
.reading .value {
    font-size: 2.4rem; font-weight: 700; line-height: 1.15; color: var(--accent);
    font-variant-numeric: tabular-nums;
}
.reading.now .value { font-size: 3.1rem; }
.reading .category { font-size: 0.9rem; font-weight: 600; color: #344054; }
.reading .foot { font-size: 0.72rem; color: #667085; margin-top: 0.4rem; }

/* The legend is a key, not a control: quiet, aligned, out of the main column. */
.legend-row {
    display: flex; align-items: center; gap: 0.45rem;
    font-size: 0.74rem; line-height: 1.5; color: #344054;
}
.legend-swatch {
    width: 10px; height: 10px; border-radius: 2px; flex: 0 0 10px;
    border: 1px solid rgba(0,0,0,0.15);
}
.legend-range {
    font-variant-numeric: tabular-nums; color: #667085;
    min-width: 3.4rem; font-size: 0.7rem;
}
.legend-name { font-weight: 600; }
.sidebar-label {
    font-size: 0.68rem; letter-spacing: 0.07em; text-transform: uppercase;
    color: #667085; font-weight: 700; margin: 0.9rem 0 0.2rem;
}
</style>
"""

# The training pipeline registers new versions daily. Without a TTL the models --
# and the metrics shown beside them -- are whatever was in the registry when this
# process started, which can be many days old on a long-lived deployment.
MODEL_CACHE_SECONDS = 3600


@st.cache_resource(ttl=MODEL_CACHE_SECONDS)
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
    feature_row, history = load_observation(city_name)
    return build_forecast(city_name, cached_models(), feature_row, history=history)


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


@st.cache_data(ttl=600)
def shap_ranking():
    """The committed SHAP ranking, or None if the explainability job has not run.

    Read from disk rather than recomputed: SHAP on a boosted ensemble takes minutes,
    which is a weekly job's work, not a page load's.
    """
    try:
        with open(SHAP_RANKING_PATH, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def plot_shap(ranking, shown=SHAP_FEATURES_SHOWN):
    """Feature importance as a chart a reader can hover, with real feature names."""
    features = ranking["features"][:shown][::-1]
    fig = go.Figure(
        go.Bar(
            x=[f["mean_abs_shap"] for f in features],
            y=[f["label"] for f in features],
            orientation="h",
            marker_color="#4c78a8",
            customdata=[f["column"] for f in features],
            hovertemplate="%{y}<br>mean |SHAP| %{x:.2f} AQI<br><i>%{customdata}</i><extra></extra>",
        )
    )
    fig.update_layout(
        xaxis_title="Mean absolute contribution to the predicted change (AQI)",
        margin=dict(t=10, b=10, l=10, r=10),
        height=26 * len(features) + 90,
    )
    return fig


def render_explainability():
    """Why the model predicts a change, in the features' own names.

    The ranking is over the predicted *change* in AQI, not its level, so a feature
    high in this list is one that moves the forecast away from persistence.
    """
    ranking = shap_ranking()
    if ranking is None:
        if os.path.exists(SHAP_PLOT_PATH):
            st.image(SHAP_PLOT_PATH)
            st.caption(
                "Static plot from an earlier explainability run. Re-run the "
                "Explainability workflow to get the interactive version."
            )
        else:
            st.info(
                f"No SHAP output at {SHAP_RANKING_PATH} yet — run the Explainability "
                "workflow, or `python -m training_pipeline.explain`."
            )
        return

    st.plotly_chart(plot_shap(ranking), use_container_width=True)
    st.caption(
        f"Mean absolute SHAP contribution over {ranking['explained_rows']:,} held-out rows, "
        f"for the {ranking['horizon_hours']}h model. Attributions are on the predicted "
        "**change** in AQI rather than its level, so a feature ranked highly here is one "
        "that moves the forecast away from persistence."
    )


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


REPO_URL = "https://github.com/imadkhan4478/AQI_Predictor"
REPORT_URL = f"{REPO_URL}/blob/main/docs/AQI_Predictor_Technical_Report.pdf"


def metric(metrics, name):
    """Look up a registry metric tolerantly.

    Hopsworks rewrites metric keys on the way in: `persistence_MAE` comes back as
    `persistence_mae`, `persistence_R2` as `persistence__r2` with a doubled
    underscore, while top-level `MAE` and `R2` survive untouched. An exact-match
    lookup found neither, so the comparison columns rendered as "not recorded" and
    the panel reported that a model beating its baseline beat nothing.

    Matching on a normalised form -- lowercased, separators stripped -- works
    whatever the exact rule turns out to be, and stops a storage-layer rename from
    silently blanking the project's central claim again.
    """
    wanted = _key(name)
    for key, value in (metrics or {}).items():
        if _key(key) == wanted:
            return value
    return None


def _key(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def has_baseline(metrics):
    """Whether any baseline figure was recorded for this model version.

    "The blend lost" and "nobody wrote the comparison down" must never look alike
    on the page -- the same reason a missing blend weight withholds a horizon
    instead of defaulting to 1.0.
    """
    return any(
        metric(metrics, f"persistence_{name}") is not None for name in ("R2", "MAE", "RMSE")
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
    if _better(metric(metrics, "R2"), metric(metrics, "persistence_R2"), higher_is_better=True):
        wins.append("R2")
    if _better(metric(metrics, "MAE"), metric(metrics, "persistence_MAE"), higher_is_better=False):
        wins.append("MAE")
    if _better(metric(metrics, "RMSE"), metric(metrics, "persistence_RMSE"), higher_is_better=False):
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
        if not has_baseline(metrics):
            verdict = "no baseline recorded"
        elif wins:
            verdict = ", ".join(wins)
        else:
            verdict = "nothing"
        rows.append(
            {
                "Horizon": f"{info['horizon_hours']}h",
                "R2": _number(metric(metrics, "R2")),
                "R2 baseline": _number(metric(metrics, "persistence_R2")),
                "MAE": _number(metric(metrics, "MAE"), 1),
                "MAE baseline": _number(metric(metrics, "persistence_MAE"), 1),
                "RMSE": _number(metric(metrics, "RMSE"), 1),
                "RMSE baseline": _number(metric(metrics, "persistence_RMSE"), 1),
                "Beats baseline on": verdict,
                "Blend weight": _number(info["blend_weight"], 2),
                "Version": info["version"],
            }
        )
    return rows


BASELINE_COLUMNS = {
    "R2 baseline": "R2",
    "MAE baseline": "MAE",
    "RMSE baseline": "RMSE",
}


def drop_empty_baselines(rows):
    """(rows, names of baselines dropped) with all-empty baseline columns removed.

    `persistence_RMSE` is only stored on versions registered after it was added, so
    that column is uniformly blank until the next retrain. An empty column reads as
    unfinished work; naming the omission in the caption reads as a known state.
    """
    missing = [
        column
        for column in BASELINE_COLUMNS
        if all(row.get(column) == "—" for row in rows)
    ]
    trimmed = [{k: v for k, v in row.items() if k not in missing} for row in rows]
    return trimmed, [BASELINE_COLUMNS[column] for column in missing]


def render_evaluation(details):
    """The held-out comparison, on the page rather than only in the report.

    This is the project's strongest claim and it lived in a PDF. A reader cannot
    judge a forecast of 117 without knowing what the naive alternative scores, so
    the baseline sits in the same table as the model at the same size.
    """
    st.subheader("Model and evaluation")
    rows, missing = drop_empty_baselines(evaluation_rows(details))
    st.dataframe(rows, hide_index=True, use_container_width=True)
    if missing:
        st.caption(
            "No baseline is recorded for "
            + ", ".join(missing)
            + " on these model versions — that figure is stored from the next retrain "
            "onward, so the column is omitted rather than shown empty."
        )
    st.caption(
        "Walk-forward evaluation: retrain at successive monthly origins and score only "
        "the following month, never a single frozen split. **Baseline** is persistence — "
        "tomorrow equals today — which for hourly AQI is a strong competitor, not a "
        "straw man. Each forecast is `current AQI + blend weight x predicted change`, so "
        "a weight of 0 would be the baseline exactly and the weight is the share of the "
        f"forecast that is model rather than persistence. Full method in the "
        f"[technical report]({REPORT_URL})."
    )


def reading_card(aqi, category, color, when, foot="", emphasis=False):
    """One AQI reading, rendered the same way wherever it appears."""
    return f"""<div class="reading {'now' if emphasis else ''}" style="--accent:{color};">
        <div class="when">{when}</div>
        <div class="value">{aqi}</div>
        <div class="category">{category}</div>
        <div class="foot">{foot}</div>
    </div>"""


def render_hero(reading, when="Observed", foot=""):
    st.markdown(
        reading_card(
            reading["aqi"], reading["category"], reading["color"], when, foot, emphasis=True
        ),
        unsafe_allow_html=True,
    )


def render_legend():
    """The EPA scale as one row per category.

    Laid out vertically because the category names are long: as inline chips,
    "Unhealthy for Sensitive Groups" and "Unhealthy" wrap into each other and the
    key becomes harder to read than the thing it explains.
    """
    rows = "".join(
        f'<div class="legend-row">'
        f'<span class="legend-swatch" style="background:{color};"></span>'
        f'<span class="legend-range">{lower}&ndash;{upper}</span>'
        f'<span class="legend-name">{name}</span>'
        f"</div>"
        for lower, upper, name, _, color in _CATEGORIES
    )
    st.markdown(rows, unsafe_allow_html=True)


def forecast_points(payload):
    """Forecast values stamped with the wall-clock time each one describes.

    "+24h" is a horizon, not a time. A reader planning tomorrow needs the date,
    so the horizon becomes a timestamp anchored to the observation the forecast
    was made from -- which is also why that observation's age is stated above.
    """
    observed_at = datetime.fromisoformat(payload["observed_at"])
    points = []
    for point in payload["forecast"]:
        when = observed_at + timedelta(hours=point["horizon_hours"])
        points.append({**point, "at": when})
    return points


def plot_forecast(payload):
    """Observed history and the forecast on one timeline.

    Three future numbers alone cannot show whether a forecast continues the recent
    trend or breaks from it. Putting the observations behind them on the same axis
    makes the forecast something a reader can judge rather than just read.
    """
    observed_at = datetime.fromisoformat(payload["observed_at"])
    local = ZoneInfo(DISPLAY_TIMEZONE)
    points = forecast_points(payload)

    fig = go.Figure()

    for lower, upper, name, _, color in _CATEGORIES:
        fig.add_hrect(
            y0=lower,
            y1=min(upper, AXIS_CEILING),
            fillcolor=color,
            opacity=0.10,
            line_width=0,
            layer="below",
            annotation_text=name,
            annotation_position="top left",
            annotation=dict(font_size=9, font_color=color, opacity=0.9),
        )

    history = payload.get("history") or []
    if history:
        fig.add_trace(
            go.Scatter(
                x=[datetime.fromisoformat(h["timestamp"]).astimezone(local) for h in history],
                y=[h["aqi"] for h in history],
                name="Observed",
                mode="lines",
                line=dict(color="#9aa4b2", width=2),
                hovertemplate="%{x|%d %b %H:%M}<br>AQI %{y}<extra>Observed</extra>",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[observed_at.astimezone(local)] + [p["at"].astimezone(local) for p in points],
            y=[payload["current"]["aqi"]] + [p["aqi"] for p in points],
            name="Forecast",
            mode="lines+markers+text",
            line=dict(color="#e8590c", width=3, dash="dot"),
            marker=dict(size=11, color=[payload["current"]["color"]] + [p["color"] for p in points]),
            text=[""] + [str(p["aqi"]) for p in points],
            textposition="top center",
            customdata=[payload["current"]["category"]] + [p["category"] for p in points],
            hovertemplate="%{x|%d %b %H:%M}<br>AQI %{y}<br>%{customdata}<extra>Forecast</extra>",
        )
    )

    fig.add_vline(
        x=observed_at.astimezone(local).timestamp() * 1000,
        line=dict(color="#475467", width=1.5, dash="dash"),
        annotation_text="forecast origin",
        annotation_position="top right",
        annotation=dict(font_size=10, font_color="#475467"),
    )

    fig.update_layout(
        yaxis_title="AQI",
        yaxis_range=[0, AXIS_CEILING],
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        margin=dict(t=60, b=10, l=10, r=10),
        height=420,
        hovermode="x unified",
    )
    fig.update_xaxes(tickformat="%d %b\n%H:%M")
    return fig


def render_forecast_cards(payload, details):
    """The current reading and the three horizons, as one row of equal cards.

    Typical error is shown to one decimal beside the baseline's. Rounded to whole
    AQI the 24h pair reads "±30 · baseline ±30", concealing that the blend is 30.2
    against persistence's 29.8 -- slightly worse. A comparison a reader cannot lose
    is the whole point of showing it.
    """
    by_horizon = {info["horizon_hours"]: info for info in details}
    points = forecast_points(payload)
    local = ZoneInfo(DISPLAY_TIMEZONE)
    _, observed_text, tz_label = observation_age(payload)

    columns = st.columns(1 + len(points), gap="medium")
    with columns[0]:
        render_hero(
            payload["current"],
            when=f"Observed · {observed_text} {tz_label}",
            foot="Measured, not forecast",
        )

    for column, point in zip(columns[1:], points):
        metrics = (by_horizon.get(point["horizon_hours"], {}).get("metrics")) or {}
        foot = typical_error_text(metrics)
        when = f"+{point['horizon_hours']}h · {point['at'].astimezone(local):%a %d %b, %H:%M}"
        with column:
            st.markdown(
                reading_card(point["aqi"], point["category"], point["color"], when, foot),
                unsafe_allow_html=True,
            )


def typical_error_text(metrics):
    """The horizon's measured error beside the baseline's, to one decimal.

    Rounded to whole AQI the 24h pair reads "±30 · baseline ±30", concealing that
    the blend is 30.2 against persistence's 29.8 -- slightly worse. A comparison a
    reader cannot lose is the whole point of showing it.
    """
    mae = metric(metrics, "MAE")
    baseline = metric(metrics, "persistence_MAE")
    if mae is None:
        return "Typical error not recorded for this version"
    if baseline is None:
        return f"Typical error ±{mae:.1f} AQI"
    return f"Typical error ±{mae:.1f} · baseline ±{baseline:.1f}"


def render_sidebar(payload, details):
    """Project identity, the EPA key, and what is deployed.

    Reference material a reader consults once. Keeping it out of the main column
    leaves that column for the forecast and the evidence behind it.
    """
    with st.sidebar:
        st.markdown("### 10Pearls AQI Predictor")
        st.caption("Data Science Internship Programme  \nImad Khan · BS Data Science, GIKI")
        st.divider()

        st.markdown('<div class="sidebar-label">City</div>', unsafe_allow_html=True)
        st.write(f"**{payload['city']}**, Punjab · Pakistan")

        dominant = payload["current"].get("dominant_pollutant")
        if dominant:
            st.markdown(
                '<div class="sidebar-label">Dominant pollutant</div>', unsafe_allow_html=True
            )
            st.write(f"**{dominant}**")

        st.markdown('<div class="sidebar-label">EPA AQI scale</div>', unsafe_allow_html=True)
        render_legend()

        st.markdown('<div class="sidebar-label">Deployed models</div>', unsafe_allow_html=True)
        for info in details:
            weight = info["blend_weight"]
            weight_text = "—" if weight is None else f"{weight:.2f}"
            st.caption(
                f"{info['horizon_hours']}h · v{info['version']} · blend weight {weight_text}"
            )

        st.markdown('<div class="sidebar-label">Documentation</div>', unsafe_allow_html=True)
        st.caption(f"[Technical report]({REPORT_URL})  \n[Source and experiment log]({REPO_URL})")


def main():
    city_name = os.environ["CITY_NAME"]
    st.set_page_config(
        page_title=f"10Pearls AQI Predictor — {city_name}", page_icon="🌫️", layout="wide"
    )
    st.markdown(CARD_CSS, unsafe_allow_html=True)

    st.title(f"Air quality forecast — {city_name}")
    st.caption(
        "US EPA Air Quality Index at 24, 48 and 72 hours · gradient boosting blended with a "
        "persistence baseline, retrained daily · feature pipeline scheduled hourly"
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

    details = fetch_model_details()
    render_sidebar(payload, details)
    render_forecast_cards(payload, details)

    if payload["alert"]:
        alert_fn = st.error if payload["alert"]["severity"] == "critical" else st.warning
        alert_fn(payload["alert"]["message"])

    st.subheader("Observed AQI and 3-day forecast")
    st.plotly_chart(plot_forecast(payload), use_container_width=True)
    st.caption(
        "The dashed rule is the **forecast origin** — the newest observation with a complete "
        "set of lag features, which is what the models were given. Readings to the right of "
        "it arrived afterwards and the forecast did not have them; where they diverge, that "
        "divergence is the error as it happens."
    )

    render_evaluation(details)

    with st.expander("Why does the model predict this? (SHAP feature importance)"):
        render_explainability()

    with st.expander("Raw registry metrics"):
        for info in details:
            st.write(f"**{info['model_name']}** — version {info['version']}")
            st.json(info["metrics"])


if __name__ == "__main__":
    main()
