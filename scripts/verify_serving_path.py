"""Offline smoke test of the slow paths, without Hopsworks or Streamlit.

`pytest tests` covers the fast logic. This exercises what a unit test should not:
real model fits against a synthetic multi-year feature store, so SHAP and the
walk-forward evaluation are checked before a runner spends 20 minutes on them.

    python scripts/verify_serving_path.py
"""

import json
import os
import sys
import tempfile
import types

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CITY = "Lahore"
HOURS = 900
RNG = np.random.default_rng(0)


def synthetic_store(hours=HOURS, start="2026-01-01"):
    """Hourly rows shaped like the real feature group, with a deliberate 4-hour
    gap so the reindex-and-interpolate path in add_lag_features is exercised."""
    index = pd.date_range(start, periods=hours, freq="h", tz="UTC")
    index = index.delete(range(500, 504))

    daily = 40 * np.sin(2 * np.pi * index.hour / 24)
    aqi = np.clip(180 + daily + RNG.normal(0, 25, len(index)).cumsum() * 0.1, 5, 500)

    return pd.DataFrame(
        {
            "city_name": CITY,
            "timestamp": index,
            "hour": index.hour,
            "day": index.day,
            "month": index.month,
            "temp": 25.0 + RNG.normal(0, 5, len(index)),
            "feels_like": 26.0 + RNG.normal(0, 5, len(index)),
            "humidity": RNG.integers(20, 90, len(index)).astype(float),
            "pressure": RNG.integers(995, 1020, len(index)).astype(float),
            "wind_speed": np.abs(RNG.normal(2, 1, len(index))),
            "wind_deg": RNG.integers(0, 360, len(index)).astype(float),
            "clouds": RNG.integers(0, 100, len(index)).astype(float),
            "co": 300.0 + RNG.normal(0, 50, len(index)),
            "no": np.abs(RNG.normal(1, 0.5, len(index))),
            "no2": np.abs(RNG.normal(20, 5, len(index))),
            "o3": np.abs(RNG.normal(60, 20, len(index))),
            "so2": np.abs(RNG.normal(10, 3, len(index))),
            "pm2_5": aqi / 2.5,
            "pm10": aqi / 1.5,
            "nh3": np.abs(RNG.normal(5, 2, len(index))),
            "dominant_pollutant": "pm2_5",
            "aqi": aqi.round(),
            "aqi_change_rate": np.diff(aqi, prepend=aqi[0]),
        }
    )


STORE = synthetic_store()


def install_stubs():
    """Stub Streamlit, Plotly and the Hopsworks client. Everything under test
    runs for real."""
    streamlit = types.ModuleType("streamlit")
    streamlit.cache_data = lambda *a, **k: (lambda fn: fn)
    streamlit.cache_resource = lambda fn=None, **k: fn if fn else (lambda f: f)
    sys.modules["streamlit"] = streamlit

    graph_objects = types.ModuleType("plotly.graph_objects")
    graph_objects.Figure = object
    graph_objects.Bar = object
    plotly = types.ModuleType("plotly")
    plotly.graph_objects = graph_objects
    sys.modules["plotly"] = plotly
    sys.modules["plotly.graph_objects"] = graph_objects

    for name in ("hopsworks", "hopsworks_common", "hopsworks_common.client", "hopsworks_common.client.exceptions"):
        module = types.ModuleType(name)
        sys.modules.setdefault(name, module)
    sys.modules["hopsworks"].login = lambda **kwargs: None
    sys.modules["hopsworks_common.client.exceptions"].RestAPIError = type(
        "RestAPIError", (Exception,), {}
    )

    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda *a, **k: None
    sys.modules.setdefault("dotenv", dotenv)

    import feature_pipeline.hopsworks_store as store

    store.read_features_df = lambda: STORE.copy()


def check(label, condition, detail=""):
    print(f"  {'PASS' if condition else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    if not condition:
        raise SystemExit(1)


def main():
    install_stubs()

    from app.load_model import _read_blend_weight
    from serving import forecast as serving
    from training_pipeline.baseline_models import build_gradient_boosting
    from training_pipeline.data import DELTA_COLUMN, FEATURE_COLUMNS, load_training_data

    print("\n1. Live row is assembled with every training feature present")
    row = serving.load_feature_row(CITY)
    missing = [c for c in FEATURE_COLUMNS if c not in row.index]
    check("no missing feature columns", not missing, str(missing))
    check("no NaN in features", not row[FEATURE_COLUMNS].isna().any())
    check("lag features present", "aqi_lag_24" in row.index and "aqi_rstd_72" in row.index)
    check("timestamp survives the reindex", pd.notna(row["timestamp"]), str(row["timestamp"]))

    print("\n2. Blend arithmetic turns a predicted delta into a forecast level")
    X, y_delta, timestamps, current_aqi = load_training_data(24, target=DELTA_COLUMN)
    model = build_gradient_boosting()
    model.fit(X.iloc[:-50], y_delta.iloc[:-50])

    features = row[FEATURE_COLUMNS].to_frame().T.astype(float)
    anchor = int(row["aqi"])
    raw_delta = float(model.predict(features)[0])

    at_zero = serving.blend_forecast({"model": model, "blend_weight": 0.0}, features, anchor)
    at_half = serving.blend_forecast({"model": model, "blend_weight": 0.5}, features, anchor)
    at_one = serving.blend_forecast({"model": model, "blend_weight": 1.0}, features, anchor)

    check("w=0 is exactly persistence", at_zero == anchor, f"{at_zero} == {anchor}")
    check(
        "w=1 is anchor + full delta",
        at_one == int(round(min(max(anchor + raw_delta, 0), 500))),
        f"{at_one} vs {anchor} + {raw_delta:.2f}",
    )
    check("w=0.5 lies between", min(at_zero, at_one) <= at_half <= max(at_zero, at_one),
          f"{at_zero} <= {at_half} <= {at_one}")
    check("forecast is a plausible AQI", 0 <= at_half <= 500, str(at_half))
    check(
        "missing weight withholds the forecast",
        serving.blend_forecast({"model": model, "blend_weight": None}, features, anchor) is None,
    )

    print("\n3. Blend weight is read from the artifact, not guessed")
    with tempfile.TemporaryDirectory() as directory:
        check("absent everywhere -> None", _read_blend_weight(directory, {}) is None)
        check("falls back to registry metrics", _read_blend_weight(directory, {"blend_weight": 0.4}) == 0.4)
        with open(os.path.join(directory, "blend.json"), "w", encoding="utf-8") as handle:
            json.dump({"blend_weight": 0.65, "horizon_hours": 24}, handle)
        check(
            "blend.json wins over metrics",
            _read_blend_weight(directory, {"blend_weight": 0.4}) == 0.65,
        )

    print("\n4. SHAP works on the deployed model type")
    import matplotlib

    matplotlib.use("Agg")
    from training_pipeline import explain

    explain.MAX_EXPLAINED_ROWS = 100
    output = os.path.join(tempfile.gettempdir(), "shap_check.png")
    explain.run(horizon_hours=24, output_path=output)
    check("plot written", os.path.exists(output) and os.path.getsize(output) > 5000,
          f"{os.path.getsize(output)} bytes")

    print("\n5. The daily job's walk-forward evaluation runs end to end")
    # Three years of synthetic history, because the evaluation needs 12 monthly
    # retrain origins each with >= MIN_TRAIN_ROWS of prior data behind it.
    # data.py binds read_features_df at import time, so swap it there.
    from training_pipeline import data as training_data
    from training_pipeline import register

    training_data.read_features_df = lambda: synthetic_store(hours=26_280, start="2023-01-01")
    X, y_delta, timestamps, current_aqi = load_training_data(24, target=DELTA_COLUMN)
    metrics, baseline, weight = register.walk_forward_evaluate(
        X, y_delta, timestamps, current_aqi, 24
    )
    check("blend weight in range", 0.0 <= weight <= 1.0, f"w={weight:.2f}")
    check("metrics are finite", all(np.isfinite(v) for v in metrics.values()), str(metrics))
    check(
        "blend is at least as good as persistence",
        metrics["R2"] >= baseline["R2"] - 1e-9,
        f"model R2={metrics['R2']:.3f} vs persistence {baseline['R2']:.3f}",
    )

    print("\nAll serving-path checks passed.\n")


if __name__ == "__main__":
    main()
