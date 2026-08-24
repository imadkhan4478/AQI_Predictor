"""Compare every candidate model against the naive baseline.

Results are written to reports/ rather than only printed.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from training_pipeline.baseline_models import (
    build_gradient_boosting,
    build_random_forest,
    build_ridge,
    predict_persistence,
)
from training_pipeline.data import DELTA_COLUMN, load_training_data, time_based_split
from training_pipeline.evaluate import evaluate
from training_pipeline.statistical_model import predict_delta as arima_predict_delta

# TensorFlow lives in requirements-deep.txt: it cannot be resolved alongside
# hopsworks in one pip pass. Optional so the comparison still runs without it.
try:
    from training_pipeline.deep_model import train_and_predict as train_predict_nn
except ImportError:  # pragma: no cover - depends on the installed environment
    train_predict_nn = None

load_dotenv()

RESULTS_PATH = Path("reports/model_comparison.json")
HORIZONS_HOURS = [24, 48, 72]


def _rowwise(build_fn):
    """Adapter for the scikit-learn models, which see one row at a time."""

    def predict(context):
        model = build_fn()
        model.fit(context["X_train"], context["y_train"])
        return model.predict(context["X_test"])

    return predict


def _neural(context):
    return train_predict_nn(context["X_train"], context["y_train"], context["X_test"])


def run(horizons=HORIZONS_HOURS):
    all_results = {}

    for horizon_hours in horizons:
        # Every candidate predicts the CHANGE in AQI, anchored to the latest
        # reading so the numbers are directly comparable to persistence.
        X, y_delta, timestamps, current_aqi = load_training_data(horizon_hours, target=DELTA_COLUMN)
        X_train, X_test, y_train, y_test = time_based_split(X, y_delta, timestamps, horizon_hours)

        anchor = np.asarray(current_aqi.loc[X_test.index], dtype=float)
        truth = anchor + np.asarray(y_test, dtype=float)

        # ARIMA needs one continuous evenly-spaced series rather than independent
        # rows. Built from the data already in memory: re-reading the feature
        # group per horizon meant six full transfers per run, and the read is
        # the least reliable step in the pipeline.
        aqi_series = (
            pd.Series(np.asarray(current_aqi, dtype=float), index=timestamps)
            .sort_index()
            .asfreq("h")
            .interpolate()
            .dropna()
        )

        # One context passed to every candidate, so models needing a continuous
        # series (ARIMA) and models needing rows (the rest) share one loop.
        context = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "train_timestamps": timestamps.loc[X_train.index],
            "test_timestamps": timestamps.loc[X_test.index],
            "current_aqi_test": anchor,
            "aqi_series": aqi_series,
            "horizon_hours": horizon_hours,
        }

        candidates = {
            "Ridge": _rowwise(build_ridge),
            "RandomForest": _rowwise(build_random_forest),
            "GradientBoosting": _rowwise(build_gradient_boosting),
            "ARIMA": arima_predict_delta,
        }
        if train_predict_nn is not None:
            candidates["NeuralNet"] = _neural
        else:
            # Announced, not silently omitted: a table with a row missing is
            # worse than one that says why.
            print("NeuralNet skipped -- TensorFlow not installed (pip install -r requirements-deep.txt)")

        results = {"Persistence": evaluate(truth, predict_persistence(anchor))}
        for name, predict in candidates.items():
            predicted_delta = predict(context)
            results[name] = evaluate(truth, anchor + predicted_delta)

        baseline_r2 = results["Persistence"]["R2"]
        print(f"\nHorizon {horizon_hours}h (held-out, purged split, {len(X_test)} rows):")
        for name, metrics in results.items():
            beats = "" if name == "Persistence" else ("  beats baseline" if metrics["R2"] > baseline_r2 else "  below baseline")
            print(f"  {name:17s} RMSE={metrics['RMSE']:6.2f}  MAE={metrics['MAE']:6.2f}  R2={metrics['R2']:+.3f}{beats}")

        all_results[f"{horizon_hours}h"] = results

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nWrote {RESULTS_PATH}")
    return all_results


if __name__ == "__main__":
    run()
