"""Compare every candidate model against the naive baseline.

Results are written to reports/ rather than only printed.
"""

import json
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from training_pipeline.baseline_models import (
    build_gradient_boosting,
    build_random_forest,
    build_ridge,
    predict_persistence,
)
from training_pipeline.data import DELTA_COLUMN, load_training_data, time_based_split
from training_pipeline.evaluate import evaluate

# TensorFlow lives in requirements-deep.txt: it cannot be resolved alongside
# hopsworks in one pip pass. Optional so the comparison still runs without it.
try:
    from training_pipeline.deep_model import train_and_predict as train_predict_nn
except ImportError:  # pragma: no cover - depends on the installed environment
    train_predict_nn = None

load_dotenv()

RESULTS_PATH = Path("reports/model_comparison.json")
HORIZONS_HOURS = [24, 48, 72]


def _sklearn(build_fn):
    def train_predict(X_train, y_train, X_test):
        model = build_fn()
        model.fit(X_train, y_train)
        return model.predict(X_test)

    return train_predict


def run(horizons=HORIZONS_HOURS):
    all_results = {}

    for horizon_hours in horizons:
        # Every candidate predicts the CHANGE in AQI, anchored to the latest
        # reading so the numbers are directly comparable to persistence.
        X, y_delta, timestamps, current_aqi = load_training_data(horizon_hours, target=DELTA_COLUMN)
        X_train, X_test, y_train, y_test = time_based_split(X, y_delta, timestamps, horizon_hours)

        anchor = np.asarray(current_aqi.loc[X_test.index], dtype=float)
        truth = anchor + np.asarray(y_test, dtype=float)

        candidates = {
            "Ridge": _sklearn(build_ridge),
            "RandomForest": _sklearn(build_random_forest),
            "GradientBoosting": _sklearn(build_gradient_boosting),
        }
        if train_predict_nn is not None:
            candidates["NeuralNet"] = train_predict_nn
        else:
            # Announced, not silently omitted: a table with a row missing is
            # worse than one that says why.
            print("NeuralNet skipped -- TensorFlow not installed (pip install -r requirements-deep.txt)")

        results = {
            "Persistence": evaluate(truth, predict_persistence(current_aqi.loc[X_test.index]))
        }
        for name, train_predict in candidates.items():
            predicted_delta = train_predict(X_train, y_train, X_test)
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
