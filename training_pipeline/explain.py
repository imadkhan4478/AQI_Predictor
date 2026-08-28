"""SHAP feature importance for the deployed model.

Attributions are on the predicted *change* in AQI, not its level: a feature that
pushes the delta up is one that makes air quality worse over the horizon.
"""

import json

import numpy as np
import shap
from dotenv import load_dotenv
from matplotlib import pyplot as plt

from training_pipeline.baseline_models import build_gradient_boosting
from training_pipeline.feature_names import display_name
from training_pipeline.data import (
    DELTA_COLUMN,
    FORECAST_HORIZON_HOURS,
    load_training_data,
    time_based_split,
)

load_dotenv()

OUTPUT_PATH = "reports/shap_feature_importance.png"

# The same ranking as data. A PNG cannot be hovered, sorted or read on a phone, and
# its axis labels are raw column names; this lets the dashboard draw the chart with
# readable labels instead of embedding an image.
RANKING_PATH = "reports/shap_feature_importance.json"

# Feature ranking stabilises long before the last few thousand rows, and SHAP on
# a boosted ensemble is slow.
MAX_EXPLAINED_ROWS = 2000


def mean_absolute_contributions(shap_values, columns):
    """Mean |SHAP| per feature, largest first, with readable labels.

    Mean absolute value is the standard summary for a bar-style importance plot: it
    answers "how much does this feature move the prediction", not "in which
    direction", which varies row by row.
    """
    magnitudes = np.abs(np.asarray(shap_values)).mean(axis=0)
    ranked = sorted(zip(columns, magnitudes), key=lambda pair: pair[1], reverse=True)
    return [
        {"column": column, "label": display_name(column), "mean_abs_shap": float(value)}
        for column, value in ranked
    ]


def run(horizon_hours=FORECAST_HORIZON_HOURS, output_path=OUTPUT_PATH, ranking_path=RANKING_PATH):
    X, y_delta, timestamps, _ = load_training_data(horizon_hours, target=DELTA_COLUMN)
    X_train, X_test, y_train, _ = time_based_split(X, y_delta, timestamps, horizon_hours)

    model = build_gradient_boosting()
    model.fit(X_train, y_train)

    explained = X_test.head(MAX_EXPLAINED_ROWS)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(explained)

    ranking = mean_absolute_contributions(shap_values, explained.columns)
    with open(ranking_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "horizon_hours": horizon_hours,
                "explained_rows": int(len(explained)),
                "features": ranking,
            },
            handle,
            indent=2,
        )
    print(f"Saved SHAP ranking ({horizon_hours}h) to {ranking_path}")

    plt.figure()
    shap.summary_plot(shap_values, explained, plot_type="bar", show=False)
    plt.title(f"Drivers of the predicted {horizon_hours}h AQI change")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved SHAP feature-importance plot ({horizon_hours}h) to {output_path}")


if __name__ == "__main__":
    run()
