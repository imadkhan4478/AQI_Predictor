"""Register the winning model (Random Forest) in the Hopsworks Model Registry."""

import os
import shutil
import tempfile

import hopsworks
import joblib
from dotenv import load_dotenv

from training_pipeline.baseline_models import build_random_forest
from training_pipeline.data import FEATURE_COLUMNS, load_training_data, time_based_split
from training_pipeline.evaluate import evaluate

load_dotenv()

MODEL_NAME = "aqi_random_forest"


def run():
    X, y, timestamps = load_training_data()
    X_train, X_test, y_train, y_test = time_based_split(X, y, timestamps)

    # Honest metrics come from the held-out split (never trained on).
    honest_model = build_random_forest()
    honest_model.fit(X_train, y_train)
    metrics = evaluate(y_test, honest_model.predict(X_test))
    print(f"Held-out evaluation metrics (what gets registered): {metrics}")

    # The deployed artifact is retrained on ALL available data -- the held-out
    # split already proved it beats the naive baseline, so using every row
    # for the final model gives it the most real-world data to learn from.
    final_model = build_random_forest()
    final_model.fit(X, y)

    model_dir = tempfile.mkdtemp()
    try:
        joblib.dump(final_model, os.path.join(model_dir, "model.pkl"))

        project = hopsworks.login(
            project=os.environ["HOPSWORKS_PROJECT_NAME"],
            api_key_value=os.environ["HOPSWORKS_API_KEY"],
            cert_folder=os.path.join(tempfile.gettempdir(), "hopsworks_certs"),
        )
        mr = project.get_model_registry()
        model = mr.python.create_model(
            name=MODEL_NAME,
            metrics=metrics,
            description=(
                "Random Forest predicting AQI 3 days ahead for Lahore. "
                f"Features: {', '.join(FEATURE_COLUMNS)}. "
                "Winner among Ridge/RandomForest/NeuralNet/ARIMA on held-out RMSE/MAE/R2."
            ),
            input_example=X.head(1),
        )
        model.save(model_dir)
        print(f"Registered model '{MODEL_NAME}' version {model.version} in the Hopsworks Model Registry")
    finally:
        shutil.rmtree(model_dir, ignore_errors=True)


if __name__ == "__main__":
    run()
