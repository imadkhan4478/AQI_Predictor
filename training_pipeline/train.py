"""Train and evaluate all candidate models, report which one wins."""

from dotenv import load_dotenv

from training_pipeline.baseline_models import build_random_forest, build_ridge
from training_pipeline.data import load_training_data, time_based_split
from training_pipeline.deep_model import train_and_predict as train_predict_nn
from training_pipeline.evaluate import evaluate
from training_pipeline.statistical_model import train_and_predict as train_predict_arima

load_dotenv()


def _sklearn_train_predict(build_fn):
    def train_predict(X_train, y_train, X_test):
        model = build_fn()
        model.fit(X_train, y_train)
        return model.predict(X_test)

    return train_predict


def run():
    X, y, timestamps = load_training_data()
    X_train, X_test, y_train, y_test = time_based_split(X, y, timestamps)
    train_timestamps = timestamps.loc[X_train.index]
    test_timestamps = timestamps.loc[X_test.index]
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows (most recent, held out)")

    # Every candidate is a (X_train, y_train, X_test) -> y_pred function, so
    # sklearn/TensorFlow/ARIMA all plug into the same comparison loop below.
    # ARIMA ignores the passed-in features entirely -- it only uses timestamps,
    # captured here via closure since it needs the split's actual time bounds.
    candidates = {
        "Ridge": _sklearn_train_predict(build_ridge),
        "RandomForest": _sklearn_train_predict(build_random_forest),
        "NeuralNet": train_predict_nn,
        "ARIMA": lambda X_train, y_train, X_test: train_predict_arima(train_timestamps, test_timestamps),
    }

    results = {}
    for name, train_predict in candidates.items():
        y_pred = train_predict(X_train, y_train, X_test)
        results[name] = evaluate(y_test, y_pred)

    print("\nModel comparison (on held-out, most-recent 20% of data):")
    for name, metrics in results.items():
        print(f"  {name:15s} RMSE={metrics['RMSE']:.2f}  MAE={metrics['MAE']:.2f}  R2={metrics['R2']:.3f}")

    return results


if __name__ == "__main__":
    run()
