"""Load the latest registered forecast model(s) from the Hopsworks Model Registry."""

import os
import tempfile

import hopsworks
import joblib


def _connect():
    return hopsworks.login(
        project=os.environ["HOPSWORKS_PROJECT_NAME"],
        api_key_value=os.environ["HOPSWORKS_API_KEY"],
        cert_folder=os.path.join(tempfile.gettempdir(), "hopsworks_certs"),
    )


def load_latest_model(model_name):
    """Fetch the highest-version copy of a registered model. NOTE: get_model()'s
    version=None default actually means version 1, not "latest" -- so once daily
    retraining is producing new versions, we must pick the max ourselves."""
    project = _connect()
    mr = project.get_model_registry()

    versions = mr.get_models(model_name)
    latest = max(versions, key=lambda m: m.version)

    local_dir = latest.download()
    model = joblib.load(os.path.join(local_dir, "model.pkl"))
    return model, latest.version, latest.training_metrics
