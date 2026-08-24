"""Load the latest registered forecast model(s) from the Hopsworks Model Registry."""

import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

import hopsworks
import joblib

# hopsworks.login() has no timeout parameter of its own -- on a flaky
# connection it can hang for 20+ minutes relying on the OS's default socket
# timeout. We can't change that internal timeout, so instead we run the call
# in a background thread and impose our own wait limit on top of it.
#
# IMPORTANT: don't use the executor as a context manager here -- `with
# ThreadPoolExecutor() as executor:` calls shutdown(wait=True) on exit, which
# blocks until the background thread actually finishes, silently undoing the
# timeout we just imposed. We shut down with wait=False instead: the thread
# itself can't be forcibly killed (it keeps running until it finishes or the
# process exits), but the caller stops waiting on it and moves on to retry.
CONNECT_TIMEOUT_SECONDS = 20
MAX_CONNECT_ATTEMPTS = 3


def _connect():
    kwargs = dict(
        project=os.environ["HOPSWORKS_PROJECT_NAME"],
        api_key_value=os.environ["HOPSWORKS_API_KEY"],
        cert_folder=os.path.join(tempfile.gettempdir(), "hopsworks_certs"),
    )
    last_error = None
    for attempt in range(1, MAX_CONNECT_ATTEMPTS + 1):
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(hopsworks.login, **kwargs)
        try:
            result = future.result(timeout=CONNECT_TIMEOUT_SECONDS)
            executor.shutdown(wait=False)
            return result
        except FutureTimeoutError:
            executor.shutdown(wait=False)
            last_error = f"timed out after {CONNECT_TIMEOUT_SECONDS}s"
        except Exception as e:
            executor.shutdown(wait=False)
            last_error = f"{type(e).__name__}: {e}"

        if attempt < MAX_CONNECT_ATTEMPTS:
            print(
                f"Hopsworks connection failed ({last_error}), attempt {attempt}/{MAX_CONNECT_ATTEMPTS}. Retrying...",
                flush=True,
            )

    raise ConnectionError(f"Could not connect to Hopsworks after {MAX_CONNECT_ATTEMPTS} attempts. Last error: {last_error}")


def _read_blend_weight(local_dir, metrics):
    """The scalar `w` in `prediction = current_aqi + w * predicted_delta`.

    Preferred source is blend.json, written next to the artifact by
    training_pipeline/register.py, because it travels with the model file and
    cannot drift from it. Registry metrics are the fallback for models
    registered before blend.json existed.

    Deliberately no numeric default: w silently defaulting to 1.0 would ship an
    unshrunk model that loses to the naive baseline at every horizon, and it
    would look like a working forecast. Returning None makes the caller decide
    visibly instead.
    """
    blend_path = os.path.join(local_dir, "blend.json")
    if os.path.exists(blend_path):
        with open(blend_path, encoding="utf-8") as handle:
            weight = json.load(handle).get("blend_weight")
            if weight is not None:
                return float(weight)

    weight = (metrics or {}).get("blend_weight")
    return None if weight is None else float(weight)


def load_latest_model(model_name):
    """Fetch the highest-version copy of a registered model.

    Returns (model, version, metrics, blend_weight). The model predicts the
    *change* in AQI, not its level, so blend_weight is not optional
    bookkeeping -- without it the caller cannot turn the output into a forecast.

    NOTE: get_model()'s version=None default actually means version 1, not
    "latest" -- so once daily retraining is producing new versions, we must pick
    the max ourselves.
    """
    project = _connect()
    mr = project.get_model_registry()

    versions = mr.get_models(model_name)
    latest = max(versions, key=lambda m: m.version)

    local_dir = latest.download()
    model = joblib.load(os.path.join(local_dir, "model.pkl"))
    metrics = latest.training_metrics
    return model, latest.version, metrics, _read_blend_weight(local_dir, metrics)
