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
        try:
            with open(blend_path, encoding="utf-8") as handle:
                weight = json.load(handle).get("blend_weight")
            if weight is not None:
                return float(weight)
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as error:
            # A truncated or malformed blend.json is exactly what a failed
            # upload leaves behind, so it must not raise here. Fall through to
            # the registry metric, and ultimately to None -- a withheld horizon
            # the dashboard reports, rather than a crash on load.
            print(
                f"Ignoring unreadable {blend_path} ({type(error).__name__}); "
                "falling back to registry metrics.",
                flush=True,
            )

    weight = (metrics or {}).get("blend_weight")
    return None if weight is None else float(weight)


def load_latest_model(model_name):
    """Fetch the newest *usable* version of a registered model.

    Returns (model, version, metrics, blend_weight). The model predicts the
    *change* in AQI, not its level, so blend_weight is not optional
    bookkeeping -- without it the caller cannot turn the output into a forecast.

    Walks versions from highest downwards instead of trusting the highest,
    because a registry write can fail partway through and leave a version whose
    metadata row exists but whose files do not. That happened for real: an
    upload returned HTTP 500 from Hopsworks' filesystem metadata layer after
    model.pkl had landed but before the schema did. Serving last week's intact
    model is correct behaviour; refusing to start because the newest version is
    half-written is not.

    NOTE: get_model()'s version=None default actually means version 1, not
    "latest" -- so once daily retraining is producing new versions, the version
    must be chosen explicitly.
    """
    project = _connect()
    mr = project.get_model_registry()

    versions = sorted(mr.get_models(model_name), key=lambda m: m.version, reverse=True)
    if not versions:
        raise LookupError(f"No versions of model {model_name!r} are registered")

    failures = []
    for candidate in versions:
        try:
            local_dir = candidate.download()
            model = joblib.load(os.path.join(local_dir, "model.pkl"))
        except Exception as error:
            failures.append(f"v{candidate.version}: {type(error).__name__}: {error}")
            print(
                f"Skipping {model_name} v{candidate.version} -- artifact could not be "
                f"loaded ({type(error).__name__}). Trying the previous version.",
                flush=True,
            )
            continue

        metrics = candidate.training_metrics
        return model, candidate.version, metrics, _read_blend_weight(local_dir, metrics)

    raise LookupError(
        f"No usable version of {model_name!r} could be loaded. Attempts: " + "; ".join(failures)
    )
