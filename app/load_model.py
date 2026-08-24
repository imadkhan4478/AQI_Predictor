"""Load registered forecast models from the Hopsworks Model Registry."""

import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

import hopsworks
import joblib

CONNECT_TIMEOUT_SECONDS = 20
MAX_CONNECT_ATTEMPTS = 3


def _connect():
    """Log in to Hopsworks with a client-side timeout.

    hopsworks.login() has no timeout of its own and can hang for 20+ minutes on
    the OS socket default, so it runs in a thread we stop waiting on. The
    executor is shut down with wait=False deliberately: wait=True blocks until
    the hung thread finishes, undoing the timeout.
    """
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
    """The scalar w in `current_aqi + w * predicted_delta`.

    Prefers blend.json beside the artifact, since it travels with the model file;
    registry metrics are the fallback. Returns None rather than a default, so a
    missing weight surfaces as a withheld horizon instead of an unshrunk model.
    """
    blend_path = os.path.join(local_dir, "blend.json")
    if os.path.exists(blend_path):
        try:
            with open(blend_path, encoding="utf-8") as handle:
                weight = json.load(handle).get("blend_weight")
            if weight is not None:
                return float(weight)
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as error:
            # A truncated blend.json is what a failed upload leaves behind, so it
            # must not raise here.
            print(
                f"Ignoring unreadable {blend_path} ({type(error).__name__}); "
                "falling back to registry metrics.",
                flush=True,
            )

    weight = (metrics or {}).get("blend_weight")
    return None if weight is None else float(weight)


def load_latest_model(model_name):
    """Newest usable version, as (model, version, metrics, blend_weight).

    Walks versions downwards rather than trusting the highest, because a failed
    registry write can leave a version whose metadata exists but whose files do
    not. Note that get_model()'s version=None default means version 1, not the
    latest, so the version is always chosen explicitly.
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
