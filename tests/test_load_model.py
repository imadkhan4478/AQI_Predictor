"""Tests for model loading against a substituted registry.

The behaviour under test is version selection and blend-weight resolution, both
of which have already failed in production: the registry's get_model() defaults
to version 1 rather than the newest, and a registry write can leave a version
whose metadata exists but whose files do not.
"""

import json
import os

import pytest

from app import load_model


class StubVersion:
    """One registered model version. `files` is what its download() directory
    would contain; None means the download itself fails."""

    def __init__(self, version, files, metrics=None):
        self.version = version
        self._files = files
        self.training_metrics = metrics if metrics is not None else {"R2": 0.8}
        self.downloaded = False

    def download(self):
        self.downloaded = True
        if self._files is None:
            raise OSError(f"v{self.version}: download failed")
        return self._files


@pytest.fixture
def registry(monkeypatch):
    """Substitute the Hopsworks connection with an in-memory registry."""
    store = {}

    class Registry:
        def get_models(self, name):
            return store.get(name, [])

    class Project:
        def get_model_registry(self):
            return Registry()

    monkeypatch.setattr(load_model, "_connect", lambda: Project())
    monkeypatch.setattr(load_model.joblib, "load", lambda path: f"model-from:{path}")
    return store


def version_dir(tmp_path, version, blend_weight=None, with_model=True):
    directory = tmp_path / f"v{version}"
    directory.mkdir()
    if with_model:
        (directory / "model.pkl").write_bytes(b"stub")
    if blend_weight is not None:
        (directory / "blend.json").write_text(
            json.dumps({"blend_weight": blend_weight, "horizon_hours": 24}), encoding="utf-8"
        )
    return str(directory)


def test_picks_the_highest_version_not_the_first(registry, tmp_path):
    """get_model()'s default of version 1 silently pinned the dashboard to the
    very first model ever trained, ignoring every daily retrain."""
    registry["aqi_forecast_24h"] = [
        StubVersion(1, version_dir(tmp_path, 1, blend_weight=0.1)),
        StubVersion(3, version_dir(tmp_path, 3, blend_weight=0.5)),
        StubVersion(2, version_dir(tmp_path, 2, blend_weight=0.3)),
    ]

    _, version, _, blend_weight = load_model.load_latest_model("aqi_forecast_24h")
    assert version == 3
    assert blend_weight == 0.5


def test_falls_back_when_the_newest_version_is_half_written(registry, tmp_path):
    """A registry write that fails partway leaves a version with metadata and no
    usable files. Serving the previous intact model beats refusing to start."""
    broken = StubVersion(4, None)
    intact = StubVersion(3, version_dir(tmp_path, 3, blend_weight=0.5))
    registry["aqi_forecast_24h"] = [intact, broken]

    _, version, _, blend_weight = load_model.load_latest_model("aqi_forecast_24h")
    assert broken.downloaded  # it tried the newest first
    assert version == 3
    assert blend_weight == 0.5


def test_raises_when_no_version_is_usable(registry, tmp_path):
    registry["aqi_forecast_24h"] = [StubVersion(2, None), StubVersion(1, None)]
    with pytest.raises(LookupError, match="No usable version"):
        load_model.load_latest_model("aqi_forecast_24h")


def test_raises_when_nothing_is_registered(registry):
    with pytest.raises(LookupError, match="No versions"):
        load_model.load_latest_model("aqi_forecast_24h")


def test_blend_json_beside_the_artifact_wins_over_registry_metrics(registry, tmp_path):
    """blend.json travels with the model file and cannot drift from it; the
    registry metric is only a fallback for models registered before it existed."""
    registry["aqi_forecast_24h"] = [
        StubVersion(1, version_dir(tmp_path, 1, blend_weight=0.65), metrics={"blend_weight": 0.4})
    ]
    *_, blend_weight = load_model.load_latest_model("aqi_forecast_24h")
    assert blend_weight == 0.65


def test_falls_back_to_registry_metrics_when_blend_json_is_absent(registry, tmp_path):
    registry["aqi_forecast_24h"] = [
        StubVersion(1, version_dir(tmp_path, 1), metrics={"blend_weight": 0.4})
    ]
    *_, blend_weight = load_model.load_latest_model("aqi_forecast_24h")
    assert blend_weight == 0.4


def test_missing_weight_stays_none(registry, tmp_path):
    """Never 1.0 by default: that is the unshrunk model, which loses to
    persistence at every horizon."""
    registry["aqi_forecast_24h"] = [StubVersion(1, version_dir(tmp_path, 1), metrics={})]
    *_, blend_weight = load_model.load_latest_model("aqi_forecast_24h")
    assert blend_weight is None


def test_corrupt_blend_json_does_not_take_down_the_loader(registry, tmp_path):
    """A truncated upload is the failure mode this whole path exists for, so a
    malformed blend.json must not raise -- the horizon is withheld instead."""
    directory = tmp_path / "corrupt"
    directory.mkdir()
    (directory / "model.pkl").write_bytes(b"stub")
    (directory / "blend.json").write_text("{not json", encoding="utf-8")
    registry["aqi_forecast_24h"] = [StubVersion(1, str(directory), metrics={})]

    *_, blend_weight = load_model.load_latest_model("aqi_forecast_24h")
    assert blend_weight is None


def test_reads_blend_weight_from_a_directory_without_touching_the_registry(tmp_path):
    assert load_model._read_blend_weight(str(tmp_path), {}) is None
    assert os.path.isdir(str(tmp_path))
