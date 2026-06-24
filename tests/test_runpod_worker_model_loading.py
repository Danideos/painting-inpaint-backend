from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

from painting_inpaint_backend.core.model_loading import (
    load_lora_if_available,
    normalize_model_id,
    resolve_hf_snapshot_path,
    resolve_lora_config,
    resolve_lora_path,
    resolve_model_load_target,
    set_lora_scale,
)

LORA_ENV_NAMES = (
    "LORA_REPO_ID",
    "LORA_FILENAME",
    "LORA_REVISION",
    "LORA_REQUIRED",
    "LORA_PATH",
    "LORA_CACHE_DIR",
    "LORA_ADAPTER_NAME",
    "HF_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
)


class _FakeLoraPipeline:
    def __init__(self):
        self.load_calls = []
        self.adapter_calls = []
        self.enable_lora_calls = 0

    def load_lora_weights(self, path, *, weight_name, adapter_name):
        self.load_calls.append(
            {
                "path": path,
                "weight_name": weight_name,
                "adapter_name": adapter_name,
            }
        )
        return None

    def enable_lora(self):
        self.enable_lora_calls += 1
        raise RuntimeError("enable_lora should not be called when adapter weights are supported")

    def set_adapters(self, adapter_names, *, adapter_weights):
        self.adapter_calls.append(
            {
                "adapter_names": list(adapter_names),
                "adapter_weights": list(adapter_weights),
            }
        )


class _InferenceModeRequiredLoraPipeline:
    def __init__(self):
        self.adapter_calls = []

    def set_adapters(self, adapter_names, *, adapter_weights):
        import torch

        if not torch.is_inference_mode_enabled():
            raise RuntimeError(
                "Setting requires_grad=True on inference tensor outside "
                "InferenceMode is not allowed."
            )
        self.adapter_calls.append(
            {
                "adapter_names": list(adapter_names),
                "adapter_weights": list(adapter_weights),
            }
        )


class _FakeTorch:
    def __init__(self):
        self._inference_mode_enabled = False

    def inference_mode(self):
        torch_module = self

        class _InferenceModeContext:
            def __enter__(self):
                self._previous = torch_module._inference_mode_enabled
                torch_module._inference_mode_enabled = True

            def __exit__(self, exc_type, exc, traceback):
                torch_module._inference_mode_enabled = self._previous

        return _InferenceModeContext()

    def is_inference_mode_enabled(self):
        return self._inference_mode_enabled


def _clear_lora_env(monkeypatch):
    for name in LORA_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def _write_snapshot(root: Path, model_cache_name: str, snapshot_id: str) -> Path:
    repo = root / model_cache_name
    snapshot = repo / "snapshots" / snapshot_id
    snapshot.mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(snapshot_id + "\n", encoding="utf-8")
    return snapshot


def test_normalize_model_id_preserves_flux_fill_canonical_case():
    assert (
        normalize_model_id("black-forest-labs/flux.1-fill-dev")
        == "black-forest-labs/FLUX.1-Fill-dev"
    )
    assert (
        normalize_model_id("  black-forest-labs/FLUX.1-Fill-dev  ")
        == "black-forest-labs/FLUX.1-Fill-dev"
    )


def test_normalize_model_id_preserves_unknown_model_ids():
    assert normalize_model_id("example/My-Other-Model") == "example/My-Other-Model"


def test_resolve_hf_snapshot_path_uses_refs_main(tmp_path):
    snapshot = _write_snapshot(
        tmp_path / "hub",
        "models--black-forest-labs--FLUX.1-Fill-dev",
        "abc123",
    )

    resolved = resolve_hf_snapshot_path(
        "black-forest-labs/FLUX.1-Fill-dev",
        cache_roots=[tmp_path / "hub"],
    )

    assert resolved == snapshot


def test_resolve_hf_snapshot_path_accepts_lowercase_flux_fill_alias(tmp_path):
    snapshot = _write_snapshot(
        tmp_path / "hub",
        "models--black-forest-labs--FLUX.1-Fill-dev",
        "abc123",
    )

    resolved = resolve_hf_snapshot_path(
        "black-forest-labs/flux.1-fill-dev",
        cache_roots=[tmp_path / "hub"],
    )

    assert resolved == snapshot


def test_resolve_hf_snapshot_path_accepts_lowercase_cache_folder(tmp_path):
    snapshot = _write_snapshot(
        tmp_path / "hub",
        "models--black-forest-labs--flux.1-fill-dev",
        "abc123",
    )

    resolved = resolve_hf_snapshot_path(
        "black-forest-labs/FLUX.1-Fill-dev",
        cache_roots=[tmp_path / "hub"],
    )

    assert resolved == snapshot


def test_resolve_hf_snapshot_path_falls_back_to_snapshot_folder(tmp_path):
    snapshots = (
        tmp_path
        / "hub"
        / "models--black-forest-labs--FLUX.1-Fill-dev"
        / "snapshots"
    )
    older = snapshots / "old"
    newer = snapshots / "new"
    older.mkdir(parents=True)
    newer.mkdir()

    resolved = resolve_hf_snapshot_path(
        "black-forest-labs/FLUX.1-Fill-dev",
        cache_roots=[tmp_path / "hub"],
    )

    assert resolved in {older, newer}


def test_model_path_uses_explicit_local_directory(monkeypatch, tmp_path):
    model_path = tmp_path / "flux-fill"
    model_path.mkdir()
    monkeypatch.setenv("MODEL_PATH", str(model_path))

    resolution = resolve_model_load_target()

    assert resolution.load_target == str(model_path)
    assert resolution.source == "explicit_local_path"
    assert resolution.local_files_only is True
    assert resolution.snapshot_path == str(model_path)


def test_model_path_rejects_missing_explicit_directory(monkeypatch, tmp_path):
    missing_path = tmp_path / "missing"
    monkeypatch.setenv("MODEL_PATH", str(missing_path))

    with pytest.raises(FileNotFoundError, match="MODEL_PATH"):
        resolve_model_load_target()


def test_lora_path_is_optional_for_smoke_tests(tmp_path):
    assert resolve_lora_path(lora_path=tmp_path / "missing.safetensors", required=False) is None


def test_lora_path_can_be_required_for_production(tmp_path):
    with pytest.raises(FileNotFoundError, match="LORA_REQUIRED=1"):
        resolve_lora_path(lora_path=tmp_path / "missing.safetensors", required=True)


def test_no_lora_configuration_preserves_optional_flow(monkeypatch, tmp_path):
    _clear_lora_env(monkeypatch)
    config = resolve_lora_config(
        lora_path=tmp_path / "missing.safetensors",
        required=False,
    )

    report = load_lora_if_available(_FakeLoraPipeline(), config=config)

    assert report["loaded"] is False
    assert report["source"] == "unavailable"
    assert report["error"] is None
    assert report["repo_id"] is None
    assert report["effective_scale"] is None


def test_remote_lora_download_load_and_scale_reuse(monkeypatch, tmp_path):
    _clear_lora_env(monkeypatch)
    monkeypatch.setenv("LORA_REPO_ID", "example/durer-lora")
    monkeypatch.setenv("LORA_FILENAME", "weights/pytorch_lora_weights.safetensors")
    monkeypatch.setenv("LORA_CACHE_DIR", str(tmp_path / "lora-cache"))
    monkeypatch.setenv("HF_TOKEN", "test-token")
    config = resolve_lora_config(required=False)
    pipe = _FakeLoraPipeline()
    download_calls = []

    def fake_download(**kwargs):
        download_calls.append(kwargs)
        path = tmp_path / "downloaded" / "weights" / "pytorch_lora_weights.safetensors"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"fake-lora")
        return str(path)

    report = load_lora_if_available(pipe, config=config, download_fn=fake_download)
    default_scale_again = set_lora_scale(
        pipe,
        adapter_name=report["adapter_name"],
        lora_scale=1.0,
    )
    half_scale = set_lora_scale(pipe, adapter_name=report["adapter_name"], lora_scale=0.5)
    zero_scale = set_lora_scale(pipe, adapter_name=report["adapter_name"], lora_scale=0)

    assert len(download_calls) == 1
    assert download_calls[0]["repo_id"] == "example/durer-lora"
    assert download_calls[0]["filename"] == "weights/pytorch_lora_weights.safetensors"
    assert download_calls[0]["revision"] == "main"
    assert download_calls[0]["token"] == "test-token"
    assert report["loaded"] is True
    assert report["repo_id"] == "example/durer-lora"
    assert report["filename"] == "weights/pytorch_lora_weights.safetensors"
    assert report["revision"] == "main"
    assert report["effective_scale"] == 1.0
    assert len(pipe.load_calls) == 1
    assert default_scale_again == {"mode": "cached_adapter_scale", "effective_scale": 1.0}
    assert half_scale == {"mode": "set_adapters", "effective_scale": 0.5}
    assert zero_scale == {"mode": "set_adapters", "effective_scale": 0.0}
    assert pipe.enable_lora_calls == 0
    assert [call["adapter_weights"] for call in pipe.adapter_calls] == [[1.0], [0.5], [0.0]]


def test_lora_scale_changes_run_inside_inference_mode_on_warm_pipeline(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())
    pipe = _InferenceModeRequiredLoraPipeline()

    first_scale = set_lora_scale(pipe, adapter_name="durer", lora_scale=1.0)
    same_scale = set_lora_scale(pipe, adapter_name="durer", lora_scale=1.0)
    changed_scale = set_lora_scale(pipe, adapter_name="durer", lora_scale=0.5)

    assert first_scale == {"mode": "set_adapters", "effective_scale": 1.0}
    assert same_scale == {"mode": "cached_adapter_scale", "effective_scale": 1.0}
    assert changed_scale == {"mode": "set_adapters", "effective_scale": 0.5}
    assert [call["adapter_weights"] for call in pipe.adapter_calls] == [[1.0], [0.5]]


def test_incomplete_remote_config_is_optional_or_required(monkeypatch, tmp_path):
    _clear_lora_env(monkeypatch)
    monkeypatch.setenv("LORA_REPO_ID", "example/durer-lora")
    monkeypatch.setenv("LORA_CACHE_DIR", str(tmp_path / "cache"))
    optional_config = resolve_lora_config(required=False)

    optional_report = load_lora_if_available(
        _FakeLoraPipeline(),
        config=optional_config,
    )

    assert optional_report["loaded"] is False
    assert "LORA_FILENAME" in optional_report["error"]
    with pytest.raises(RuntimeError, match="Required LoRA configuration is invalid"):
        load_lora_if_available(
            _FakeLoraPipeline(),
            config=replace(optional_config, required=True),
        )


def test_remote_download_failure_is_optional_or_required(monkeypatch, tmp_path):
    _clear_lora_env(monkeypatch)
    monkeypatch.setenv("LORA_REPO_ID", "invalid/repo")
    monkeypatch.setenv("LORA_FILENAME", "missing.safetensors")
    monkeypatch.setenv("LORA_CACHE_DIR", str(tmp_path / "cache"))
    optional_config = resolve_lora_config(required=False)

    def failing_download(**kwargs):
        raise RuntimeError("repository or file not found")

    optional_report = load_lora_if_available(
        _FakeLoraPipeline(),
        config=optional_config,
        download_fn=failing_download,
    )

    assert optional_report["loaded"] is False
    assert "repository or file not found" in optional_report["error"]
    with pytest.raises(RuntimeError, match="Required LoRA download/load failed"):
        load_lora_if_available(
            _FakeLoraPipeline(),
            config=replace(optional_config, required=True),
            download_fn=failing_download,
        )


def test_dockerignore_excludes_lora_safetensors_by_default():
    dockerignore = Path(".dockerignore").read_text(encoding="utf-8")

    assert "**/*.safetensors" in dockerignore
    assert "!assets/loras/*.safetensors" not in dockerignore
