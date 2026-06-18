"""Config loading helpers for training experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping from disk."""

    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {cfg_path}")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return ``base`` recursively merged with ``override``."""

    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_relative_config(config_dir: Path, relative_path: str | Path | None) -> dict[str, Any]:
    if not relative_path:
        return {}
    return load_yaml((config_dir / relative_path).resolve())


def load_training_experiment_config(path: str | Path) -> dict[str, Any]:
    """Load an experiment config plus referenced dataset and method configs."""

    experiment_path = Path(path).resolve()
    experiment = load_yaml(experiment_path)
    config_dir = experiment_path.parent

    dataset_cfg = _load_relative_config(config_dir, experiment.get("dataset_config"))
    method_cfg = _load_relative_config(config_dir, experiment.get("method_config"))

    if "dataset" in experiment:
        dataset_cfg = deep_merge(dataset_cfg, experiment["dataset"])
    if "method" in experiment:
        method_cfg = deep_merge(method_cfg, experiment["method"])
    if "training" in experiment:
        method_cfg["training"] = deep_merge(method_cfg.get("training", {}), experiment["training"])

    return {
        "name": experiment.get("name", experiment_path.stem),
        "dataset": dataset_cfg,
        "method": method_cfg,
        "experiment": experiment,
        "source_config": str(experiment_path),
    }


def load_lora_dataset_config(path: str | Path) -> dict[str, Any]:
    """Load either a direct dataset config or the dataset part of an experiment config."""

    cfg_path = Path(path).resolve()
    cfg = load_yaml(cfg_path)
    if "dataset_config" not in cfg:
        return cfg

    resolved = load_training_experiment_config(cfg_path)
    return resolved["dataset"]
