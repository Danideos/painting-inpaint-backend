"""Run-folder orchestration for LoRA training launches."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
import time
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from painting_inpaint.experiments.manifest import (
    create_run_dir,
    git_commit,
    package_versions,
    write_json,
    write_manifest,
)
from painting_inpaint.paths import resolve_data_path, runs_root
from painting_inpaint.training.config import load_training_experiment_config
from painting_inpaint.training.diffusers_lora import build_flux2_klein_lora_command


def collect_training_environment() -> dict[str, Any]:
    """Collect lightweight training environment metadata without importing model libraries."""

    packages = [
        "accelerate",
        "bitsandbytes",
        "datasets",
        "diffusers",
        "peft",
        "safetensors",
        "torch",
        "torchao",
        "transformers",
    ]
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "git_commit": git_commit(),
        "package_versions": package_versions(packages),
    }


def _save_resolved_config(run_dir: Path, resolved: dict[str, Any], source_config: Path) -> Path:
    config_dir = run_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_config, config_dir / source_config.name)
    resolved_path = config_dir / "resolved_config.json"
    resolved_path.write_text(
        json.dumps(resolved, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return resolved_path


def _copy_dataset_manifest(prepared_dataset_dir: Path, run_dir: Path) -> Path | None:
    source = prepared_dataset_dir / "dataset_manifest.json"
    if not source.exists():
        return None
    dest = run_dir / "dataset_manifest.json"
    shutil.copy2(source, dest)
    return dest


def _validate_prepared_dataset(dataset: dict[str, Any]) -> Path:
    prepared_dir = resolve_data_path(dataset.get("prepared_dir"))
    if prepared_dir is None:
        raise ValueError("Dataset config must define prepared_dir.")
    prepared_dir = prepared_dir.resolve()
    metadata_filename = dataset.get("metadata_filename", "metadata.jsonl")
    metadata_path = prepared_dir / "train" / metadata_filename
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Prepared dataset metadata not found: {metadata_path}. "
            "Run scripts/prepare_lora_dataset.py first."
        )
    return prepared_dir


def _validate_runtime_dependencies(method: dict[str, Any]) -> None:
    training = method.get("training", {})
    missing = []
    if training.get("do_fp8_training") and find_spec("torchao") is None:
        missing.append("torchao")
    if training.get("use_8bit_adam") and find_spec("bitsandbytes") is None:
        missing.append("bitsandbytes")

    if missing:
        packages = ", ".join(missing)
        raise RuntimeError(
            f"Missing training runtime package(s): {packages}. "
            "Install the method-specific LoRA training dependencies before running."
        )


def _tail_text(path: Path, lines: int = 80) -> str:
    if not path.exists():
        return ""
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])


def _run_streaming_command(command: list[str], cwd: Path, log_path: Path) -> int:
    """Run a command while teeing combined output to the notebook and a log file."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        if process.stdout is not None:
            for line in process.stdout:
                print(line, end="", flush=True)
                log_file.write(line)
                log_file.flush()
        return process.wait()


def run_lora_training(
    config_path: str | Path,
    *,
    dry_run: bool = True,
    diffusers_dir: str | Path | None = None,
    run_root: str | Path | None = None,
) -> Path:
    """Create a training run folder and optionally launch the configured trainer."""

    source_config = Path(config_path).resolve()
    resolved = load_training_experiment_config(source_config)
    dataset = resolved["dataset"]
    method = resolved["method"]
    prepared_dataset_dir = _validate_prepared_dataset(dataset)

    method_name = method.get("name") or method.get("trainer") or resolved["name"]
    root = Path(run_root).expanduser() if run_root is not None else runs_root() / "training"
    run_dir = create_run_dir(method_name, root=root)
    resolved_config_path = _save_resolved_config(run_dir, resolved, source_config)

    trainer_output_dir = run_dir / "trainer_output"
    command = build_flux2_klein_lora_command(
        resolved,
        prepared_dataset_dir=prepared_dataset_dir,
        output_dir=trainer_output_dir,
        diffusers_dir=diffusers_dir,
    )
    dataset_manifest_path = _copy_dataset_manifest(command.prepared_dataset_dir, run_dir)

    write_json(
        run_dir / "trainer_command.json",
        {
            "command": command.command,
            "redacted_command": command.redacted_command,
            "redacted_display": command.redacted_display,
            "cwd": str(command.trainer_script.parent),
        },
    )

    manifest = {
        "name": resolved["name"],
        "dataset": dataset.get("name"),
        "method": method.get("name"),
        "trainer": method.get("trainer"),
        "model_id": method.get("model_id"),
        "dry_run": dry_run,
        "executed": False,
        "source_config": str(source_config),
        "resolved_config": str(resolved_config_path.relative_to(run_dir)),
        "prepared_dataset_dir": str(command.prepared_dataset_dir),
        "dataset_manifest": (
            str(dataset_manifest_path.relative_to(run_dir)) if dataset_manifest_path else None
        ),
        "trainer_script": str(command.trainer_script),
        "trainer_repo": method.get("trainer_repo"),
        "trainer_ref": method.get("trainer_ref"),
        "trainer_output_dir": str(trainer_output_dir.relative_to(run_dir)),
        "trainer_command": "trainer_command.json",
        "environment": collect_training_environment(),
    }

    if dry_run:
        write_manifest(run_dir, manifest)
        return run_dir

    _validate_runtime_dependencies(method)

    trainer_output_dir.mkdir(parents=True, exist_ok=True)
    training_log_path = run_dir / "training.log"
    started = time.perf_counter()
    manifest["executed"] = True
    manifest["training_log"] = str(training_log_path.relative_to(run_dir))
    returncode = _run_streaming_command(
        command.command,
        cwd=command.trainer_script.parent,
        log_path=training_log_path,
    )
    manifest["returncode"] = returncode

    if returncode != 0:
        manifest["runtime_seconds"] = time.perf_counter() - started
        manifest["log_tail"] = _tail_text(training_log_path)
        write_manifest(run_dir, manifest)
        raise RuntimeError(
            f"LoRA trainer failed with exit code {returncode}. "
            f"See {training_log_path} for the full log.\n\n"
            f"Last log lines:\n{manifest['log_tail']}"
        )

    manifest["runtime_seconds"] = time.perf_counter() - started
    manifest["outputs"] = {
        "trainer_output": str(trainer_output_dir.relative_to(run_dir)),
        "lora_weights": "trainer_output/pytorch_lora_weights.safetensors",
    }
    write_manifest(run_dir, manifest)
    return run_dir
