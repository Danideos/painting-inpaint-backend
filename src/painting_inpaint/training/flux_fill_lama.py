"""Run-file helpers for the forked Flux Fill LaMa-mask LoRA trainer."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from painting_inpaint.experiments.manifest import create_run_dir, package_versions, write_json
from painting_inpaint.paths import resolve_data_path, runs_root

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class FluxFillLamaRuntime:
    """Generated files and paths for one Flux Fill LaMa LoRA training run."""

    run_dir: Path
    config_dir: Path
    trainer_command_path: Path
    metadata_path: Path
    config_snapshot_path: Path
    debug_samples_dir: Path
    trainer_workdir: Path


def load_flux_fill_lama_config(path: str | Path) -> dict[str, Any]:
    """Load a Flux Fill LaMa LoRA training config from YAML."""

    config_path = Path(path)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"Training config must be a mapping: {config_path}")
    for section in ["trainer", "model", "dataset", "training", "debug", "validation"]:
        if section not in cfg:
            raise ValueError(f"Training config is missing required section: {section}")
    return cfg


def _bool_flag(command: list[str], enabled: bool, flag: str) -> None:
    if enabled:
        command.append(flag)


def _append_option(command: list[str], flag: str, value: Any) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def _shell_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _expanded_path(value: str | Path) -> Path:
    import os

    return Path(os.path.expandvars(str(value))).expanduser()


def _trainer_workdir(trainer_cfg: dict[str, Any]) -> Path:
    return _expanded_path(trainer_cfg["checkout_dir"]) / trainer_cfg["script_subdir"]


def _resolve_optional_data_path(value: str | None) -> Path | None:
    if value in (None, ""):
        return None
    return resolve_data_path(value)


def validate_flux_fill_lama_training_inputs(cfg: dict[str, Any]) -> Path:
    """Fail clearly if required training inputs are missing."""

    dataset_dir = resolve_data_path(cfg["dataset"]["source_dir"])
    if dataset_dir is None:
        raise ValueError("Flux Fill LaMa dataset source_dir is missing.")
    if not dataset_dir.exists():
        raise FileNotFoundError(
            "Flux Fill LaMa dataset directory does not exist: "
            f"{dataset_dir}. Copy the LaMa-Durer dataset there before submitting training."
        )
    if not dataset_dir.is_dir():
        raise NotADirectoryError(f"Flux Fill LaMa dataset path is not a directory: {dataset_dir}")

    image_count = sum(
        1
        for path in dataset_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if image_count == 0:
        raise FileNotFoundError(
            "Flux Fill LaMa dataset directory contains no supported image files: "
            f"{dataset_dir}. Expected one of: {', '.join(sorted(IMAGE_SUFFIXES))}"
        )

    return dataset_dir


def create_flux_fill_lama_run_files(
    cfg: dict[str, Any],
    *,
    run_dir: str | Path | None = None,
    run_root: str | Path | None = None,
) -> FluxFillLamaRuntime:
    """Create command and metadata files for the forked Flux Fill LaMa trainer."""

    method_name = cfg.get("name", "flux1_fill_dev_lama_durer_lama_lora")
    root = Path(run_root) if run_root is not None else runs_root() / "training"
    out_dir = Path(run_dir) if run_dir is not None else create_run_dir(method_name, root=root)
    out_dir.mkdir(parents=True, exist_ok=True)

    config_dir = out_dir / "flux_fill_lama"
    config_dir.mkdir(parents=True, exist_ok=True)
    trainer = cfg["trainer"]
    model = cfg["model"]
    dataset = cfg["dataset"]
    training = cfg["training"]
    debug = cfg["debug"]
    validation = cfg["validation"]

    dataset_dir = resolve_data_path(dataset["source_dir"])
    debug_samples_dir = out_dir / "debug_preprocessed_samples"
    trainer_output_dir = out_dir / "trainer_output"
    trainer_workdir = _trainer_workdir(trainer)
    script_name = trainer.get("script_name", "train_dreambooth_inpaint_lora_flux.py")

    command = [
        "accelerate",
        "launch",
        script_name,
        "--pretrained_model_name_or_path",
        model["pretrained_model_name_or_path"],
        "--instance_data_dir",
        str(dataset_dir),
        "--instance_prompt",
        dataset["instance_prompt"],
        "--output_dir",
        str(trainer_output_dir),
        "--resolution",
        str(dataset["resolution"]),
        "--mask_source",
        dataset.get("mask_source", "lama"),
        "--rank",
        str(training["rank"]),
        "--train_batch_size",
        str(training["train_batch_size"]),
        "--gradient_accumulation_steps",
        str(training["gradient_accumulation_steps"]),
        "--max_train_steps",
        str(training["max_train_steps"]),
        "--checkpointing_steps",
        str(training["checkpointing_steps"]),
        "--checkpoints_total_limit",
        str(training["checkpoints_total_limit"]),
        "--mixed_precision",
        training["mixed_precision"],
        "--learning_rate",
        str(training["learning_rate"]),
        "--seed",
        str(training["seed"]),
        "--report_to",
        training.get("report_to", "tensorboard"),
    ]
    _bool_flag(command, bool(training.get("gradient_checkpointing")), "--gradient_checkpointing")
    _append_option(command, "--resume_from_checkpoint", training.get("resume_from_checkpoint"))

    if debug.get("save_preprocessed_samples", True):
        command.extend(
            [
                "--debug_save_preprocessed_samples",
                str(debug_samples_dir),
                "--debug_num_preprocessed_samples",
                str(debug.get("num_preprocessed_samples", 16)),
            ]
        )

    validation_image = _resolve_optional_data_path(validation.get("image"))
    validation_mask = _resolve_optional_data_path(validation.get("mask"))
    validation_enabled = bool(
        validation_image
        and validation_mask
        and validation_image.exists()
        and validation_mask.exists()
    )
    if validation_enabled:
        _append_option(command, "--validation_prompt", validation.get("prompt"))
        _append_option(command, "--validation_image", validation_image)
        _append_option(command, "--validation_mask", validation_mask)
        _append_option(command, "--num_validation_images", validation.get("num_validation_images"))

    command_payload = {
        "command": command,
        "command_text": _shell_command(command),
        "cwd": str(trainer_workdir),
    }
    metadata = {
        "method": method_name,
        "trainer_repo": trainer["repo"],
        "trainer_commit": trainer["commit"],
        "trainer_branch_reference": trainer["branch_reference"],
        "trainer_checkout_dir": str(_expanded_path(trainer["checkout_dir"])),
        "trainer_workdir": str(trainer_workdir),
        "base_model": model["base_model"],
        "model_id": model["pretrained_model_name_or_path"],
        "dataset_name": dataset["name"],
        "dataset_path": str(dataset_dir),
        "mask_source": dataset.get("mask_source", "lama"),
        "resolution": dataset["resolution"],
        "instance_prompt": dataset["instance_prompt"],
        "training": training,
        "debug_samples_dir": str(debug_samples_dir),
        "trainer_output_dir": str(trainer_output_dir),
        "validation": {
            "enabled": validation_enabled,
            "prompt": validation.get("prompt"),
            "image": str(validation_image) if validation_image else None,
            "mask": str(validation_mask) if validation_mask else None,
            "skip_reason": None
            if validation_enabled
            else "validation image/mask not configured or not found",
        },
        "packages": package_versions(
            ["accelerate", "diffusers", "peft", "torch", "transformers"]
        ),
    }

    trainer_command_path = config_dir / "trainer_command.json"
    metadata_path = out_dir / "metadata.json"
    config_snapshot_path = config_dir / "config_snapshot.yaml"
    write_json(trainer_command_path, command_payload)
    write_json(metadata_path, metadata)
    config_snapshot_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    return FluxFillLamaRuntime(
        run_dir=out_dir,
        config_dir=config_dir,
        trainer_command_path=trainer_command_path,
        metadata_path=metadata_path,
        config_snapshot_path=config_snapshot_path,
        debug_samples_dir=debug_samples_dir,
        trainer_workdir=trainer_workdir,
    )


def clone_flux_fill_lama_trainer(
    cfg: dict[str, Any],
    *,
    force_reclone: bool = False,
) -> Path:
    """Clone or update the pinned forked Flux Fill LaMa trainer."""

    trainer = cfg["trainer"]
    checkout_dir = _expanded_path(trainer["checkout_dir"])

    if force_reclone and checkout_dir.exists():
        import shutil

        shutil.rmtree(checkout_dir)

    if not checkout_dir.exists():
        checkout_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", trainer["repo"], str(checkout_dir)], check=True)

    subprocess.run(["git", "fetch", "--all", "--tags"], cwd=checkout_dir, check=True)
    subprocess.run(["git", "checkout", "--detach", trainer["commit"]], cwd=checkout_dir, check=True)
    actual_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout_dir,
        text=True,
    ).strip()
    if actual_commit != trainer["commit"]:
        raise RuntimeError(f"Trainer commit mismatch: {actual_commit} != {trainer['commit']}")

    return checkout_dir


def verify_flux_fill_lama_trainer(runtime: FluxFillLamaRuntime, cfg: dict[str, Any]) -> None:
    """Verify that the pinned trainer has the patched LaMa-mask options."""

    script_name = cfg["trainer"].get("script_name", "train_dreambooth_inpaint_lora_flux.py")
    trainer_script = runtime.trainer_workdir / script_name
    trainer_text = trainer_script.read_text(encoding="utf-8")
    for required in [
        "--mask_source",
        "--debug_save_preprocessed_samples",
        "--debug_num_preprocessed_samples",
    ]:
        if required not in trainer_text:
            raise RuntimeError(f"Patched trainer option missing: {required}")


def install_flux_fill_lama_trainer_dependencies(
    trainer_checkout_dir: str | Path,
    *,
    constraints_path: str | Path | None = None,
) -> None:
    """Install the forked trainer and its training requirements into the active env."""

    checkout_dir = _expanded_path(trainer_checkout_dir)
    requirements_path = (
        checkout_dir
        / "examples"
        / "research_projects"
        / "dreambooth_inpaint"
        / "requirements_flux.txt"
    )
    if not requirements_path.exists():
        raise FileNotFoundError(f"Missing Flux Fill trainer requirements: {requirements_path}")

    base = [sys.executable, "-m", "pip", "install", "--no-cache-dir"]
    constraints = []
    if constraints_path is not None:
        constraints_path = Path(constraints_path)
        if constraints_path.exists():
            constraints = ["-c", str(constraints_path)]

    subprocess.run([*base, "--no-deps", "-e", str(checkout_dir)], check=True)
    subprocess.run([*base, *constraints, "-r", str(requirements_path)], check=True)
    subprocess.run(
        [
            *base,
            *constraints,
            "--upgrade-strategy",
            "only-if-needed",
            "transformers>=4.41.2,<4.56",
        ],
        check=True,
    )
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=False)


def _run_streaming(command: list[str], *, cwd: str | Path, log_path: str | Path) -> int:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return process.wait()


def _command_with_max_train_steps(command: list[str], max_train_steps: int) -> list[str]:
    adjusted = list(command)
    if "--max_train_steps" not in adjusted:
        raise ValueError("Trainer command is missing --max_train_steps")
    adjusted[adjusted.index("--max_train_steps") + 1] = str(max_train_steps)
    return adjusted


def run_flux_fill_lama_training(
    config_path: str | Path,
    *,
    run: bool = False,
    run_dir: str | Path | None = None,
    run_root: str | Path | None = None,
    resume_from_checkpoint: str | None = None,
    trainer_checkout_dir: str | Path | None = None,
    force_reclone_trainer: bool = False,
    install_trainer: bool = True,
    constraints_path: str | Path | None = None,
    debug_preprocess: bool = True,
) -> FluxFillLamaRuntime:
    """Create and optionally execute a Flux Fill LaMa LoRA training run."""

    cfg = load_flux_fill_lama_config(config_path)
    if resume_from_checkpoint is not None:
        cfg["training"]["resume_from_checkpoint"] = resume_from_checkpoint
    if trainer_checkout_dir is not None:
        cfg["trainer"]["checkout_dir"] = str(trainer_checkout_dir)

    runtime = create_flux_fill_lama_run_files(cfg, run_dir=run_dir, run_root=run_root)
    if not run:
        return runtime

    validate_flux_fill_lama_training_inputs(cfg)

    checkout_dir = clone_flux_fill_lama_trainer(cfg, force_reclone=force_reclone_trainer)
    verify_flux_fill_lama_trainer(runtime, cfg)
    if install_trainer:
        install_flux_fill_lama_trainer_dependencies(
            checkout_dir,
            constraints_path=constraints_path,
        )

    command_payload = json.loads(runtime.trainer_command_path.read_text(encoding="utf-8"))
    command = command_payload["command"]
    cwd = command_payload["cwd"]

    if debug_preprocess:
        debug_command = _command_with_max_train_steps(command, 1)
        debug_log = runtime.run_dir / "debug_preprocess.log"
        returncode = _run_streaming(debug_command, cwd=cwd, log_path=debug_log)
        if returncode != 0:
            raise RuntimeError(f"Debug preprocessing command failed. See {debug_log}")

    training_log = runtime.run_dir / "training.log"
    returncode = _run_streaming(command, cwd=cwd, log_path=training_log)
    if returncode != 0:
        raise RuntimeError(f"Training failed. See {training_log}")

    return runtime
