"""Command construction for Diffusers LoRA trainers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from painting_inpaint.paths import PROJECT_ROOT, resolve_data_path, resolve_project_path
from painting_inpaint.training.config import deep_merge

DEFAULT_FLUX2_KLEIN_TRAINER_SCRIPT = (
    "examples/dreambooth/train_dreambooth_lora_flux2_klein.py"
)
SENSITIVE_ARGS = {"--hub_token", "--token", "--hf_token"}
BOOLEAN_TRAINING_FLAGS = {
    "allow_tf32": "--allow_tf32",
    "cache_latents": "--cache_latents",
    "center_crop": "--center_crop",
    "do_fp8_training": "--do_fp8_training",
    "gradient_checkpointing": "--gradient_checkpointing",
    "offload": "--offload",
    "random_flip": "--random_flip",
    "skip_final_inference": "--skip_final_inference",
    "upcast_before_saving": "--upcast_before_saving",
    "use_8bit_adam": "--use_8bit_adam",
}
VALUE_TRAINING_ARGS = {
    "adam_beta1": "--adam_beta1",
    "adam_beta2": "--adam_beta2",
    "adam_epsilon": "--adam_epsilon",
    "adam_weight_decay": "--adam_weight_decay",
    "aspect_ratio_buckets": "--aspect_ratio_buckets",
    "bnb_quantization_config_path": "--bnb_quantization_config_path",
    "checkpointing_steps": "--checkpointing_steps",
    "checkpoints_total_limit": "--checkpoints_total_limit",
    "dataloader_num_workers": "--dataloader_num_workers",
    "gradient_accumulation_steps": "--gradient_accumulation_steps",
    "guidance_scale": "--guidance_scale",
    "learning_rate": "--learning_rate",
    "lora_alpha": "--lora_alpha",
    "lora_dropout": "--lora_dropout",
    "lora_layers": "--lora_layers",
    "lr_num_cycles": "--lr_num_cycles",
    "lr_power": "--lr_power",
    "lr_scheduler": "--lr_scheduler",
    "lr_warmup_steps": "--lr_warmup_steps",
    "max_grad_norm": "--max_grad_norm",
    "max_sequence_length": "--max_sequence_length",
    "max_train_steps": "--max_train_steps",
    "mixed_precision": "--mixed_precision",
    "num_train_epochs": "--num_train_epochs",
    "num_validation_images": "--num_validation_images",
    "optimizer": "--optimizer",
    "rank": "--rank",
    "report_to": "--report_to",
    "resolution": "--resolution",
    "seed": "--seed",
    "text_encoder_lr": "--text_encoder_lr",
    "train_batch_size": "--train_batch_size",
    "validation_epochs": "--validation_epochs",
    "weighting_scheme": "--weighting_scheme",
}


@dataclass(frozen=True)
class TrainingCommand:
    """A shell-free command representation for one trainer launch."""

    command: list[str]
    redacted_command: list[str]
    trainer_script: Path
    output_dir: Path
    prepared_dataset_dir: Path

    @property
    def redacted_display(self) -> str:
        """Return a display string suitable for logs."""

        return " ".join(self.redacted_command)


def resolve_diffusers_trainer_script(
    method_config: dict[str, Any],
    *,
    diffusers_dir: str | Path | None = None,
) -> Path:
    """Resolve the Flux.2 Klein DreamBooth LoRA trainer script path."""

    script_value = method_config.get("trainer_script", DEFAULT_FLUX2_KLEIN_TRAINER_SCRIPT)
    script_path = resolve_project_path(script_value)
    if script_path is not None and script_path.is_absolute() and script_path.exists():
        return script_path

    relative_script = Path(script_value)
    candidates: list[Path] = []
    if diffusers_dir is not None:
        candidates.append(Path(diffusers_dir).expanduser())
    if method_config.get("diffusers_dir"):
        candidates.append(Path(method_config["diffusers_dir"]).expanduser())
    if os.environ.get("PAINTING_INPAINT_DIFFUSERS_DIR"):
        candidates.append(Path(os.environ["PAINTING_INPAINT_DIFFUSERS_DIR"]).expanduser())
    candidates.extend(
        [
            PROJECT_ROOT / ".tmp" / "diffusers",
            Path("/content/diffusers"),
        ]
    )

    for base in candidates:
        candidate = (base / relative_script).resolve()
        if candidate.exists():
            return candidate

    searched = ", ".join(str((base / relative_script).resolve()) for base in candidates)
    raise FileNotFoundError(
        f"Could not find Diffusers trainer script {relative_script}. Searched: {searched}"
    )


def redact_command(command: list[str]) -> list[str]:
    """Return a copy of ``command`` with secret-looking argument values redacted."""

    redacted: list[str] = []
    redact_next = False
    for token in command:
        if redact_next:
            redacted.append("[REDACTED]")
            redact_next = False
            continue

        if token in SENSITIVE_ARGS:
            redacted.append(token)
            redact_next = True
            continue

        if any(token.startswith(f"{arg}=") for arg in SENSITIVE_ARGS):
            name, _, _value = token.partition("=")
            redacted.append(f"{name}=[REDACTED]")
            continue

        lowered = token.lower()
        if "token=" in lowered or "hf_token" in lowered:
            name, sep, _value = token.partition("=")
            redacted.append(f"{name}{sep}[REDACTED]" if sep else "[REDACTED]")
            continue

        redacted.append(token)
    return redacted


def _append_value_arg(command: list[str], arg_name: str, value: Any) -> None:
    if value is None:
        return
    command.extend([arg_name, str(value)])


def _training_config(resolved: dict[str, Any]) -> dict[str, Any]:
    method = resolved.get("method", {})
    experiment = resolved.get("experiment", {})
    training = dict(method.get("training", {}))
    if isinstance(experiment.get("training"), dict):
        training = deep_merge(training, experiment["training"])
    return training


def build_flux2_klein_lora_command(
    resolved: dict[str, Any],
    *,
    prepared_dataset_dir: str | Path | None = None,
    output_dir: str | Path,
    diffusers_dir: str | Path | None = None,
) -> TrainingCommand:
    """Build an ``accelerate launch`` command for Diffusers Flux.2 Klein LoRA training."""

    dataset = resolved.get("dataset", {})
    method = resolved.get("method", {})
    if method.get("trainer") != "diffusers_flux2_klein_lora":
        raise ValueError(f"Unsupported LoRA trainer: {method.get('trainer')!r}")

    training = _training_config(resolved)
    if training.get("remote_text_encoder"):
        raise ValueError("remote_text_encoder is not supported for Flux.2 Klein training.")

    dataset_dir = (
        Path(prepared_dataset_dir).expanduser()
        if prepared_dataset_dir is not None
        else resolve_data_path(dataset.get("prepared_dir"))
    )
    if dataset_dir is None:
        raise ValueError("Dataset config must define prepared_dir.")
    dataset_dir = dataset_dir.resolve()
    trainer_script = resolve_diffusers_trainer_script(method, diffusers_dir=diffusers_dir)
    trainer_output_dir = Path(output_dir).expanduser().resolve()

    trigger_word = str(dataset.get("trigger_word", "")).strip()
    instance_prompt = training.get(
        "instance_prompt",
        trigger_word,
    )
    validation_prompt = training.get("validation_prompt")

    command = [
        "accelerate",
        "launch",
        str(trainer_script),
        "--pretrained_model_name_or_path",
        str(method.get("model_id")),
        "--dataset_name",
        str(dataset_dir),
        "--image_column",
        str(dataset.get("image_column", "image")),
        "--caption_column",
        str(dataset.get("caption_column", "text")),
        "--output_dir",
        str(trainer_output_dir),
        "--instance_prompt",
        str(instance_prompt),
    ]
    if validation_prompt:
        command.extend(["--validation_prompt", str(validation_prompt)])

    for key, arg_name in VALUE_TRAINING_ARGS.items():
        _append_value_arg(command, arg_name, training.get(key))

    for key, arg_name in BOOLEAN_TRAINING_FLAGS.items():
        if training.get(key):
            command.append(arg_name)

    return TrainingCommand(
        command=command,
        redacted_command=redact_command(command),
        trainer_script=trainer_script,
        output_dir=trainer_output_dir,
        prepared_dataset_dir=dataset_dir,
    )
