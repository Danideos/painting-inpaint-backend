from __future__ import annotations

import base64
import json
from contextlib import nullcontext
from io import BytesIO
from types import SimpleNamespace

from PIL import Image

from painting_inpaint_backend.core.image_io import image_to_base64
from painting_inpaint_backend.core.inference import InferenceService
from painting_inpaint_backend.core.model_loading import LoadedPipeline
from painting_inpaint_backend.core.progress import ProgressReporter, sanitize_for_progress
from painting_inpaint_backend.runpod.service import run_job_input_streaming


class _FakeTorch:
    @staticmethod
    def inference_mode():
        return nullcontext()


class _StepCallbackPipeline:
    def __init__(self, *, steps: int = 3):
        self.steps = steps
        self._last_schedule_debug = {"effective_num_steps": steps}
        self._last_latent_init_debug = {}

    def __call__(
        self,
        *,
        image,
        callback_on_step_end=None,
        **_kwargs,
    ):
        self._last_schedule_debug = {"effective_num_steps": self.steps}
        if callback_on_step_end is not None:
            for index in range(self.steps):
                callback_on_step_end(self, index, 1000 - index, {})
        return SimpleNamespace(images=[Image.new("RGB", image.size, "blue")])


def _worker_with_fake_pipeline() -> InferenceService:
    worker = InferenceService()
    worker._loaded = LoadedPipeline(
        pipe=_StepCallbackPipeline(),
        torch=_FakeTorch(),
        torch_dtype="fake",
        model={"model_id": "fake"},
        lora={"loaded": False, "adapter_name": "durer"},
        supports_negative_prompt=False,
        timings={},
    )
    return worker


def _payload() -> dict[str, object]:
    image = Image.new("RGB", (4, 4), "red")
    mask = Image.new("L", (4, 4), 255)
    return {
        "prompt": "DURER_RESTO",
        "image_base64": image_to_base64(image),
        "mask_base64": image_to_base64(mask),
        "num_inference_steps": 3,
        "partial_noise": 1.0,
    }


def _decoded_size(encoded: str) -> tuple[int, int]:
    raw = base64.b64decode(encoded)
    with Image.open(BytesIO(raw)) as image:
        return image.size


def test_progress_reporter_suppresses_base64_and_secret_fields():
    reporter = ProgressReporter(keep_history=True)

    reporter.emit(
        "input_decode_done",
        stage="input",
        message="done",
        metadata={
            "image_base64": "secret-image",
            "mask_base64": "secret-mask",
            "HF_TOKEN": "secret-token",
            "safe": {"width": 4},
        },
    )

    serialized = json.dumps(reporter.history)
    assert "image_base64" not in serialized
    assert "mask_base64" not in serialized
    assert "secret-image" not in serialized
    assert "secret-token" not in serialized
    assert reporter.history[0]["metadata"] == {"safe": {"width": 4}}
    assert sanitize_for_progress({"payload": {"prompt": "x"}}) == {}


def test_non_streaming_worker_output_keeps_old_shape():
    response = _worker_with_fake_pipeline().run(_payload())

    assert response["image_base64"]
    assert response["output_format"] == "png"
    assert response["width"] == 4
    assert response["height"] == 4
    assert response["gpu_memory"] == {
        "pre_inference_cuda_available": False,
        "inference_cuda_available": False,
    }
    assert "type" not in response
    assert "event" not in response
    assert "run_report" not in response
    assert _decoded_size(response["image_base64"]) == (4, 4)


def test_streaming_worker_yields_progress_and_final_event():
    non_streaming = _worker_with_fake_pipeline().run(_payload())

    events = list(
        run_job_input_streaming(
            _payload(),
            job_id="job-1",
            service=_worker_with_fake_pipeline(),
        )
    )

    assert events[0]["event"] == "job_received"
    assert any(event["event"] == "inference_step" for event in events)
    assert events[-1]["type"] == "final"
    assert events[-1]["event"] == "job_done"
    assert set(events[-1]["output"].keys()) == set(non_streaming.keys())


def test_inference_step_callback_event_shape():
    reporter = ProgressReporter(job_id="job-1", keep_history=True)

    _worker_with_fake_pipeline().run(_payload(), reporter=reporter)

    step_events = [event for event in reporter.history if event["event"] == "inference_step"]
    assert len(step_events) == 3
    assert step_events[0]["type"] == "progress"
    assert step_events[0]["stage"] == "inference"
    assert step_events[0]["progress"] == {
        "current": 1,
        "total": 3,
        "fraction": 1 / 3,
    }
    assert step_events[0]["metadata"]["timestep"] == 1000
