from __future__ import annotations

import pytest

from runpod_worker.partial_noise import (
    FluxFillPartialNoisePipeline,
    normalize_partial_noise,
    partial_noise_start_index,
)


class _FakeSeries(list):
    def detach(self):
        return self

    def clone(self):
        return _FakeSeries(self)

    def __getitem__(self, item):
        value = super().__getitem__(item)
        return _FakeSeries(value) if isinstance(item, slice) else value


class _FakeScheduler:
    order = 1

    def __init__(self, length: int):
        self.timesteps = _FakeSeries(range(length))
        self.sigmas = _FakeSeries([1.0 - index / length for index in range(length)])
        self.begin_index = None

    def set_begin_index(self, value: int) -> None:
        self.begin_index = value


def _pipeline_with_schedule(length: int, partial_noise: float | None):
    pipe = FluxFillPartialNoisePipeline.__new__(FluxFillPartialNoisePipeline)
    pipe.scheduler = _FakeScheduler(length)
    pipe.partial_noise = normalize_partial_noise(partial_noise)
    pipe.partial_noise_fraction = pipe.partial_noise
    pipe.print_timestep_schedule = False
    return pipe


def test_normalize_partial_noise_defaults_to_full_schedule():
    assert normalize_partial_noise(None) == 1.0


@pytest.mark.parametrize(
    ("partial_noise", "expected_index"),
    [
        (1.0, 0),
        (0.5, 5),
        (0.3, 7),
    ],
)
def test_partial_noise_start_index_matches_notebook_formula(partial_noise, expected_index):
    assert partial_noise_start_index(10, partial_noise) == expected_index


@pytest.mark.parametrize("value", [0, -0.1, 1.1, "bad"])
def test_partial_noise_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="partial_noise"):
        normalize_partial_noise(value)


def test_pipeline_get_timesteps_slices_full_schedule():
    pipe = _pipeline_with_schedule(10, 0.5)

    timesteps, effective_steps = pipe.get_timesteps(
        num_inference_steps=10,
        strength=1.0,
        device="cpu",
    )

    assert list(timesteps) == [5, 6, 7, 8, 9]
    assert effective_steps == 5
    assert pipe.scheduler.begin_index == 5
    assert pipe._last_schedule_debug["partial_noise"] == 0.5
    assert pipe._last_schedule_debug["diffusers_strength_argument_ignored"] == 1.0
    assert pipe._last_schedule_debug["t_start"] == 5
    assert pipe._last_schedule_debug["effective_num_steps"] == 5
    assert pipe._last_schedule_debug["schedule"][5]["status"] == "START"


def test_pipeline_default_partial_noise_uses_all_steps():
    pipe = _pipeline_with_schedule(4, None)

    timesteps, effective_steps = pipe.get_timesteps(
        num_inference_steps=4,
        strength=1.0,
        device="cpu",
    )

    assert list(timesteps) == [0, 1, 2, 3]
    assert effective_steps == 4
    assert pipe.scheduler.begin_index == 0
