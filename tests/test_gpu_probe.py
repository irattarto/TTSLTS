import json
import subprocess
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort
import pytest

from ttslts import gpu_probe, onnx_util


def fake_session(*inputs):
    return SimpleNamespace(
        get_inputs=lambda: [SimpleNamespace(name=n, type=t, shape=s) for n, t, s in inputs]
    )


def test_made_up_inputs_fit_the_model():
    session = fake_session(
        ("tokens", "tensor(int64)", [1, "n"]),
        ("style", "tensor(float)", [1, 256]),
        ("speed", "tensor(float)", [1]),
        ("mask", "tensor(bool)", [1, "n"]),
    )
    feed = gpu_probe.make_inputs(session, 24, np.random.default_rng(0))
    assert feed["tokens"].shape == (1, 24) and feed["tokens"].dtype == np.int64
    assert 0 < feed["tokens"].min() and feed["tokens"].max() < 8  # small ids fit any vocabulary
    assert feed["style"].shape == (1, 256) and feed["style"].dtype == np.float32
    assert 0.5 <= float(feed["speed"][0]) <= 1.5  # a speed must stay a sensible speed
    assert feed["mask"].dtype == bool


def test_compare_finds_garbage_and_accepts_noise():
    cpu = [np.random.default_rng(0).standard_normal((1, 100)).astype(np.float32)]
    assert gpu_probe.compare([cpu[0] + 1e-4], cpu) == ""
    assert "NaN" in gpu_probe.compare([np.full((1, 100), np.nan, np.float32)], cpu)
    assert "different" in gpu_probe.compare([cpu[0] * -40], cpu)
    assert "shape" in gpu_probe.compare([np.zeros((1, 50), np.float32)], cpu)


@pytest.fixture
def probing(monkeypatch):
    monkeypatch.delenv("TTSLTS_NO_GPU_PROBE", raising=False)
    monkeypatch.setattr(gpu_probe, "gpu_signature", lambda: "Test GPU 1.0")


def test_a_verdict_is_saved_and_not_asked_for_again(tmp_path, probing):
    model = tmp_path / "m.onnx"
    model.write_bytes(b"x")
    calls = []

    def runner(path):
        calls.append(path)
        return {"verdict": "incompatible", "reason": "fails"}

    assert gpu_probe.verdict(str(model), runner)["verdict"] == "incompatible"
    assert gpu_probe.verdict(str(model), runner)["reason"] == "fails"
    assert len(calls) == 1
    model.write_bytes(b"changed")  # another file: checked again
    gpu_probe.verdict(str(model), runner)
    assert len(calls) == 2


def test_a_new_graphics_card_or_driver_is_checked_again(tmp_path, probing, monkeypatch):
    model = tmp_path / "m.onnx"
    model.write_bytes(b"x")
    calls = []
    runner = lambda p: calls.append(p) or {"verdict": "ok", "reason": ""}  # noqa: E731
    gpu_probe.verdict(str(model), runner)
    monkeypatch.setattr(gpu_probe, "gpu_signature", lambda: "Test GPU 2.0")
    gpu_probe.verdict(str(model), runner)
    assert len(calls) == 2


def test_unknown_results_are_not_saved(tmp_path, probing):
    model = tmp_path / "m.onnx"
    model.write_bytes(b"x")
    calls = []
    runner = lambda p: calls.append(p) or {"verdict": "unknown", "reason": "timeout"}  # noqa: E731
    gpu_probe.verdict(str(model), runner)
    gpu_probe.verdict(str(model), runner)
    assert len(calls) == 2


def test_a_crash_of_the_check_is_a_verdict(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=-1073741819, stdout="warnings only\n", stderr=""
        ),
    )
    result = gpu_probe.probe_in_new_process("m.onnx")
    assert result["verdict"] == "incompatible" and "crashed" in result["reason"]


def test_the_checks_answer_is_read_from_its_output(monkeypatch):
    out = "noise\n" + json.dumps({"verdict": "ok", "reason": ""}) + "\n"
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout=out, stderr="")
    )
    assert gpu_probe.probe_in_new_process("m.onnx")["verdict"] == "ok"


def test_a_model_judged_incompatible_runs_on_the_cpu(tmp_path, monkeypatch):
    monkeypatch.setattr(
        onnx_util,
        "providers",
        lambda use_gpu: (["DmlExecutionProvider"] if use_gpu else []) + ["CPUExecutionProvider"],
    )
    verdicts = {"bad.onnx": "incompatible", "good.onnx": "ok"}
    monkeypatch.setattr(
        gpu_probe, "verdict", lambda path, **kw: {"verdict": verdicts[path], "reason": "test"}
    )

    class Loaded:
        def __init__(self, path, sess_options=None, providers=None):
            self.chosen = providers

        def get_providers(self):
            return self.chosen

    monkeypatch.setattr(ort, "InferenceSession", Loaded)
    assert onnx_util.make_session("bad.onnx", True).get_providers() == ["CPUExecutionProvider"]
    assert onnx_util.make_session("good.onnx", True).get_providers()[0] == "DmlExecutionProvider"


def test_a_check_that_has_to_run_is_announced(tmp_path, probing):
    model = tmp_path / "kokoro.onnx"
    model.write_bytes(b"x")
    said = []
    runner = lambda p: {"verdict": "ok", "reason": ""}  # noqa: E731
    gpu_probe.verdict(str(model), runner, announce=said.append)
    gpu_probe.verdict(str(model), runner, announce=said.append)  # saved now: nothing to announce
    assert len(said) == 1 and "kokoro.onnx" in said[0]
