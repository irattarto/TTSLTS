import onnxruntime as ort
import pytest

from ttslts import onnx_util


def test_gpu_failure_falls_back_to_the_cpu(monkeypatch):
    calls = []

    class FakeSession:
        def __init__(self, path, sess_options=None, providers=None):
            calls.append(providers)
            if providers != ["CPUExecutionProvider"]:
                raise RuntimeError("no usable GPU")

        def get_providers(self):
            return ["CPUExecutionProvider"]

    monkeypatch.setattr(
        onnx_util, "providers", lambda use_gpu: ["DmlExecutionProvider", "CPUExecutionProvider"]
    )
    monkeypatch.setattr(ort, "InferenceSession", FakeSession)
    session = onnx_util.make_session("model.onnx", True)
    assert calls == [["DmlExecutionProvider", "CPUExecutionProvider"], ["CPUExecutionProvider"]]
    assert session.get_providers() == ["CPUExecutionProvider"]


def test_cpu_only_errors_are_not_hidden(monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("bad model")

    monkeypatch.setattr(onnx_util, "providers", lambda use_gpu: ["CPUExecutionProvider"])
    monkeypatch.setattr(ort, "InferenceSession", broken)
    with pytest.raises(RuntimeError, match="bad model"):
        onnx_util.make_session("model.onnx", True)


class FlakySession:
    """Stands in for a GPU session whose runs fail."""

    def run(self, outputs, feed):
        raise RuntimeError("80070057 The parameter is incorrect")

    def get_providers(self):
        return ["DmlExecutionProvider", "CPUExecutionProvider"]


def test_failed_gpu_run_is_repeated_on_the_cpu(monkeypatch):
    class CpuSession:
        def __init__(self, path, sess_options=None, providers=None):
            assert providers == ["CPUExecutionProvider"]

        def run(self, outputs, feed):
            return ["cpu result"]

    messages = []
    onnx_util.set_fallback_notifier(messages.append)
    monkeypatch.setattr(ort, "InferenceSession", CpuSession)
    session = onnx_util.ResilientSession("model.onnx", None, FlakySession())
    try:
        assert session.run(None, {}) == ["cpu result"]
        assert session.on_cpu and len(messages) == 1
        assert session.run(None, {}) == ["cpu result"]  # stays on the CPU, no second warning
        assert len(messages) == 1
    finally:
        onnx_util.set_fallback_notifier(None)


def test_errors_on_the_cpu_are_not_hidden():
    class Broken:
        def run(self, outputs, feed):
            raise RuntimeError("bad input")

    session = onnx_util.ResilientSession("model.onnx", None, Broken())
    session.on_cpu = True
    with pytest.raises(RuntimeError, match="bad input"):
        session.run(None, {})


def test_a_model_that_failed_on_the_gpu_goes_straight_to_the_cpu_next_time(monkeypatch):
    class Session:
        def __init__(self, path, sess_options=None, providers=None):
            self.chosen = providers

        def run(self, outputs, feed):
            if self.chosen[0] != "CPUExecutionProvider":
                raise RuntimeError("unsupported on this GPU")
            return ["ok"]

        def get_providers(self):
            return self.chosen

    monkeypatch.setattr(ort, "InferenceSession", Session)
    monkeypatch.setattr(
        onnx_util,
        "providers",
        lambda use_gpu: (["DmlExecutionProvider"] if use_gpu else []) + ["CPUExecutionProvider"],
    )
    first = onnx_util.make_session("kokoro.onnx", True)
    assert first.run(None, {}) == ["ok"] and first.on_cpu  # fell back during the first run
    second = onnx_util.make_session("kokoro.onnx", True)
    assert second.get_providers() == ["CPUExecutionProvider"]  # remembered
    other = onnx_util.make_session("other.onnx", True)
    assert other.get_providers()[0] == "DmlExecutionProvider"  # other models still try the GPU
