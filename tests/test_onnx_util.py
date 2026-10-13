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
