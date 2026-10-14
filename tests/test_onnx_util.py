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


class GpuSession:
    """Stands in for a GPU session; ``error`` is what its runs fail with (None: they work)."""

    def __init__(self, error=None):
        self.error = error
        self.runs = 0

    def run(self, outputs, feed):
        self.runs += 1
        if self.error:
            raise RuntimeError(self.error)
        return ["gpu result"]

    def get_providers(self):
        return ["DmlExecutionProvider", "CPUExecutionProvider"]


@pytest.fixture
def cpu_sessions(monkeypatch):
    class CpuSession:
        def __init__(self, path, sess_options=None, providers=None):
            assert providers == ["CPUExecutionProvider"]

        def run(self, outputs, feed):
            return ["cpu result"]

    monkeypatch.setattr(ort, "InferenceSession", CpuSession)
    messages = []
    onnx_util.set_fallback_notifier(messages.append)
    yield messages
    onnx_util.set_fallback_notifier(None)


def test_error_classes():
    assert onnx_util.is_transient(RuntimeError("E_OUTOFMEMORY 8007000E"))
    assert onnx_util.is_transient(RuntimeError("DXGI_ERROR_DEVICE_REMOVED 887A0005"))
    assert not onnx_util.is_transient(RuntimeError("80070057 The parameter is incorrect"))


def test_a_busy_gpu_is_only_avoided_for_a_moment_and_never_given_up_on(cpu_sessions):
    gpu = GpuSession("out of memory")
    session = onnx_util.ResilientSession("model.onnx", None, gpu)
    assert session.run(None, {}) == ["cpu result"] and session.on_cpu
    assert session.run(None, {}) == ["cpu result"] and gpu.runs == 1  # not retried during backoff
    assert len(cpu_sessions) == 1 and "busy" in cpu_sessions[0]
    gpu.error = None
    session._cpu_until = 0.0  # the backoff is over
    assert session.run(None, {}) == ["gpu result"] and not session.on_cpu  # back on the GPU
    assert onnx_util._skip_list() == []  # nothing remembered for next time


def test_the_same_unsupported_failure_twice_sends_a_model_to_the_cpu_for_good(
    cpu_sessions, monkeypatch
):
    session = onnx_util.ResilientSession("kokoro.onnx", None, GpuSession("80070057"))
    session.run(None, {})
    assert onnx_util._skip_list() == []  # once could have been bad luck
    session._cpu_until = 0.0
    session.run(None, {})
    assert onnx_util._skip_list() == ["kokoro.onnx"]

    monkeypatch.setattr(
        onnx_util,
        "providers",
        lambda use_gpu: (["DmlExecutionProvider"] if use_gpu else []) + ["CPUExecutionProvider"],
    )

    class Loaded:
        def __init__(self, path, sess_options=None, providers=None):
            self.chosen = providers

        def get_providers(self):
            return self.chosen

    monkeypatch.setattr(ort, "InferenceSession", Loaded)
    assert onnx_util.make_session("kokoro.onnx", True).get_providers() == ["CPUExecutionProvider"]
    assert onnx_util.make_session("other.onnx", True).get_providers()[0] == "DmlExecutionProvider"

    onnx_util.clear_gpu_failures()  # "Try the GPU again for every model"
    assert onnx_util.make_session("kokoro.onnx", True).get_providers()[0] == "DmlExecutionProvider"


def test_errors_on_the_cpu_are_not_hidden(monkeypatch):
    class Broken:
        def __init__(self, path, sess_options=None, providers=None):
            pass

        def run(self, outputs, feed):
            raise RuntimeError("bad input")

    monkeypatch.setattr(ort, "InferenceSession", Broken)
    session = onnx_util.ResilientSession("model.onnx", None, GpuSession("out of memory"))
    with pytest.raises(RuntimeError, match="bad input"):
        session.run(None, {})


def test_a_failure_the_cpu_shares_is_not_blamed_on_the_gpu(monkeypatch):
    class Broken:
        def __init__(self, path, sess_options=None, providers=None):
            pass

        def run(self, outputs, feed):
            raise RuntimeError("invalid input shape")

    monkeypatch.setattr(ort, "InferenceSession", Broken)
    session = onnx_util.ResilientSession("model.onnx", None, GpuSession("invalid input shape"))
    for _ in range(3):
        with pytest.raises(RuntimeError, match="invalid input"):
            session.run(None, {})
    assert onnx_util._skip_list() == [] and not session.on_cpu  # the input is at fault, not the GPU


def test_gpu_runs_never_overlap():
    """Two threads on DirectML at once crashed the program: runs take turns."""
    import threading
    import time

    active, overlaps = [0], []

    class Slow(GpuSession):
        def run(self, outputs, feed):
            active[0] += 1
            overlaps.append(active[0] > 1)
            time.sleep(0.01)
            active[0] -= 1
            return ["gpu result"]

    a = onnx_util.ResilientSession("a.onnx", None, Slow())
    b = onnx_util.ResilientSession("b.onnx", None, Slow())  # a different model counts too
    threads = [
        threading.Thread(target=lambda s=s: [s.run(None, {}) for _ in range(10)]) for s in (a, b)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(overlaps) == 20 and not any(overlaps)
