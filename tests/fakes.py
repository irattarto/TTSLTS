"""Minimal stand-ins for onnxruntime sessions."""

from types import SimpleNamespace

import numpy as np


class FakeInput(SimpleNamespace):
    pass


class FakeSession:
    def __init__(self, inputs, run_fn, meta=None):
        self._inputs = [FakeInput(name=n, type=t, shape=s) for n, t, s in inputs]
        self._run = run_fn
        self._meta = meta or {}
        self.feeds = []

    def get_inputs(self):
        return self._inputs

    def get_modelmeta(self):
        return SimpleNamespace(custom_metadata_map=self._meta)

    def run(self, outputs, feed):
        self.feeds.append(feed)
        return [self._run(feed)]


def fake_contentvec():
    # 16 kHz audio -> one 768-dim frame per 320 samples, like ContentVec.
    return FakeSession(
        [("source", "tensor(float)", [1, 1, "n"])],
        lambda feed: np.zeros((1, feed["source"].shape[-1] // 320, 768), dtype=np.float32),
    )


def rvc_session(names, sr_hop=400, meta=None, dtype="tensor(float)"):
    types = {
        "feats": dtype,
        "phone": dtype,
        "pitchf": dtype,
        "nsff0": dtype,
        "rnd": dtype,
    }
    inputs = [(n, types.get(n, "tensor(int64)"), None) for n in names]
    feat_name = "feats" if "feats" in names else "phone"
    return FakeSession(
        inputs,
        lambda feed: np.ones((1, 1, feed[feat_name].shape[1] * sr_hop), dtype=np.float32),
        meta,
    )
