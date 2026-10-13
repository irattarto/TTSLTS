"""RVC v2 inference with ONNX Runtime.

Three models are involved; the voice model is the user's own:

* the RVC v2 voice model, as ``.onnx`` (RVC WebUI / w-okada export) or the ``.pth`` checkpoint
  that voices are shared as (converted once by ``rvc_pth``),
* a ContentVec / HuBERT feature extractor in ONNX (768-dim, e.g. ``vec-768-layer-12.onnx``),
* optionally an RMVPE pitch extractor in ONNX; without it a built-in YIN estimator is used.

The two base models can be fetched and prepared by the app itself (``rvc_base``).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ttslts.audio.dsp import estimate_f0, f0_to_coarse

log = logging.getLogger(__name__)

SR_16K = 16000
HOP_16K = 160  # 10 ms pitch / feature frames at 16 kHz

# Input names used by the two common RVC ONNX exporters.
_FEATS = ("feats", "phone")
_LENGTHS = ("p_len", "phone_lengths")
_PITCH = ("pitch",)
_PITCHF = ("pitchf", "nsff0")
_SID = ("sid", "ds")
_NOISE = ("rnd",)


def _np_dtype(onnx_type: str):
    return np.float16 if onnx_type == "tensor(float16)" else np.float32


@dataclass
class RvcInfo:
    sample_rate: int
    uses_f0: bool
    input_names: list[str]


class ContentVec:
    """Wraps a ContentVec ONNX model: 16 kHz audio -> (1, T, 768) features at 20 ms."""

    def __init__(self, session):
        self.session = session
        inp = session.get_inputs()[0]
        self._name = inp.name
        self._rank = len(inp.shape) if inp.shape is not None else 3
        self._dtype = _np_dtype(inp.type)

    def __call__(self, audio16k: np.ndarray) -> np.ndarray:
        x = audio16k.astype(self._dtype)
        x = x.reshape((1, 1, -1) if self._rank == 3 else (1, -1))
        out = np.asarray(self.session.run(None, {self._name: x})[0], dtype=np.float32)
        if out.ndim == 2:
            out = out[None]
        if out.shape[-1] != 768 and out.shape[1] == 768:
            out = out.transpose(0, 2, 1)
        return out


class RmvpePitch:
    """Wraps a w-okada style RMVPE ONNX export (inputs: waveform, threshold)."""

    def __init__(self, session, threshold: float = 0.3):
        self.session = session
        self.threshold = threshold
        self._names = [i.name for i in session.get_inputs()]

    def __call__(self, audio16k: np.ndarray) -> np.ndarray:
        feed = {self._names[0]: audio16k.astype(np.float32)[None]}
        if len(self._names) > 1:
            feed[self._names[1]] = np.array([self.threshold], dtype=np.float32)
        f0 = np.asarray(self.session.run(None, feed)[0], dtype=np.float32).reshape(-1)
        n = len(audio16k) // HOP_16K + 1
        return np.pad(f0, (0, max(0, n - len(f0))))[:n]


def read_info(session, fallback_sr: int) -> RvcInfo:
    names = [i.name for i in session.get_inputs()]
    sr = fallback_sr
    try:
        meta = session.get_modelmeta().custom_metadata_map or {}
        if "metadata" in meta:  # w-okada exports embed a JSON blob
            sr = int(json.loads(meta["metadata"]).get("samplingRate", sr))
        elif "sr" in meta:
            sr = int(meta["sr"])
    except Exception as exc:  # metadata is optional
        log.debug("No usable RVC metadata: %s", exc)
    uses_f0 = any(n in names for n in _PITCH + _PITCHF)
    return RvcInfo(sample_rate=sr, uses_f0=uses_f0, input_names=names)


class RvcModel:
    """Converts 16 kHz mono speech into the target voice at the model's sample rate."""

    def __init__(
        self,
        session,
        contentvec: ContentVec,
        pitch_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        fallback_sr: int = 40000,
    ):
        self.session = session
        self.contentvec = contentvec
        self.pitch_fn = pitch_fn or estimate_f0
        self.index = None  # a RetrievalIndex of the voice's real features, if it has one
        self.index_rate = 0.0  # 0..1: how much of each frame comes from the index
        self.protect = 0.33  # 0..0.5: keeps unvoiced consonants clear when the index is used
        self.info = read_info(session, fallback_sr)
        self._inputs = {i.name: i for i in session.get_inputs()}
        known = set(_FEATS + _LENGTHS + _PITCH + _PITCHF + _SID + _NOISE)
        unknown = [n for n in self._inputs if n not in known]
        if unknown:
            raise ValueError(f"Unrecognised RVC ONNX inputs: {unknown}")
        if not any(n in self._inputs for n in _FEATS):
            raise ValueError("RVC ONNX model has no feature input (feats/phone)")

    @property
    def sample_rate(self) -> int:
        return self.info.sample_rate

    def build_feed(
        self, audio16k: np.ndarray, pitch_shift: float = 0.0, speaker_id: int = 0
    ) -> dict:
        feats = self.contentvec(audio16k)
        original = feats
        use_index = self.index is not None and self.index_rate > 0
        if use_index:
            retrieved = self.index.search(feats[0])[None]
            feats = self.index_rate * retrieved + (1.0 - self.index_rate) * feats
        feats = np.repeat(feats, 2, axis=1)  # 20 ms -> 10 ms frames
        original = np.repeat(original, 2, axis=1)
        n = feats.shape[1]
        f0 = None
        if self.info.uses_f0:
            f0 = np.asarray(self.pitch_fn(audio16k), dtype=np.float32)
            f0 = f0 * np.float32(2.0 ** (pitch_shift / 12.0))
            n = min(n, len(f0))
            f0 = f0[:n]
        feats = feats[:, :n]
        if use_index and f0 is not None and self.protect < 0.5:
            # Unvoiced frames (consonants, breath) keep the speech's own features, so the
            # index does not smear them into voiced sounds.
            keep = np.where(f0 > 0, 1.0, self.protect).astype(np.float32)[None, :, None]
            feats = feats * keep + original[:, :n] * (1.0 - keep)

        feed = {}
        for name, inp in self._inputs.items():
            if name in _FEATS:
                feed[name] = feats.astype(_np_dtype(inp.type))
            elif name in _LENGTHS:
                feed[name] = np.array([n], dtype=np.int64)
            elif name in _PITCH:
                feed[name] = f0_to_coarse(f0)[None]
            elif name in _PITCHF:
                feed[name] = f0[None].astype(_np_dtype(inp.type))
            elif name in _SID:
                feed[name] = np.array([speaker_id], dtype=np.int64)
            elif name in _NOISE:
                channels = inp.shape[1] if inp.shape and isinstance(inp.shape[1], int) else 192
                feed[name] = np.random.randn(1, channels, n).astype(_np_dtype(inp.type))
        return feed

    def convert(
        self, audio16k: np.ndarray, pitch_shift: float = 0.0, speaker_id: int = 0
    ) -> np.ndarray:
        audio16k = np.asarray(audio16k, dtype=np.float32)
        if len(audio16k) < HOP_16K * 4:
            return np.zeros(0, dtype=np.float32)
        feed = self.build_feed(audio16k, pitch_shift, speaker_id)
        out = self.session.run(None, feed)[0]
        return np.asarray(out, dtype=np.float32).reshape(-1)


def load_rvc(
    model_path: str,
    contentvec_path: str,
    rmvpe_path: str = "",
    use_gpu: bool = True,
    fallback_sr: int = 40000,
) -> RvcModel:
    from ttslts.onnx_util import make_session

    if not model_path:
        raise ValueError("No RVC voice model selected (Settings > Models)")
    if not contentvec_path:
        raise ValueError("No ContentVec model selected (Settings > Models)")
    if model_path.lower().endswith(".pth"):
        from ttslts.vc.rvc_pth import ensure_onnx

        model_path = ensure_onnx(model_path)  # one-time conversion, cached
    contentvec = ContentVec(make_session(contentvec_path, use_gpu))
    pitch_fn = None
    if rmvpe_path:
        from ttslts.vc.rmvpe_mel import RmvpeMelPitch, is_mel_model

        session = make_session(rmvpe_path, use_gpu)
        # The original RVC export takes a mel spectrogram; w-okada's takes the waveform.
        pitch_fn = RmvpeMelPitch(session) if is_mel_model(session) else RmvpePitch(session)
    return RvcModel(make_session(model_path, use_gpu), contentvec, pitch_fn, fallback_sr)
