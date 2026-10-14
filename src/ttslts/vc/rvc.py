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
from pathlib import Path

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
_SKIP = ("skip",)  # frames to leave out of decoding (voices made by TTSLTS only)
_NSF_NOISE = ("nsf_noise",)  # the decoder's noise source (voices made by TTSLTS only)
NOISE_BANK_FRAMES = 3000  # 30 s of fixed noise, looked up by time (see ``NoiseBank``)
DECODER_MARGIN = 12  # frames (10 ms) of left context the decoder still needs


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


class NoiseBank:
    """Fixed random noise, looked up by the moment it is used for.

    The model needs random noise for every frame. The real-time voice changer converts each
    moment more than once (in overlapping windows) and crossfades the results; with new noise
    every time the two versions differ, so the joins wobble and an unvoiced "sz" loses about
    3 dB in the middle of every crossfade. Taking the noise of a moment from the same place in
    a fixed bank makes the overlapping versions agree.
    """

    def __init__(self, seed: int = 0, frames: int = NOISE_BANK_FRAMES):
        self.frames = frames
        self._rng = np.random.default_rng(seed)
        self._banks: dict[tuple[int, int], np.ndarray] = {}

    def take(self, first_frame: int, n: int, rows: int, per_frame: int = 1) -> np.ndarray:
        """``(rows, n * per_frame)`` noise for the frames ``first_frame .. first_frame + n``."""
        key = (rows, per_frame)
        if key not in self._banks:
            shape = (rows, self.frames * per_frame)
            self._banks[key] = self._rng.standard_normal(shape, dtype=np.float32)
        bank = self._banks[key]
        start = (first_frame % self.frames) * per_frame
        return bank[:, (start + np.arange(n * per_frame)) % bank.shape[1]]


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
        self.noise = NoiseBank()
        self._inputs = {i.name: i for i in session.get_inputs()}
        known = set(_FEATS + _LENGTHS + _PITCH + _PITCHF + _SID + _NOISE + _SKIP + _NSF_NOISE)
        unknown = [n for n in self._inputs if n not in known]
        if unknown:
            raise ValueError(f"Unrecognised RVC ONNX inputs: {unknown}")
        if not any(n in self._inputs for n in _FEATS):
            raise ValueError("RVC ONNX model has no feature input (feats/phone)")

    @property
    def sample_rate(self) -> int:
        return self.info.sample_rate

    @property
    def can_skip(self) -> bool:
        """Whether the model decodes just the newest part of what it is given.

        Only on the CPU: on a GPU (DirectML) leaving frames out makes every run slower
        (110-145 ms instead of 51 ms in a measurement), so the whole window is decoded there.
        """
        if not any(n in self._inputs for n in _SKIP):
            return False
        try:
            return self.session.get_providers()[0] == "CPUExecutionProvider"
        except (AttributeError, IndexError):
            return False

    def build_feed(
        self,
        audio16k: np.ndarray,
        pitch_shift: float = 0.0,
        speaker_id: int = 0,
        keep_16k: int | None = None,
        position: int = 0,
    ) -> dict:
        """Model inputs for ``audio16k``; with ``keep_16k`` only that many newest 16 kHz samples
        of output are needed, so the older frames are neither indexed nor decoded.
        ``position`` is where ``audio16k`` starts in a stream (16 kHz samples); it picks the
        noise, so overlapping windows of a stream get the same noise for the same moment."""
        feats = self.contentvec(audio16k)
        original = feats
        wanted = None  # frames (10 ms) from the end that must be produced
        if keep_16k is not None:
            wanted = -(-keep_16k // HOP_16K) + DECODER_MARGIN
        use_index = self.index is not None and self.index_rate > 0
        if use_index:
            first = 0 if wanted is None else max(0, feats.shape[1] - wanted // 2 - 1)
            mixed = feats.copy()
            retrieved = self.index.search(feats[0, first:])[None]
            mixed[:, first:] = (
                self.index_rate * retrieved + (1.0 - self.index_rate) * feats[:, first:]
            )
            feats = mixed
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

        first_frame = round(position / HOP_16K)
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
                rnd = self.noise.take(first_frame, n, channels)
                feed[name] = rnd[None].astype(_np_dtype(inp.type))
            elif name in _NSF_NOISE:
                upp = inp.shape[1]  # output samples per frame: (1, upp, frames)
                nsf = self.noise.take(first_frame, n, 1, upp).reshape(n, upp).T
                feed[name] = np.ascontiguousarray(nsf[None])
            elif name in _SKIP:
                cut = max(0, n - wanted) if wanted is not None and self.can_skip else 0
                feed[name] = np.array([cut], dtype=np.int64)
        return feed

    def convert(
        self,
        audio16k: np.ndarray,
        pitch_shift: float = 0.0,
        speaker_id: int = 0,
        keep_16k: int | None = None,
        position: int = 0,
    ) -> np.ndarray:
        """Convert speech. With ``keep_16k`` only the newest ``keep_16k / 16000`` seconds of the
        result are returned (and, if the model allows, only those are computed). ``position``:
        see ``build_feed``."""
        audio16k = np.asarray(audio16k, dtype=np.float32)
        if len(audio16k) < HOP_16K * 4:
            return np.zeros(0, dtype=np.float32)
        feed = self.build_feed(audio16k, pitch_shift, speaker_id, keep_16k, position)
        out = self.session.run(None, feed)[0]
        out = np.asarray(out, dtype=np.float32).reshape(-1)
        if keep_16k is not None:
            want = round(keep_16k * self.sample_rate / SR_16K)
            out = out[-want:] if len(out) >= want else np.pad(out, (want - len(out), 0))
        return out


def _has_input(model_path: str, name: str) -> bool:
    """Whether an ONNX file has an input of that name (read without starting a session)."""
    import onnx

    try:
        graph = onnx.load(model_path, load_external_data=False).graph
    except Exception:
        return True  # unreadable here: let the session report the real problem
    return any(i.name == name for i in graph.input)


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
    sibling = Path(model_path).with_suffix(".pth")
    if sibling.exists() and not _has_input(model_path, _NSF_NOISE[0]):
        # an ONNX file written by an older version: rebuild it from its .pth, which can decode
        # just the newest audio and takes its noise as an input (see ``NoiseBank``)
        from ttslts.vc.rvc_pth import UnsupportedVoice, ensure_onnx

        try:
            model_path = ensure_onnx(str(sibling))
        except (UnsupportedVoice, OSError, ValueError) as exc:
            log.warning("Keeping %s as it is: %s", model_path, exc)
    return RvcModel(make_session(model_path, use_gpu), contentvec, pitch_fn, fallback_sr)
