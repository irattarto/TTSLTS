"""Feature retrieval for RVC voices ("index"): pull the converted speech toward the real voice.

While training, the ContentVec features of the recordings are kept. When converting, every frame
of the incoming speech is mixed with its closest matches from those real frames, so the output
leans on what the target voice actually sounded like. This is what the ``.index`` file does in the
original RVC; here it is a plain numpy file next to the voice (``<name>.index.npy``) and the
search is a brute-force nearest-neighbour lookup, so no extra libraries are needed.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

MAX_VECTORS = 40000  # keeps the lookup fast; neighbouring frames of speech are very alike
NEIGHBOURS = 8
_BUILD_LOCK = threading.Lock()  # two loads of one voice at once build its index once


def index_path(model_path: str | Path) -> Path:
    """Index file of a voice file: ``voice.onnx`` or ``voice.pth`` -> ``voice.index.npy``."""
    path = Path(model_path)
    return path.with_name(f"{path.stem}.index.npy")


def build(feats_dir: Path, out: Path, max_vectors: int = MAX_VECTORS, seed: int = 0) -> int:
    """Collect the training features into one index file; returns how many frames it holds."""
    files = sorted(feats_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No training features in {feats_dir}")
    vectors = np.concatenate([np.load(f) for f in files]).astype(np.float32)
    if len(vectors) > max_vectors:
        keep = np.sort(np.random.default_rng(seed).choice(len(vectors), max_vectors, replace=False))
        vectors = vectors[keep]
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f"{out.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with open(tmp, "wb") as f:
        np.save(f, vectors.astype(np.float16))
    tmp.replace(out)
    return len(vectors)


def build_once(feats_dir: Path, out: Path) -> None:
    """``build`` unless the index is there already (also if another thread just made it)."""
    with _BUILD_LOCK:
        if not out.exists():
            build(feats_dir, out)


def _search_graph(vectors: np.ndarray, k: int):
    """ONNX graph: query frames ``(T, 768)`` -> squared distances and ids of the ``k`` closest."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    norms = (vectors.astype(np.float64) ** 2).sum(axis=1).astype(np.float32)[None, :]
    inits = [
        numpy_helper.from_array(np.ascontiguousarray(vectors.T), "vt"),
        numpy_helper.from_array(norms, "norms"),
        numpy_helper.from_array(np.asarray(-2.0, dtype=np.float32), "minus2"),
        numpy_helper.from_array(np.asarray([k], dtype=np.int64), "k"),
    ]
    nodes = [
        helper.make_node("ReduceSumSquare", ["q"], ["qq"], axes=[1], keepdims=1),
        helper.make_node("MatMul", ["q", "vt"], ["qv"]),
        helper.make_node("Mul", ["qv", "minus2"], ["cross"]),
        helper.make_node("Add", ["cross", "norms"], ["partial"]),
        helper.make_node("Add", ["partial", "qq"], ["dist"]),
        helper.make_node("TopK", ["dist", "k"], ["near", "ids"], axis=1, largest=0, sorted=0),
    ]
    graph = helper.make_graph(
        nodes,
        "retrieval",
        [helper.make_tensor_value_info("q", TensorProto.FLOAT, ["frames", vectors.shape[1]])],
        [
            helper.make_tensor_value_info("near", TensorProto.FLOAT, ["frames", k]),
            helper.make_tensor_value_info("ids", TensorProto.INT64, ["frames", k]),
        ],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


class RetrievalIndex:
    """The voice's real features; ``use_gpu`` searches them on the graphics card.

    The search reads the whole index (about 120 MB) for every chunk, which on the CPU takes
    25-150 ms depending on what else is running; on the GPU it is a few milliseconds. Meant for
    the real-time voice changer, which always searches the same number of frames (a DirectML
    session is only fast for the first size it sees).
    """

    def __init__(self, vectors: np.ndarray, use_gpu: bool = False):
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self._norms = (self.vectors**2).sum(axis=1)
        self._session = None
        if use_gpu:
            self._session = self._gpu_session()

    def _gpu_session(self):
        try:
            import onnxruntime as ort

            from ttslts.onnx_util import providers, session_options

            chosen = providers(True)
            if chosen == ["CPUExecutionProvider"]:
                return None
            from ttslts.onnx_util import GPU_LOCK, ResilientSession

            graph = _search_graph(self.vectors, min(NEIGHBOURS, len(self.vectors)))
            data, opts = graph.SerializeToString(), session_options(chosen)
            with GPU_LOCK:  # created while no other thread uses the card
                session = ort.InferenceSession(data, opts, providers=chosen)
            if session.get_providers()[0] == "CPUExecutionProvider":
                return None
            # like the other GPU sessions: one run at a time, made anew if a newer one is freed
            return ResilientSession("voice index search", opts, session, source=data)
        except Exception as exc:  # the numpy search always works
            log.warning("Searching the voice index on the CPU: %s", exc)
            return None

    @property
    def on_gpu(self) -> bool:
        return self._session is not None

    @classmethod
    def load(cls, path: str | Path, use_gpu: bool = False) -> RetrievalIndex:
        return cls(np.load(path), use_gpu)

    def __len__(self) -> int:
        return len(self.vectors)

    def _nearest(self, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Squared distances and ids of the ``k`` closest index frames of every frame."""
        if self._session is not None and k == min(NEIGHBOURS, len(self.vectors)):
            try:
                near, ids = self._session.run(None, {"q": q})
                return np.asarray(near, dtype=np.float32), np.asarray(ids)
            except Exception as exc:
                log.warning("Voice index search failed on the GPU, using the CPU: %s", exc)
                self._session = None
        dist = (q**2).sum(axis=1)[:, None] - 2.0 * (q @ self.vectors.T) + self._norms[None, :]
        ids = np.argpartition(dist, k - 1, axis=1)[:, :k]
        return np.take_along_axis(dist, ids, axis=1), ids

    def search(self, feats: np.ndarray, k: int = NEIGHBOURS) -> np.ndarray:
        """Replace each (T, 768) frame by the distance-weighted mean of its ``k`` closest frames."""
        q = np.ascontiguousarray(feats, dtype=np.float32)
        k = min(k, len(self.vectors))
        near, ids = self._nearest(q, k)
        weight = (1.0 / np.maximum(near, 1e-6)) ** 2  # the original RVC weighting
        weight /= weight.sum(axis=1, keepdims=True)
        return np.einsum("tk,tkd->td", weight, self.vectors[ids])
