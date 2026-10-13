"""Feature retrieval for RVC voices ("index"): pull the converted speech toward the real voice.

While training, the ContentVec features of the recordings are kept. When converting, every frame
of the incoming speech is mixed with its closest matches from those real frames, so the output
leans on what the target voice actually sounded like. This is what the ``.index`` file does in the
original RVC; here it is a plain numpy file next to the voice (``<name>.index.npy``) and the
search is a brute-force nearest-neighbour lookup, so no extra libraries are needed.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

MAX_VECTORS = 40000  # keeps the lookup fast; neighbouring frames of speech are very alike
NEIGHBOURS = 8


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
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, vectors.astype(np.float16))
    tmp.replace(out)
    return len(vectors)


class RetrievalIndex:
    def __init__(self, vectors: np.ndarray):
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self._norms = (self.vectors**2).sum(axis=1)

    @classmethod
    def load(cls, path: str | Path) -> RetrievalIndex:
        return cls(np.load(path))

    def __len__(self) -> int:
        return len(self.vectors)

    def search(self, feats: np.ndarray, k: int = NEIGHBOURS) -> np.ndarray:
        """Replace each (T, 768) frame by the distance-weighted mean of its ``k`` closest frames."""
        q = np.asarray(feats, dtype=np.float32)
        k = min(k, len(self.vectors))
        dist = (q**2).sum(axis=1)[:, None] - 2.0 * (q @ self.vectors.T) + self._norms[None, :]
        nearest = np.argpartition(dist, k - 1, axis=1)[:, :k]
        near = np.maximum(np.take_along_axis(dist, nearest, axis=1), 1e-6)
        weight = (1.0 / near) ** 2  # the original RVC weighting
        weight /= weight.sum(axis=1, keepdims=True)
        return np.einsum("tk,tkd->td", weight, self.vectors[nearest])
