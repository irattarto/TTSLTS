"""Reads tensors from a PyTorch ``.pt`` / ``.bin`` zip checkpoint without PyTorch.

A checkpoint is a zip holding ``data.pkl`` (a pickle describing the tensors) and one raw file
per storage. Unpickling arbitrary files is a code-execution risk, so only a fixed allow-list of
globals is accepted and every other global is rejected.
"""

from __future__ import annotations

import pickle
import zipfile
from collections import OrderedDict

import numpy as np

_STORAGE_DTYPES = {
    "HalfStorage": np.float16,
    "FloatStorage": np.float32,
    "DoubleStorage": np.float64,
    "LongStorage": np.int64,
    "IntStorage": np.int32,
    "ShortStorage": np.int16,
    "CharStorage": np.int8,
    "ByteStorage": np.uint8,
    "BoolStorage": np.bool_,
}


class _Storage:
    def __init__(self, dtype, data: bytes):
        self.array = np.frombuffer(data, dtype=dtype)


class _StorageType:
    def __init__(self, name: str):
        self.dtype = _STORAGE_DTYPES[name]


def _rebuild_tensor(storage, offset, size, stride, *_ignored):
    size, stride = tuple(size), tuple(stride)
    if not size:  # 0-d tensor
        return np.array(storage.array[offset])
    view = np.lib.stride_tricks.as_strided(
        storage.array[offset:],
        shape=size,
        strides=tuple(s * storage.array.itemsize for s in stride),
    )
    return np.array(view)  # own, contiguous copy


def _rebuild_parameter(data, *_ignored):
    return data


class _Unpickler(pickle.Unpickler):
    def __init__(self, file, archive: zipfile.ZipFile, prefix: str):
        super().__init__(file)
        self._archive = archive
        self._prefix = prefix

    def find_class(self, module, name):  # noqa: D102
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor
        if module == "torch._utils" and name == "_rebuild_parameter":
            return _rebuild_parameter
        if module == "torch" and name in _STORAGE_DTYPES:
            return _StorageType(name)
        raise pickle.UnpicklingError(f"Refusing to load {module}.{name} from a checkpoint")

    def persistent_load(self, pid):  # noqa: D102
        kind, storage_type, key, _location, _numel = pid
        if kind != "storage":
            raise pickle.UnpicklingError(f"Unsupported persistent id {kind!r}")
        data = self._archive.read(f"{self._prefix}data/{key}")
        return _Storage(storage_type.dtype, data)


def load_object(path: str):
    """Return the unpickled top-level object of a PyTorch zip checkpoint (tensors as arrays)."""
    with zipfile.ZipFile(path) as archive:
        pkl = next((n for n in archive.namelist() if n.endswith("data.pkl")), None)
        if pkl is None:
            raise ValueError("Not a PyTorch zip checkpoint (no data.pkl)")
        prefix = pkl[: -len("data.pkl")]
        with archive.open(pkl) as f:
            return _Unpickler(f, archive, prefix).load()


def load_state_dict(path: str) -> dict[str, np.ndarray]:
    """Return ``{name: ndarray}`` for a PyTorch zip checkpoint (state dict layout)."""
    obj = load_object(path)
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        obj = obj["model"]
    if not isinstance(obj, dict):
        raise ValueError("Checkpoint is not a state dict")
    return dict(obj)
