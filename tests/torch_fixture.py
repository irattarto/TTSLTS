"""Writes PyTorch-format zip checkpoints without PyTorch (fake ``torch`` modules for pickling)."""

from __future__ import annotations

import pickle
import sys
import types
import zipfile
from collections import OrderedDict

import numpy as np

_STORAGE_NAMES = {np.dtype(np.float16): "HalfStorage", np.dtype(np.float32): "FloatStorage"}
_STORAGE_NAMES[np.dtype(np.int64)] = "LongStorage"


def write_checkpoint(path, tensors: dict[str, np.ndarray], wrap: str | None = None) -> None:
    torch_mod = types.ModuleType("torch")
    utils_mod = types.ModuleType("torch._utils")
    storage_classes = {}
    for name in set(_STORAGE_NAMES.values()):
        cls = type(name, (), {})
        cls.__module__ = "torch"
        setattr(torch_mod, name, cls)
        storage_classes[name] = cls

    def _rebuild_tensor_v2(*args):  # pragma: no cover - only referenced by name
        raise AssertionError

    _rebuild_tensor_v2.__module__ = "torch._utils"
    _rebuild_tensor_v2.__qualname__ = "_rebuild_tensor_v2"
    utils_mod._rebuild_tensor_v2 = _rebuild_tensor_v2

    class Storage:
        def __init__(self, key, array):
            self.key, self.array = key, array

    class Tensor:
        def __init__(self, storage, shape):
            self.storage, self.shape = storage, shape

        def __reduce__(self):
            strides = tuple(_strides(self.shape))  # torch strides count elements, not bytes
            return (
                utils_mod._rebuild_tensor_v2,
                (self.storage, 0, self.shape, strides, False, OrderedDict()),
            )

    def _strides(shape):
        out, acc = [], 1
        for dim in reversed(shape):
            out.append(acc)
            acc *= dim
        return [s for s in reversed(out)]

    storages: list[Storage] = []
    state = OrderedDict()
    for name, array in tensors.items():
        array = np.ascontiguousarray(array)
        storage = Storage(str(len(storages)), array)
        storages.append(storage)
        shape = tuple(array.shape)
        state[name] = Tensor(storage, shape)
    payload = OrderedDict(model=state) if wrap else state

    class Pickler(pickle.Pickler):
        def persistent_id(self, obj):
            if isinstance(obj, Storage):
                cls = storage_classes[_STORAGE_NAMES[obj.array.dtype]]
                return ("storage", cls, obj.key, "cpu", int(obj.array.size))
            return None

    saved = {k: sys.modules.get(k) for k in ("torch", "torch._utils")}
    sys.modules["torch"], sys.modules["torch._utils"] = torch_mod, utils_mod
    torch_mod._utils = utils_mod
    try:
        import io

        buf = io.BytesIO()
        Pickler(buf, protocol=2).dump(payload)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        z.writestr("archive/data.pkl", buf.getvalue())
        for storage in storages:
            z.writestr(f"archive/data/{storage.key}", storage.array.tobytes())
        z.writestr("archive/version", b"3\n")
