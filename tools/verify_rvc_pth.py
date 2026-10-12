"""Checks the RVC v2 ``.pth`` importer against the original PyTorch model code (run in CI).

``REF_SRC`` points at a folder holding the original ``rvc/lib/infer_pack`` sources. The ONNX graph
is compared with ``SynthesizerTrnMs768NSFsid.infer`` using identical inputs and with the random
sources fixed, for random weights in several layouts (written with ``torch.save`` and read back by
the app's own reader) and for the official pretrained v2 generator.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, os.environ["REF_SRC"])
from rvc.lib.infer_pack.models import SynthesizerTrnMs768NSFsid  # noqa: E402

from ttslts.vc import rvc_base, rvc_pth  # noqa: E402
from ttslts.vc.torch_zip import load_object  # noqa: E402

failures: list[str] = []


def check(ok: bool, message: str) -> None:
    print(("PASS " if ok else "FAIL ") + message, flush=True)
    if not ok:
        failures.append(message)


CONFIGS = {
    "40k": [1025, 32, 192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11], [[1, 3, 5]] * 3,
            [10, 10, 2, 2], 512, [16, 16, 4, 4], 109, 256, 40000],
    "48k": [1025, 32, 192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11], [[1, 3, 5]] * 3,
            [12, 10, 2, 2], 512, [24, 20, 4, 4], 109, 256, 48000],
    "32k, resblock 2": [513, 32, 192, 192, 768, 2, 6, 3, 0, "2", [3, 5, 7], [[1, 2]] * 3,
                        [10, 8, 2, 2], 256, [20, 16, 4, 4], 20, 256, 32000],
}  # fmt: skip


def compare(name: str, torch_model, state, cfg, frames: int) -> None:
    session = ort.InferenceSession(
        rvc_pth.build_model(state, cfg, noise=False).SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    rng = np.random.default_rng(frames)
    f0 = rng.uniform(90, 350, (1, frames)).astype(np.float32)
    f0[:, 5:11] = 0  # an unvoiced stretch
    rnd = rng.standard_normal((1, cfg.inter_channels, frames)).astype(np.float32)
    feed = {
        "phone": rng.standard_normal((1, frames, 768)).astype(np.float32),
        "phone_lengths": np.array([frames], dtype=np.int64),
        "pitch": rng.integers(1, 255, (1, frames)).astype(np.int64),
        "nsff0": f0,
        "sid": np.array([2], dtype=np.int64),
        "rnd": rnd,
    }
    mine = session.run(None, feed)[0]

    original = torch.randn_like
    torch.randn_like = lambda t, *a, **k: (
        torch.from_numpy(rnd) if tuple(t.shape) == tuple(rnd.shape) else torch.zeros_like(t)
    )
    try:
        with torch.no_grad():
            theirs = torch_model.infer(
                torch.from_numpy(feed["phone"]),
                torch.from_numpy(feed["phone_lengths"]),
                torch.from_numpy(feed["pitch"]),
                torch.from_numpy(feed["nsff0"]),
                torch.from_numpy(feed["sid"]),
            )[0].numpy()
    finally:
        torch.randn_like = original
    same_shape = mine.shape == theirs.shape
    err = float(np.abs(mine - theirs).max()) if same_shape else float("inf")
    corr = float(np.corrcoef(mine.ravel(), theirs.ravel())[0, 1]) if same_shape else 0.0
    peak = float(np.abs(theirs).max())
    check(
        same_shape and err < 5e-3 and corr > 0.9999,
        f"{name}, {frames} frames: shape {mine.shape}, peak {peak:.3f}, "
        f"max error {err:.2e}, correlation {corr:.6f}",
    )


work = Path(tempfile.mkdtemp(prefix="rvc-pth-"))

# 1. Random weights, written by torch.save (float32 and float16) and read by the app's reader.
for name, config in CONFIGS.items():
    torch.manual_seed(len(name))
    model = SynthesizerTrnMs768NSFsid(*config[:-1], config[-1], is_half=False).eval()
    with torch.no_grad():
        for p in model.parameters():
            if float(p.abs().max()) < 1e-6:  # layers RVC initialises to zero
                p.normal_(0, 0.02)
    for half in (False, True):
        weights = {k: (v.half() if half else v) for k, v in model.state_dict().items()}
        path = work / f"{len(name)}-{int(half)}.pth"
        torch.save(
            {"weight": weights, "config": config, "f0": 1, "version": "v2", "sr": name[:3]}, path
        )
        state, cfg = rvc_pth.validate(load_object(str(path)))
        if half:  # the reference sees the same rounded values
            model.load_state_dict({k: v.float() for k, v in weights.items()})
        check(cfg.sample_rate == config[-1], f"{name}: sample rate read as {cfg.sample_rate}")
        for frames in (4, 37, 120):
            compare(f"{name} ({'fp16' if half else 'fp32'} file)", model, state, cfg, frames)

# 2. The official pretrained v2 generator (real trained weights, original file layout).
pretrained = work / "f0G40k.pth"
rvc_base._fetch("pretrained_v2/f0G40k.pth", pretrained, None)
raw = torch.load(pretrained, map_location="cpu", weights_only=True)["model"]
config = CONFIGS["40k"]
model = SynthesizerTrnMs768NSFsid(*config[:-1], config[-1], is_half=False).eval()
check(not model.load_state_dict(raw, strict=False).unexpected_keys, "pretrained keys fit the model")
obj = load_object(str(pretrained))
state, cfg = rvc_pth.validate({"weight": obj["model"], "config": config, "f0": 1, "version": "v2"})
check(
    all(np.array_equal(state[k], v.numpy()) for k, v in raw.items() if k in state),
    "reader returns the pretrained tensors unchanged",
)
for frames in (8, 60, 150):
    compare("official f0G40k", model, state, cfg, frames)

sys.exit(1 if failures else 0)
