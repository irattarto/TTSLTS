"""Checks the RVC base-model pipeline against the real original files (run in CI).

* downloads from the original repo through the app's own code (pinned revision, size checks),
* checks the torch-free checkpoint reader against ``torch.load``,
* checks the generated ContentVec ONNX against Hugging Face ``HubertModel`` (PyTorch),
* checks the official ``rmvpe.onnx`` pipeline recovers the pitch of synthetic voiced signals.
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from transformers import HubertModel

from ttslts.onnx_util import make_session
from ttslts.vc import rmvpe_mel, rvc_base
from ttslts.vc.rvc import ContentVec
from ttslts.vc.torch_zip import load_state_dict

work = Path(tempfile.mkdtemp(prefix="rvc-verify-"))
failures: list[str] = []


def check(ok: bool, message: str) -> None:
    print(("PASS " if ok else "FAIL ") + message, flush=True)
    if not ok:
        failures.append(message)


# 1. The app's own download + conversion path.
base = work / "base"
contentvec_path, rmvpe_path = rvc_base.download_and_convert(
    progress=lambda name, done, total: None, status=print, dest=base
)
check(contentvec_path.exists() and rmvpe_path.exists(), "download_and_convert produced both files")
license_text = (base / "LICENSE-RVC.txt").read_text(encoding="utf-8")
check(license_text.startswith("MIT License"), "license saved")

# 2. Reader versus torch.load, then the converted graph versus PyTorch.
ref_dir = work / "hubert_base"
rvc_base._fetch(rvc_base.WEIGHTS_FILE, ref_dir / "pytorch_model.bin", None)
rvc_base._fetch(rvc_base.CONFIG_FILE, ref_dir / "config.json", None)
mine = load_state_dict(str(ref_dir / "pytorch_model.bin"))
theirs = torch.load(ref_dir / "pytorch_model.bin", map_location="cpu", weights_only=True)
check(set(mine) == set(theirs), f"reader finds the same {len(theirs)} tensors")
same = all(
    mine[k].shape == tuple(v.shape) and np.array_equal(mine[k], v.numpy())
    for k, v in theirs.items()
)
check(same, "reader values are bit-identical to torch.load")

model = HubertModel.from_pretrained(ref_dir).float().eval()
session = make_session(str(contentvec_path), False)
rng = np.random.default_rng(0)
t = np.arange(16000 * 3) / 16000
signals = {
    "noise": rng.standard_normal(16000 * 3).astype(np.float32) * 0.1,
    "tone+noise": (0.3 * np.sin(2 * np.pi * 180 * t) + 0.05 * rng.standard_normal(len(t))).astype(
        np.float32
    ),
    "short": rng.standard_normal(8000 + 123).astype(np.float32) * 0.1,
}
for name, audio in signals.items():
    with torch.no_grad():
        expected = model(torch.from_numpy(audio)[None]).last_hidden_state.numpy()
    got = session.run(None, {session.get_inputs()[0].name: audio[None]})[0]
    same_shape = got.shape == expected.shape
    diff = float(np.abs(got - expected).max()) if same_shape else float("inf")
    print(f"  {name}: shape {got.shape} vs {expected.shape}, max abs diff {diff:.2e}", flush=True)
    check(same_shape and diff < 5e-3, f"ContentVec ONNX matches PyTorch on {name}")

wrapped = ContentVec(session)(signals["tone+noise"])
check(wrapped.ndim == 3 and wrapped.shape[2] == 768, f"ContentVec wrapper output {wrapped.shape}")

# 3. RMVPE on synthetic voiced signals.
pitch = rmvpe_mel.RmvpeMelPitch(make_session(str(rmvpe_path), False))
for target in (110.0, 220.0, 330.0):
    tt = np.arange(16000 * 2) / 16000
    voiced = sum(np.sin(2 * np.pi * target * k * tt) / k for k in range(1, 9))
    voiced = (0.3 * voiced / np.abs(voiced).max()).astype(np.float32)
    f0 = pitch(voiced)
    mid = f0[20:-20]
    ratio = float((mid > 0).mean())
    median = float(np.median(mid[mid > 0])) if (mid > 0).any() else 0.0
    print(f"  target {target} Hz: median {median:.1f} Hz, voiced {ratio:.0%}", flush=True)
    check(ratio > 0.8 and abs(median - target) / target < 0.03, f"RMVPE recovers {target:.0f} Hz")
silence = pitch(np.zeros(16000, dtype=np.float32))
check(float((silence > 0).mean()) < 0.1, "RMVPE reports silence as unvoiced")

shutil.rmtree(work, ignore_errors=True)
if failures:
    print(f"{len(failures)} check(s) failed", file=sys.stderr)
    sys.exit(1)
print("All checks passed")
