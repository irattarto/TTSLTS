"""One-off probe: what do the original RVC project's base model files look like?"""

import json
import urllib.request

from huggingface_hub import hf_hub_download

REPO = "lj1995/VoiceConversionWebUI"

with urllib.request.urlopen(f"https://huggingface.co/api/models/{REPO}?blobs=true") as r:
    info = json.load(r)
print("license:", info.get("cardData", {}).get("license"), "| sha:", info.get("sha"))
for s in sorted(info["siblings"], key=lambda s: s["rfilename"]):
    name = s["rfilename"]
    if "/" not in name or name.startswith(("uvr5", "pretrained")) is False:
        print(f"  {name}  {s.get('size')}")

import onnxruntime as ort  # noqa: E402

for name in ("rmvpe.onnx", "vec-768-layer-12.onnx", "hubert_base.onnx"):
    try:
        path = hf_hub_download(REPO, name)
    except Exception as e:  # noqa: BLE001
        print(f"{name}: not available ({type(e).__name__})")
        continue
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    print(f"{name}: opset-free probe")
    for i in sess.get_inputs():
        print("   in ", i.name, i.shape, i.type)
    for o in sess.get_outputs():
        print("   out", o.name, o.shape, o.type)

import torch  # noqa: E402

for name in ("hubert_base.pt", "rmvpe.pt"):
    try:
        path = hf_hub_download(REPO, name)
    except Exception as e:  # noqa: BLE001
        print(f"{name}: not available ({type(e).__name__})")
        continue
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:  # noqa: BLE001
        print(f"{name}: torch.load failed: {type(e).__name__}: {str(e)[:300]}")
        continue
    print(f"{name}: type={type(ckpt).__name__}")
    if isinstance(ckpt, dict):
        print("   top-level keys:", list(ckpt.keys())[:12])
        sd = ckpt.get("model", ckpt)
        if isinstance(sd, dict):
            keys = list(sd.keys())
            print("   tensors:", len(keys))
            for k in keys[:6] + keys[-4:]:
                v = sd[k]
                print("   ", k, tuple(getattr(v, "shape", ())))
