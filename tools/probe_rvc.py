"""One-off probe: license text and layout of the original RVC base model files."""

import json

from huggingface_hub import hf_hub_download

REPO = "lj1995/VoiceConversionWebUI"


def show(name: str, limit: int = 6000) -> None:
    path = hf_hub_download(REPO, name)
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    print(f"===== {name} ({len(text)} chars) =====")
    print(text[:limit])


show("使用需遵守的协议-LICENSE.txt")
show("README.md")
show("hubert_base/config.json")
show("hubert_base/preprocessor_config.json")

import torch  # noqa: E402

path = hf_hub_download(REPO, "hubert_base/pytorch_model.bin")
sd = torch.load(path, map_location="cpu", weights_only=True)
keys = list(sd.keys())
print("hubert_base/pytorch_model.bin tensors:", len(keys))
for k in keys[:8] + keys[-6:]:
    print("  ", k, tuple(sd[k].shape))
print(json.dumps({"has_final_proj": any(k.startswith("final_proj") for k in keys)}))

import onnxruntime as ort  # noqa: E402

sess = ort.InferenceSession(hf_hub_download(REPO, "rmvpe.onnx"), providers=["CPUExecutionProvider"])
meta = sess.get_modelmeta()
print("rmvpe.onnx producer:", meta.producer_name, meta.producer_version)
print("rmvpe.onnx metadata:", meta.custom_metadata_map)
