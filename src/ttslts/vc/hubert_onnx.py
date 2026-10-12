"""Builds a ContentVec / HuBERT-base ONNX model from the original PyTorch weights.

The RVC project ships ContentVec as a PyTorch checkpoint only. To avoid needing PyTorch, the
network is written out directly as an ONNX graph from the weights (Hugging Face ``Hubert``
layout, post-layer-norm variant, ``feat_extract_norm="group"``). The graph maps 16 kHz audio
``(1, samples)`` to the layer-12 features ``(1, frames, 768)`` that RVC v2 uses, one frame per
20 ms.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np

OPSET = 17


@dataclass
class HubertConfig:
    conv_dim: tuple[int, ...] = (512,) * 7
    conv_kernel: tuple[int, ...] = (10, 3, 3, 3, 3, 2, 2)
    conv_stride: tuple[int, ...] = (5, 2, 2, 2, 2, 2, 2)
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    intermediate_size: int = 3072
    pos_kernel: int = 128
    pos_groups: int = 16
    eps: float = 1e-5

    @classmethod
    def from_json(cls, path: str) -> HubertConfig:
        with open(path, encoding="utf-8") as f:
            c = json.load(f)
        if c.get("do_stable_layer_norm") or c.get("feat_extract_norm", "group") != "group":
            raise ValueError("Unsupported HuBERT variant (expected the base model)")
        if c.get("conv_bias"):
            raise ValueError("Unsupported HuBERT variant (convolution bias)")
        return cls(
            conv_dim=tuple(c["conv_dim"]),
            conv_kernel=tuple(c["conv_kernel"]),
            conv_stride=tuple(c["conv_stride"]),
            hidden_size=c["hidden_size"],
            num_layers=c["num_hidden_layers"],
            num_heads=c["num_attention_heads"],
            intermediate_size=c["intermediate_size"],
            pos_kernel=c["num_conv_pos_embeddings"],
            pos_groups=c["num_conv_pos_embedding_groups"],
            eps=c.get("layer_norm_eps", 1e-5),
        )


def _weight_norm(state: dict[str, np.ndarray], prefix: str) -> np.ndarray:
    """Positional-conv weight: ``g * v / ||v||`` with the norm over every dim except dim 2."""
    para = f"{prefix}.parametrizations.weight"
    for g_name, v_name in (
        (f"{prefix}.weight_g", f"{prefix}.weight_v"),
        (f"{para}.original0", f"{para}.original1"),
    ):
        if g_name in state and v_name in state:
            g, v = state[g_name].astype(np.float32), state[v_name].astype(np.float32)
            norm = np.sqrt((v**2).sum(axis=(0, 1), keepdims=True))
            return g * v / norm
    raise KeyError(f"No weight-norm parameters under {prefix}")


def build_model(state: dict[str, np.ndarray], cfg: HubertConfig | None = None):
    """Return an ``onnx.ModelProto`` for the given HuBERT state dict."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    cfg = cfg or HubertConfig()
    nodes: list = []
    inits: list = []
    counter = [0]

    def f32(name: str) -> np.ndarray:
        return np.ascontiguousarray(state[name].astype(np.float32))

    def init(name: str, array: np.ndarray) -> str:
        inits.append(numpy_helper.from_array(array, name))
        return name

    def node(op: str, inputs, **attrs) -> str:
        counter[0] += 1
        out = f"t{counter[0]}"
        nodes.append(helper.make_node(op, list(inputs), [out], name=f"{op}_{counter[0]}", **attrs))
        return out

    def const_i64(values) -> str:
        counter[0] += 1
        return init(f"c{counter[0]}", np.asarray(values, dtype=np.int64))

    def const_f32(value: float) -> str:
        counter[0] += 1
        return init(f"c{counter[0]}", np.asarray(value, dtype=np.float32))

    def gelu(x: str) -> str:
        scaled = node("Mul", [x, const_f32(1.0 / math.sqrt(2.0))])
        erf = node("Erf", [scaled])
        plus = node("Add", [erf, const_f32(1.0)])
        return node("Mul", [node("Mul", [x, plus]), const_f32(0.5)])

    def linear(x: str, prefix: str) -> str:
        w = init(f"{prefix}.w", f32(f"{prefix}.weight").T.copy())
        b = init(f"{prefix}.b", f32(f"{prefix}.bias"))
        return node("Add", [node("MatMul", [x, w]), b])

    def layer_norm(x: str, prefix: str) -> str:
        gamma = init(f"{prefix}.g", f32(f"{prefix}.weight"))
        beta = init(f"{prefix}.beta", f32(f"{prefix}.bias"))
        return node("LayerNormalization", [x, gamma, beta], axis=-1, epsilon=cfg.eps)

    # --- convolutional feature extractor: (1, samples) -> (1, frames, 512) -------------------
    x = node("Unsqueeze", ["source", const_i64([1])])
    for i, (kernel, stride) in enumerate(zip(cfg.conv_kernel, cfg.conv_stride, strict=True)):
        w = init(f"conv{i}.w", f32(f"feature_extractor.conv_layers.{i}.conv.weight"))
        x = node("Conv", [x, w], kernel_shape=[kernel], strides=[stride])
        if i == 0:  # GroupNorm with one channel per group == per-channel instance norm
            p = "feature_extractor.conv_layers.0.layer_norm"
            x = node(
                "InstanceNormalization",
                [x, init("gn.scale", f32(f"{p}.weight")), init("gn.bias", f32(f"{p}.bias"))],
                epsilon=cfg.eps,
            )
        x = gelu(x)
    x = node("Transpose", [x], perm=[0, 2, 1])

    # --- feature projection ----------------------------------------------------------------------
    x = layer_norm(x, "feature_projection.layer_norm")
    x = linear(x, "feature_projection.projection")

    # --- positional convolution embedding -----------------------------------------------------
    pos_w = init("pos.w", _weight_norm(state, "encoder.pos_conv_embed.conv"))
    pos_b = init("pos.b", f32("encoder.pos_conv_embed.conv.bias"))
    pad = cfg.pos_kernel // 2
    p = node("Transpose", [x], perm=[0, 2, 1])
    p = node(
        "Conv",
        [p, pos_w, pos_b],
        kernel_shape=[cfg.pos_kernel],
        pads=[pad, pad],
        group=cfg.pos_groups,
    )
    if cfg.pos_kernel % 2 == 0:  # drop the extra frame, as the same-pad layer does
        p = node("Slice", [p, const_i64([0]), const_i64([-1]), const_i64([2])])
    p = gelu(p)
    p = node("Transpose", [p], perm=[0, 2, 1])
    x = layer_norm(node("Add", [x, p]), "encoder.layer_norm")

    # --- transformer layers (post-layer-norm) ----------------------------------------------
    head_dim = cfg.hidden_size // cfg.num_heads
    split = const_i64([0, 0, cfg.num_heads, head_dim])
    merge = const_i64([0, 0, cfg.hidden_size])
    scale = const_f32(head_dim**-0.5)

    def heads(x: str, name: str, perm: list[int]) -> str:
        projected = linear(x, name)
        return node("Transpose", [node("Reshape", [projected, split])], perm=perm)

    for i in range(cfg.num_layers):
        pre = f"encoder.layers.{i}"
        q = heads(x, f"{pre}.attention.q_proj", [0, 2, 1, 3])
        k = heads(x, f"{pre}.attention.k_proj", [0, 2, 3, 1])
        v = heads(x, f"{pre}.attention.v_proj", [0, 2, 1, 3])
        scores = node("Mul", [node("MatMul", [q, k]), scale])
        context = node("MatMul", [node("Softmax", [scores], axis=-1), v])
        context = node("Reshape", [node("Transpose", [context], perm=[0, 2, 1, 3]), merge])
        attn = linear(context, f"{pre}.attention.out_proj")
        x = layer_norm(node("Add", [x, attn]), f"{pre}.layer_norm")
        hidden = gelu(linear(x, f"{pre}.feed_forward.intermediate_dense"))
        ff = linear(hidden, f"{pre}.feed_forward.output_dense")
        x = layer_norm(node("Add", [x, ff]), f"{pre}.final_layer_norm")

    nodes.append(helper.make_node("Identity", [x], ["embed"], name="output"))
    graph = helper.make_graph(
        nodes,
        "contentvec",
        [helper.make_tensor_value_info("source", TensorProto.FLOAT, [1, "samples"])],
        [helper.make_tensor_value_info("embed", TensorProto.FLOAT, [1, "frames", cfg.hidden_size])],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", OPSET)],
        producer_name="ttslts",
    )
    model.ir_version = 8
    onnx.checker.check_model(model, full_check=False)
    return model


def convert(weights_path: str, config_path: str, out_path: str) -> None:
    """Convert the original ``hubert_base`` weights into an ONNX file."""
    import onnx

    from ttslts.vc.torch_zip import load_state_dict

    cfg = HubertConfig.from_json(config_path)
    model = build_model(load_state_dict(weights_path), cfg)
    onnx.save(model, out_path)


def random_state(cfg: HubertConfig, seed: int = 0) -> dict[str, np.ndarray]:
    """A random state dict with the real key names; used for self-tests."""
    rng = np.random.default_rng(seed)
    h, ff = cfg.hidden_size, cfg.intermediate_size

    def w(*shape, scale=0.05):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    s: dict[str, np.ndarray] = {}
    prev = 1
    for i, (dim, k) in enumerate(zip(cfg.conv_dim, cfg.conv_kernel, strict=True)):
        s[f"feature_extractor.conv_layers.{i}.conv.weight"] = w(dim, prev, k, scale=0.3)
        prev = dim
    s["feature_extractor.conv_layers.0.layer_norm.weight"] = 1 + w(cfg.conv_dim[0])
    s["feature_extractor.conv_layers.0.layer_norm.bias"] = w(cfg.conv_dim[0])
    s["feature_projection.layer_norm.weight"] = 1 + w(cfg.conv_dim[-1])
    s["feature_projection.layer_norm.bias"] = w(cfg.conv_dim[-1])
    s["feature_projection.projection.weight"] = w(h, cfg.conv_dim[-1])
    s["feature_projection.projection.bias"] = w(h)
    s["encoder.pos_conv_embed.conv.weight_g"] = 1 + w(1, 1, cfg.pos_kernel)
    s["encoder.pos_conv_embed.conv.weight_v"] = w(h, h // cfg.pos_groups, cfg.pos_kernel)
    s["encoder.pos_conv_embed.conv.bias"] = w(h)
    s["encoder.layer_norm.weight"] = 1 + w(h)
    s["encoder.layer_norm.bias"] = w(h)
    for i in range(cfg.num_layers):
        p = f"encoder.layers.{i}"
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            s[f"{p}.attention.{name}.weight"] = w(h, h)
            s[f"{p}.attention.{name}.bias"] = w(h)
        s[f"{p}.layer_norm.weight"] = 1 + w(h)
        s[f"{p}.layer_norm.bias"] = w(h)
        s[f"{p}.feed_forward.intermediate_dense.weight"] = w(ff, h)
        s[f"{p}.feed_forward.intermediate_dense.bias"] = w(ff)
        s[f"{p}.feed_forward.output_dense.weight"] = w(h, ff)
        s[f"{p}.feed_forward.output_dense.bias"] = w(h)
        s[f"{p}.final_layer_norm.weight"] = 1 + w(h)
        s[f"{p}.final_layer_norm.bias"] = w(h)
    return s


def tiny_config() -> HubertConfig:
    return HubertConfig(
        conv_dim=(8,) * 7,
        hidden_size=16,
        num_layers=2,
        num_heads=2,
        intermediate_size=32,
        pos_kernel=8,
        pos_groups=2,
    )


def selftest() -> str:
    """Build a tiny model and run it with ONNX Runtime; returns a short report."""
    import onnxruntime as ort

    cfg = tiny_config()
    model = build_model(random_state(cfg), cfg)
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    audio = np.random.default_rng(1).standard_normal((1, 4000)).astype(np.float32)
    out = session.run(None, {"source": audio})[0]
    ok = out.ndim == 3 and out.shape[0] == 1 and out.shape[2] == cfg.hidden_size
    if not ok or not np.isfinite(out).all():
        raise RuntimeError(f"Unexpected self-test output {out.shape}")
    return f"onnx {onnx_version()} ok, output {tuple(out.shape)}"


def onnx_version() -> str:
    import onnx

    return onnx.__version__
