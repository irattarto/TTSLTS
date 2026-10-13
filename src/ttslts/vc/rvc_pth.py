"""Imports an RVC v2 voice checkpoint (``.pth``) by writing its inference network as ONNX.

RVC voices are trained and shared as PyTorch checkpoints (``{"weight": ..., "config": [...],
"f0": 1, "version": "v2"}``). To avoid shipping PyTorch, the checkpoint is read with the safe
reader in ``torch_zip`` and the *inference* path of ``SynthesizerTrnMs768NSFsid`` (text encoder
with relative-position attention, reversed flow, NSF-HiFiGAN decoder) is written out directly as
an ONNX graph. The posterior encoder is only needed for training and is left out.

The graph takes the inputs the voice changer already feeds (``phone``, ``phone_lengths``,
``pitch``, ``nsff0``, ``sid``, ``rnd``) and returns ``audio`` ``(1, 1, samples)``. Batch size is
one, so the padding mask is all ones and is not modelled. Verified against the original
PyTorch code in CI (``tools/verify_rvc_pth.py``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

OPSET = 17
SR_NAMES = {"32k": 32000, "40k": 40000, "48k": 48000}
LRELU = 0.1
HALF_HARMONICS = 0  # RVC uses a single (fundamental) source


class UnsupportedVoice(ValueError):
    """The checkpoint is not an RVC v2 voice with pitch."""


@dataclass
class RvcConfig:
    inter_channels: int
    hidden_channels: int
    filter_channels: int
    n_heads: int
    n_layers: int
    kernel_size: int
    resblock: str
    resblock_kernel_sizes: tuple[int, ...]
    resblock_dilation_sizes: tuple[tuple[int, ...], ...]
    upsample_rates: tuple[int, ...]
    upsample_initial_channel: int
    upsample_kernel_sizes: tuple[int, ...]
    spk_embed_dim: int
    gin_channels: int
    sample_rate: int

    @property
    def upp(self) -> int:
        return math.prod(self.upsample_rates)

    @classmethod
    def from_checkpoint(cls, config: list, sr=None) -> RvcConfig:
        if len(config) < 18:
            raise UnsupportedVoice("Unexpected RVC config layout")
        (
            _spec,
            _segment,
            inter,
            hidden,
            filt,
            heads,
            layers,
            kernel,
            _drop,
            resblock,
            rb_kernels,
            rb_dilations,
            up_rates,
            up_channel,
            up_kernels,
            spk,
            gin,
            cfg_sr,
        ) = config[:18]
        sr = sr if sr not in (None, "") else cfg_sr
        if isinstance(sr, str):
            sr = SR_NAMES.get(sr) or int(sr)
        return cls(
            inter_channels=int(inter),
            hidden_channels=int(hidden),
            filter_channels=int(filt),
            n_heads=int(heads),
            n_layers=int(layers),
            kernel_size=int(kernel),
            resblock=str(resblock),
            resblock_kernel_sizes=tuple(int(k) for k in rb_kernels),
            resblock_dilation_sizes=tuple(tuple(int(d) for d in ds) for ds in rb_dilations),
            upsample_rates=tuple(int(u) for u in up_rates),
            upsample_initial_channel=int(up_channel),
            upsample_kernel_sizes=tuple(int(k) for k in up_kernels),
            spk_embed_dim=int(spk),
            gin_channels=int(gin),
            sample_rate=int(sr),
        )


def _plain(array) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(array).astype(np.float32))


def _conv_weight(state: dict, prefix: str) -> np.ndarray:
    """``prefix.weight`` or the weight-norm form ``g * v / ||v||`` (norm over all but dim 0)."""
    if f"{prefix}.weight" in state:
        return _plain(state[f"{prefix}.weight"])
    para = f"{prefix}.parametrizations.weight"
    for g_name, v_name in (
        (f"{prefix}.weight_g", f"{prefix}.weight_v"),
        (f"{para}.original0", f"{para}.original1"),
    ):
        if g_name in state and v_name in state:
            g, v = _plain(state[g_name]), _plain(state[v_name])
            axes = tuple(range(1, v.ndim))
            return g * v / np.sqrt((v**2).sum(axis=axes, keepdims=True))
    raise KeyError(f"Missing weights for {prefix}")


def validate(obj) -> tuple[dict, RvcConfig]:
    """Returns ``(state, config)`` for a loaded checkpoint or explains why it is unsupported."""
    if not isinstance(obj, dict) or "weight" not in obj or "config" not in obj:
        raise UnsupportedVoice("This is not an RVC voice checkpoint (no weight/config).")
    version = str(obj.get("version", "v1"))
    if version != "v2":
        raise UnsupportedVoice(
            f"This is an RVC {version} voice; only v2 voices (768-dim features) are supported."
        )
    if int(obj.get("f0", 1)) != 1:
        raise UnsupportedVoice("This voice was trained without pitch; only pitch (f0) voices work.")
    state = obj["weight"]
    if "enc_p.emb_phone.weight" not in state or state["enc_p.emb_phone.weight"].shape[1] != 768:
        raise UnsupportedVoice("Not a v2 (768-dim) RVC voice.")
    cfg = RvcConfig.from_checkpoint(list(obj["config"]), obj.get("sr"))
    if cfg.resblock not in ("1", "2"):
        raise UnsupportedVoice(f"Unknown resblock type {cfg.resblock!r}")
    return state, cfg


def build_model(state: dict, cfg: RvcConfig, noise: bool = True):
    """Return an ``onnx.ModelProto``; ``noise=False`` removes the random sources (for tests)."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    nodes: list = []
    inits: list = []
    counter = [0]

    def init(name: str, array: np.ndarray) -> str:
        counter[0] += 1
        unique = f"{name}.{counter[0]}"
        inits.append(numpy_helper.from_array(array, unique))
        return unique

    def op(kind: str, inputs, **attrs) -> str:
        counter[0] += 1
        out = f"t{counter[0]}"
        nodes.append(
            helper.make_node(kind, list(inputs), [out], name=f"{kind}_{counter[0]}", **attrs)
        )
        return out

    def i64(values) -> str:
        return init("i", np.asarray(values, dtype=np.int64))

    def f32(value) -> str:
        return init("f", np.asarray(value, dtype=np.float32))

    def lrelu(x: str, slope: float) -> str:
        return op("LeakyRelu", [x], alpha=slope)

    def conv(x, prefix, *, stride=1, dilation=1, pad=0, bias=True, pads=None):
        weight = _conv_weight(state, prefix)
        args = [x, init(prefix, weight)]
        if bias:
            args.append(init(prefix + ".b", _plain(state[f"{prefix}.bias"])))
        k = weight.shape[-1]
        return op(
            "Conv",
            args,
            kernel_shape=[k],
            strides=[stride],
            dilations=[dilation],
            pads=pads if pads is not None else [pad, pad],
        )

    def linear(x, prefix):
        w = init(prefix, _plain(state[f"{prefix}.weight"]).T.copy())
        return op(
            "Add", [op("MatMul", [x, w]), init(prefix + ".b", _plain(state[f"{prefix}.bias"]))]
        )

    def layer_norm(x, prefix):  # x is (1, C, T); normalise over C
        t = op("Transpose", [x], perm=[0, 2, 1])
        t = op(
            "LayerNormalization",
            [
                t,
                init(prefix, _plain(state[f"{prefix}.gamma"])),
                init(prefix + ".b", _plain(state[f"{prefix}.beta"])),
            ],
            axis=-1,
            epsilon=1e-5,
        )
        return op("Transpose", [t], perm=[0, 2, 1])

    # ---- inputs -----------------------------------------------------------------------------
    heads, kc = cfg.n_heads, cfg.hidden_channels // cfg.n_heads
    window = 10
    n_frames = op("Gather", [op("Shape", ["phone"]), i64(1)], axis=0)  # scalar T
    frames_vec = op("Unsqueeze", [n_frames, i64([0])])
    ar = op("Range", [i64(0), n_frames, i64(1)])  # (T,)
    col = op("Unsqueeze", [ar, i64([0])])  # (1, T): j
    row = op("Unsqueeze", [ar, i64([1])])  # (T, 1): i
    diff = op("Sub", [col, row])  # j - i

    # key-side relative index: (T, T) -> embedding slot, validity
    rel_idx = op("Clip", [op("Add", [diff, i64(window)]), i64(0), i64(2 * window)])
    rel_valid = op(
        "Cast",
        [op("LessOrEqual", [op("Abs", [diff]), i64(window)])],
        to=TensorProto.FLOAT,
    )
    rel_idx4 = op(
        "Expand",
        [
            op("Unsqueeze", [rel_idx, i64([0, 1])]),
            op("Concat", [i64([1, heads]), frames_vec, frames_vec], axis=0),
        ],
    )
    # value-side: for query i and slot m the absolute position j = i + m - window
    slots = i64(np.arange(2 * window + 1))
    absj = op("Sub", [op("Add", [row, op("Unsqueeze", [slots, i64([0])])]), i64(window)])  # (T, 21)
    abs_valid = op(
        "Cast",
        [op("And", [op("GreaterOrEqual", [absj, i64(0)]), op("Less", [absj, n_frames])])],
        to=TensorProto.FLOAT,
    )
    abs_idx = op("Clip", [absj, i64(0), op("Sub", [n_frames, i64(1)])])
    abs_idx4 = op(
        "Expand",
        [
            op("Unsqueeze", [abs_idx, i64([0, 1])]),
            op("Concat", [i64([1, heads]), frames_vec, i64([2 * window + 1])], axis=0),
        ],
    )

    # ---- text encoder -------------------------------------------------------------------------
    emb = linear("phone", "enc_p.emb_phone")
    pitch_emb = op(
        "Gather",
        [init("enc_p.emb_pitch", _plain(state["enc_p.emb_pitch.weight"])), "pitch"],
        axis=0,
    )
    x = op("Mul", [op("Add", [emb, pitch_emb]), f32(math.sqrt(cfg.hidden_channels))])
    x = lrelu(x, LRELU)
    x = op("Transpose", [x], perm=[0, 2, 1])  # (1, H, T)

    def attention(x, p):
        q = conv(x, f"{p}.conv_q", pads=[0, 0])
        k = conv(x, f"{p}.conv_k", pads=[0, 0])
        v = conv(x, f"{p}.conv_v", pads=[0, 0])
        split = i64([0, heads, kc, -1])
        q = op("Transpose", [op("Reshape", [q, split])], perm=[0, 1, 3, 2])  # (1,h,T,k)
        kt = op("Reshape", [k, split])  # (1,h,k,T)
        v = op("Transpose", [op("Reshape", [v, split])], perm=[0, 1, 3, 2])
        q = op("Mul", [q, f32(1.0 / math.sqrt(kc))])
        scores = op("MatMul", [q, kt])

        emb_k = _plain(state[f"{p}.emb_rel_k"])[None]  # (1, 1, 21, k)
        emb_v = _plain(state[f"{p}.emb_rel_v"])[None]
        logits = op(
            "MatMul", [q, init(p + ".ek", emb_k.transpose(0, 1, 3, 2).copy())]
        )  # (1,h,T,21)
        local = op("GatherElements", [logits, rel_idx4], axis=3)  # (1,h,T,T)
        scores = op("Add", [scores, op("Mul", [local, rel_valid])])
        probs = op("Softmax", [scores], axis=-1)
        out = op("MatMul", [probs, v])
        rel_w = op(
            "Mul", [op("GatherElements", [probs, abs_idx4], axis=3), abs_valid]
        )  # (1,h,T,21)
        out = op("Add", [out, op("MatMul", [rel_w, init(p + ".ev", emb_v)])])
        out = op(
            "Reshape",
            [op("Transpose", [out], perm=[0, 1, 3, 2]), i64([0, cfg.hidden_channels, -1])],
        )
        return conv(out, f"{p}.conv_o", pads=[0, 0])

    def ffn(x, p):
        k = cfg.kernel_size
        pads = [(k - 1) // 2, k // 2]
        h = op("Relu", [conv(x, f"{p}.conv_1", pads=pads)])
        return conv(h, f"{p}.conv_2", pads=pads)

    for i in range(cfg.n_layers):
        x = layer_norm(
            op("Add", [x, attention(x, f"enc_p.encoder.attn_layers.{i}")]),
            f"enc_p.encoder.norm_layers_1.{i}",
        )
        x = layer_norm(
            op("Add", [x, ffn(x, f"enc_p.encoder.ffn_layers.{i}")]),
            f"enc_p.encoder.norm_layers_2.{i}",
        )
    stats = conv(x, "enc_p.proj", pads=[0, 0])
    ic = cfg.inter_channels
    m_p = op("Slice", [stats, i64([0]), i64([ic]), i64([1])])
    logs_p = op("Slice", [stats, i64([ic]), i64([2 * ic]), i64([1])])
    z = op("Add", [m_p, op("Mul", [op("Exp", [logs_p]), op("Mul", ["rnd", f32(0.66666)])])])

    # ---- speaker embedding ---------------------------------------------------------------
    g = op("Gather", [init("emb_g", _plain(state["emb_g.weight"])), "sid"], axis=0)  # (1, gin)
    g = op("Unsqueeze", [g, i64([2])])  # (1, gin, 1)

    # ---- flow (reverse) --------------------------------------------------------------------------
    half = ic // 2
    hc = cfg.hidden_channels

    def wn(h, p):
        cond = conv(g, f"{p}.cond_layer", pads=[0, 0])
        out = None
        n_layers = 3
        for i in range(n_layers):
            dil = 1**i
            x_in = conv(h, f"{p}.in_layers.{i}", dilation=dil, pad=int((5 * dil - dil) / 2))
            x_in = op(
                "Add",
                [x_in, op("Slice", [cond, i64([i * 2 * hc]), i64([(i + 1) * 2 * hc]), i64([1])])],
            )
            t = op("Tanh", [op("Slice", [x_in, i64([0]), i64([hc]), i64([1])])])
            s = op("Sigmoid", [op("Slice", [x_in, i64([hc]), i64([2 * hc]), i64([1])])])
            rs = conv(op("Mul", [t, s]), f"{p}.res_skip_layers.{i}", pads=[0, 0])
            if i < n_layers - 1:
                h = op("Add", [h, op("Slice", [rs, i64([0]), i64([hc]), i64([1])])])
                skip = op("Slice", [rs, i64([hc]), i64([2 * hc]), i64([1])])
            else:
                skip = rs
            out = skip if out is None else op("Add", [out, skip])
        return out

    flip = i64(np.arange(ic)[::-1].copy())
    for layer in reversed(range(4)):
        z = op("Gather", [z, flip], axis=1)  # Flip
        p = f"flow.flows.{layer * 2}"
        x0 = op("Slice", [z, i64([0]), i64([half]), i64([1])])
        x1 = op("Slice", [z, i64([half]), i64([ic]), i64([1])])
        h = conv(x0, f"{p}.pre", pads=[0, 0])
        h = wn(h, f"{p}.enc")
        m = conv(h, f"{p}.post", pads=[0, 0])
        z = op("Concat", [x0, op("Sub", [x1, m])], axis=1)

    # ---- only decode the frames from "skip" on (the voice changer needs just the newest) ----
    z = op("Slice", [z, "skip", i64([2**31 - 1]), i64([2])])
    nsff0_dec = op("Slice", ["nsff0", "skip", i64([2**31 - 1]), i64([1])])
    dec_frames = op("Sub", [frames_vec, "skip"])

    # ---- NSF source (single harmonic) ---------------------------------------------------------
    upp = cfg.upp
    sr = float(cfg.sample_rate)
    f0 = op(
        "Unsqueeze", [op("Unsqueeze", [op("Squeeze", [nsff0_dec, i64([0])]), i64([0])]), i64([0])]
    )  # (1,1,T)
    rad = op("Mod", [op("Div", [f0, f32(sr)]), f32(1.0)], fmod=1)
    phase = op("Mul", [op("CumSum", [rad, i64(2)]), f32(float(upp))])
    out_len = op("Mul", [dec_frames, i64([upp])])
    sizes4 = op("Concat", [i64([1, 1, 1]), out_len], axis=0)
    empty = init("empty", np.zeros(0, dtype=np.float32))

    def resize(x, mode, **extra):  # x is (1, 1, T) -> (1, 1, T * upp)
        x4 = op("Unsqueeze", [x, i64([2])])
        y = op("Resize", [x4, empty, empty, sizes4], mode=mode, **extra)
        return op("Squeeze", [y, i64([2])])

    phase_up = resize(phase, "linear", coordinate_transformation_mode="align_corners")
    rad_up = resize(
        rad, "nearest", coordinate_transformation_mode="asymmetric", nearest_mode="floor"
    )
    wrapped = op("Mod", [phase_up, f32(1.0)], fmod=1)
    step = op(
        "Sub",
        [
            op("Slice", [wrapped, i64([1]), i64([2**31 - 1]), i64([2])]),
            op("Slice", [wrapped, i64([0]), i64([-1]), i64([2])]),
        ],
    )
    jump = op("Mul", [op("Cast", [op("Less", [step, f32(-0.5)])], to=TensorProto.FLOAT), f32(-1.0)])
    shift = op("Pad", [jump, i64([0, 0, 1, 0, 0, 0])])
    wave = op(
        "Sin", [op("Mul", [op("CumSum", [op("Add", [rad_up, shift]), i64(2)]), f32(2.0 * math.pi)])]
    )
    wave = op("Mul", [wave, f32(0.1)])
    voiced = op("Cast", [op("Greater", [f0, f32(0.0)])], to=TensorProto.FLOAT)
    voiced_up = resize(
        voiced, "nearest", coordinate_transformation_mode="asymmetric", nearest_mode="floor"
    )
    noise_amp = op(
        "Add",
        [
            op("Mul", [voiced_up, f32(0.003)]),
            op("Mul", [op("Sub", [f32(1.0), voiced_up]), f32(0.1 / 3)]),
        ],
    )
    if noise:
        rnd_src = op("RandomNormalLike", [wave])
    else:
        rnd_src = op("Mul", [wave, f32(0.0)])
    wave = op("Add", [op("Mul", [wave, voiced_up]), op("Mul", [noise_amp, rnd_src])])
    lw = _plain(state["dec.m_source.l_linear.weight"]).reshape(())
    lb = _plain(state["dec.m_source.l_linear.bias"]).reshape(())
    har = op("Tanh", [op("Add", [op("Mul", [wave, f32(float(lw))]), f32(float(lb))])])  # (1,1,L)

    # ---- NSF-HiFiGAN decoder ------------------------------------------------------------------
    x = conv(z, "dec.conv_pre", pad=3)
    x = op("Add", [x, conv(g, "dec.cond", pads=[0, 0])])
    n_k = len(cfg.resblock_kernel_sizes)
    for i, (u, k) in enumerate(zip(cfg.upsample_rates, cfg.upsample_kernel_sizes, strict=True)):
        x = lrelu(x, LRELU)
        w = init(f"dec.ups.{i}", _conv_weight(state, f"dec.ups.{i}"))
        x = op(
            "ConvTranspose",
            [x, w, init(f"dec.ups.{i}.b", _plain(state[f"dec.ups.{i}.bias"]))],
            kernel_shape=[k],
            strides=[u],
            pads=[(k - u) // 2, (k - u) // 2],
        )
        if i + 1 < len(cfg.upsample_rates):
            stride_f0 = math.prod(cfg.upsample_rates[i + 1 :])
            noise_in = conv(har, f"dec.noise_convs.{i}", stride=stride_f0, pad=stride_f0 // 2)
        else:
            noise_in = conv(har, f"dec.noise_convs.{i}", pads=[0, 0])
        x = op("Add", [x, noise_in])
        total = None
        for j in range(n_k):
            block = f"dec.resblocks.{i * n_k + j}"
            kernel = cfg.resblock_kernel_sizes[j]
            dils = cfg.resblock_dilation_sizes[j]
            y = x
            if cfg.resblock == "1":
                for d, dil in enumerate(dils):
                    pad = (kernel * dil - dil) // 2
                    t = conv(lrelu(y, LRELU), f"{block}.convs1.{d}", dilation=dil, pad=pad)
                    t = conv(lrelu(t, LRELU), f"{block}.convs2.{d}", pad=(kernel - 1) // 2)
                    y = op("Add", [t, y])
            else:
                for d, dil in enumerate(dils):
                    pad = (kernel * dil - dil) // 2
                    y = op(
                        "Add",
                        [conv(lrelu(y, LRELU), f"{block}.convs.{d}", dilation=dil, pad=pad), y],
                    )
            total = y if total is None else op("Add", [total, y])
        x = op("Div", [total, f32(float(n_k))])
    x = lrelu(x, 0.01)
    x = conv(x, "dec.conv_post", pad=3, bias=False)
    nodes.append(helper.make_node("Tanh", [x], ["audio"], name="output"))

    float_t = TensorProto.FLOAT
    graph = helper.make_graph(
        nodes,
        "rvc_v2",
        [
            helper.make_tensor_value_info("phone", float_t, [1, "frames", 768]),
            helper.make_tensor_value_info("phone_lengths", TensorProto.INT64, [1]),
            helper.make_tensor_value_info("pitch", TensorProto.INT64, [1, "frames"]),
            helper.make_tensor_value_info("nsff0", float_t, [1, "frames"]),
            helper.make_tensor_value_info("sid", TensorProto.INT64, [1]),
            helper.make_tensor_value_info("rnd", float_t, [1, ic, "frames"]),
            # frames dropped from the start before decoding (0 decodes everything)
            helper.make_tensor_value_info("skip", TensorProto.INT64, [1]),
        ],
        [helper.make_tensor_value_info("audio", float_t, [1, 1, "samples"])],
        initializer=inits,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", OPSET)], producer_name="ttslts"
    )
    model.ir_version = 8
    meta = model.metadata_props.add()
    meta.key, meta.value = "sr", str(cfg.sample_rate)
    onnx.checker.check_model(model, full_check=False)
    return model


def convert(pth_path: str, out_path: str) -> int:
    """Convert an RVC v2 ``.pth`` voice to ONNX; returns the voice's sample rate."""
    import onnx

    from ttslts.vc.torch_zip import load_object

    state, cfg = validate(load_object(pth_path))
    onnx.save(build_model(state, cfg), out_path)
    return cfg.sample_rate


def ensure_onnx(pth_path: str, cache_dir: str | None = None) -> str:
    """ONNX file for a ``.pth`` voice, converting once and reusing the result afterwards."""
    import hashlib
    import os
    from pathlib import Path

    src = Path(pth_path)
    stat = src.stat()
    digest = hashlib.sha1(f"{src.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode()).hexdigest()
    if cache_dir is None:
        from ttslts.paths import models_dir

        cache = models_dir() / "voices"
    else:
        cache = Path(cache_dir)
    out = cache / f"{src.stem}-{digest[:10]}-v2.onnx"  # v2: graph with the skip input
    if out.exists():
        return str(out)
    cache.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    convert(str(src), str(tmp))
    os.replace(tmp, out)
    return str(out)
