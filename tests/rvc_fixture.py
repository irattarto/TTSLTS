"""A tiny random RVC v2 voice (same key names and layout as a real checkpoint)."""

from __future__ import annotations

import numpy as np

# spec, segment, inter, hidden, filter, heads, layers, kernel, dropout, resblock, rb kernels,
# rb dilations, upsample rates, upsample channel, upsample kernels, speakers, gin, sample rate
TINY_CONFIG = [
    513, 32, 16, 16, 32, 2, 2, 3, 0, "1", [3, 5], [[1, 3, 5], [1, 3, 5]],
    [4, 2], 16, [8, 4], 3, 8, 16000,
]  # fmt: skip


def random_voice(config=None, seed: int = 0, **extra) -> dict:
    config = list(config or TINY_CONFIG)
    (
        _,
        _,
        ic,
        hc,
        fc,
        heads,
        layers,
        kernel,
        _,
        resblock,
        rb_k,
        rb_d,
        up_r,
        up_c,
        up_k,
        spk,
        gin,
        sr,
    ) = config
    rng = np.random.default_rng(seed)

    def w(*shape, scale=0.1):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    s: dict[str, np.ndarray] = {}

    def conv(name, out, inp, k, bias=True, norm=False):
        if norm:
            s[f"{name}.weight_g"] = 1 + w(out, 1, 1)
            s[f"{name}.weight_v"] = w(out, inp, k)
        else:
            s[f"{name}.weight"] = w(out, inp, k)
        if bias:
            s[f"{name}.bias"] = w(out)

    s["enc_p.emb_phone.weight"], s["enc_p.emb_phone.bias"] = w(hc, 768, scale=0.05), w(hc)
    s["enc_p.emb_pitch.weight"] = w(256, hc)
    kc = hc // heads
    for i in range(layers):
        for n in ("q", "k", "v", "o"):
            conv(f"enc_p.encoder.attn_layers.{i}.conv_{n}", hc, hc, 1)
        s[f"enc_p.encoder.attn_layers.{i}.emb_rel_k"] = w(1, 21, kc)
        s[f"enc_p.encoder.attn_layers.{i}.emb_rel_v"] = w(1, 21, kc)
        for n in ("norm_layers_1", "norm_layers_2"):
            s[f"enc_p.encoder.{n}.{i}.gamma"] = 1 + w(hc)
            s[f"enc_p.encoder.{n}.{i}.beta"] = w(hc)
        conv(f"enc_p.encoder.ffn_layers.{i}.conv_1", fc, hc, kernel)
        conv(f"enc_p.encoder.ffn_layers.{i}.conv_2", hc, fc, kernel)
    conv("enc_p.proj", 2 * ic, hc, 1)
    s["emb_g.weight"] = w(spk, gin)
    for f in (0, 2, 4, 6):
        p = f"flow.flows.{f}"
        conv(f"{p}.pre", hc, ic // 2, 1)
        conv(f"{p}.enc.cond_layer", 2 * hc * 3, gin, 1, norm=True)
        for i in range(3):
            conv(f"{p}.enc.in_layers.{i}", 2 * hc, hc, 5, norm=True)
            conv(f"{p}.enc.res_skip_layers.{i}", 2 * hc if i < 2 else hc, hc, 1, norm=True)
        conv(f"{p}.post", ic // 2, hc, 1)
    s["dec.m_source.l_linear.weight"], s["dec.m_source.l_linear.bias"] = w(1, 1), w(1)
    conv("dec.conv_pre", up_c, ic, 7)
    conv("dec.cond", up_c, gin, 1)
    ch = up_c
    for i, k in enumerate(up_k):
        ch = up_c // 2 ** (i + 1)
        s[f"dec.ups.{i}.weight_g"] = 1 + w(up_c // 2**i, 1, 1)
        s[f"dec.ups.{i}.weight_v"] = w(up_c // 2**i, ch, k)
        s[f"dec.ups.{i}.bias"] = w(ch)
        if i + 1 < len(up_r):
            stride = int(np.prod(up_r[i + 1 :]))
            conv(f"dec.noise_convs.{i}", ch, 1, stride * 2)
        else:
            conv(f"dec.noise_convs.{i}", ch, 1, 1)
        for j, (k2, dils) in enumerate(zip(rb_k, rb_d, strict=True)):
            base = f"dec.resblocks.{i * len(rb_k) + j}"
            for d in range(len(dils)):
                if resblock == "1":
                    conv(f"{base}.convs1.{d}", ch, ch, k2, norm=True)
                    conv(f"{base}.convs2.{d}", ch, ch, k2, norm=True)
                else:
                    conv(f"{base}.convs.{d}", ch, ch, k2, norm=True)
    conv("dec.conv_post", 1, ch, 7, bias=False)
    return {"weight": s, "config": config, "f0": 1, "version": "v2", "sr": sr, **extra}


def feed(config=None, frames: int = 9, seed: int = 1) -> dict:
    config = config or TINY_CONFIG
    rng = np.random.default_rng(seed)
    f0 = rng.uniform(100, 300, (1, frames)).astype(np.float32)
    f0[:, 2:4] = 0
    return {
        "phone": rng.standard_normal((1, frames, 768)).astype(np.float32),
        "phone_lengths": np.array([frames], dtype=np.int64),
        "pitch": rng.integers(1, 255, (1, frames)).astype(np.int64),
        "nsff0": f0,
        "sid": np.array([1], dtype=np.int64),
        "skip": np.zeros(1, dtype=np.int64),
        "rnd": rng.standard_normal((1, config[2], frames)).astype(np.float32),
    }
