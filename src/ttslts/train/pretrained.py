"""Downloads the pretrained RVC v2 generator and discriminator that training starts from.

Training from scratch needs hours of audio; starting from the original project's pretrained
weights gives a good voice from a few minutes. The files come straight from the original
author's Hugging Face repository (pinned revision, same as the other RVC base models).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ttslts.vc import rvc_base

SIZES: dict[str, int | None] = {
    "pretrained_v2/f0G32k.pth": 73950049,
    "pretrained_v2/f0D32k.pth": 142875703,
    "pretrained_v2/f0G40k.pth": 73106273,
    "pretrained_v2/f0D40k.pth": 142875703,
    "pretrained_v2/f0G48k.pth": 75465569,
    "pretrained_v2/f0D48k.pth": 142875703,
}

CONSENT_TITLE = "Pretrained RVC voice models: terms of use"
CONSENT_TEXT = f"""\
Training starts from the pretrained RVC v2 voice models of the original RVC project:
{rvc_base.SOURCE_URL}

Their license is the MIT license with this extra condition, written by the author:

    本软件仅供研究使用，使用软件者、传播软件导出的声音者自负全责。
    如不认可该条款，则不能使用/引用软件包内所有代码和文件。

Unofficial English translation: "This software is for research use only. Those who use the
software, and those who distribute voices exported from it, bear full responsibility
themselves. If you do not accept this clause, you may not use or reference any code or files
in the package."

By continuing you accept these terms: use them for research and personal experiments, and you
are responsible for the voices you create and share. Only train on recordings of voices you
have permission to use.

Download size: about 215 MB (for the sample rate you picked)."""


def dest_dir() -> Path:
    return rvc_base.dest_dir().parent / "rvc-pretrained"


def names(sample_rate: int) -> tuple[str, str]:
    k = sample_rate // 1000
    return f"pretrained_v2/f0G{k}k.pth", f"pretrained_v2/f0D{k}k.pth"


def paths_for(sample_rate: int, dest: Path | None = None) -> tuple[Path, Path]:
    dest = dest or dest_dir()
    g, d = names(sample_rate)
    return dest / Path(g).name, dest / Path(d).name


def is_installed(sample_rate: int, dest: Path | None = None) -> bool:
    return all(p.exists() for p in paths_for(sample_rate, dest))


def download(
    sample_rate: int,
    progress: Callable[[str, int, int], None] | None = None,
    dest: Path | None = None,
) -> tuple[Path, Path]:
    """Fetch the generator and discriminator for ``sample_rate``; returns their paths."""
    dest = dest or dest_dir()
    dest.mkdir(parents=True, exist_ok=True)
    license_copy = dest / "LICENSE-RVC.txt"
    if not license_copy.exists():
        rvc_base._fetch(rvc_base.LICENSE_FILE, license_copy, None)
    g_path, d_path = paths_for(sample_rate, dest)
    for name, target in zip(names(sample_rate), (g_path, d_path), strict=True):
        if not target.exists():
            rvc_base._fetch(name, target, progress, SIZES)
    return g_path, d_path
