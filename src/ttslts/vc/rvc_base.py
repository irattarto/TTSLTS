"""Downloads the RVC base models (ContentVec, RMVPE) from the original RVC project.

Nothing is redistributed: the files come straight from the original author's Hugging Face
repository, pinned to a fixed revision, to the user's own machine. ContentVec is only
published there as PyTorch weights, so it is converted to ONNX locally (``hubert_onnx``).
"""

from __future__ import annotations

import logging
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path

from ttslts import paths

log = logging.getLogger(__name__)

REPO = "lj1995/VoiceConversionWebUI"
# Pinned so the files cannot change underneath the size checks below.
REVISION = "e6d0c1a17da07c33557852f9dfa2bd44cc75737d"
SOURCE_URL = f"https://huggingface.co/{REPO}"

LICENSE_FILE = "使用需遵守的协议-LICENSE.txt"
WEIGHTS_FILE = "hubert_base/pytorch_model.bin"
CONFIG_FILE = "hubert_base/config.json"
RMVPE_FILE = "rmvpe.onnx"
# name -> expected size in bytes (None: not checked)
FILES: dict[str, int | None] = {
    LICENSE_FILE: None,
    CONFIG_FILE: 1492,
    WEIGHTS_FILE: 189206711,
    RMVPE_FILE: 361688443,
}

CONSENT_TITLE = "RVC base models: terms of use"
CONSENT_TEXT = f"""\
ContentVec and RMVPE are downloaded from the original RVC project:
{SOURCE_URL}

Their license is the MIT license with this extra condition, written by the author:

    本软件仅供研究使用，使用软件者、传播软件导出的声音者自负全责。
    如不认可该条款，则不能使用/引用软件包内所有代码和文件。

Unofficial English translation: "This software is for research use only. Those who use the
software, and those who distribute voices exported from it, bear full responsibility
themselves. If you do not accept this clause, you may not use or reference any code or files
in the package."

By continuing you accept these terms: use the models for research and personal experiments,
and you are responsible for the voices you create and share. The license text is saved next
to the downloaded files. ContentVec is converted to ONNX on this computer.

Download size: about 550 MB."""


def dest_dir() -> Path:
    return paths.models_dir() / "rvc-base"


def _url(name: str) -> str:
    return f"https://huggingface.co/{REPO}/resolve/{REVISION}/{urllib.parse.quote(name)}"


def _fetch(
    name: str,
    target: Path,
    progress: Callable[[str, int, int], None] | None,
    sizes: dict[str, int | None] | None = None,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(_url(name), headers={"User-Agent": "TTSLTS"})
    with urllib.request.urlopen(request, timeout=60) as resp, open(tmp, "wb") as f:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while chunk := resp.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if progress:
                progress(Path(name).name, done, total)
    expected = (FILES if sizes is None else sizes).get(name)
    if expected is not None and tmp.stat().st_size != expected:
        size = tmp.stat().st_size
        tmp.unlink(missing_ok=True)
        raise OSError(f"{name}: downloaded {size} bytes, expected {expected}")
    tmp.replace(target)


def download_and_convert(
    progress: Callable[[str, int, int], None] | None = None,
    status: Callable[[str], None] | None = None,
    dest: Path | None = None,
) -> tuple[Path, Path]:
    """Download RMVPE and ContentVec, convert ContentVec; returns (contentvec, rmvpe) paths."""
    from ttslts.vc import hubert_onnx

    dest = dest or dest_dir()
    dest.mkdir(parents=True, exist_ok=True)
    contentvec = dest / "contentvec.onnx"
    rmvpe = dest / RMVPE_FILE

    license_copy = dest / "LICENSE-RVC.txt"
    if not license_copy.exists():
        _fetch(LICENSE_FILE, license_copy, None)
    if not rmvpe.exists():
        _fetch(RMVPE_FILE, rmvpe, progress)
    if not contentvec.exists():
        weights = dest / "pytorch_model.bin"
        config = dest / "config.json"
        if not weights.exists():
            _fetch(WEIGHTS_FILE, weights, progress)
        if not config.exists():
            _fetch(CONFIG_FILE, config, None)
        if status:
            status("Converting ContentVec to ONNX…")
        tmp = contentvec.with_suffix(".onnx.part")
        hubert_onnx.convert(str(weights), str(config), str(tmp))
        tmp.replace(contentvec)
        weights.unlink(missing_ok=True)  # the ONNX file replaces it
    return contentvec, rmvpe
