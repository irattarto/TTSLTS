"""The private Python + PyTorch environment that training runs in.

PyTorch is far too large to ship in the installers (the CUDA build is several GB), so the first
time you train, the app sets up its own environment under ``%APPDATA%\\TTSLTS\\runtime``:

* Windows: the embeddable Python from python.org, with pip from pypa.io;
* Linux / macOS: a ``venv`` made from the system ``python3``;

and installs PyTorch (CUDA build when an NVIDIA GPU is present, otherwise the small CPU build)
plus the few packages the training job needs. When TTSLTS itself runs from a source checkout that
already has PyTorch, that interpreter is used and nothing is downloaded.

The training job imports the app's own modules from ``source_dir()``: the package source is
shipped inside the packaged app as plain files for exactly this reason.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

from ttslts import paths

log = logging.getLogger(__name__)

PYTHON_VERSION = "3.12.10"
EMBED_URL = (
    f"https://www.python.org/ftp/python/{PYTHON_VERSION}/python-{PYTHON_VERSION}-embed-amd64.zip"
)
GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"
TORCH_INDEX_CUDA = "https://download.pytorch.org/whl/cu126"
TORCH_INDEX_CPU = "https://download.pytorch.org/whl/cpu"
PACKAGES = ["numpy", "scipy", "onnx", "onnxruntime", "soundfile"]
CHECK = "import torch, numpy, scipy, onnx, onnxruntime, soundfile; print(torch.cuda.is_available())"

CONSENT_TITLE = "Set up the training environment"
CONSENT_TEXT = """\
Training needs PyTorch, which is too big to ship with the app. TTSLTS will set up its own
private Python environment in its data folder (nothing is installed system-wide) and download:

  - Python (Python Software Foundation license), from python.org
  - pip, from bootstrap.pypa.io
  - PyTorch (BSD-3-Clause), from download.pytorch.org
  - numpy, scipy, onnx, onnxruntime and soundfile, from PyPI

Download size: about {size}. This is done once; the environment is reused for every training.
You can delete the "runtime" folder in the TTSLTS data folder at any time to remove it."""


def has_nvidia_gpu() -> bool:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return False
    try:
        out = subprocess.run(
            [exe, "-L"], capture_output=True, text=True, timeout=15, creationflags=_no_window()
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return out.returncode == 0 and "GPU" in out.stdout


def consent_text() -> str:
    return CONSENT_TEXT.format(size="3 GB" if has_nvidia_gpu() else "0.5 GB")


def _no_window() -> int:
    return subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


def runtime_dir() -> Path:
    return paths.config_dir() / "runtime"


def source_dir() -> Path:
    """Folder that holds the ``ttslts`` package source for the training process to import."""
    bundled = Path(getattr(sys, "_MEIPASS", "")) / "pysrc"
    if getattr(sys, "frozen", False) and bundled.exists():
        return bundled
    return Path(__file__).resolve().parents[2]  # .../src


def _own_python_has(module: str = "torch") -> bool:
    """Whether the app runs from a source checkout whose own Python already has ``module``."""
    import importlib.util

    return not getattr(sys, "frozen", False) and importlib.util.find_spec(module) is not None


def _python_exe(root: Path | None = None) -> Path:
    root = root or runtime_dir()
    if sys.platform == "win32":
        return root / "python" / "python.exe"
    return root / "venv" / "bin" / "python"


def is_installed(root: Path | None = None, module: str = "torch") -> bool:
    return _own_python_has(module) or (
        (root or runtime_dir()).joinpath("ready.json").exists() and _python_exe(root).exists()
    )


def python_command(root: Path | None = None, module: str = "torch") -> list[str]:
    """The interpreter that runs training jobs (and what it needs in its environment)."""
    if _own_python_has(module):
        exe = Path(sys.executable)
        if exe.name.lower() == "pythonw.exe" and exe.with_name("python.exe").exists():
            exe = exe.with_name("python.exe")  # the job reports on stdout
        return [str(exe)]
    return [str(_python_exe(root))]


def prepare_for_run(root: Path | None = None, module: str = "torch") -> dict[str, str]:
    """Make the interpreter find the app's sources; returns extra environment variables."""
    src = str(source_dir())
    if _own_python_has(module):
        return {}  # a source install already has them
    pth = next((_python_exe(root).parent).glob("python*._pth"), None)
    if pth is not None:  # the embeddable Python ignores PYTHONPATH and reads this file instead
        lines = [ln for ln in pth.read_text(encoding="utf-8").splitlines() if "pysrc" not in ln]
        pth.write_text("\n".join([*lines, src + "  # pysrc"]) + "\n", encoding="utf-8")
        return {}
    return {"PYTHONPATH": src}


def _download(url: str, target: Path, progress: Callable[[str, int, int], None] | None) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "TTSLTS"})
    with urllib.request.urlopen(request, timeout=60) as resp, open(target, "wb") as f:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while chunk := resp.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if progress:
                progress(target.name, done, total)


def _run(
    cmd: list[str], status: Callable[[str], None] | None, env: dict[str, str] | None = None
) -> None:
    """Run a command, passing its output lines on as status text."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=_no_window(),
        env={**os.environ, **(env or {})},
    )
    tail: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        if line:
            tail = [*tail[-15:], line]
            if status:
                status(line[:120])
    if proc.wait() != 0:
        raise RuntimeError(f"{Path(cmd[0]).name} failed:\n" + "\n".join(tail[-8:]))


def install(
    progress: Callable[[str, int, int], None] | None = None,
    status: Callable[[str], None] | None = None,
    root: Path | None = None,
    cuda: bool | None = None,
    torch_packages: list[str] | None = None,
    packages: list[str] | None = None,
    check: str = CHECK,
) -> Path:
    """Create the training environment (or, with other packages, another PyTorch environment
    such as the F5 voices' one); returns its Python interpreter."""
    root = root or runtime_dir()
    say = status or (lambda _text: None)
    cuda = has_nvidia_gpu() if cuda is None else cuda
    root.mkdir(parents=True, exist_ok=True)
    (root / "ready.json").unlink(missing_ok=True)

    if sys.platform == "win32":
        python_dir = root / "python"
        if not python_dir.exists():
            say("Downloading Python…")
            archive = root / "python-embed.zip"
            _download(EMBED_URL, archive, progress)
            tmp = root / "python.part"
            shutil.rmtree(tmp, ignore_errors=True)
            with zipfile.ZipFile(archive) as z:
                z.extractall(tmp)
            archive.unlink()
            for pth in tmp.glob("python*._pth"):  # let site-packages (pip, torch) be found
                text = pth.read_text(encoding="utf-8").replace("#import site", "import site")
                pth.write_text(text, encoding="utf-8")
            tmp.replace(python_dir)
        python = _python_exe(root)
        if not (python_dir / "Scripts" / "pip.exe").exists():
            say("Installing pip…")
            get_pip = root / "get-pip.py"
            _download(GET_PIP_URL, get_pip, None)
            _run([str(python), str(get_pip), "--no-warn-script-location"], status)
            get_pip.unlink(missing_ok=True)
    else:
        python = _python_exe(root)
        if not python.exists():
            say("Creating the Python environment…")
            _run([sys.executable if not getattr(sys, "frozen", False) else "python3", "-m", "venv",
                  str(root / "venv")], status)  # fmt: skip

    pip = [
        str(python),
        "-m",
        "pip",
        "install",
        "--no-warn-script-location",
        "--disable-pip-version-check",
    ]
    say("Downloading PyTorch (this is the big one)…")
    index = TORCH_INDEX_CUDA if cuda else TORCH_INDEX_CPU
    _run([*pip, *(torch_packages or ["torch"]), "--index-url", index], status)
    say("Installing the other packages…")
    _run([*pip, *(packages or PACKAGES)], status)

    say("Checking the installation…")
    env = {"PYTHONPATH": str(source_dir())}
    check = subprocess.run(
        [str(python), "-c", check],
        capture_output=True,
        text=True,
        creationflags=_no_window(),
        env={**os.environ, **env},
    )
    if check.returncode != 0:
        raise RuntimeError("The training environment did not start:\n" + check.stderr[-600:])
    (root / "ready.json").write_text(
        json.dumps({"python": PYTHON_VERSION, "cuda": check.stdout.strip() == "True"}),
        encoding="utf-8",
    )
    return python
