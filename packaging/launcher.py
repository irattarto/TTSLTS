"""PyInstaller entry point.

``--version`` exits after the imports and ``--selftest`` also builds and runs a tiny ONNX
graph; both exist so CI can smoke-test a packaged build without opening a window.
"""

import sys

from ttslts import __version__
from ttslts.app import main

args = sys.argv[1:]
if "--version" in args:
    print(f"TTSLTS {__version__}")
    raise SystemExit(0)
if "--gpu-probe" in args:  # checks one model on the GPU in its own process
    from ttslts import gpu_probe

    raise SystemExit(gpu_probe.main(args[args.index("--gpu-probe") + 1 :]))
if "--selftest" in args:
    import os
    import tempfile

    import numpy as np

    from ttslts.audio.wav import write_audio
    from ttslts.vc import hubert_onnx

    mp3 = os.path.join(tempfile.mkdtemp(), "selftest.mp3")
    write_audio(mp3, np.zeros(24000, np.float32), 24000)  # the bundled libsndfile writes MP3
    print(f"TTSLTS {__version__}: {hubert_onnx.selftest()}, mp3 {os.path.getsize(mp3)} bytes")
    raise SystemExit(0)

raise SystemExit(main())
