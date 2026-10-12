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
if "--selftest" in args:
    from ttslts.vc import hubert_onnx

    print(f"TTSLTS {__version__}: {hubert_onnx.selftest()}")
    raise SystemExit(0)

raise SystemExit(main())
