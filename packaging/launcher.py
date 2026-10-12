"""PyInstaller entry point. `--version` exits after the imports, as a packaging smoke test."""

import sys

from ttslts import __version__
from ttslts.app import main

if "--version" in sys.argv[1:]:
    print(f"TTSLTS {__version__}")
    raise SystemExit(0)

raise SystemExit(main())
