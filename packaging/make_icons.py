"""Renders the app icon to PNG and ICO files for the installers.

Usage: python packaging/make_icons.py OUTPUT_DIR
"""

import sys
from pathlib import Path

from PySide6.QtGui import QGuiApplication

from ttslts.ui.icon import app_icon


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    app = QGuiApplication.instance() or QGuiApplication(sys.argv[:1])  # noqa: F841
    icon = app_icon()
    if not icon.pixmap(256, 256).toImage().save(str(out_dir / "ttslts.png")):
        raise SystemExit("could not write ttslts.png")
    if not icon.pixmap(256, 256).toImage().save(str(out_dir / "ttslts.ico")):
        raise SystemExit("could not write ttslts.ico")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
