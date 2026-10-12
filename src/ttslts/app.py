"""Application entry point."""

from __future__ import annotations

import logging
import sys

from PySide6.QtWidgets import QApplication

from ttslts import APP_NAME
from ttslts.controller import Controller
from ttslts.settings import Settings
from ttslts.ui.icon import app_icon
from ttslts.ui.main_window import MainWindow
from ttslts.ui.tray import Tray


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName(APP_NAME)
    app.setWindowIcon(app_icon())
    # The tray keeps the app alive; quitting is explicit (tray menu or close without tray).
    app.setQuitOnLastWindowClosed(False)

    settings = Settings.load()
    controller = Controller(settings)
    window = MainWindow(controller, tray_available=Tray.available())
    tray = None
    if window.tray_available:
        tray = Tray(window, controller)
        window.tray = tray
        tray.show()
    app.aboutToQuit.connect(controller.shutdown)

    if settings.start_minimized and tray is not None:
        window.hide()
    else:
        window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
