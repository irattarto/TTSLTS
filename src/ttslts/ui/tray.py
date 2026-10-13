"""System tray icon and its menu."""

from __future__ import annotations

from PySide6.QtGui import QAction, QGuiApplication
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

from ttslts import APP_NAME
from ttslts.controller import Controller
from ttslts.ui.icon import app_icon


class Tray(QSystemTrayIcon):
    def __init__(self, window, controller: Controller):
        super().__init__(app_icon(), window)
        self.window = window
        self.controller = controller
        self.setToolTip(APP_NAME)

        menu = QMenu()
        self.show_action = QAction("Show window", menu, triggered=self.window.restore)
        self.vc_action = QAction("Voice changer", menu, checkable=True)
        self.vc_action.triggered.connect(self._toggle_vc)
        self.clip_action = QAction("Speak clipboard", menu, triggered=self._speak_clipboard)
        self.stop_action = QAction("Stop speaking", menu, triggered=controller.stop_speaking)
        self.quit_action = QAction("Quit", menu, triggered=self.window.quit)
        for action, tip in (
            (self.show_action, "Show the TTSLTS window"),
            (self.vc_action, "Turn the real-time voice changer on or off"),
            (self.clip_action, "Read the text currently on the clipboard aloud"),
            (self.stop_action, "Stop the speech that is playing"),
            (self.quit_action, "Exit TTSLTS completely (closing the window only hides it)"),
        ):
            action.setToolTip(tip)
        menu.setToolTipsVisible(True)
        for a in (
            self.show_action,
            None,
            self.vc_action,
            self.clip_action,
            self.stop_action,
            None,
            self.quit_action,
        ):
            menu.addSeparator() if a is None else menu.addAction(a)
        self._menu = menu
        self.setContextMenu(menu)

        self.activated.connect(self._on_activated)
        controller.vcRunning.connect(self._on_vc_running)

    def _on_activated(self, reason):
        if reason in (
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        ):
            self.window.toggle_visible()

    def _toggle_vc(self, checked: bool):
        self.controller.start_vc() if checked else self.controller.stop_vc()

    def _on_vc_running(self, running: bool):
        self.vc_action.setChecked(running)
        self.setIcon(app_icon(active=running))
        self.setToolTip(f"{APP_NAME} – voice changer {'on' if running else 'off'}")

    def _speak_clipboard(self):
        text = QGuiApplication.clipboard().text()
        if text.strip():
            self.controller.speak(text)

    @staticmethod
    def available() -> bool:
        return QApplication.instance() is not None and QSystemTrayIcon.isSystemTrayAvailable()
