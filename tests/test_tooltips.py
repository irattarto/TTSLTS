"""Every control and tray menu entry explains itself on hover."""

import pytest
from PySide6.QtWidgets import (
    QAbstractButton,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QFormLayout,
    QLineEdit,
    QListWidget,
    QPlainTextEdit,
    QProgressBar,
    QTabBar,
)

from ttslts.controller import Controller
from ttslts.settings import Settings
from ttslts.ui.main_window import MainWindow
from ttslts.ui.tray import Tray


def _ancestors(widget):
    parent = widget.parentWidget()
    while parent is not None:
        yield parent
        parent = parent.parentWidget()


@pytest.fixture
def window():
    QApplication.instance() or QApplication([])
    w = MainWindow(Controller(Settings(), save=lambda s: None), tray_available=True)
    yield w
    w._quitting = True
    w.deleteLater()


def test_every_control_has_a_tooltip(window):
    kinds = (
        QAbstractButton,
        QAbstractSpinBox,
        QComboBox,
        QLineEdit,
        QListWidget,
        QPlainTextEdit,
        QProgressBar,
    )
    missing = []
    for kind in kinds:
        for widget in window.findChildren(kind):
            internal = any(
                isinstance(a, (QTabBar, QAbstractSpinBox, QComboBox)) for a in _ancestors(widget)
            )
            if not internal and not widget.toolTip():
                missing.append(f"{type(widget).__name__} {getattr(widget, 'text', lambda: '')()!r}")
    assert not missing, f"controls without a tooltip: {missing}"


def test_form_labels_share_their_fields_tooltip(window):
    checked = 0
    for form in window.findChildren(QFormLayout):
        for row in range(form.rowCount()):
            label = form.itemAt(row, QFormLayout.ItemRole.LabelRole)
            field = form.itemAt(row, QFormLayout.ItemRole.FieldRole)
            if label is None or label.widget() is None or not label.widget().text():
                continue
            widget = field.widget() or field.layout().itemAt(0).widget()
            assert label.widget().toolTip() == widget.toolTip(), label.widget().text()
            checked += 1
    assert checked > 15


def test_tray_menu_entries_have_tooltips(window):
    tray = Tray(window, window.controller)
    actions = [a for a in tray._menu.actions() if not a.isSeparator()]
    assert len(actions) == 5 and all(a.toolTip() for a in actions)
    assert tray._menu.toolTipsVisible()
