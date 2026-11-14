"""Hover help that stays on list and table views.

A list or table hides the tooltip at once when the item under the mouse has none of its own, so a
tooltip set on the whole view only flashed up ("csak felvillan"). ``keep_view_tooltip`` shows the
item's own tooltip, or else the view's, for as long as the mouse rests there.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import QAbstractItemView, QToolTip


class _ViewTooltip(QObject):
    def __init__(self, view: QAbstractItemView):
        super().__init__(view)
        self._view = view

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 (Qt's name)
        if event.type() != QEvent.Type.ToolTip:
            return False
        view = self._view
        index = view.indexAt(event.pos())
        tip = index.data(Qt.ItemDataRole.ToolTipRole) if index.isValid() else None
        tip = tip or view.toolTip()
        if tip:
            QToolTip.showText(event.globalPos(), tip, view.viewport())
        else:
            QToolTip.hideText()
        return True


def keep_view_tooltip(view: QAbstractItemView) -> None:
    view.viewport().installEventFilter(_ViewTooltip(view))
