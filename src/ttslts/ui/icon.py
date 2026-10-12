"""App icon drawn at runtime, so no binary assets are needed."""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap


def app_icon(active: bool = False) -> QIcon:
    icon = QIcon()
    for size in (16, 24, 32, 48, 64, 128, 256):
        pm = QPixmap(size, size)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#d9480f" if active else "#3b5bdb"))
        p.drawEllipse(QRectF(0.5, 0.5, size - 1, size - 1))
        # Sound-wave bars
        pen = QPen(QColor("white"), max(1.5, size / 11))
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        heights = (0.25, 0.5, 0.75, 0.5, 0.25)
        for i, h in enumerate(heights):
            x = size * (0.22 + i * 0.14)
            half = size * h / 2
            p.drawLine(int(x), int(size / 2 - half / 1.2), int(x), int(size / 2 + half / 1.2))
        p.end()
        icon.addPixmap(pm)
    return icon
