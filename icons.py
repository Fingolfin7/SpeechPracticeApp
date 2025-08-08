from PyQt5 import QtCore, QtGui
from PyQt5.QtGui import QColor, QPainter, QPixmap


def make_record_icon(size: int = 20) -> QtGui.QIcon:
    pix = QPixmap(size, size)
    pix.fill(QtCore.Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(QColor("#ff4d57"))
    p.setPen(QColor("#0f141a"))
    p.drawEllipse(0, 0, size, size)
    p.end()
    return QtGui.QIcon(pix)


def make_play_icon(size: int = 20) -> QtGui.QIcon:
    w = h = size
    pix = QPixmap(w, h)
    pix.fill(QtCore.Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(QColor("#1d2633"))
    p.setPen(QColor("#0f141a"))
    p.drawEllipse(0, 0, w, h)
    p.setBrush(QColor("#e6eaf0"))
    p.setPen(QtCore.Qt.NoPen)
    margin = int(size * 0.28)
    points = [
        QtCore.QPoint(margin, margin),
        QtCore.QPoint(w - margin, h // 2),
        QtCore.QPoint(margin, h - margin),
    ]
    p.drawPolygon(QtGui.QPolygon(points))
    p.end()
    return QtGui.QIcon(pix)


def make_pause_icon(size: int = 20) -> QtGui.QIcon:
    w = h = size
    pix = QPixmap(w, h)
    pix.fill(QtCore.Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(QColor("#1d2633"))
    p.setPen(QColor("#0f141a"))
    p.drawEllipse(0, 0, w, h)
    bar_w = max(2, int(size * 0.18))
    gap = int(size * 0.14)
    x1 = w // 2 - gap - bar_w
    x2 = w // 2 + gap
    y = int(size * 0.24)
    bar_h = h - 2 * y
    p.setBrush(QColor("#e6eaf0"))
    p.setPen(QtCore.Qt.NoPen)
    p.drawRoundedRect(x1, y, bar_w, bar_h, 2, 2)
    p.drawRoundedRect(x2, y, bar_w, bar_h, 2, 2)
    p.end()
    return QtGui.QIcon(pix)


def make_stop_icon(size: int = 20) -> QtGui.QIcon:
    w = h = size
    pix = QPixmap(w, h)
    pix.fill(QtCore.Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(QColor("#1d2633"))
    p.setPen(QColor("#0f141a"))
    p.drawEllipse(0, 0, w, h)
    p.setBrush(QColor("#e6eaf0"))
    p.setPen(QtCore.Qt.NoPen)
    s = int(size * 0.42)
    x = (w - s) // 2
    y = (h - s) // 2
    p.drawRoundedRect(x, y, s, s, 2, 2)
    p.end()
    return QtGui.QIcon(pix)


