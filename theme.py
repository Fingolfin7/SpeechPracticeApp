from PyQt5 import QtGui, QtWidgets


def apply_modern_theme(app: QtWidgets.QApplication) -> None:
    """Apply a clean, modern dark theme with a cyan accent."""
    app.setStyle("Fusion")

    # Base font
    try:
        app.setFont(QtGui.QFont("Segoe UI", 10))
    except Exception:
        pass

    # Dark palette
    bg = QtGui.QColor("#0f141a")
    panel = QtGui.QColor("#151b22")
    text = QtGui.QColor("#e6eaf0")
    accent = QtGui.QColor("#00d0ff")

    pal = QtGui.QPalette()
    pal.setColor(QtGui.QPalette.Window, bg)
    pal.setColor(QtGui.QPalette.WindowText, text)
    pal.setColor(QtGui.QPalette.Base, panel)
    pal.setColor(QtGui.QPalette.AlternateBase, bg)
    pal.setColor(QtGui.QPalette.Text, text)
    pal.setColor(QtGui.QPalette.Button, panel)
    pal.setColor(QtGui.QPalette.ButtonText, text)
    pal.setColor(QtGui.QPalette.ToolTipBase, panel)
    pal.setColor(QtGui.QPalette.ToolTipText, text)
    pal.setColor(QtGui.QPalette.Highlight, accent)
    pal.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor("#0f141a"))
    app.setPalette(pal)

    # App-wide stylesheet (widgets keep subtle borders/padding)
    app.setStyleSheet(
        """
            QMainWindow{background:#0f141a;}
            QMenuBar{background:#0f141a;color:#e6eaf0;}
            QMenuBar::item:selected{background:#151b22;}
            QMenu{background:#151b22;color:#e6eaf0;border:1px solid #202833;}
            QMenu::item:selected{background:#1f2a36;}
            QSplitter::handle{background:#0f141a;width:6px;margin:0 4px;}
            QSplitter::handle:hover{background:#1a2230;}
            QLabel{color:#e6eaf0;}
            QListWidget{background:#151b22;color:#e6eaf0;border:1px solid #202833;border-radius:6px;}
            QListWidget::item:selected{background:#0e639c;}
            QTextEdit{background:#151b22;color:#e6eaf0;border:1px solid #202833;border-radius:6px;}
            QPushButton{background:#1d2633;color:#e6eaf0;border:1px solid #263241;border-radius:8px;padding:8px 14px;}
            QPushButton:hover{background:#223043;}
            QPushButton:disabled{color:#6f7c91;border-color:#2b3747;}
            QPushButton#PrimaryButton{background:#00d0ff;color:#0f141a;font-weight:600;border:0;padding:8px 18px;border-radius:18px;}
            QPushButton#PrimaryButton:hover{background:#5ee0ff;}
            QPushButton#RecordBtn{background:#e5484d;border:0;width:44px;height:44px;border-radius:22px;color:#0f141a;}
            QPushButton#RecordBtn:pressed{background:#ff5b61;}
            QPushButton#CircleBtn{background:#1d2633;width:44px;height:44px;border-radius:22px;}
            QWidget#Transport{background:#151b22;border:1px solid #263241;border-radius:28px;}
            QToolTip{background-color:#151b22;color:#e6eaf0;border:1px solid #202833;}
        """
    )


