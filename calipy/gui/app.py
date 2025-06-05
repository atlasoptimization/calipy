# calipy/gui/app.py  – PySide2 edition
import sys
from PySide2.QtWidgets import (
    QApplication, QMainWindow, QDockWidget, QTextEdit
)
from PySide2.QtGui import QAction
from PySide2.QtCore import Qt

from NodeGraphQt import NodeGraph, NodeGraphWidget


class CalipyDesigner(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Calipy Designer")
        self.resize(1200, 800)

        # --- Node-graph canvas -------------------------------------------
        self.graph = NodeGraph()
        self.graph_widget = NodeGraphWidget(self.graph)
        self.setCentralWidget(self.graph_widget)

        # --- Inspector dock (right) --------------------------------------
        self.inspector_dock = QDockWidget("Inspector", self)
        self.inspector_dock.setAllowedAreas(Qt.RightDockWidgetArea)
        self.inspector_widget = QTextEdit("Properties appear here.")
        self.inspector_dock.setWidget(self.inspector_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, self.inspector_dock)

        # --- Console dock (bottom) ---------------------------------------
        self.console_dock = QDockWidget("Console", self)
        self.console_dock.setAllowedAreas(Qt.BottomDockWidgetArea)
        self.console_widget = QTextEdit()
        self.console_widget.setReadOnly(True)
        self.console_dock.setWidget(self.console_widget)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.console_dock)

        # --- Menu bar -----------------------------------------------------
        self._build_menu()

        # --- Demo node ----------------------------------------------------
        self._populate_test_node()

    # ---------------------------------------------------------------------
    # UI helpers
    # ---------------------------------------------------------------------
    def _build_menu(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        help_menu = menubar.addMenu("Help")
        about_action = QAction("About", self)
        about_action.triggered.connect(
            lambda: self.console_widget.append(
                "Calipy Designer — Modular Effect GUI (PySide2)"
            )
        )
        help_menu.addAction(about_action)

    def _populate_test_node(self):
        from NodeGraphQt.nodes import basic_nodes

        self.graph.register_node(basic_nodes.TextInputNode)
        node = self.graph.create_node("nodes.TextInputNode", name="HelloNode")
        node.set_pos(200, 150)


# -------------------------------------------------------------------------
# Entry-point
# -------------------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    win = CalipyDesigner()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
