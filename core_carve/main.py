import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget
from PyQt5.QtCore import Qt

from core_carve.tab_geometry import GeometryTab
from core_carve.tab_blank import BlankTab
from core_carve.tab_gcode import GcodeTab


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Core Carve — SkiNC G-code Generator")
        self.resize(1400, 900)

        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)
        self.setCentralWidget(self.tabs)

        self.geometry_tab = GeometryTab()
        self.tabs.addTab(self.geometry_tab, "1 · Geometry")

        # Blank and G-code tabs (created when geometry is loaded)
        self.blank_tab = None
        self.gcode_tab = None

        # Connect to geometry updates
        self.tabs.currentChanged.connect(self._check_geometry_loaded)

    def _check_geometry_loaded(self):
        """Check if geometry is loaded and create blank and G-code tabs if needed."""
        if self.geometry_tab._geom is None:
            return
        params = self.geometry_tab.panel.get_params()
        if self.blank_tab is None:
            self.blank_tab = BlankTab(self.geometry_tab._geom, params)
            self.tabs.addTab(self.blank_tab, "2 · Core Blank")
            self.gcode_tab = GcodeTab(self.geometry_tab._geom, params, self.blank_tab)
            self.tabs.addTab(self.gcode_tab, "3 · G-code")
        else:
            self.blank_tab.geom = self.geometry_tab._geom
            self.blank_tab.params = params
            self.blank_tab._update_layout()
            self.gcode_tab.geom = self.geometry_tab._geom
            self.gcode_tab.params = params


def main():
    app = QApplication(sys.argv)
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
