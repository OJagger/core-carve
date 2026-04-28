import json
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QFileDialog, QMessageBox
from PyQt5.QtCore import Qt

from core_carve.tab_design import DesignTab
from core_carve.tab_base import BaseTab
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

        self.design_tab = DesignTab()
        self.tabs.addTab(self.design_tab, "Outline Design")
        self.design_tab.set_outline_callback(self._receive_designed_outline)

        self.base_tab = BaseTab()
        self.tabs.addTab(self.base_tab, "Base Design")

        self.geometry_tab = GeometryTab()
        self.tabs.addTab(self.geometry_tab, "Core Design")

        # Blank and G-code tabs (created when geometry is loaded)
        self.blank_tab = None
        self.gcode_tab = None

        # Wire geometry tab "Save ski" to combined save (includes planform params)
        self.geometry_tab.panel.btn_save_json.clicked.disconnect(
            self.geometry_tab._save_json
        )
        self.geometry_tab.panel.btn_save_json.clicked.connect(self._save_ski_file)

        # Connect to geometry updates
        self.tabs.currentChanged.connect(self._check_geometry_loaded)

    def _save_ski_file(self):
        """Save all ski params (planform + core) to one JSON file."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Ski Definition", "ski_params.json",
            "JSON Files (*.json);;All Files (*)"
        )
        if not path:
            return
        try:
            ski_data = {}
            ski_data.update(self.design_tab.panel.get_params().to_dict())
            ski_data["base"] = self.base_tab.panel.get_params().to_dict()
            from dataclasses import asdict
            ski_data["core"] = asdict(self.geometry_tab.panel.get_params())
            with open(path, "w") as f:
                json.dump(ski_data, f, indent=2)
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", str(exc))

    def _receive_designed_outline(self, outline):
        """Accept a designed outline from the Design tab and push it to other tabs."""
        self.base_tab.set_outline(outline)
        self.geometry_tab._outline = outline
        self.geometry_tab._update_geometry()
        self.tabs.setCurrentWidget(self.base_tab)

    def _check_geometry_loaded(self):
        """Check if geometry is loaded and create blank and G-code tabs if needed."""
        if self.geometry_tab._geom is None:
            return
        params = self.geometry_tab.panel.get_params()
        if self.blank_tab is None:
            self.blank_tab = BlankTab(self.geometry_tab._geom, params)
            self.tabs.addTab(self.blank_tab, "Core Blank")
            self.gcode_tab = GcodeTab(self.geometry_tab._geom, params, self.blank_tab)
            self.tabs.addTab(self.gcode_tab, "G-code")
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
