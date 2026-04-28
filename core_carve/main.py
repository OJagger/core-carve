import json
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QFileDialog, QMessageBox
from PyQt5.QtCore import Qt

from core_carve.tab_design import DesignTab
from core_carve.tab_base import BaseTab
from core_carve.tab_geometry import GeometryTab
from core_carve.tab_camber import CamberTab
from core_carve.tab_blank import BlankTab
from core_carve.tab_gcode import GcodeTab
from core_carve.tab_profile import ProfileTab


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
        self.design_tab.set_ski_definition_loaded_callback(self._load_ski_definition)

        self.base_tab = BaseTab()
        self.tabs.addTab(self.base_tab, "Base Design")

        self.geometry_tab = GeometryTab()
        self.tabs.addTab(self.geometry_tab, "Core Design")

        self.camber_tab = CamberTab()
        self.tabs.addTab(self.camber_tab, "Camber Design")

        # Blank, G-code, and Profile tabs (created when geometry is loaded)
        self.blank_tab = None
        self.gcode_tab = None
        self.profile_tab = None

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
            ski_data["camber"] = self.camber_tab.panel.get_params().to_dict()
            from dataclasses import asdict
            ski_data["core"] = asdict(self.geometry_tab.panel.get_params())
            with open(path, "w") as f:
                json.dump(ski_data, f, indent=2)
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", str(exc))

    def _load_ski_definition(self, path: str) -> bool:
        """Load complete ski definition file and populate all tabs."""
        try:
            with open(path) as f:
                ski_data = json.load(f)

            # Load outline design params
            if "length" in ski_data:
                outline_params = self.design_tab.panel.get_params()
                from core_carve.ski_design import SkiPlanformParams
                updated_params = SkiPlanformParams(
                    length=ski_data.get("length", outline_params.length),
                    waist_w=ski_data.get("waist_w", outline_params.waist_w),
                    sidecut_radius=ski_data.get("dimensions", {}).get("sidecut_radius", outline_params.sidecut_radius),
                    tip_l=ski_data.get("tip_l", outline_params.tip_l),
                    tip_w=ski_data.get("tip_w", outline_params.tip_w),
                    tail_l=ski_data.get("tail_l", outline_params.tail_l),
                    tail_w=ski_data.get("tail_w", outline_params.tail_w),
                    setback=ski_data.get("setback", outline_params.setback),
                    tip_trans_len=ski_data.get("tip_trans_len", outline_params.tip_trans_len),
                    tail_trans_len=ski_data.get("tail_trans_len", outline_params.tail_trans_len),
                    tip_apex_arm=ski_data.get("control_arms", {}).get("tip_apex_arm", outline_params.tip_apex_arm),
                    tip_junc_arm=ski_data.get("control_arms", {}).get("tip_junc_arm", outline_params.tip_junc_arm),
                    tip_trans_junc_arm=ski_data.get("control_arms", {}).get("tip_trans_junc_arm", outline_params.tip_trans_junc_arm),
                    tip_trans_arc_arm=ski_data.get("control_arms", {}).get("tip_trans_arc_arm", outline_params.tip_trans_arc_arm),
                    tail_trans_arc_arm=ski_data.get("control_arms", {}).get("tail_trans_arc_arm", outline_params.tail_trans_arc_arm),
                    tail_trans_junc_arm=ski_data.get("control_arms", {}).get("tail_trans_junc_arm", outline_params.tail_trans_junc_arm),
                    tail_junc_arm=ski_data.get("control_arms", {}).get("tail_junc_arm", outline_params.tail_junc_arm),
                    tail_apex_arm=ski_data.get("control_arms", {}).get("tail_apex_arm", outline_params.tail_apex_arm),
                )
                self.design_tab._update_from_params(updated_params)

            # Load base design params
            if "base" in ski_data:
                from core_carve.base_design import BaseParams
                base_params = BaseParams(**ski_data["base"])
                self.base_tab.panel.set_params(base_params)
                self.base_tab._update_preview()

            # Load camber design params
            if "camber" in ski_data:
                from core_carve.camber_design import CamberParams
                camber_params = CamberParams(**ski_data["camber"])
                self.camber_tab.panel.set_params(camber_params)
                self.camber_tab._update_preview()

            # Load core design params
            if "core" in ski_data:
                from core_carve.ski_geometry import SkiParams
                core_params = SkiParams(**ski_data["core"])
                self.geometry_tab.panel.set_params(core_params)
                self.geometry_tab._update_geometry()

            return True
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load ski definition: {str(e)}")
            return False

    def _receive_designed_outline(self, outline):
        """Accept a designed outline from the Design tab and push it to other tabs."""
        self.base_tab.set_outline(outline)
        self.geometry_tab._outline = outline
        self.geometry_tab._update_geometry()
        self.tabs.setCurrentWidget(self.base_tab)

    def _check_geometry_loaded(self):
        """Check if geometry is loaded and create blank, G-code, and profile tabs if needed."""
        if self.geometry_tab._geom is None:
            return
        params = self.geometry_tab.panel.get_params()
        if self.blank_tab is None:
            self.blank_tab = BlankTab(self.geometry_tab._geom, params)
            self.tabs.addTab(self.blank_tab, "Core Blank")
            self.gcode_tab = GcodeTab(self.geometry_tab._geom, params, self.blank_tab)
            self.tabs.addTab(self.gcode_tab, "Sidewall slot machining")
            self.profile_tab = ProfileTab(self.geometry_tab._geom, params, self.blank_tab)
            self.tabs.addTab(self.profile_tab, "Core profiling")
        else:
            self.blank_tab.geom = self.geometry_tab._geom
            self.blank_tab.params = params
            self.blank_tab._update_layout()
            self.gcode_tab.geom = self.geometry_tab._geom
            self.gcode_tab.params = params
            self.profile_tab.geom = self.geometry_tab._geom
            self.profile_tab.params = params
            self.camber_tab.set_ski_length(self.geometry_tab._geom.ski_length)


def main():
    app = QApplication(sys.argv)
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
