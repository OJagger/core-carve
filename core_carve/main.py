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
        self.base_tab.set_load_ski_callback(self._load_ski_definition)
        self.base_tab.set_save_ski_callback(self._save_ski_file)

        self.geometry_tab = GeometryTab()
        self.tabs.addTab(self.geometry_tab, "Core Design")
        self.geometry_tab.set_load_ski_callback(self._load_ski_definition)
        self.geometry_tab.set_save_ski_callback(self._save_ski_file)

        self.camber_tab = CamberTab()
        self.tabs.addTab(self.camber_tab, "Camber Design")
        self.camber_tab.set_load_ski_callback(self._load_ski_definition)
        self.camber_tab.set_save_ski_callback(self._save_ski_file)

        # Blank, G-code, and Profile tabs (created when geometry is loaded)
        self.blank_tab = None
        self.gcode_tab = None
        self.profile_tab = None

        # Splitters whose vis-pane size is kept in sync across all tabs
        self._viz_splitters = [
            self.design_tab.splitter if hasattr(self.design_tab, "splitter") else None,
            self.base_tab.splitter,
            self.geometry_tab.splitter,
            self.camber_tab.splitter,
        ]
        self._viz_splitters = [s for s in self._viz_splitters if s is not None]
        self._syncing_splitters = False
        for s in self._viz_splitters:
            s.splitterMoved.connect(self._sync_splitters)

        # Connect to geometry updates
        self.tabs.currentChanged.connect(self._check_geometry_loaded)

    def _sync_splitters(self, pos: int, index: int):
        """Keep all viz-pane splitters at the same fractional position."""
        if self._syncing_splitters:
            return
        sender = self.sender()
        sizes = sender.sizes()
        total = sum(sizes)
        if total == 0:
            return
        ratio = sizes[0] / total
        self._syncing_splitters = True
        try:
            for s in self._viz_splitters:
                if s is sender:
                    continue
                t = sum(s.sizes())
                if t > 0:
                    s.setSizes([int(ratio * t), t - int(ratio * t)])
        finally:
            self._syncing_splitters = False

    def _save_ski_file(self, path: str = None):
        """Save all ski params to one hierarchical JSON file."""
        if path is None:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Ski Definition", "ski_params.json",
                "JSON Files (*.json);;All Files (*)"
            )
        if not path:
            return
        try:
            from dataclasses import asdict
            ski_data = {
                "outline": self.design_tab.panel.get_params().to_dict(),
                "base": self.base_tab.panel.get_params().to_dict(),
                "camber": self.camber_tab.panel.get_params().to_dict(),
                "core": asdict(self.geometry_tab.panel.get_params()),
            }
            with open(path, "w") as f:
                json.dump(ski_data, f, indent=2)
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", str(exc))

    def _load_ski_definition(self, path: str) -> bool:
        """Load complete ski definition file and populate all tabs."""
        try:
            with open(path) as f:
                ski_data = json.load(f)

            # Load outline design params — support new hierarchical format and old flat format
            from core_carve.ski_design import SkiPlanformParams
            outline_src = ski_data.get("outline", ski_data)
            if outline_src:
                try:
                    updated_params = SkiPlanformParams.from_dict(outline_src)
                    self.design_tab._update_from_params(updated_params)
                    if self.design_tab._result is not None:
                        self.geometry_tab._outline = self.design_tab._result.outline.copy()
                        self.base_tab.set_outline(self.design_tab._result.outline.copy())
                except Exception:
                    pass

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

            # Load core design params (supports nested "core" key or flat top-level fields)
            from core_carve.ski_geometry import SkiParams
            flat = ski_data.get("core", ski_data)
            core_kwargs = {k: v for k, v in flat.items() if k in SkiParams.__dataclass_fields__}
            if core_kwargs:
                core_params = SkiParams(**core_kwargs)
                self.geometry_tab.panel.set_params(core_params)
                self.geometry_tab._update_geometry()

            # Create / update downstream tabs (blank, gcode, profile) and sync camber length
            self._check_geometry_loaded()
            if self.geometry_tab._geom is not None:
                self.camber_tab.set_ski_length(self.geometry_tab._geom.ski_length)
                self.camber_tab.set_geometry(self.geometry_tab._geom, self.geometry_tab.panel.get_params())

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
            self.camber_tab.set_geometry(self.geometry_tab._geom, self.geometry_tab.panel.get_params())
            # Register late-created splitters and sync them to the current ratio
            for new_tab in (self.blank_tab, self.gcode_tab, self.profile_tab):
                s = new_tab.splitter
                self._viz_splitters.append(s)
                s.splitterMoved.connect(self._sync_splitters)
            # Apply current ratio from first splitter
            if self._viz_splitters:
                ref = self._viz_splitters[0]
                ref_sizes = ref.sizes()
                ref_total = sum(ref_sizes)
                if ref_total > 0:
                    ratio = ref_sizes[0] / ref_total
                    for s in (self.blank_tab.splitter, self.gcode_tab.splitter, self.profile_tab.splitter):
                        t = sum(s.sizes())
                        if t > 0:
                            s.setSizes([int(ratio * t), t - int(ratio * t)])
        else:
            self.blank_tab.geom = self.geometry_tab._geom
            self.blank_tab.params = params
            self.blank_tab._update_layout()
            self.gcode_tab.geom = self.geometry_tab._geom
            self.gcode_tab.params = params
            self.profile_tab.geom = self.geometry_tab._geom
            self.profile_tab.params = params
            self.camber_tab.set_ski_length(self.geometry_tab._geom.ski_length)
            self.camber_tab.set_geometry(self.geometry_tab._geom, self.geometry_tab.panel.get_params())


def main():
    app = QApplication(sys.argv)
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
