"""Tab — Base Design: metal edge layout, step cutouts, and G-code export."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QPushButton, QLabel, QLineEdit, QGroupBox, QFormLayout,
    QMessageBox, QScrollArea, QSizePolicy, QFileDialog, QComboBox, QProgressBar,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from core_carve.base_design import BaseParams, compute_base_outline, export_base_dxf, compute_base_gcode


class _GcodeWorker(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, outline, params):
        super().__init__()
        self._outline = outline
        self._params = params

    def run(self):
        try:
            result = compute_base_gcode(self._outline, self._params)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


# ── Canvas ────────────────────────────────────────────────────────────────────

class BaseCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(facecolor="#1e1e1e", figsize=(12, 6))
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ax = None
        self._setup_axes()

    def _setup_axes(self):
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#2b2b2b")
        self.ax.tick_params(colors="#cccccc", labelsize=8)
        for spine in self.ax.spines.values():
            spine.set_edgecolor("#555555")
        self.ax.xaxis.label.set_color("#cccccc")
        self.ax.yaxis.label.set_color("#cccccc")
        self.ax.title.set_color("#eeeeee")
        self.fig.tight_layout(pad=2.0)
        self.draw()

    def plot_base_outline(self, outline, params):
        """Plot the base outline polygon with edge step cutouts (horizontal top-down view)."""
        self._setup_axes()
        ax = self.ax

        if outline is None or len(outline) == 0:
            ax.text(0.5, 0.5, "Load outline from Outline Design tab",
                    ha="center", va="center", transform=ax.transAxes, color="#999")
            self.fig.tight_layout(pad=2.0)
            self.draw()
            return

        # Ski displayed horizontally: X-axis = along ski (outline Y), Y-axis = across ski (outline X)
        # Draw ski outline (reference, dashed)
        ax.plot(outline[:, 1], outline[:, 0], color="#4488ff", linewidth=1,
                linestyle="--", alpha=0.4, label="Ski outline")

        # Compute and draw the base outline with edge step cutouts
        base_poly = compute_base_outline(outline, params)
        if len(base_poly) > 0:
            # Close the polygon for plotting
            poly_closed = np.vstack([base_poly, base_poly[0]])
            ax.plot(poly_closed[:, 1], poly_closed[:, 0], color="#80c0ff",
                    linewidth=1.0, label="Base outline")

        y_min, y_max = outline[:, 1].min(), outline[:, 1].max()
        edge_start_y = y_min + params.tip_offset
        edge_end_y = y_max - params.tail_offset
        ax.axvline(x=edge_start_y, color="#ffaa00", linestyle="--", linewidth=1,
                   alpha=0.5, label="Edge start/end")
        ax.axvline(x=edge_end_y, color="#ffaa00", linestyle="--", linewidth=1, alpha=0.5)

        ax.set_aspect("equal")
        ax.set_xlabel("Along ski (mm)")
        ax.set_ylabel("Across ski (mm)")
        ax.set_title("Base Design with Edge Step Cutouts (Top View)")
        ax.grid(True, alpha=0.2, color="#555555")
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3,
                  facecolor="#333333", labelcolor="#dddddd")

        self.fig.tight_layout(pad=2.0)
        self.draw()


# ── Parameter panel ──────────────────────────────────────────────────────────

class _FloatField(QLineEdit):
    def __init__(self, default: float):
        super().__init__(str(default))
        self.setFixedWidth(90)

    def value(self) -> float:
        try:
            return float(self.text())
        except ValueError:
            return 0.0


class BaseParameterPanel(QWidget):
    def __init__(self):
        super().__init__()
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(6)

        # ── Edge parameters ──────────────────────────────────────────────────
        edge_group = QGroupBox("Metal Edge Parameters")
        edge_lay = QFormLayout(edge_group)

        self.f_edge_width = _FloatField(2.0)
        self.f_tip_offset = _FloatField(100.0)
        self.f_tail_offset = _FloatField(100.0)

        edge_lay.addRow("Edge groove width (mm):", self.f_edge_width)
        edge_lay.addRow("Tip edge offset (mm):", self.f_tip_offset)
        edge_lay.addRow("Tail edge offset (mm):", self.f_tail_offset)
        root.addWidget(edge_group)

        # ── Base material ────────────────────────────────────────────────────
        base_group = QGroupBox("Base Material")
        base_lay = QFormLayout(base_group)

        self.f_base_length = _FloatField(2000.0)
        self.f_base_width = _FloatField(100.0)
        self.f_base_thickness = _FloatField(2.0)

        base_lay.addRow("Base length (mm):", self.f_base_length)
        base_lay.addRow("Base width (mm):", self.f_base_width)
        base_lay.addRow("Base thickness (mm):", self.f_base_thickness)
        root.addWidget(base_group)

        # ── Cutting tool ─────────────────────────────────────────────────────
        tool_group = QGroupBox("Cutting Tool")
        tool_lay = QFormLayout(tool_group)

        self.combo_cutter = QComboBox()
        self.combo_cutter.addItems(["Router", "Drag knife"])
        self.f_cutting_feed = _FloatField(2000.0)

        # Router-specific
        self.f_tool_diameter = _FloatField(6.0)
        self.f_spindle_speed = _FloatField(18000.0)
        self.f_plunge_feed = _FloatField(300.0)
        self.f_clearance = _FloatField(10.0)

        # Drag knife-specific
        self.f_kerf_width = _FloatField(0.5)

        tool_lay.addRow("Cutter type:", self.combo_cutter)
        tool_lay.addRow("Cutting feed (mm/min):", self.f_cutting_feed)
        tool_lay.addRow("Tool diameter (mm):", self.f_tool_diameter)
        tool_lay.addRow("Spindle speed (RPM):", self.f_spindle_speed)
        tool_lay.addRow("Plunge feed (mm/min):", self.f_plunge_feed)
        tool_lay.addRow("Clearance height (mm):", self.f_clearance)
        tool_lay.addRow("Kerf width (mm):", self.f_kerf_width)
        root.addWidget(tool_group)

        # Show/hide router vs drag-knife fields
        self.combo_cutter.currentTextChanged.connect(self._on_cutter_changed)
        self._on_cutter_changed("Router")

        # ── Buttons ──────────────────────────────────────────────────────────
        button_lay = QHBoxLayout()
        self.btn_load_params = QPushButton("Load params…")
        self.btn_save_params = QPushButton("Save params…")
        self.btn_export_dxf = QPushButton("Export DXF…")
        self.btn_generate_gcode = QPushButton("Generate G-code")
        self.btn_save_gcode = QPushButton("Save G-code…")

        button_lay.addWidget(self.btn_load_params)
        button_lay.addWidget(self.btn_save_params)
        button_lay.addStretch()
        button_lay.addWidget(self.btn_export_dxf)
        button_lay.addWidget(self.btn_generate_gcode)
        button_lay.addWidget(self.btn_save_gcode)
        root.addLayout(button_lay)

        self.lbl_status = QLabel("✓ Ready")
        self.lbl_status.setStyleSheet("color: #60cc60;")
        root.addWidget(self.lbl_status)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(12)
        root.addWidget(self.progress_bar)

        root.addStretch()

    def _on_cutter_changed(self, text: str):
        is_router = text == "Router"
        self.f_tool_diameter.setEnabled(is_router)
        self.f_spindle_speed.setEnabled(is_router)
        self.f_plunge_feed.setEnabled(is_router)
        self.f_clearance.setEnabled(is_router)
        self.f_kerf_width.setEnabled(not is_router)

    def get_params(self) -> BaseParams:
        cutter = "router" if self.combo_cutter.currentText() == "Router" else "drag_knife"
        return BaseParams(
            edge_width=self.f_edge_width.value(),
            tip_offset=self.f_tip_offset.value(),
            tail_offset=self.f_tail_offset.value(),
            base_length=self.f_base_length.value(),
            base_width=self.f_base_width.value(),
            base_thickness=self.f_base_thickness.value(),
            cutter_type=cutter,
            tool_diameter=self.f_tool_diameter.value(),
            spindle_speed=self.f_spindle_speed.value(),
            cutting_feed=self.f_cutting_feed.value(),
            plunge_feed=self.f_plunge_feed.value(),
            clearance_height=self.f_clearance.value(),
            kerf_width=self.f_kerf_width.value(),
            cutting_speed=self.f_cutting_feed.value(),
        )

    def set_params(self, p: BaseParams):
        self.f_edge_width.setText(str(p.edge_width))
        self.f_tip_offset.setText(str(p.tip_offset))
        self.f_tail_offset.setText(str(p.tail_offset))
        self.f_base_length.setText(str(p.base_length))
        self.f_base_width.setText(str(p.base_width))
        self.f_base_thickness.setText(str(p.base_thickness))
        cutter_text = "Router" if p.cutter_type == "router" else "Drag knife"
        self.combo_cutter.setCurrentText(cutter_text)
        self.f_tool_diameter.setText(str(p.tool_diameter))
        self.f_spindle_speed.setText(str(p.spindle_speed))
        self.f_cutting_feed.setText(str(p.cutting_feed))
        self.f_plunge_feed.setText(str(p.plunge_feed))
        self.f_clearance.setText(str(p.clearance_height))
        self.f_kerf_width.setText(str(p.kerf_width))


# ── Tab widget ────────────────────────────────────────────────────────────────

class BaseTab(QWidget):
    def __init__(self):
        super().__init__()
        self._outline = None
        self._gcode_string = None
        self._build_ui()
        self._connect_signals()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.splitter = QSplitter(Qt.Vertical)

        self.canvas = BaseCanvas()
        self.splitter.addWidget(self.canvas)

        self.panel = BaseParameterPanel()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.panel)
        self.splitter.addWidget(scroll)

        self.splitter.setStretchFactor(0, 2)
        self.splitter.setStretchFactor(1, 1)

        layout.addWidget(self.splitter)

    def _connect_signals(self):
        self.panel.f_edge_width.textChanged.connect(self._update_preview)
        self.panel.f_tip_offset.textChanged.connect(self._update_preview)
        self.panel.f_tail_offset.textChanged.connect(self._update_preview)

        self.panel.btn_export_dxf.clicked.connect(self._export_dxf)
        self.panel.btn_load_params.clicked.connect(self._load_params)
        self.panel.btn_save_params.clicked.connect(self._save_params)
        self.panel.btn_generate_gcode.clicked.connect(self._generate_gcode)
        self.panel.btn_save_gcode.clicked.connect(self._save_gcode)
        self._load_ski_callback = None
        self._save_ski_callback = None

    def set_load_ski_callback(self, fn):
        self._load_ski_callback = fn

    def set_save_ski_callback(self, fn):
        self._save_ski_callback = fn

    def set_outline(self, outline):
        self._outline = outline
        self._update_preview()

    def _update_preview(self):
        try:
            params = self.panel.get_params()
            self.canvas.plot_base_outline(self._outline, params)
            self.panel.lbl_status.setText("✓ Ready")
            self.panel.lbl_status.setStyleSheet("color: #60cc60;")
        except Exception as e:
            self.panel.lbl_status.setText(f"✗ Error: {str(e)}")
            self.panel.lbl_status.setStyleSheet("color: #ff6060;")

    def _export_dxf(self):
        if self._outline is None or len(self._outline) == 0:
            QMessageBox.warning(self, "No outline", "Load outline from Outline Design tab first")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Base DXF", "base.dxf",
            "DXF Files (*.dxf);;All Files (*)"
        )
        if not path:
            return

        try:
            params = self.panel.get_params()
            export_base_dxf(self._outline, params, path)
            QMessageBox.information(self, "Exported", f"Base outline exported to {Path(path).name}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def _generate_gcode(self):
        if self._outline is None or len(self._outline) == 0:
            QMessageBox.warning(self, "No outline", "Load outline from Outline Design tab first")
            return

        params = self.panel.get_params()
        self.panel.btn_generate_gcode.setEnabled(False)
        self.panel.lbl_status.setText("Generating G-code…")
        self.panel.lbl_status.setStyleSheet("color: #aaaaaa;")
        self.panel.progress_bar.setVisible(True)

        self._worker = _GcodeWorker(self._outline, params)
        self._worker.finished.connect(self._on_gcode_ready)
        self._worker.error.connect(self._on_gcode_error)
        self._worker.start()

    def _on_gcode_ready(self, gcode_string: str):
        self._gcode_string = gcode_string
        n_lines = gcode_string.count("\n") + 1
        self.panel.btn_generate_gcode.setEnabled(True)
        self.panel.progress_bar.setVisible(False)
        self.panel.lbl_status.setText(f"✓ G-code generated ({n_lines} lines)")
        self.panel.lbl_status.setStyleSheet("color: #60cc60;")

    def _on_gcode_error(self, msg: str):
        self.panel.btn_generate_gcode.setEnabled(True)
        self.panel.progress_bar.setVisible(False)
        self.panel.lbl_status.setText(f"✗ Error: {msg}")
        self.panel.lbl_status.setStyleSheet("color: #ff6060;")

    def _save_gcode(self):
        if not self._gcode_string:
            QMessageBox.warning(self, "No G-code", "Generate G-code first")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save G-code", "base.nc",
            "G-code Files (*.nc *.gcode);;All Files (*)"
        )
        if not path:
            return

        try:
            with open(path, "w") as f:
                f.write(self._gcode_string)
            QMessageBox.information(self, "Saved", f"G-code saved to {Path(path).name}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def _load_params(self):
        if self._load_ski_callback is not None:
            path, _ = QFileDialog.getOpenFileName(
                self, "Load Ski Definition", "", "JSON Files (*.json);;All Files (*)"
            )
            if path:
                self._load_ski_callback(path)
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Parameters", "", "JSON Files (*.json);;All Files (*)"
        )
        if not path:
            return
        try:
            params = BaseParams.from_json(path)
            self.panel.set_params(params)
            self._update_preview()
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))

    def _save_params(self):
        if self._save_ski_callback is not None:
            self._save_ski_callback()
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Parameters", "base_params.json",
            "JSON Files (*.json);;All Files (*)"
        )
        if not path:
            return
        try:
            self.panel.get_params().to_json(path)
            QMessageBox.information(self, "Saved", f"Parameters saved to {Path(path).name}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))
