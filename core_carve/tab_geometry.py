"""Tab 1 — Ski Geometry: DXF planform input, parameter entry, and visualisation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QPushButton, QLabel, QLineEdit, QFileDialog,
    QGroupBox, QFormLayout, QMessageBox, QScrollArea,
    QSizePolicy,
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as mpatches

from core_carve.ski_geometry import (
    SkiParams, SkiGeometry, load_planform_dxf, compute_geometry, half_widths_at_y
)


# ── Matplotlib canvas ──────────────────────────────────────────────────────────

class SkiCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(facecolor="#1e1e1e")
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ax_plan = None
        self.ax_side = None
        self._setup_axes()

    def _setup_axes(self):
        self.fig.clear()
        # Two stacked subplots: planform (top, more height) + side profile (bottom)
        self.ax_plan, self.ax_side = self.fig.subplots(
            2, 1, gridspec_kw={"height_ratios": [3, 1]}
        )
        for ax in (self.ax_plan, self.ax_side):
            ax.set_facecolor("#2b2b2b")
            ax.tick_params(colors="#cccccc", labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor("#555555")
            ax.xaxis.label.set_color("#cccccc")
            ax.yaxis.label.set_color("#cccccc")
            ax.title.set_color("#eeeeee")
        self.fig.tight_layout(pad=2.0, rect=[0, 0.08, 1, 1])
        self.draw()

    def plot_geometry(self, outline: np.ndarray, geom: SkiGeometry, params: SkiParams):
        self._setup_axes()
        self._plot_planform(outline, geom, params)
        self._plot_side_profile(geom, params)
        self.fig.tight_layout(pad=2.0, rect=[0, 0.08, 1, 1])
        self.draw()

    # ── planform ──────────────────────────────────────────────────────────────

    def _plot_planform(self, outline: np.ndarray, geom: SkiGeometry, params: SkiParams):
        ax = self.ax_plan
        y = outline[:, 1]
        x = outline[:, 0]

        # --- ski outline
        ax.plot(y, x, color="#80c0ff", linewidth=1.5, label="Ski outline")

        # Sample Y positions along the ski for derived edges
        y_core = np.linspace(geom.core_tip_x, geom.core_tail_x, 600)
        left_out, right_out = half_widths_at_y(outline, y_core)

        # --- sidewall outer edge = outline ± sidewall_overlap
        sw_left = left_out - params.sidewall_overlap
        sw_right = right_out + params.sidewall_overlap
        ax.plot(y_core, sw_left, color="#f0a040", linewidth=1.0,
                linestyle="--", label="Sidewall outer edge")
        ax.plot(y_core, sw_right, color="#f0a040", linewidth=1.0, linestyle="--")

        # --- core inner edge = outline ∓ sidewall_width (inboard of sidewall outer)
        core_left = sw_left + params.sidewall_width
        core_right = sw_right - params.sidewall_width
        ax.plot(y_core, core_left, color="#60cc60", linewidth=1.0,
                linestyle="-.", label="Core edge")
        ax.plot(y_core, core_right, color="#60cc60", linewidth=1.0, linestyle="-.")

        # --- core/sidewall ends (vertical lines at tip and tail infill)
        for lbl, xv in [("Core tip end", geom.core_tip_x),
                         ("Core tail end", geom.core_tail_x)]:
            ax.axvline(xv, color="#ff6060", linewidth=1.2, linestyle=":")

        # --- waist & geometric centre
        ax.axvline(geom.waist_x, color="#ffd700", linewidth=1.5,
                   linestyle="-", label=f"Waist  (x={geom.waist_x:.0f} mm)")
        ax.axvline(geom.geometric_centre_x, color="#cc88ff", linewidth=1.2,
                   linestyle="--",
                   label=f"Geo. centre  (x={geom.geometric_centre_x:.0f} mm)")

        ax.set_xlabel("Length along ski (mm)")
        ax.set_ylabel("Half-width (mm)")
        ax.set_title("Ski Planform Geometry")
        ax.set_aspect("equal")
        ax.legend(fontsize=8, facecolor="#333333", labelcolor="#dddddd",
                  loc="upper center", bbox_to_anchor=(0.5, -0.38), ncol=6, frameon=True)

    # ── side profile ──────────────────────────────────────────────────────────

    def _plot_side_profile(self, geom: SkiGeometry, params: SkiParams):
        ax = self.ax_side
        EXAG = 10  # vertical exaggeration

        y_prof = np.linspace(geom.core_tip_x, geom.core_tail_x, 800)
        thickness = geom.thickness_at(y_prof)

        ax.fill_between(y_prof, 0, thickness * EXAG,
                        color="#a0c8ff", alpha=0.5, label="Core")
        ax.plot(y_prof, thickness * EXAG, color="#80c0ff", linewidth=1.5)

        # underfoot region shading
        ax.axvspan(geom.underfoot_start_x, geom.underfoot_end_x,
                   color="#ffd700", alpha=0.15, label="Underfoot zone")
        ax.axvline(geom.waist_x, color="#ffd700", linewidth=1.0, linestyle="-")
        ax.axvline(geom.geometric_centre_x, color="#cc88ff",
                   linewidth=1.0, linestyle="--")

        # Y-axis tick labels: divide back by exaggeration to show real mm
        y_ticks = ax.get_yticks()
        ax.set_yticklabels([f"{v / EXAG:.1f}" for v in y_ticks], fontsize=7)

        ax.set_xlabel("Length along ski (mm)")
        ax.set_ylabel("Thickness (mm)")
        ax.set_title("Core Thickness Profile")
        ax.legend(fontsize=8, facecolor="#333333", labelcolor="#dddddd",
                  loc="upper right")


# ── Parameter input panel ─────────────────────────────────────────────────────

class _FloatField(QLineEdit):
    def __init__(self, default: float):
        super().__init__(str(default))
        self.setFixedWidth(90)

    def value(self) -> float:
        return float(self.text())


class ParameterPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(6)

        # ── File buttons ──────────────────────────────────────────────────────
        file_group = QGroupBox("Files")
        file_lay = QHBoxLayout(file_group)

        self.btn_load_dxf = QPushButton("Load Planform DXF…")
        self.lbl_dxf = QLabel("No file loaded")
        self.lbl_dxf.setStyleSheet("color: #888;")

        self.btn_load_json = QPushButton("Load Params JSON…")
        self.btn_save_json = QPushButton("Save Params JSON…")

        for w in (self.btn_load_dxf, self.lbl_dxf,
                  self.btn_load_json, self.btn_save_json):
            file_lay.addWidget(w)
        file_lay.addStretch()
        root.addWidget(file_group)

        # ── Derived info (two columns) ────────────────────────────────────────
        info_group = QGroupBox("Ski planform geometry")
        info_lay = QHBoxLayout(info_group)

        # Left column
        left_info = QFormLayout()
        self.lbl_ski_length = QLabel("—")
        self.lbl_core_length = QLabel("—")
        self.lbl_waist_pos = QLabel("—")
        self.lbl_setback = QLabel("—")
        left_info.addRow("Ski length:", self.lbl_ski_length)
        left_info.addRow("Core length:", self.lbl_core_length)
        left_info.addRow("Waist position:", self.lbl_waist_pos)
        left_info.addRow("Setback:", self.lbl_setback)

        # Right column
        right_info = QFormLayout()
        self.lbl_tip_width = QLabel("—")
        self.lbl_waist_width = QLabel("—")
        self.lbl_tail_width = QLabel("—")
        right_info.addRow("Tip width:", self.lbl_tip_width)
        right_info.addRow("Waist width:", self.lbl_waist_width)
        right_info.addRow("Tail width:", self.lbl_tail_width)

        info_lay.addLayout(left_info)
        info_lay.addLayout(right_info)
        root.addWidget(info_group)

        # ── Parameters (two columns) ──────────────────────────────────────────
        params_group = QGroupBox("Core geometry")
        params_lay = QHBoxLayout(params_group)

        defaults = SkiParams()
        self.f_tip_infill      = _FloatField(defaults.tip_infill)
        self.f_tail_infill     = _FloatField(defaults.tail_infill)
        self.f_sw_width        = _FloatField(defaults.sidewall_width)
        self.f_sw_overlap      = _FloatField(defaults.sidewall_overlap)
        self.f_tip_thick       = _FloatField(defaults.tip_thickness)
        self.f_uf_thick        = _FloatField(defaults.underfoot_thickness)
        self.f_tail_thick      = _FloatField(defaults.tail_thickness)
        self.f_uf_length       = _FloatField(defaults.underfoot_length)

        # Left column: infill and sidewall
        left_params = QFormLayout()
        left_params.addRow("Tip infill (mm):", self.f_tip_infill)
        left_params.addRow("Tail infill (mm):", self.f_tail_infill)
        left_params.addRow("Sidewall width (mm):", self.f_sw_width)
        left_params.addRow("Sidewall overlap (mm):", self.f_sw_overlap)

        # Right column: thickness
        right_params = QFormLayout()
        right_params.addRow("Tip thickness (mm):", self.f_tip_thick)
        right_params.addRow("Underfoot thickness (mm):", self.f_uf_thick)
        right_params.addRow("Tail thickness (mm):", self.f_tail_thick)
        right_params.addRow("Underfoot length (mm):", self.f_uf_length)

        params_lay.addLayout(left_params)
        params_lay.addLayout(right_params)
        root.addWidget(params_group)

        self.btn_update = QPushButton("Update Geometry")
        self.btn_update.setStyleSheet(
            "QPushButton { background: #2a6099; color: white; padding: 6px 20px; "
            "border-radius: 4px; font-weight: bold; }"
            "QPushButton:hover { background: #3a80b9; }"
        )
        root.addWidget(self.btn_update, alignment=Qt.AlignRight)
        root.addStretch()

    # ── Public helpers ────────────────────────────────────────────────────────

    def get_params(self) -> SkiParams:
        return SkiParams(
            tip_infill=self.f_tip_infill.value(),
            tail_infill=self.f_tail_infill.value(),
            sidewall_width=self.f_sw_width.value(),
            sidewall_overlap=self.f_sw_overlap.value(),
            tip_thickness=self.f_tip_thick.value(),
            underfoot_thickness=self.f_uf_thick.value(),
            tail_thickness=self.f_tail_thick.value(),
            underfoot_length=self.f_uf_length.value(),
        )

    def set_params(self, p: SkiParams):
        self.f_tip_infill.setText(str(p.tip_infill))
        self.f_tail_infill.setText(str(p.tail_infill))
        self.f_sw_width.setText(str(p.sidewall_width))
        self.f_sw_overlap.setText(str(p.sidewall_overlap))
        self.f_tip_thick.setText(str(p.tip_thickness))
        self.f_uf_thick.setText(str(p.underfoot_thickness))
        self.f_tail_thick.setText(str(p.tail_thickness))
        self.f_uf_length.setText(str(p.underfoot_length))

    def update_derived(self, geom: SkiGeometry):
        self.lbl_ski_length.setText(f"{geom.ski_length:.1f} mm")
        core_len = geom.core_tail_x - geom.core_tip_x
        self.lbl_core_length.setText(f"{core_len:.1f} mm")
        self.lbl_waist_pos.setText(f"{geom.waist_x:.1f} mm from tip")
        sign = "+" if geom.setback >= 0 else ""
        self.lbl_setback.setText(f"{sign}{geom.setback:.1f} mm")

        # Compute widths at key positions
        tip_width = geom.width_at_y(geom.core_tip_x)
        waist_width = geom.width_at_y(geom.waist_x)
        tail_width = geom.width_at_y(geom.core_tail_x)
        self.lbl_tip_width.setText(f"{tip_width:.1f} mm")
        self.lbl_waist_width.setText(f"{waist_width:.1f} mm")
        self.lbl_tail_width.setText(f"{tail_width:.1f} mm")


# ── Tab widget ────────────────────────────────────────────────────────────────

class GeometryTab(QWidget):
    def __init__(self):
        super().__init__()
        self._outline: np.ndarray | None = None
        self._geom: SkiGeometry | None = None
        self._build_ui()
        self._connect_signals()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Vertical)

        # Top: visualisation canvas
        self.canvas = SkiCanvas()
        splitter.addWidget(self.canvas)

        # Bottom: parameter panel in a scroll area
        self.panel = ParameterPanel()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.panel)
        splitter.addWidget(scroll)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        layout.addWidget(splitter)

    def _connect_signals(self):
        self.panel.btn_load_dxf.clicked.connect(self._load_dxf)
        self.panel.btn_load_json.clicked.connect(self._load_json)
        self.panel.btn_save_json.clicked.connect(self._save_json)
        self.panel.btn_update.clicked.connect(self._update_geometry)

    def load_test_files(self):
        """Auto-load test DXF and JSON files (for development)."""
        try:
            from pathlib import Path
            data_dir = Path(__file__).parent.parent / "data"
            dxf_path = data_dir / "Ski_planform.dxf"
            json_path = data_dir / "ski_params.json"

            if dxf_path.exists():
                self._outline = load_planform_dxf(str(dxf_path))
                self.panel.lbl_dxf.setText(dxf_path.name)
                self.panel.lbl_dxf.setStyleSheet("color: #60cc60;")

            if json_path.exists():
                params = SkiParams.from_json(str(json_path))
                self.panel.set_params(params)

            if self._outline is not None:
                self._update_geometry()
        except Exception:
            pass  # Silent fail for test files

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _load_dxf(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Planform DXF", "", "DXF Files (*.dxf);;All Files (*)"
        )
        if not path:
            return
        try:
            self._outline = load_planform_dxf(path)
            self.panel.lbl_dxf.setText(Path(path).name)
            self.panel.lbl_dxf.setStyleSheet("color: #60cc60;")
            self._update_geometry()
        except Exception as exc:
            QMessageBox.critical(self, "DXF Error", str(exc))

    def _load_json(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Parameters JSON", "", "JSON Files (*.json);;All Files (*)"
        )
        if not path:
            return
        try:
            params = SkiParams.from_json(path)
            self.panel.set_params(params)
            self._update_geometry()
        except Exception as exc:
            QMessageBox.critical(self, "JSON Error", str(exc))

    def _save_json(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Parameters JSON", "ski_params.json",
            "JSON Files (*.json);;All Files (*)"
        )
        if not path:
            return
        try:
            self.panel.get_params().to_json(path)
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", str(exc))

    def _update_geometry(self):
        if self._outline is None:
            return
        try:
            params = self.panel.get_params()
            self._geom = compute_geometry(self._outline, params)
            self.panel.update_derived(self._geom)
            self.canvas.plot_geometry(self._outline, self._geom, params)
            # Notify parent window that geometry is ready
            parent = self.parent()
            while parent:
                if hasattr(parent, '_check_geometry_loaded'):
                    parent._check_geometry_loaded()
                    break
                parent = parent.parent()
        except Exception as exc:
            QMessageBox.critical(self, "Geometry Error", str(exc))
