"""Tab — Base Design: metal edge layout and DXF export."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QPushButton, QLabel, QLineEdit, QGroupBox, QFormLayout,
    QMessageBox, QScrollArea, QSizePolicy, QFileDialog,
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as patches

from core_carve.base_design import BaseParams, compute_edge_cutouts, export_base_dxf


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
        """Plot ski outline with edge cutout regions (horizontal ski view, top-down)."""
        self._setup_axes()
        ax = self.ax

        if outline is None or len(outline) == 0:
            ax.text(0.5, 0.5, "Load outline from Outline Design tab",
                   ha="center", va="center", transform=ax.transAxes, color="#999")
            self.fig.tight_layout(pad=2.0)
            self.draw()
            return

        # Ski is displayed horizontally: X=along ski, Y=across ski (swapped from outline coords)
        # outline is (x, y) where x=across, y=along
        # So we plot with Y as X-axis and X as Y-axis to show horizontal view

        # Get edge cutout regions
        left_edge, right_edge = compute_edge_cutouts(outline, params)

        # Draw overall ski outline in dashed lines (light color, behind)
        ax.plot(outline[:, 1], outline[:, 0], color="#4488ff", linewidth=1.5, linestyle="--", alpha=0.4, label="Ski outline")

        # Draw base outline (solid blue lines)
        ax.plot(outline[:, 1], outline[:, 0], color="#80c0ff", linewidth=2.5, label="Base outline")

        # Get y range (along ski)
        y_min, y_max = outline[:, 1].min(), outline[:, 1].max()
        tip_line_y = y_min + params.tip_offset
        tail_line_y = y_max - params.tail_offset

        # Overlay dashed lines in cutout regions (left edge)
        if len(left_edge) > 0:
            # Find indices where cutout applies
            mask = (left_edge[:, 1] >= tip_line_y) & (left_edge[:, 1] <= tail_line_y)
            if np.any(mask):
                cutout_y = left_edge[mask, 1]
                left_outline_at_y = np.interp(cutout_y, outline[:, 1], outline[:, 0])
                ax.plot(cutout_y, left_outline_at_y, color="#4488ff", linewidth=1, linestyle="--", alpha=0.5)

        # Overlay dashed lines in cutout regions (right edge)
        if len(right_edge) > 0:
            mask = (right_edge[:, 1] >= tip_line_y) & (right_edge[:, 1] <= tail_line_y)
            if np.any(mask):
                cutout_y = right_edge[mask, 1]
                right_outline_at_y = np.interp(cutout_y, outline[:, 1], outline[:, 0])
                ax.plot(cutout_y, right_outline_at_y, color="#4488ff", linewidth=1, linestyle="--", alpha=0.5)

        # Mark tip and tail offset lines (dashed vertical lines in horizontal view)
        ax.axvline(x=tip_line_y, color="#ffaa00", linestyle="--", linewidth=1, alpha=0.5, label="Tip offset")
        ax.axvline(x=tail_line_y, color="#ffaa00", linestyle="--", linewidth=1, alpha=0.5, label="Tail offset")

        ax.set_aspect("equal")
        ax.set_xlabel("Along ski (mm)")
        ax.set_ylabel("Across ski (mm)")
        ax.set_title("Base Design with Edge Cutouts (Top View)")
        ax.grid(True, alpha=0.2, color="#555555")

        # Move legend below the plot
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

        edge_lay.addRow("Edge width (mm):", self.f_edge_width)
        edge_lay.addRow("Tip offset (mm):", self.f_tip_offset)
        edge_lay.addRow("Tail offset (mm):", self.f_tail_offset)
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

        # ── Buttons ──────────────────────────────────────────────────────────
        button_lay = QHBoxLayout()
        self.btn_export_dxf = QPushButton("Export DXF…")
        self.btn_load_params = QPushButton("Load params…")
        self.btn_save_params = QPushButton("Save params…")

        button_lay.addWidget(self.btn_load_params)
        button_lay.addWidget(self.btn_save_params)
        button_lay.addStretch()
        button_lay.addWidget(self.btn_export_dxf)
        root.addLayout(button_lay)

        self.lbl_status = QLabel("✓ Ready")
        self.lbl_status.setStyleSheet("color: #60cc60;")
        root.addWidget(self.lbl_status)

        root.addStretch()

    def get_params(self) -> BaseParams:
        return BaseParams(
            edge_width=self.f_edge_width.value(),
            tip_offset=self.f_tip_offset.value(),
            tail_offset=self.f_tail_offset.value(),
            base_length=self.f_base_length.value(),
            base_width=self.f_base_width.value(),
            base_thickness=self.f_base_thickness.value(),
        )

    def set_params(self, p: BaseParams):
        self.f_edge_width.setText(str(p.edge_width))
        self.f_tip_offset.setText(str(p.tip_offset))
        self.f_tail_offset.setText(str(p.tail_offset))
        self.f_base_length.setText(str(p.base_length))
        self.f_base_width.setText(str(p.base_width))
        self.f_base_thickness.setText(str(p.base_thickness))


# ── Tab widget ────────────────────────────────────────────────────────────────

class BaseTab(QWidget):
    def __init__(self):
        super().__init__()
        self._outline = None
        self._build_ui()
        self._connect_signals()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Vertical)

        # Top: canvas
        self.canvas = BaseCanvas()
        splitter.addWidget(self.canvas)

        # Bottom: parameter panel
        self.panel = BaseParameterPanel()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.panel)
        splitter.addWidget(scroll)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter)

    def _connect_signals(self):
        # Parameters → preview
        self.panel.f_edge_width.textChanged.connect(self._update_preview)
        self.panel.f_tip_offset.textChanged.connect(self._update_preview)
        self.panel.f_tail_offset.textChanged.connect(self._update_preview)

        # Buttons
        self.panel.btn_export_dxf.clicked.connect(self._export_dxf)
        self.panel.btn_load_params.clicked.connect(self._load_params)
        self.panel.btn_save_params.clicked.connect(self._save_params)

    def set_outline(self, outline):
        """Receive outline from Outline Design tab."""
        self._outline = outline
        self._update_preview()

    def _update_preview(self):
        """Redraw the base design preview."""
        try:
            params = self.panel.get_params()
            self.canvas.plot_base_outline(self._outline, params)
            self.panel.lbl_status.setText("✓ Ready to export")
            self.panel.lbl_status.setStyleSheet("color: #60cc60;")
        except Exception as e:
            self.panel.lbl_status.setText(f"✗ Error: {str(e)}")
            self.panel.lbl_status.setStyleSheet("color: #ff6060;")

    def _export_dxf(self):
        """Export base outline to DXF."""
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
            QMessageBox.information(self, "Exported", f"Base design exported to {Path(path).name}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def _load_params(self):
        """Load base params from JSON."""
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
        """Save base params to JSON."""
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
