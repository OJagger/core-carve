"""Tab — Camber Design: vertical ski shape with rocker and camber."""
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

from core_carve.camber_design import CamberParams, compute_camber_line


# ── Canvas ────────────────────────────────────────────────────────────────────

class CamberCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(facecolor="#1e1e1e", figsize=(12, 6))
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ax = None
        self._setup_axes()
        self._ski_length = 1800.0
        self._dragging_idx = None
        self._mpl_connect_events()

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

    def _mpl_connect_events(self):
        """Connect mouse events for interactive control points."""
        self.mpl_connect("button_press_event", self._on_press)
        self.mpl_connect("button_release_event", self._on_release)
        self.mpl_connect("motion_notify_event", self._on_motion)

    def _on_press(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        self._dragging_idx = None

    def _on_release(self, event):
        self._dragging_idx = None

    def _on_motion(self, event):
        if event.inaxes != self.ax or self._dragging_idx is None or event.xdata is None:
            return

    def plot_camber(self, params: CamberParams, ski_length: float = None):
        """Plot camber line with control points and constraint markers."""
        if ski_length is None:
            ski_length = self._ski_length

        self._setup_axes()
        ax = self.ax

        # Compute camber line
        y_points, z_points = compute_camber_line(ski_length, params)

        # Plot camber line
        ax.plot(y_points, z_points, color="#60cc60", linewidth=2, label="Camber line")

        # Plot reference line (snow contact points)
        ax.axhline(y=0, color="#999999", linestyle="-", linewidth=1, alpha=0.5, label="Snow contact")

        # Plot key feature points
        tip_end_y = params.tip_rocker_length
        tail_start_y = ski_length - params.tail_rocker_length
        center_y = ski_length / 2.0

        # Mark rocker and camber boundaries with dashed offset lines
        ax.axvline(x=tip_end_y, color="#ffaa00", linestyle="--", linewidth=1, alpha=0.5, label="Rocker/camber boundary")
        ax.axvline(x=tail_start_y, color="#ffaa00", linestyle="--", linewidth=1, alpha=0.5)

        # Plot control points and constraint markers
        # Tip rocker endpoints
        ax.plot(0, params.tip_rocker_height, "ro", markersize=8, label="Tip rocker control")
        ax.plot(tip_end_y, 0, "bs", markersize=8, label="Rocker/camber junction (z=0, constrained)")

        # Camber section
        ax.plot(center_y, params.camber_amount, "g^", markersize=8, label="Camber peak")

        # Tail rocker endpoints
        ax.plot(tail_start_y, 0, "bs", markersize=8)
        ax.plot(ski_length, params.tail_rocker_height, "ro", markersize=8, label="Tail rocker control")

        # Draw control arms (spline control structure) - horizontal constraints at boundaries
        # Tip rocker: shows the control arm is horizontal at the boundary
        ax.arrow(tip_end_y - 20, 0, 40, 0, head_width=1, head_length=10, fc="#ff8800", ec="#ff8800", alpha=0.4, linestyle='--')
        ax.text(tip_end_y, -3, "Horizontal\nconstraint", ha="center", fontsize=7, color="#ff8800")

        # Tail rocker: horizontal constraint at boundary
        ax.arrow(tail_start_y - 20, 0, 40, 0, head_width=1, head_length=10, fc="#ff8800", ec="#ff8800", alpha=0.4, linestyle='--')

        # Annotations
        ax.text(0, params.tip_rocker_height + 2, f"Tip: {params.tip_rocker_height:.0f}mm", ha="center", fontsize=8, color="#cccccc")
        ax.text(ski_length, params.tail_rocker_height + 2, f"Tail: {params.tail_rocker_height:.0f}mm", ha="center", fontsize=8, color="#cccccc")
        ax.text(center_y, params.camber_amount + 2, f"Camber: {params.camber_amount:.1f}mm", ha="center", fontsize=8, color="#cccccc")

        ax.set_xlim(-50, ski_length + 50)
        ax.set_ylim(-10, max(params.tip_rocker_height, params.camber_amount, params.tail_rocker_height) + 10)
        ax.set_xlabel("Along ski (mm)")
        ax.set_ylabel("Vertical height (mm)")
        ax.set_title("Camber Profile (Side View)")
        ax.grid(True, alpha=0.2, color="#555555")
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, facecolor="#333333", labelcolor="#dddddd")

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


class CamberParameterPanel(QWidget):
    def __init__(self):
        super().__init__()
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(6)

        # ── Tip rocker ───────────────────────────────────────────────────────
        tip_group = QGroupBox("Tip Rocker")
        tip_lay = QFormLayout(tip_group)

        self.f_tip_length = _FloatField(150.0)
        self.f_tip_height = _FloatField(30.0)

        tip_lay.addRow("Rocker length (mm):", self.f_tip_length)
        tip_lay.addRow("Rocker height (mm):", self.f_tip_height)
        root.addWidget(tip_group)

        # ── Camber underfoot ─────────────────────────────────────────────────
        camber_group = QGroupBox("Camber Underfoot")
        camber_lay = QFormLayout(camber_group)

        self.f_camber_amount = _FloatField(20.0)

        camber_lay.addRow("Camber amount (mm):", self.f_camber_amount)
        root.addWidget(camber_group)

        # ── Tail rocker ──────────────────────────────────────────────────────
        tail_group = QGroupBox("Tail Rocker")
        tail_lay = QFormLayout(tail_group)

        self.f_tail_length = _FloatField(150.0)
        self.f_tail_height = _FloatField(30.0)

        tail_lay.addRow("Rocker length (mm):", self.f_tail_length)
        tail_lay.addRow("Rocker height (mm):", self.f_tail_height)
        root.addWidget(tail_group)

        # ── Buttons ──────────────────────────────────────────────────────────
        button_lay = QHBoxLayout()
        self.btn_load_params = QPushButton("Load params…")
        self.btn_save_params = QPushButton("Save params…")

        button_lay.addWidget(self.btn_load_params)
        button_lay.addWidget(self.btn_save_params)
        button_lay.addStretch()
        root.addLayout(button_lay)

        self.lbl_status = QLabel("✓ Ready")
        self.lbl_status.setStyleSheet("color: #60cc60;")
        root.addWidget(self.lbl_status)

        root.addStretch()

    def get_params(self) -> CamberParams:
        return CamberParams(
            tip_rocker_length=self.f_tip_length.value(),
            tip_rocker_height=self.f_tip_height.value(),
            camber_amount=self.f_camber_amount.value(),
            tail_rocker_length=self.f_tail_length.value(),
            tail_rocker_height=self.f_tail_height.value(),
        )

    def set_params(self, p: CamberParams):
        self.f_tip_length.setText(str(p.tip_rocker_length))
        self.f_tip_height.setText(str(p.tip_rocker_height))
        self.f_camber_amount.setText(str(p.camber_amount))
        self.f_tail_length.setText(str(p.tail_rocker_length))
        self.f_tail_height.setText(str(p.tail_rocker_height))


# ── Tab widget ────────────────────────────────────────────────────────────────

class CamberTab(QWidget):
    def __init__(self, ski_length: float = 1800.0):
        super().__init__()
        self._ski_length = ski_length
        self._build_ui()
        self._connect_signals()
        self._update_preview()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Vertical)

        # Top: canvas
        self.canvas = CamberCanvas()
        splitter.addWidget(self.canvas)

        # Bottom: parameter panel
        self.panel = CamberParameterPanel()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.panel)
        splitter.addWidget(scroll)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter)

    def _connect_signals(self):
        # Parameters → preview
        self.panel.f_tip_length.textChanged.connect(self._update_preview)
        self.panel.f_tip_height.textChanged.connect(self._update_preview)
        self.panel.f_camber_amount.textChanged.connect(self._update_preview)
        self.panel.f_tail_length.textChanged.connect(self._update_preview)
        self.panel.f_tail_height.textChanged.connect(self._update_preview)

        # Buttons
        self.panel.btn_load_params.clicked.connect(self._load_params)
        self.panel.btn_save_params.clicked.connect(self._save_params)

    def set_ski_length(self, ski_length: float):
        """Update ski length (from geometry)."""
        self._ski_length = ski_length
        self._update_preview()

    def _update_preview(self):
        """Redraw camber preview."""
        try:
            params = self.panel.get_params()
            self.canvas.plot_camber(params, self._ski_length)
            self.panel.lbl_status.setText("✓ Camber design ready")
            self.panel.lbl_status.setStyleSheet("color: #60cc60;")
        except Exception as e:
            self.panel.lbl_status.setText(f"✗ Error: {str(e)}")
            self.panel.lbl_status.setStyleSheet("color: #ff6060;")

    def _load_params(self):
        """Load camber params from JSON."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Parameters", "", "JSON Files (*.json);;All Files (*)"
        )
        if not path:
            return

        try:
            params = CamberParams.from_json(path)
            self.panel.set_params(params)
            self._update_preview()
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))

    def _save_params(self):
        """Save camber params to JSON."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Parameters", "camber_params.json",
            "JSON Files (*.json);;All Files (*)"
        )
        if not path:
            return

        try:
            self.panel.get_params().to_json(path)
            QMessageBox.information(self, "Saved", f"Parameters saved to {Path(path).name}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))
