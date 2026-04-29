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

from core_carve.camber_design import CamberParams, compute_camber_line


# ── Canvas ────────────────────────────────────────────────────────────────────

class CamberCanvas(FigureCanvas):
    def __init__(self, tab: "CamberTab"):
        self.fig = Figure(facecolor="#1e1e1e", figsize=(12, 4))
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._tab = tab
        self.ax = None
        self._ski_length = 1800.0
        self._params: CamberParams | None = None
        self._drag_target: str | None = None  # "tip_junc", "tail_junc", "center"
        self._setup_axes()
        self.mpl_connect("button_press_event", self._on_press)
        self.mpl_connect("button_release_event", self._on_release)
        self.mpl_connect("motion_notify_event", self._on_motion)

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

    # ── Drag logic ────────────────────────────────────────────────────────────

    def _control_points(self, params: CamberParams, ski_length: float) -> dict[str, tuple[float, float]]:
        """Return {name: (y, z)} for each draggable control point."""
        tip_junc = params.tip_rocker_length
        tail_junc = ski_length - params.tail_rocker_length
        center_y = (tip_junc + tail_junc) / 2.0
        return {
            "tip_junc": (tip_junc, 0.0),
            "tail_junc": (tail_junc, 0.0),
            "center": (center_y, params.camber_amount),
        }

    def _on_press(self, event):
        if event.inaxes != self.ax or self._params is None or event.xdata is None:
            return
        pts = self._control_points(self._params, self._ski_length)
        threshold = self._ski_length * 0.04
        best, best_dist = None, threshold
        for name, (py, pz) in pts.items():
            # Distance in normalised axes (y spans ski_length, z spans much less — weight y only)
            d = abs(event.xdata - py)
            if d < best_dist:
                best_dist = d
                best = name
        self._drag_target = best

    def _on_release(self, event):
        self._drag_target = None

    def _on_motion(self, event):
        if event.inaxes != self.ax or self._drag_target is None or event.xdata is None:
            return
        p = self._tab.panel
        y_new = float(np.clip(event.xdata, 0, self._ski_length))
        if self._drag_target == "tip_junc":
            max_val = self._ski_length - self._params.tail_rocker_length - 10
            p.f_tip_length.setText(f"{min(y_new, max_val):.1f}")
        elif self._drag_target == "tail_junc":
            min_val = self._params.tip_rocker_length + 10
            from_tail = self._ski_length - max(y_new, min_val)
            p.f_tail_length.setText(f"{from_tail:.1f}")
        elif self._drag_target == "center":
            # Centre point only moves vertically (camber amount)
            if event.ydata is not None:
                z_new = max(0.0, float(event.ydata))
                p.f_camber_amount.setText(f"{z_new:.1f}")

    # ── Drawing ───────────────────────────────────────────────────────────────

    def plot_camber(self, params: CamberParams, ski_length: float | None = None):
        """Plot camber line with draggable control points (ski-outline style)."""
        if ski_length is not None:
            self._ski_length = ski_length
        self._params = params

        self._setup_axes()
        ax = self.ax
        L = self._ski_length

        y_pts, z_pts = compute_camber_line(L, params)
        tip_junc = params.tip_rocker_length
        tail_junc = L - params.tail_rocker_length
        center_y = (tip_junc + tail_junc) / 2.0
        z_max = max(params.tip_rocker_height, params.tail_rocker_height, params.camber_amount, 1.0)
        arm_len = L * 0.04

        # Camber curve
        ax.plot(y_pts, z_pts, color="#80c0ff", linewidth=2.0, label="Camber line", zorder=3)

        # Snow contact reference
        ax.axhline(y=0, color="#555555", linewidth=1, alpha=0.8, label="Snow contact")

        # ── Control arms (dashed lines to draggable points) ───────────────────
        # Tip end → tip junction arm
        ax.plot([0, tip_junc], [params.tip_rocker_height, 0.0],
                color="#ff6633", lw=0.8, ls="--", alpha=0.7)
        # Tip junction horizontal arm
        ax.plot([tip_junc - arm_len, tip_junc + arm_len], [0.0, 0.0],
                color="#ff6633", lw=0.8, ls="--", alpha=0.7)
        # Camber peak horizontal arm
        ax.plot([center_y - arm_len, center_y + arm_len], [params.camber_amount, params.camber_amount],
                color="#ff6633", lw=0.8, ls="--", alpha=0.7)
        # Tail junction horizontal arm
        ax.plot([tail_junc - arm_len, tail_junc + arm_len], [0.0, 0.0],
                color="#ff6633", lw=0.8, ls="--", alpha=0.7)
        # Tail end → tail junction arm
        ax.plot([tail_junc, L], [0.0, params.tail_rocker_height],
                color="#ff6633", lw=0.8, ls="--", alpha=0.7)

        # ── Control points (red circles, matching ski-outline style) ──────────
        # Spline endpoints (tip and tail ends)
        ax.plot(0, params.tip_rocker_height, "o", color="#ff6633", markersize=7, zorder=5)
        ax.plot(L, params.tail_rocker_height, "o", color="#ff6633", markersize=7, zorder=5)

        # Draggable junction points
        ax.plot(tip_junc, 0.0, "o", color="#ff6633", markersize=7, zorder=5,
                label="Drag: junction / peak")
        ax.plot(tail_junc, 0.0, "o", color="#ff6633", markersize=7, zorder=5)
        ax.plot(center_y, params.camber_amount, "o", color="#ff6633", markersize=7, zorder=5)

        # Labels
        ax.annotate(f" {params.tip_rocker_height:.0f} mm", (0, params.tip_rocker_height),
                    fontsize=7, color="#aaaaaa", va="bottom")
        ax.annotate(f" {params.tail_rocker_height:.0f} mm", (L, params.tail_rocker_height),
                    fontsize=7, color="#aaaaaa", va="bottom")
        ax.annotate(f" {params.camber_amount:.1f} mm", (center_y, params.camber_amount),
                    fontsize=7, color="#aaaaaa", va="bottom")

        ax.set_xlim(-50, L + 100)
        # Use at least 60 mm on z axis to avoid over-exaggeration
        ax.set_ylim(-3, max(z_max * 1.2, 60))
        ax.set_xlabel("Along ski (mm)")
        ax.set_ylabel("Vertical height (mm)")
        ax.set_title("Camber Profile — drag red circles to adjust junctions/heights")
        ax.grid(True, alpha=0.2, color="#555555")
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.18),
                  ncol=3, facecolor="#333333", labelcolor="#dddddd")

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

        tip_group = QGroupBox("Tip Rocker")
        tip_lay = QFormLayout(tip_group)
        self.f_tip_length = _FloatField(150.0)
        self.f_tip_height = _FloatField(30.0)
        tip_lay.addRow("Junction position (mm from tip):", self.f_tip_length)
        tip_lay.addRow("Rocker height (mm):", self.f_tip_height)
        root.addWidget(tip_group)

        camber_group = QGroupBox("Camber Underfoot")
        camber_lay = QFormLayout(camber_group)
        self.f_camber_amount = _FloatField(5.0)
        camber_lay.addRow("Camber height (mm):", self.f_camber_amount)
        root.addWidget(camber_group)

        tail_group = QGroupBox("Tail Rocker")
        tail_lay = QFormLayout(tail_group)
        self.f_tail_length = _FloatField(150.0)
        self.f_tail_height = _FloatField(20.0)
        tail_lay.addRow("Junction position (mm from tail):", self.f_tail_length)
        tail_lay.addRow("Rocker height (mm):", self.f_tail_height)
        root.addWidget(tail_group)

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

        self.splitter = QSplitter(Qt.Vertical)

        self.canvas = CamberCanvas(self)
        self.splitter.addWidget(self.canvas)

        self.panel = CamberParameterPanel()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.panel)
        self.splitter.addWidget(scroll)

        self.splitter.setStretchFactor(0, 2)
        self.splitter.setStretchFactor(1, 1)

        layout.addWidget(self.splitter)

    def _connect_signals(self):
        for field in (
            self.panel.f_tip_length,
            self.panel.f_tip_height,
            self.panel.f_camber_amount,
            self.panel.f_tail_length,
            self.panel.f_tail_height,
        ):
            field.textChanged.connect(self._update_preview)

        self.panel.btn_load_params.clicked.connect(self._load_params)
        self.panel.btn_save_params.clicked.connect(self._save_params)
        self._load_ski_callback = None
        self._save_ski_callback = None

    def set_load_ski_callback(self, fn):
        self._load_ski_callback = fn

    def set_save_ski_callback(self, fn):
        self._save_ski_callback = fn

    def set_ski_length(self, ski_length: float):
        self._ski_length = ski_length
        self._update_preview()

    def _update_preview(self):
        try:
            params = self.panel.get_params()
            self.canvas.plot_camber(params, self._ski_length)
            self.panel.lbl_status.setText("✓ Camber design ready")
            self.panel.lbl_status.setStyleSheet("color: #60cc60;")
        except Exception as e:
            self.panel.lbl_status.setText(f"✗ Error: {str(e)}")
            self.panel.lbl_status.setStyleSheet("color: #ff6060;")

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
            params = CamberParams.from_json(path)
            self.panel.set_params(params)
            self._update_preview()
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))

    def _save_params(self):
        if self._save_ski_callback is not None:
            self._save_ski_callback()
            return
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
