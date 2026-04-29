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

from core_carve.camber_design import CamberParams, compute_camber_line, bezier_control_points


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
        self._drag_target: str | None = None
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

    # Draggable control point names and the field they update
    # (target_name, panel_field_name, axis: 'y' or 'z')
    _DRAG_TARGETS = [
        ("seg1_p1", "f_tip_apex_arm", "y"),    # P1 of tip rocker
        ("seg1_p2", "f_tip_junc_arm", "y"),    # P2 of tip rocker (arm from junction)
        ("seg1_p3", "f_tip_length", "y"),       # junction (tip_rocker_length)
        ("seg2_p1", "f_camber_arm", "y"),       # P1 of camber rise
        ("seg2_p3", "f_camber_amount", "z"),    # camber peak
        ("seg3_p3", "f_tail_length", "y"),      # tail junction
        ("seg4_p1", "f_tail_junc_arm", "y"),    # P1 of tail rocker
        ("seg4_p2", "f_tail_apex_arm", "y"),    # P2 of tail rocker
    ]

    def _on_press(self, event):
        if event.inaxes != self.ax or self._params is None or event.xdata is None:
            return
        cps = bezier_control_points(self._ski_length, self._params)
        threshold = self._ski_length * 0.03
        best, best_dist = None, threshold
        for name, _field, _axis in self._DRAG_TARGETS:
            py, pz = cps[name]
            # Use x-distance for horizontal targets, combined for z targets
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
        params = self._params
        L = self._ski_length

        y_new = float(np.clip(event.xdata, 0, L))
        tip_junc = params.tip_rocker_length

        if self._drag_target == "seg1_p1":
            # P1 of tip rocker: arm from tip end (y=0)
            arm = max(1.0, min(y_new, tip_junc * 0.9))
            p.f_tip_apex_arm.setText(f"{arm:.1f}")

        elif self._drag_target == "seg1_p2":
            # P2 of tip rocker: arm = tip_junc - y_new
            arm = max(1.0, min(tip_junc - y_new, tip_junc * 0.9))
            p.f_tip_junc_arm.setText(f"{arm:.1f}")

        elif self._drag_target == "seg1_p3":
            # Junction position = tip_rocker_length
            val = max(10.0, min(y_new, L * 0.4))
            p.f_tip_length.setText(f"{val:.1f}")

        elif self._drag_target == "seg2_p1":
            # P1 of camber rise: arm from tip_junc
            tail_junc = L - params.tail_rocker_length
            center_y = (tip_junc + tail_junc) / 2.0
            arm = max(1.0, min(y_new - tip_junc, (center_y - tip_junc) * 0.9))
            p.f_camber_arm.setText(f"{arm:.1f}")

        elif self._drag_target == "seg2_p3":
            # Camber peak: drag vertically
            if event.ydata is not None:
                val = max(0.0, float(event.ydata))
                p.f_camber_amount.setText(f"{val:.1f}")

        elif self._drag_target == "seg3_p3":
            # Tail junction position = tail_rocker_length (measured from tail)
            val = max(10.0, min(L - y_new, L * 0.4))
            p.f_tail_length.setText(f"{val:.1f}")

        elif self._drag_target == "seg4_p1":
            # P1 of tail rocker: arm from tail junction
            tail_junc = L - params.tail_rocker_length
            arm = max(1.0, min(y_new - tail_junc, (L - tail_junc) * 0.9))
            p.f_tail_junc_arm.setText(f"{arm:.1f}")

        elif self._drag_target == "seg4_p2":
            # P2 of tail rocker: arm from tail end (L)
            arm = max(1.0, min(L - y_new, (L - (L - params.tail_rocker_length)) * 0.9))
            p.f_tail_apex_arm.setText(f"{arm:.1f}")

    # ── Drawing ───────────────────────────────────────────────────────────────

    def plot_camber(self, params: CamberParams, ski_length: float | None = None):
        """Plot camber line with Bezier control arms and draggable control points."""
        if ski_length is not None:
            self._ski_length = ski_length
        self._params = params

        self._setup_axes()
        ax = self.ax
        L = self._ski_length

        y_pts, z_pts = compute_camber_line(L, params)
        cps = bezier_control_points(L, params)

        z_max = max(params.tip_rocker_height, params.tail_rocker_height,
                    params.camber_amount, 1.0)

        # Camber curve
        ax.plot(y_pts, z_pts, color="#80c0ff", linewidth=2.0, label="Camber line", zorder=3)

        # Snow contact reference
        ax.axhline(y=0, color="#555555", linewidth=1, alpha=0.8)

        # ── Control arms (dashed, one per segment endpoint pair) ─────────────
        arm_pairs = [
            ("seg1_p0", "seg1_p1"),  # tip end → P1
            ("seg1_p3", "seg1_p2"),  # tip junction → P2
            ("seg2_p0", "seg2_p1"),  # tip junction → P1
            ("seg2_p3", "seg2_p2"),  # peak → P2
            ("seg3_p0", "seg3_p1"),  # peak → P1
            ("seg3_p3", "seg3_p2"),  # tail junction → P2
            ("seg4_p0", "seg4_p1"),  # tail junction → P1
            ("seg4_p3", "seg4_p2"),  # tail end → P2
        ]
        for a_name, b_name in arm_pairs:
            ay, az = cps[a_name]
            by, bz = cps[b_name]
            ax.plot([ay, by], [az, bz], color="#ff6633", lw=0.8, ls="--", alpha=0.7)

        # ── Fixed endpoints (smaller circles) ────────────────────────────────
        for name in ("seg1_p0", "seg4_p3"):
            py, pz = cps[name]
            ax.plot(py, pz, "o", color="#ff6633", markersize=6, zorder=5)

        # ── Draggable control points (filled red circles) ─────────────────────
        draggable_names = {dt[0] for dt in self._DRAG_TARGETS}
        for name in draggable_names:
            py, pz = cps[name]
            ax.plot(py, pz, "o", color="#ff6633", markersize=7, zorder=6)

        # Labels on key points
        ax.annotate(f" {params.tip_rocker_height:.0f} mm", cps["seg1_p0"],
                    fontsize=7, color="#aaaaaa", va="bottom")
        ax.annotate(f" {params.camber_amount:.1f} mm", cps["seg2_p3"],
                    fontsize=7, color="#aaaaaa", va="bottom")
        ax.annotate(f" {params.tail_rocker_height:.0f} mm", cps["seg4_p3"],
                    fontsize=7, color="#aaaaaa", va="bottom")

        ax.set_xlim(-50, L + 100)
        ax.set_ylim(-3, max(z_max * 1.2, 60))
        ax.set_xlabel("Along ski (mm)")
        ax.set_ylabel("Vertical height (mm)")
        ax.set_title("Camber Profile — drag red control points to adjust")
        ax.grid(True, alpha=0.2, color="#555555")
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.18),
                  ncol=2, facecolor="#333333", labelcolor="#dddddd")

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
        self.f_tip_apex_arm = _FloatField(50.0)
        self.f_tip_junc_arm = _FloatField(40.0)
        tip_lay.addRow("Junction pos (mm from tip):", self.f_tip_length)
        tip_lay.addRow("Rocker height (mm):", self.f_tip_height)
        tip_lay.addRow("Apex arm (mm):", self.f_tip_apex_arm)
        tip_lay.addRow("Junction arm (mm):", self.f_tip_junc_arm)
        root.addWidget(tip_group)

        camber_group = QGroupBox("Camber Underfoot")
        camber_lay = QFormLayout(camber_group)
        self.f_camber_amount = _FloatField(5.0)
        self.f_camber_arm = _FloatField(100.0)
        camber_lay.addRow("Camber height (mm):", self.f_camber_amount)
        camber_lay.addRow("Camber arm (mm):", self.f_camber_arm)
        root.addWidget(camber_group)

        tail_group = QGroupBox("Tail Rocker")
        tail_lay = QFormLayout(tail_group)
        self.f_tail_length = _FloatField(150.0)
        self.f_tail_height = _FloatField(20.0)
        self.f_tail_junc_arm = _FloatField(40.0)
        self.f_tail_apex_arm = _FloatField(50.0)
        tail_lay.addRow("Junction pos (mm from tail):", self.f_tail_length)
        tail_lay.addRow("Rocker height (mm):", self.f_tail_height)
        tail_lay.addRow("Junction arm (mm):", self.f_tail_junc_arm)
        tail_lay.addRow("Apex arm (mm):", self.f_tail_apex_arm)
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
            tip_apex_arm=self.f_tip_apex_arm.value(),
            tip_junc_arm=self.f_tip_junc_arm.value(),
            camber_amount=self.f_camber_amount.value(),
            camber_arm=self.f_camber_arm.value(),
            tail_junc_arm=self.f_tail_junc_arm.value(),
            tail_apex_arm=self.f_tail_apex_arm.value(),
            tail_rocker_length=self.f_tail_length.value(),
            tail_rocker_height=self.f_tail_height.value(),
        )

    def set_params(self, p: CamberParams):
        self.f_tip_length.setText(str(p.tip_rocker_length))
        self.f_tip_height.setText(str(p.tip_rocker_height))
        self.f_tip_apex_arm.setText(str(getattr(p, "tip_apex_arm", 50.0)))
        self.f_tip_junc_arm.setText(str(getattr(p, "tip_junc_arm", 40.0)))
        self.f_camber_amount.setText(str(p.camber_amount))
        self.f_camber_arm.setText(str(getattr(p, "camber_arm", 100.0)))
        self.f_tail_junc_arm.setText(str(getattr(p, "tail_junc_arm", 40.0)))
        self.f_tail_apex_arm.setText(str(getattr(p, "tail_apex_arm", 50.0)))
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
            self.panel.f_tip_apex_arm,
            self.panel.f_tip_junc_arm,
            self.panel.f_camber_amount,
            self.panel.f_camber_arm,
            self.panel.f_tail_junc_arm,
            self.panel.f_tail_apex_arm,
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
