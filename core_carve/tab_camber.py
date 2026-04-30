"""Tab — Camber Design: vertical ski shape with rocker and camber."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QPushButton, QLabel, QLineEdit, QGroupBox, QFormLayout,
    QMessageBox, QScrollArea, QSizePolicy, QFileDialog,
    QComboBox, QSpinBox, QCheckBox,
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from core_carve.camber_design import CamberParams, compute_camber_line, bezier_control_points


# ── Canvas ────────────────────────────────────────────────────────────────────

class CamberCanvas(FigureCanvas):
    def __init__(self, tab: "CamberTab"):
        self.fig = Figure(facecolor="#1e1e1e", figsize=(12, 6))
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._tab = tab
        self.ax = None
        self.ax2 = None
        self._ski_length = 1800.0
        self._params: CamberParams | None = None
        self._drag_target: str | None = None
        self._setup_axes()
        self.mpl_connect("button_press_event", self._on_press)
        self.mpl_connect("button_release_event", self._on_release)
        self.mpl_connect("motion_notify_event", self._on_motion)

    def _setup_axes(self):
        self.fig.clear()
        gs = self.fig.add_gridspec(2, 1, height_ratios=[3, 2], hspace=0.45)
        self.ax = self.fig.add_subplot(gs[0])
        self.ax2 = self.fig.add_subplot(gs[1])
        for ax in (self.ax, self.ax2):
            ax.set_facecolor("#2b2b2b")
            ax.tick_params(colors="#cccccc", labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor("#555555")
            ax.xaxis.label.set_color("#cccccc")
            ax.yaxis.label.set_color("#cccccc")
            ax.title.set_color("#eeeeee")
        self.fig.tight_layout(pad=2.0)
        self.draw()

    # ── Drag logic ────────────────────────────────────────────────────────────

    # Draggable point names (all handled explicitly in _on_motion)
    _DRAGGABLE = {
        "seg1_p1",   # tip apex arm — both y and z
        "seg1_p2",   # tip junction arm — y only
        "seg1_p3",   # tip junction position — y only
        "seg2_p1",   # camber rise arm from junction — y only (also controls peak arm)
        "seg2_p3",   # camber peak — z only
        "seg3_p3",   # tail junction position — y only
        "seg4_p1",   # tail junction arm — y only
        "seg4_p2",   # tail apex arm — both y and z
    }

    def _on_press(self, event):
        if event.inaxes != self.ax or self._params is None or event.xdata is None:
            return
        cps = bezier_control_points(self._ski_length, self._params)
        z_scale = max(self._params.tip_rocker_height, self._params.tail_rocker_height,
                      self._params.camber_amount, 1.0)
        threshold = 0.04  # normalised distance threshold
        best, best_dist = None, threshold
        for name in self._DRAGGABLE:
            py, pz = cps[name]
            # Normalise by axis range so y and z contribute equally
            dy = (event.xdata - py) / self._ski_length
            dz = ((event.ydata or pz) - pz) / z_scale
            d = np.sqrt(dy**2 + dz**2)
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
        z_new = float(event.ydata) if event.ydata is not None else 0.0
        tip_junc = params.tip_rocker_length
        tail_junc = L - params.tail_rocker_length
        center_y = (tip_junc + tail_junc) / 2.0
        half_span = max(1.0, center_y - tip_junc)

        if self._drag_target == "seg1_p1":
            # Apex arm — free in both y and z
            arm = float(np.clip(y_new, 1.0, tip_junc * 0.9))
            p.f_tip_apex_arm.setText(f"{arm:.1f}")
            dz = z_new - params.tip_rocker_height
            p.f_tip_apex_arm_dz.setText(f"{dz:.1f}")

        elif self._drag_target == "seg1_p2":
            arm = float(np.clip(tip_junc - y_new, 1.0, tip_junc * 0.9))
            p.f_tip_junc_arm.setText(f"{arm:.1f}")

        elif self._drag_target == "seg1_p3":
            val = float(np.clip(y_new, 10.0, L * 0.4))
            p.f_tip_length.setText(f"{val:.1f}")

        elif self._drag_target == "seg2_p1":
            arm = float(np.clip(y_new - tip_junc, 1.0, half_span * 0.9))
            p.f_camber_arm.setText(f"{arm:.1f}")

        elif self._drag_target == "seg2_p3":
            val = max(0.0, z_new)
            p.f_camber_amount.setText(f"{val:.1f}")

        elif self._drag_target == "seg3_p3":
            val = float(np.clip(L - y_new, 10.0, L * 0.4))
            p.f_tail_length.setText(f"{val:.1f}")

        elif self._drag_target == "seg4_p1":
            arm = float(np.clip(y_new - tail_junc, 1.0, (L - tail_junc) * 0.9))
            p.f_tail_junc_arm.setText(f"{arm:.1f}")

        elif self._drag_target == "seg4_p2":
            # Apex arm — free in both y and z
            arm = float(np.clip(L - y_new, 1.0, (L - tail_junc) * 0.9))
            p.f_tail_apex_arm.setText(f"{arm:.1f}")
            dz = z_new - params.tail_rocker_height
            p.f_tail_apex_arm_dz.setText(f"{dz:.1f}")

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
        z_max = max(
            params.tip_rocker_height, params.tail_rocker_height,
            params.camber_amount,
            cps["seg1_p1"][1], cps["seg4_p2"][1],
            1.0,
        )

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

        # ── Fixed endpoints (smaller circles, cyan) ──────────────────────────
        for name in ("seg1_p0", "seg4_p3"):
            py, pz = cps[name]
            ax.plot(py, pz, "o", color="#00cccc", markersize=8, zorder=5)

        # ── Draggable control points (filled orange circles) ──────────────────
        for name in self._DRAGGABLE:
            py, pz = cps[name]
            ax.plot(py, pz, "o", color="#ff6633", markersize=7, zorder=6)

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

    def plot_distributions(self, result):
        """Plot mass/EI/GJ distributions below the camber line."""
        ax = self.ax2
        ax.clear()
        ax.set_facecolor("#2b2b2b")
        ax.tick_params(colors="#cccccc", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#555555")
        ax.xaxis.label.set_color("#cccccc")
        ax.yaxis.label.set_color("#cccccc")
        ax.title.set_color("#eeeeee")

        y = result.y
        # Plot EI and GJ on twin axes
        ax.plot(y, result.EI / 1e6, color="#80c0ff", linewidth=1.5, label="EI (kN·m²×10³)")
        ax2r = ax.twinx()
        ax2r.tick_params(colors="#ffaa44", labelsize=8)
        ax2r.plot(y, result.GJ / 1e6, color="#ffaa44", linewidth=1.5, linestyle="--", label="GJ (kN·m²×10³)")
        ax2r.yaxis.label.set_color("#ffaa44")
        ax2r.set_ylabel("GJ (N·m²)", color="#ffaa44")
        ax2r.spines["right"].set_edgecolor("#ffaa44")

        ax.set_xlabel("Along ski (mm)")
        ax.set_ylabel("EI (N·mm²)", color="#80c0ff")
        ax.set_title(f"Stiffness distributions  |  Total mass: {result.total_mass_g:.0f} g", color="#eeeeee")
        ax.grid(True, alpha=0.2, color="#555555")

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2r.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right",
                  facecolor="#333333", labelcolor="#dddddd")
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
        self.f_tip_apex_arm_dz = _FloatField(0.0)
        self.f_tip_junc_arm = _FloatField(40.0)
        tip_lay.addRow("Junction pos (mm from tip):", self.f_tip_length)
        tip_lay.addRow("Rocker height (mm):", self.f_tip_height)
        tip_lay.addRow("Apex arm length (mm):", self.f_tip_apex_arm)
        tip_lay.addRow("Apex arm z-offset (mm):", self.f_tip_apex_arm_dz)
        tip_lay.addRow("Junction arm (mm):", self.f_tip_junc_arm)
        root.addWidget(tip_group)

        camber_group = QGroupBox("Camber Underfoot")
        camber_lay = QFormLayout(camber_group)
        self.f_camber_amount = _FloatField(5.0)
        self.f_camber_arm = _FloatField(100.0)
        camber_lay.addRow("Camber height (mm):", self.f_camber_amount)
        camber_lay.addRow("Junction/peak arm (mm):", self.f_camber_arm)
        root.addWidget(camber_group)

        tail_group = QGroupBox("Tail Rocker")
        tail_lay = QFormLayout(tail_group)
        self.f_tail_length = _FloatField(150.0)
        self.f_tail_height = _FloatField(20.0)
        self.f_tail_junc_arm = _FloatField(40.0)
        self.f_tail_apex_arm = _FloatField(50.0)
        self.f_tail_apex_arm_dz = _FloatField(0.0)
        tail_lay.addRow("Junction pos (mm from tail):", self.f_tail_length)
        tail_lay.addRow("Rocker height (mm):", self.f_tail_height)
        tail_lay.addRow("Junction arm (mm):", self.f_tail_junc_arm)
        tail_lay.addRow("Apex arm length (mm):", self.f_tail_apex_arm)
        tail_lay.addRow("Apex arm z-offset (mm):", self.f_tail_apex_arm_dz)
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
            tip_apex_arm_dz=self.f_tip_apex_arm_dz.value(),
            tip_junc_arm=self.f_tip_junc_arm.value(),
            camber_amount=self.f_camber_amount.value(),
            camber_arm=self.f_camber_arm.value(),
            tail_junc_arm=self.f_tail_junc_arm.value(),
            tail_apex_arm=self.f_tail_apex_arm.value(),
            tail_apex_arm_dz=self.f_tail_apex_arm_dz.value(),
            tail_rocker_length=self.f_tail_length.value(),
            tail_rocker_height=self.f_tail_height.value(),
        )

    def set_params(self, p: CamberParams):
        self.f_tip_length.setText(str(p.tip_rocker_length))
        self.f_tip_height.setText(str(p.tip_rocker_height))
        self.f_tip_apex_arm.setText(str(getattr(p, "tip_apex_arm", 50.0)))
        self.f_tip_apex_arm_dz.setText(str(getattr(p, "tip_apex_arm_dz", 0.0)))
        self.f_tip_junc_arm.setText(str(getattr(p, "tip_junc_arm", 40.0)))
        self.f_camber_amount.setText(str(p.camber_amount))
        self.f_camber_arm.setText(str(getattr(p, "camber_arm", 100.0)))
        self.f_tail_junc_arm.setText(str(getattr(p, "tail_junc_arm", 40.0)))
        self.f_tail_apex_arm.setText(str(getattr(p, "tail_apex_arm", 50.0)))
        self.f_tail_apex_arm_dz.setText(str(getattr(p, "tail_apex_arm_dz", 0.0)))
        self.f_tail_length.setText(str(p.tail_rocker_length))
        self.f_tail_height.setText(str(p.tail_rocker_height))



# ── Tab widget ────────────────────────────────────────────────────────────────

class CamberTab(QWidget):
    def __init__(self, ski_length: float = 1800.0):
        super().__init__()
        self._ski_length = ski_length
        self._geom = None
        self._core_params = None
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
            self.panel.f_tip_apex_arm_dz,
            self.panel.f_tip_junc_arm,
            self.panel.f_camber_amount,
            self.panel.f_camber_arm,
            self.panel.f_tail_junc_arm,
            self.panel.f_tail_apex_arm,
            self.panel.f_tail_apex_arm_dz,
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

    def set_geometry(self, geom, core_params):
        self._geom = geom
        self._core_params = core_params
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
