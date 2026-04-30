"""Tab — Materials & Stiffness: mass breakdown and material selection."""
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

from core_carve.materials import MaterialDatabase
from core_carve.ski_mechanics import LayupConfig, PlyCfg, compute_mechanics


# ── Canvas ────────────────────────────────────────────────────────────────────

class MassBreakdownCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(facecolor="#1e1e1e", figsize=(12, 4))
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
        self.fig.tight_layout(pad=1.0)
        self.draw()

    def plot_mass_breakdown(self, components_dict: dict, total_mass: float):
        """
        Plot horizontal stacked bar chart showing mass distribution by component.

        Args:
            components_dict: {'component_name': mass_g, ...}
            total_mass: total mass in grams
        """
        self._setup_axes()
        ax = self.ax

        # Component order and colors
        order = ['tip_fill', 'bottom_laminate', 'left_edge', 'core', 'right_edge', 'top_laminate', 'tail_fill', 'topsheet', 'base']
        colors = {
            'tip_fill': '#d4b870',
            'bottom_laminate': '#80c0ff',
            'left_edge': '#a0a0a0',
            'core': '#c8a050',
            'right_edge': '#a0a0a0',
            'top_laminate': '#80c0ff',
            'tail_fill': '#d4b870',
            'topsheet': '#e0e080',
            'base': '#ffffff',
        }

        # Build data in order
        masses = []
        labels = []
        colors_list = []
        for comp in order:
            if comp in components_dict:
                m = components_dict[comp]
                if m > 0:
                    masses.append(m)
                    labels.append(comp)
                    colors_list.append(colors.get(comp, '#888888'))

        if not masses:
            ax.text(0.5, 0.5, "No mass data", ha='center', va='center',
                   transform=ax.transAxes, color='#aaaaaa')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            self.draw()
            return

        # Create horizontal stacked bar
        ax.barh(0, sum(masses), color=colors_list, height=0.4,
               edgecolor='#555555', linewidth=0.5)

        # Label each segment
        x_offset = 0
        for mass, label in zip(masses, labels):
            x_center = x_offset + mass / 2.0
            ax.text(x_center, 0, f"{label}\n{mass:.0f}g",
                   ha='center', va='center', fontsize=7, color='#000000',
                   fontweight='bold')
            x_offset += mass

        ax.set_xlim(0, sum(masses) * 1.05)
        ax.set_ylim(-0.5, 0.5)
        ax.set_ylabel('')
        ax.set_yticks([])
        ax.set_xlabel('Mass (g)', color='#cccccc')
        ax.set_title(f'Mass Breakdown  |  Total: {total_mass:.0f} g', color='#eeeeee')
        ax.grid(True, alpha=0.2, color='#555555', axis='x')

        self.fig.tight_layout(pad=1.0)
        self.draw()


class StiffnessCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(facecolor="#1e1e1e", figsize=(12, 3))
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
        self.fig.tight_layout(pad=1.0)
        self.draw()

    def plot_distributions(self, result):
        """Plot EI and GJ stiffness distributions (reused from camber tab)."""
        self._setup_axes()
        ax = self.ax

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
        ax.set_title("Stiffness Distributions", color="#eeeeee")
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


class MaterialsParameterPanel(QWidget):
    def __init__(self):
        super().__init__()
        self._build_ui()

    def _build_ui(self):
        db = MaterialDatabase()
        self._db = db

        root = QVBoxLayout(self)
        root.setSpacing(6)

        composite_names = ["(none)"] + db.names("composite")
        core_names = db.names("wood_core")
        base_names = db.names("base")
        edge_names = db.names("edge")
        sw_names = db.names("sidewall")

        # ── Component materials ────────────────────────────────────────────────
        comp_group = QGroupBox("Component Materials")
        comp_lay = QFormLayout(comp_group)
        self.cb_core = QComboBox()
        self.cb_core.addItems(core_names)
        self.cb_base = QComboBox()
        self.cb_base.addItems(base_names)
        self.cb_edge = QComboBox()
        self.cb_edge.addItems(edge_names)
        self.cb_sidewall = QComboBox()
        self.cb_sidewall.addItems(sw_names)
        comp_lay.addRow("Core wood:", self.cb_core)
        comp_lay.addRow("Base:", self.cb_base)
        comp_lay.addRow("Edge:", self.cb_edge)
        comp_lay.addRow("Sidewall:", self.cb_sidewall)
        root.addWidget(comp_group)

        # ── Top laminate ──────────────────────────────────────────────────────
        top_group = QGroupBox("Top Laminate")
        top_lay = QFormLayout(top_group)
        self.cb_top1_mat = QComboBox()
        self.cb_top1_mat.addItems(composite_names)
        self.f_top1_angle = _FloatField(0.0)
        self.sb_top1_n = QSpinBox()
        self.sb_top1_n.setRange(1, 8)
        self.sb_top1_n.setValue(1)
        top_row1 = QHBoxLayout()
        top_row1.addWidget(self.cb_top1_mat)
        top_row1.addWidget(QLabel("θ°:"))
        top_row1.addWidget(self.f_top1_angle)
        top_row1.addWidget(QLabel("n:"))
        top_row1.addWidget(self.sb_top1_n)
        top_lay.addRow("Layer 1:", top_row1)
        self.chk_top2 = QCheckBox("Layer 2")
        self.chk_top2.setChecked(False)
        self.cb_top2_mat = QComboBox()
        self.cb_top2_mat.addItems(composite_names)
        self.f_top2_angle = _FloatField(45.0)
        self.sb_top2_n = QSpinBox()
        self.sb_top2_n.setRange(1, 8)
        self.sb_top2_n.setValue(1)
        top_row2 = QHBoxLayout()
        top_row2.addWidget(self.chk_top2)
        top_row2.addWidget(self.cb_top2_mat)
        top_row2.addWidget(QLabel("θ°:"))
        top_row2.addWidget(self.f_top2_angle)
        top_row2.addWidget(QLabel("n:"))
        top_row2.addWidget(self.sb_top2_n)
        top_lay.addRow("", top_row2)
        root.addWidget(top_group)

        # ── Bottom laminate ────────────────────────────────────────────────────
        bot_group = QGroupBox("Bottom Laminate")
        bot_lay = QFormLayout(bot_group)
        self.chk_mirror = QCheckBox("Mirror top laminate")
        self.chk_mirror.setChecked(True)
        bot_lay.addRow("", self.chk_mirror)
        self.cb_bot1_mat = QComboBox()
        self.cb_bot1_mat.addItems(composite_names)
        self.f_bot1_angle = _FloatField(0.0)
        self.sb_bot1_n = QSpinBox()
        self.sb_bot1_n.setRange(1, 8)
        self.sb_bot1_n.setValue(1)
        bot_row1 = QHBoxLayout()
        bot_row1.addWidget(self.cb_bot1_mat)
        bot_row1.addWidget(QLabel("θ°:"))
        bot_row1.addWidget(self.f_bot1_angle)
        bot_row1.addWidget(QLabel("n:"))
        bot_row1.addWidget(self.sb_bot1_n)
        bot_lay.addRow("Layer 1:", bot_row1)
        self.chk_bot2 = QCheckBox("Layer 2")
        self.chk_bot2.setChecked(False)
        self.cb_bot2_mat = QComboBox()
        self.cb_bot2_mat.addItems(composite_names)
        self.f_bot2_angle = _FloatField(45.0)
        self.sb_bot2_n = QSpinBox()
        self.sb_bot2_n.setRange(1, 8)
        self.sb_bot2_n.setValue(1)
        bot_row2 = QHBoxLayout()
        bot_row2.addWidget(self.chk_bot2)
        bot_row2.addWidget(self.cb_bot2_mat)
        bot_row2.addWidget(QLabel("θ°:"))
        bot_row2.addWidget(self.f_bot2_angle)
        bot_row2.addWidget(QLabel("n:"))
        bot_row2.addWidget(self.sb_bot2_n)
        bot_lay.addRow("", bot_row2)
        root.addWidget(bot_group)

        # Toggle bottom widgets visibility with mirror checkbox
        def _toggle_mirror(checked):
            for w in (self.cb_bot1_mat, self.f_bot1_angle, self.sb_bot1_n,
                      self.chk_bot2, self.cb_bot2_mat, self.f_bot2_angle, self.sb_bot2_n):
                w.setVisible(not checked)
        self.chk_mirror.toggled.connect(_toggle_mirror)
        _toggle_mirror(True)

        # ── Topsheet ───────────────────────────────────────────────────────────
        topsheet_group = QGroupBox("Topsheet")
        topsheet_lay = QFormLayout(topsheet_group)
        self.f_topsheet_mass = _FloatField(50.0)
        topsheet_lay.addRow("Mass per unit area (g/m²):", self.f_topsheet_mass)
        root.addWidget(topsheet_group)

        # Calc button
        self.btn_calc_mechanics = QPushButton("Calculate")
        root.addWidget(self.btn_calc_mechanics)

        self.lbl_status = QLabel("✓ Ready")
        self.lbl_status.setStyleSheet("color: #60cc60;")
        root.addWidget(self.lbl_status)

        root.addStretch()

    def get_layup(self) -> LayupConfig:
        def _ply(cb_mat, f_angle, sb_n, enabled=True) -> PlyCfg:
            name = cb_mat.currentText()
            if name == "(none)":
                name = ""
            return PlyCfg(name, f_angle.value(), sb_n.value(), enabled)

        top = [_ply(self.cb_top1_mat, self.f_top1_angle, self.sb_top1_n)]
        if self.chk_top2.isChecked():
            top.append(_ply(self.cb_top2_mat, self.f_top2_angle, self.sb_top2_n))
        bot = [_ply(self.cb_bot1_mat, self.f_bot1_angle, self.sb_bot1_n)]
        if self.chk_bot2.isChecked():
            bot.append(_ply(self.cb_bot2_mat, self.f_bot2_angle, self.sb_bot2_n))
        return LayupConfig(
            top_layers=top,
            bottom_layers=bot,
            mirror_bottom=self.chk_mirror.isChecked(),
            core_material=self.cb_core.currentText(),
            base_material=self.cb_base.currentText(),
            edge_material=self.cb_edge.currentText(),
            sidewall_material=self.cb_sidewall.currentText(),
        )


# ── Tab widget ────────────────────────────────────────────────────────────────

class MaterialsTab(QWidget):
    def __init__(self):
        super().__init__()
        self._geom = None
        self._core_params = None
        self._build_ui()
        self._connect_signals()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.splitter = QSplitter(Qt.Vertical)

        # Stacked canvas: mass breakdown and stiffness
        canvas_container = QWidget()
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(0, 0, 0, 0)

        self.mass_canvas = MassBreakdownCanvas()
        canvas_layout.addWidget(self.mass_canvas)

        self.stiffness_canvas = StiffnessCanvas()
        canvas_layout.addWidget(self.stiffness_canvas)

        self.splitter.addWidget(canvas_container)

        self.panel = MaterialsParameterPanel()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.panel)
        self.splitter.addWidget(scroll)

        self.splitter.setStretchFactor(0, 2)
        self.splitter.setStretchFactor(1, 1)

        layout.addWidget(self.splitter)

    def _connect_signals(self):
        for field in (
            self.panel.f_top1_angle,
            self.panel.f_top2_angle,
            self.panel.f_bot1_angle,
            self.panel.f_bot2_angle,
            self.panel.f_topsheet_mass,
        ):
            field.textChanged.connect(self._update_preview)

        for cb in (
            self.panel.cb_core,
            self.panel.cb_base,
            self.panel.cb_edge,
            self.panel.cb_sidewall,
            self.panel.cb_top1_mat,
            self.panel.cb_top2_mat,
            self.panel.cb_bot1_mat,
            self.panel.cb_bot2_mat,
        ):
            cb.currentIndexChanged.connect(self._update_preview)

        for sb in (
            self.panel.sb_top1_n,
            self.panel.sb_top2_n,
            self.panel.sb_bot1_n,
            self.panel.sb_bot2_n,
        ):
            sb.valueChanged.connect(self._update_preview)

        for chk in (
            self.panel.chk_top2,
            self.panel.chk_bot2,
            self.panel.chk_mirror,
        ):
            chk.toggled.connect(self._update_preview)

        self.panel.btn_calc_mechanics.clicked.connect(self._update_preview)

    def set_geometry(self, geom, core_params):
        """Receive geometry from main window."""
        self._geom = geom
        self._core_params = core_params
        self._update_preview()

    def _update_preview(self):
        try:
            if self._geom is None or self._core_params is None:
                self.panel.lbl_status.setText("✓ Waiting for geometry")
                self.panel.lbl_status.setStyleSheet("color: #aaaaaa;")
                return

            db = self.panel._db
            layup = self.panel.get_layup()
            result = compute_mechanics(self._geom, self._core_params, layup, db)

            # Compute mass breakdown by component
            # For now, approximate the breakdown based on mass_per_mm
            components = self._compute_mass_breakdown(result, layup)

            self.mass_canvas.plot_mass_breakdown(components, result.total_mass_g)
            self.stiffness_canvas.plot_distributions(result)

            self.panel.lbl_status.setText("✓ Materials design ready")
            self.panel.lbl_status.setStyleSheet("color: #60cc60;")
        except Exception as e:
            self.panel.lbl_status.setText(f"✗ Error: {str(e)}")
            self.panel.lbl_status.setStyleSheet("color: #ff6060;")

    def _compute_mass_breakdown(self, result, layup) -> dict:
        """
        Estimate mass breakdown by component based on the mechanics result
        and geometry + materials.
        """
        from core_carve.ski_geometry import half_widths_at_y

        db = self.panel._db
        L = self._geom.ski_length
        y_arr = result.y

        # Get material properties
        core_mat = db.get(layup.core_material)
        base_mat = db.get(layup.base_material)
        edge_mat = db.get(layup.edge_material)
        sw_mat = db.get(layup.sidewall_material)

        rho_core = core_mat.density if core_mat else 400  # kg/m³ → g/mm³ = 1e-6
        rho_base = base_mat.density if base_mat else 0
        rho_sw = sw_mat.density if sw_mat else 0
        t_base = base_mat.thickness if base_mat else 0
        t_sw = sw_mat.thickness if sw_mat else 0
        m_edge = edge_mat.mass_per_length if edge_mat else 0  # g/m

        # Get laminate masses
        from core_carve.ski_mechanics import _laminate_props
        t_top, _, _, m_top_per_mm2 = _laminate_props(layup.top_layers, db)
        bot_layers = layup.effective_bottom()
        t_bot, _, _, m_bot_per_mm2 = _laminate_props(bot_layers, db)

        # Width and core thickness
        left_w, right_w = half_widths_at_y(self._geom.outline, y_arr)
        w_arr = np.abs(right_w - left_w)

        y_core = np.clip(y_arr, self._geom.core_tip_x, self._geom.core_tail_x)
        h_core = self._geom.thickness_at(y_core)
        h_core = np.where((y_arr >= self._geom.core_tip_x) & (y_arr <= self._geom.core_tail_x), h_core, 0.0)

        # Integrate masses
        m_core = np.trapz(rho_core * 1e-6 * w_arr * h_core, y_arr)
        m_top_lam = np.trapz(m_top_per_mm2 * w_arr, y_arr)
        m_bot_lam = np.trapz(m_bot_per_mm2 * w_arr, y_arr)
        m_base = np.trapz(rho_base * 1e-6 * w_arr * t_base, y_arr)
        m_edges = 2.0 * (m_edge / 1000.0) * L  # two edges × length
        m_sidewalls = np.trapz(2.0 * rho_sw * 1e-6 * h_core * t_sw, y_arr)

        # Topsheet: assume uniform area density
        topsheet_mass_per_area = self.panel.f_topsheet_mass.value()  # g/m²
        outline_area = self._geom.outline  # Nx2
        if len(outline_area) > 1:
            # Approximate ski area as average width × length
            avg_width = np.mean(w_arr)
            ski_area = avg_width * L / 1e6  # m²
            m_topsheet = topsheet_mass_per_area * ski_area
        else:
            m_topsheet = 0.0

        # Infill (tip and tail)
        m_tip_fill = 0.0  # Would need to know infill material; skip for now
        m_tail_fill = 0.0

        return {
            'tip_fill': m_tip_fill,
            'bottom_laminate': m_bot_lam,
            'left_edge': m_edges / 2.0,
            'core': m_core,
            'right_edge': m_edges / 2.0,
            'top_laminate': m_top_lam,
            'tail_fill': m_tail_fill,
            'topsheet': m_topsheet,
            'base': m_base,
        }
