"""Tab 2 — Core Blank Design: blank size, core positioning, and CNC layout."""
from __future__ import annotations

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QLabel, QLineEdit, QComboBox, QGroupBox, QFormLayout, QScrollArea,
    QSizePolicy,
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as patches

from core_carve.core_blank import CoreBlank, MachineOrientation, OriginCorner
from core_carve.ski_geometry import SkiGeometry


# ── Matplotlib canvas ──────────────────────────────────────────────────────────

class BlankCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(facecolor="#1e1e1e", figsize=(10, 6))
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

    def plot_blank_layout(self, blank: CoreBlank, geom: SkiGeometry, params):
        """Visualize the blank, cores, and sidewalls."""
        self._setup_axes()
        ax = self.ax

        # Core extends 25mm beyond the infill regions, but clamp to outline bounds
        outline_y_min = geom.outline[:, 1].min()
        outline_y_max = geom.outline[:, 1].max()
        core_start = max(geom.core_tip_x - 25.0, outline_y_min)
        core_end = min(geom.core_tail_x + 25.0, outline_y_max)
        core_extent = core_end - core_start

        # Center the core horizontally in the blank
        x_offset = (blank.length - core_extent) / 2 - core_start

        # --- Blank outline
        rect_blank = patches.Rectangle(
            (0, -blank.width / 2), blank.length, blank.width,
            linewidth=2, edgecolor="#80c0ff", facecolor="none", label="Blank"
        )
        ax.add_patch(rect_blank)

        # --- Machine origin
        origin_markers = {
            OriginCorner.BOTTOM_LEFT: (0, -blank.width / 2),
            OriginCorner.BOTTOM_RIGHT: (blank.length, -blank.width / 2),
            OriginCorner.TOP_LEFT: (0, blank.width / 2),
            OriginCorner.TOP_RIGHT: (blank.length, blank.width / 2),
        }
        origin_x, origin_y = origin_markers[blank.origin_corner]
        ax.plot(origin_x, origin_y, "r+", markersize=15, markeredgewidth=2,
                label="Machine origin (0,0)")

        # Get core positions
        positions = blank.get_core_positions(geom, params)

        # Sample ski outline to draw realistic core/sidewall shapes
        from core_carve.ski_geometry import half_widths_at_y
        y_samples = np.linspace(core_start, core_end, 100)
        y_samples_offset = y_samples + x_offset
        left_outline, right_outline = half_widths_at_y(geom.outline, y_samples)

        # Core edges: offset inward by (sidewall_width - sidewall_overlap)
        core_offset = params.sidewall_width - params.sidewall_overlap

        # Trim line positions (where core will be trimmed to final dimension)
        trim_tip = geom.core_tip_x + x_offset
        trim_tail = geom.core_tail_x + x_offset

        # Track min distance to blank edges for validation
        min_distance = float('inf')

        # --- Draw cores and sidewalls
        for i, (core_x, core_y) in enumerate(positions):
            # Core edges (inboard of the ski outline by the offset)
            core_left_edge = left_outline + core_offset + core_y
            core_right_edge = right_outline - core_offset + core_y

            # Sidewall outer edges
            sw_left_outer = left_outline + core_y
            sw_right_outer = right_outline + core_y

            # Draw core outline
            core_x_poly = np.concatenate([y_samples_offset, y_samples_offset[::-1]])
            core_y_poly = np.concatenate([core_left_edge, core_right_edge[::-1]])
            poly_core = patches.Polygon(
                list(zip(core_x_poly, core_y_poly)),
                linewidth=1.5, edgecolor="#60cc60", facecolor="none",
                label="Core" if i == 0 else ""
            )
            ax.add_patch(poly_core)

            # Draw sidewalls
            poly_sw_left = patches.Polygon(
                list(zip(np.concatenate([y_samples_offset, y_samples_offset[::-1]]),
                        np.concatenate([sw_left_outer, core_left_edge[::-1]]))),
                linewidth=0.5, edgecolor="#f0a040", facecolor="#f0a040",
                alpha=0.3, label="Sidewall" if i == 0 else ""
            )
            ax.add_patch(poly_sw_left)

            poly_sw_right = patches.Polygon(
                list(zip(np.concatenate([y_samples_offset, y_samples_offset[::-1]]),
                        np.concatenate([sw_right_outer, core_right_edge[::-1]]))),
                linewidth=0.5, edgecolor="#f0a040", facecolor="#f0a040",
                alpha=0.3
            )
            ax.add_patch(poly_sw_right)

            # Calculate min distance to blank edges
            dist_left = min(sw_left_outer) - (-blank.width / 2)
            dist_right = (blank.width / 2) - max(sw_right_outer)
            dist_edges = min(dist_left, dist_right)
            min_distance = min(min_distance, dist_edges)

        # --- Trim lines (between outer sidewall edges)
        if len(positions) == 1:
            # Single core: trim line spans outer sidewalls
            core_y = positions[0][1]
            sw_left = left_outline + core_y
            sw_right = right_outline + core_y
            y_min = np.nanmin(sw_left)
            y_max = np.nanmax(sw_right)
            ax.plot([trim_tip, trim_tip], [y_min, y_max], color="#ff8800", linewidth=1.5,
                   linestyle="--", label="Trim lines")
            ax.plot([trim_tail, trim_tail], [y_min, y_max], color="#ff8800", linewidth=1.5, linestyle="--")
        else:
            # Two cores: single trim line across both
            all_left = []
            all_right = []
            for core_y in [p[1] for p in positions]:
                all_left.extend(left_outline + core_y)
                all_right.extend(right_outline + core_y)
            y_min = np.nanmin(all_left)
            y_max = np.nanmax(all_right)
            ax.plot([trim_tip, trim_tip], [y_min, y_max], color="#ff8800", linewidth=1.5,
                   linestyle="--", label="Trim lines")
            ax.plot([trim_tail, trim_tail], [y_min, y_max], color="#ff8800", linewidth=1.5, linestyle="--")

        # --- Layout
        ax.set_xlim(-50, blank.length + 50)
        ax.set_ylim(-blank.width / 2.0 - 50, blank.width / 2.0 + 50)
        ax.set_aspect("equal")
        ax.set_xlabel("Along ski (mm)")
        ax.set_ylabel("Across ski (mm)")
        ax.set_title("Blank Layout (Top View)")
        ax.grid(True, alpha=0.2, color="#555555")
        ax.legend(fontsize=8, facecolor="#333333", labelcolor="#dddddd",
                 loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=5, frameon=True)

        self.fig.tight_layout(pad=2.0)
        self.draw()

        return min_distance if min_distance != float('inf') else None


# ── Parameter input panel ─────────────────────────────────────────────────────

class _FloatField(QLineEdit):
    def __init__(self, default: float):
        super().__init__(str(default))
        self.setFixedWidth(90)

    def value(self) -> float:
        return float(self.text())


class BlankParameterPanel(QWidget):
    def __init__(self, geom: SkiGeometry, parent=None):
        super().__init__(parent)
        self.geom = geom
        self._calculated_spacing = None
        self._build_ui()
        self._calculate_default_spacing()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(6)

        # ── Blank dimensions ──────────────────────────────────────────────────
        blank_group = QGroupBox("Blank Dimensions")
        blank_lay = QFormLayout(blank_group)

        self.f_blank_length = _FloatField(2000.0)
        self.f_blank_width = _FloatField(200.0)
        self.f_blank_thickness = _FloatField(15.0)

        blank_lay.addRow("Length (mm):", self.f_blank_length)
        blank_lay.addRow("Width (mm):", self.f_blank_width)
        blank_lay.addRow("Thickness (mm):", self.f_blank_thickness)
        root.addWidget(blank_group)

        # ── Core positioning ──────────────────────────────────────────────────
        pos_group = QGroupBox("Core Positioning")
        pos_lay = QHBoxLayout(pos_group)

        # Left column: number of cores, machine orientation
        left_pos = QFormLayout()
        self.combo_num_cores = QComboBox()
        self.combo_num_cores.addItem("1", 1)
        self.combo_num_cores.addItem("2", 2)

        self.combo_orientation = QComboBox()
        self.combo_orientation.addItem("X axis", MachineOrientation.X_AXIS)
        self.combo_orientation.addItem("Y axis", MachineOrientation.Y_AXIS)

        self.combo_origin = QComboBox()
        self.combo_origin.addItem("Bottom-left", OriginCorner.BOTTOM_LEFT)
        self.combo_origin.addItem("Bottom-right", OriginCorner.BOTTOM_RIGHT)
        self.combo_origin.addItem("Top-left", OriginCorner.TOP_LEFT)
        self.combo_origin.addItem("Top-right", OriginCorner.TOP_RIGHT)

        left_pos.addRow("Number of cores:", self.combo_num_cores)
        left_pos.addRow("Machine orientation:", self.combo_orientation)
        left_pos.addRow("Origin corner:", self.combo_origin)

        # Right column: position adjustments and spacing
        right_pos = QFormLayout()
        self.f_pos_offset_x = _FloatField(0.0)
        self.f_pos_offset_y = _FloatField(0.0)
        self.f_core_spacing = _FloatField(120.0)  # Placeholder; updated in _calculate_default_spacing

        right_pos.addRow("Offset along (mm):", self.f_pos_offset_x)
        right_pos.addRow("Offset across (mm):", self.f_pos_offset_y)
        right_pos.addRow("Core spacing (mm):", self.f_core_spacing)

        pos_lay.addLayout(left_pos)
        pos_lay.addLayout(right_pos)
        root.addWidget(pos_group)

        # ── Validation messages ──────────────────────────────────────────────
        self.validation_group = QGroupBox("Validation")
        validation_lay = QVBoxLayout(self.validation_group)
        self.lbl_validation = QLabel("✓ All checks passed")
        self.lbl_validation.setStyleSheet("color: #60cc60;")
        validation_lay.addWidget(self.lbl_validation)
        root.addWidget(self.validation_group)

        root.addStretch()

    def set_blank(self, blank: CoreBlank):
        self.f_blank_length.setText(str(blank.length))
        self.f_blank_width.setText(str(blank.width))
        self.f_blank_thickness.setText(str(blank.thickness))
        idx = self.combo_num_cores.findData(blank.num_cores)
        if idx >= 0:
            self.combo_num_cores.setCurrentIndex(idx)
        idx = self.combo_orientation.findData(blank.machine_orientation)
        if idx >= 0:
            self.combo_orientation.setCurrentIndex(idx)
        idx = self.combo_origin.findData(blank.origin_corner)
        if idx >= 0:
            self.combo_origin.setCurrentIndex(idx)
        self.f_pos_offset_x.setText(str(blank.position_offset_x))
        self.f_pos_offset_y.setText(str(blank.position_offset_y))
        if blank.core_spacing is not None:
            self.f_core_spacing.setText(str(blank.core_spacing))

    def get_blank(self) -> CoreBlank:
        # Use calculated spacing if not explicitly set by user
        spacing = self.f_core_spacing.value()
        if spacing is None or (self._calculated_spacing is not None and
                              abs(spacing - self._calculated_spacing) < 0.1):
            spacing = None

        return CoreBlank(
            length=self.f_blank_length.value(),
            width=self.f_blank_width.value(),
            thickness=self.f_blank_thickness.value(),
            num_cores=self.combo_num_cores.currentData(),
            machine_orientation=self.combo_orientation.currentData(),
            origin_corner=self.combo_origin.currentData(),
            position_offset_x=self.f_pos_offset_x.value(),
            position_offset_y=self.f_pos_offset_y.value(),
            core_spacing=spacing if spacing is not None else None,
        )

    def _calculate_default_spacing(self):
        """Calculate default core spacing based on ski geometry."""
        try:
            from core_carve.ski_geometry import half_widths_at_y
            outline_y_min = self.geom.outline[:, 1].min()
            outline_y_max = self.geom.outline[:, 1].max()
            core_start = max(self.geom.core_tip_x - 25.0, outline_y_min)
            core_end = min(self.geom.core_tail_x + 25.0, outline_y_max)
            y_samples = np.linspace(core_start, core_end, 100)
            left_w, right_w = half_widths_at_y(self.geom.outline, y_samples)
            max_half_width = max(np.nanmax(np.abs(left_w)), np.nanmax(np.abs(right_w)))
            self._calculated_spacing = 15.0 + 2.0 * max_half_width
            # Round to nearest 10mm
            self._calculated_spacing = round(self._calculated_spacing / 10.0) * 10.0
            self.f_core_spacing.setText(str(self._calculated_spacing))
        except Exception:
            self._calculated_spacing = None

    def update_spacing_visibility(self):
        """Show/hide core spacing field based on number of cores."""
        is_two_cores = self.combo_num_cores.currentData() == 2
        self.f_core_spacing.setVisible(is_two_cores)

    def show_validation(self, blank: CoreBlank, params, min_distance=None):
        is_valid, warnings = blank.validate(self.geom, params)
        msgs = []
        color = "#60cc60"

        if not is_valid or (min_distance is not None and min_distance < 15):
            color = "#ffa040"

        if is_valid and (min_distance is None or min_distance >= 15):
            msgs.append("✓ All checks passed")
        else:
            if warnings:
                msgs.extend(f"• {w}" for w in warnings)

        # Only show margin warning if < 15mm
        if min_distance is not None and min_distance < 15:
            msgs.append(f"• Min margin to blank edge: {min_distance:.1f} mm ⚠")

        # Check core gap if 2 cores
        if blank.num_cores == 2:
            gap = self._calculate_core_gap(blank, params)
            if gap is not None and gap < 15:
                msgs.append(f"• Min gap between cores: {gap:.1f} mm ⚠")
                color = "#ffa040"

        text = "\n".join(msgs) if msgs else "✓ All checks passed"
        self.lbl_validation.setText(text)
        self.lbl_validation.setStyleSheet(f"color: {color};")

    def _calculate_core_gap(self, blank, params):
        """Calculate minimum gap between core sidewalls."""
        try:
            from core_carve.ski_geometry import half_widths_at_y
            core_start = blank.get_core_positions(self.geom, params)
            core_x, _ = core_start[0]

            # Sample ski outline at core region
            y_samples = np.linspace(self.geom.core_tip_x - 25, self.geom.core_tail_x + 25, 100)
            left_w, right_w = half_widths_at_y(self.geom.outline, y_samples)

            # Gap = spacing - 2*max_half_width
            spacing = blank.core_spacing if blank.core_spacing is not None else self._calculated_spacing
            if spacing is None:
                return None

            max_half_width = max(np.nanmax(np.abs(left_w)), np.nanmax(np.abs(right_w)))
            gap = spacing - 2.0 * max_half_width
            return gap
        except Exception:
            return None


# ── Tab widget ────────────────────────────────────────────────────────────────

class BlankTab(QWidget):
    def __init__(self, geom: SkiGeometry, params):
        super().__init__()
        self.geom = geom
        self.params = params
        self._blank: CoreBlank | None = None
        self._build_ui()
        self._connect_signals()
        self._update_layout()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.splitter = QSplitter(Qt.Vertical)

        # Top: visualization canvas
        self.canvas = BlankCanvas()
        self.splitter.addWidget(self.canvas)

        # Bottom: parameter panel in a scroll area
        self.panel = BlankParameterPanel(self.geom)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.panel)
        self.splitter.addWidget(scroll)

        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 2)

        layout.addWidget(self.splitter)

    def _connect_signals(self):
        self.panel.combo_num_cores.currentIndexChanged.connect(self._on_num_cores_changed)
        self.panel.f_blank_length.textChanged.connect(self._update_layout)
        self.panel.f_blank_width.textChanged.connect(self._update_layout)
        self.panel.f_blank_thickness.textChanged.connect(self._update_layout)
        self.panel.combo_orientation.currentIndexChanged.connect(self._update_layout)
        self.panel.combo_origin.currentIndexChanged.connect(self._update_layout)
        self.panel.f_pos_offset_x.textChanged.connect(self._update_layout)
        self.panel.f_pos_offset_y.textChanged.connect(self._update_layout)
        self.panel.f_core_spacing.textChanged.connect(self._update_layout)

    def _on_num_cores_changed(self):
        self.panel.update_spacing_visibility()
        self._update_layout()

    def _update_layout(self):
        try:
            blank = self.panel.get_blank()
            self._blank = blank
            min_distance = self.canvas.plot_blank_layout(blank, self.geom, self.params)
            self.panel.show_validation(blank, self.params, min_distance)
        except Exception as e:
            self.panel.lbl_validation.setText(f"✗ Error: {str(e)}")
            self.panel.lbl_validation.setStyleSheet("color: #ff6060;")
