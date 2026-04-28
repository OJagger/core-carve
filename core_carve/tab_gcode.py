"""Tab 3 — G-code Generation: slot cutting toolpaths and code export."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QPushButton, QLabel, QLineEdit, QComboBox,
    QGroupBox, QFormLayout, QMessageBox, QScrollArea,
    QSizePolicy, QProgressBar,
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button

from core_carve.gcode_generator import SlotParams, generate_slot_gcode
from core_carve.ski_geometry import SkiGeometry, half_widths_at_y


class _GcodeWorker(QThread):
    finished = pyqtSignal(str, list)
    error = pyqtSignal(str)

    def __init__(self, fn, *args):
        super().__init__()
        self._fn = fn
        self._args = args

    def run(self):
        try:
            result = self._fn(*self._args)
            self.finished.emit(*result)
        except Exception as e:
            self.error.emit(str(e))


# ── Matplotlib canvas ──────────────────────────────────────────────────────────

class GcodeCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(facecolor="#1e1e1e", figsize=(12, 6))
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ax = None
        self._setup_axes()
        self._moves = None
        self._blank = None
        self._playback_idx = 0
        self._playback_speed_multiplier = 1  # 1, 2, 4, 8, 16 for frame skipping
        self._is_playing = False
        self._playback_timer = QTimer()
        self._playback_timer.timeout.connect(self._step_playback)
        self._use_3d = False
        self._3d_press = None
        self._3d_xpress = None
        self._3d_ypress = None

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

    def plot_slot_preview(self, blank, geom, params, slot_params, x_offset=0.0):
        """Show blank and slot outlines."""
        self._setup_axes()
        ax = self.ax

        # Blank outline
        rect_blank = patches.Rectangle(
            (0, -blank.width / 2), blank.length, blank.width,
            linewidth=2, edgecolor="#80c0ff", facecolor="none", label="Blank"
        )
        ax.add_patch(rect_blank)

        # Slot geometry
        core_positions = blank.get_core_positions(geom, params)
        outline_y_min = geom.outline[:, 1].min()
        outline_y_max = geom.outline[:, 1].max()
        core_start = max(geom.core_tip_x - 25.0, outline_y_min)
        core_end = min(geom.core_tail_x + 25.0, outline_y_max)
        y_samples = np.linspace(core_start, core_end, 300)
        y_samples_offset = y_samples + x_offset
        left_outline, right_outline = half_widths_at_y(geom.outline, y_samples)

        core_offset = params.sidewall_width - params.sidewall_overlap

        # Draw slots and tab markers
        for core_idx, (core_x, core_y) in enumerate(core_positions):
            # Left slot edges
            slot_left_outer = left_outline + core_y
            slot_left_inner = left_outline + core_offset + core_y

            # Right slot edges
            slot_right_inner = right_outline - core_offset + core_y
            slot_right_outer = right_outline + core_y

            # Draw slot outline (left)
            ax.plot(y_samples_offset, slot_left_outer, color="#ffa040", linewidth=1, linestyle="-")
            ax.plot(y_samples_offset, slot_left_inner, color="#ffa040", linewidth=1, linestyle="-")

            # Draw slot outline (right)
            ax.plot(y_samples_offset, slot_right_inner, color="#ffa040", linewidth=1, linestyle="-")
            ax.plot(y_samples_offset, slot_right_outer, color="#ffa040", linewidth=1, linestyle="-")

            # Tab markers
            slot_centers = [(y_samples_offset, slot_left_inner), (y_samples_offset, slot_right_inner)]
            for slot_y_samples, slot_y_vals in slot_centers:
                # Skip if any NaN values
                if np.isnan(slot_y_vals).any():
                    continue
                arc_length = np.cumsum(np.concatenate([[0], np.linalg.norm(np.diff(np.column_stack([slot_y_samples, slot_y_vals])), axis=1)]))
                total_arc = arc_length[-1]
                if total_arc > 0:
                    tab_centers = np.arange(0, total_arc, slot_params.tab_spacing)
                    for tc in tab_centers:
                        idx = np.argmin(np.abs(arc_length - tc))
                        ax.plot(slot_y_samples[idx], slot_y_vals[idx], "r.", markersize=4)

        ax.set_xlim(-50, blank.length + 50)
        ax.set_ylim(-blank.width / 2 - 50, blank.width / 2 + 50)
        ax.set_aspect("equal")
        ax.set_xlabel("Along blank (mm)")
        ax.set_ylabel("Across blank (mm)")
        ax.set_title("Slot Layout (Top View)")
        ax.grid(True, alpha=0.2, color="#555555")
        ax.legend(fontsize=8, facecolor="#333333", labelcolor="#dddddd")

        self.fig.tight_layout(pad=2.0)
        self.draw()

    def plot_toolpaths(self, moves):
        """Overlay toolpaths colour-coded by Z depth."""
        if not moves:
            return

        self._setup_axes()
        ax = self.ax

        # Group moves by Z depth
        z_values = [m.z for m in moves]
        z_min, z_max = min(z_values), max(z_values)
        z_range = z_max - z_min if z_max != z_min else 1.0

        # Plot moves
        for i in range(1, len(moves)):
            prev_move = moves[i - 1]
            curr_move = moves[i]
            z_norm = (curr_move.z - z_min) / z_range if z_range > 0 else 0.5

            # Colour gradient: blue (z=z_min, top) to red (z=z_max, bottom)
            color = cm.coolwarm(z_norm)

            linestyle = "--" if curr_move.is_rapid else "-"
            linewidth = 0.5 if curr_move.is_rapid else 1.5

            ax.plot(
                [prev_move.x, curr_move.x],
                [prev_move.y, curr_move.y],
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=0.6
            )

        ax.set_aspect("equal")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_title("Toolpath Visualization")
        ax.grid(True, alpha=0.2, color="#555555")

        self.fig.tight_layout(pad=2.0)
        self.draw()

    def start_playback(self, moves, blank, tool_diameter, speed_multiplier=1):
        """Start animating the cutter along the path."""
        self._moves = moves
        self._blank = blank
        self._tool_diameter = tool_diameter
        # Only reset index if we're at the beginning (first time playing)
        if self._playback_idx == 0 or not self._moves:
            self._playback_idx = 0
        self._playback_speed_multiplier = speed_multiplier
        self._is_playing = True
        self._playback_timer.start(50)  # Fixed 50ms interval; speed controlled by frame skipping

    def pause_playback(self):
        self._is_playing = False
        self._playback_timer.stop()

    def resume_playback(self):
        if not self._is_playing and self._moves and self._playback_idx < len(self._moves):
            self._is_playing = True
            interval = max(10, int(100 / self._playback_speed))
            self._playback_timer.start(interval)

    def reset_playback(self):
        """Reset playback to the beginning."""
        self._playback_timer.stop()
        self._playback_idx = 0
        self._is_playing = False
        self._step_playback_frame()

    def set_playback_speed(self, speed_multiplier):
        """Change playback speed (1, 2, 4, 8, 16x via frame skipping)."""
        self._playback_speed_multiplier = speed_multiplier
        # Timer continues at fixed 50ms interval; speed is controlled by frame skipping

    def _step_playback(self):
        """Advance playback by skipping frames based on speed multiplier."""
        if not self._moves or self._playback_idx >= len(self._moves):
            self._playback_timer.stop()
            self._is_playing = False
            return
        self._step_playback_frame()
        # Advance by speed_multiplier frames
        self._playback_idx += self._playback_speed_multiplier
        # If we reached or passed the end, clamp to last index
        if self._playback_idx >= len(self._moves):
            self._playback_idx = len(self._moves) - 1
            self._playback_timer.stop()
            self._is_playing = False

    def _step_playback_frame(self):
        """Draw current frame of playback."""
        self._setup_axes()
        ax = self.ax

        # Draw blank outline
        if self._blank:
            rect_blank = patches.Rectangle(
                (0, -self._blank.width / 2), self._blank.length, self._blank.width,
                linewidth=2, edgecolor="#80c0ff", facecolor="none", label="Blank"
            )
            ax.add_patch(rect_blank)

        # Draw path taken so far (rapid moves dashed, cutting moves solid)
        # Optimize by batching moves of the same type instead of drawing each segment individually
        if self._moves and len(self._moves) > 0:
            cutting_x, cutting_y = [], []
            rapid_x, rapid_y = [], []

            # Collect path segments up to current index
            for i in range(1, min(self._playback_idx, len(self._moves))):
                prev = self._moves[i - 1]
                curr = self._moves[i]
                if curr.is_rapid:
                    rapid_x.extend([prev.x, curr.x, None])
                    rapid_y.extend([prev.y, curr.y, None])
                else:
                    cutting_x.extend([prev.x, curr.x, None])
                    cutting_y.extend([prev.y, curr.y, None])

            # Draw batched paths (single plot call per type is much faster)
            if cutting_x:
                ax.plot(cutting_x, cutting_y, color="#60cc60", linewidth=1.5, linestyle="-", alpha=0.7)
            if rapid_x:
                ax.plot(rapid_x, rapid_y, color="#ff8844", linewidth=1.5, linestyle="--", alpha=0.7)

            # Draw current cutter position as red circle (only if valid index)
            if self._playback_idx < len(self._moves):
                curr_move = self._moves[self._playback_idx]
                circle = patches.Circle(
                    (curr_move.x, curr_move.y),
                    self._tool_diameter / 2,
                    edgecolor="#ff0000",
                    facecolor="none",
                    linewidth=2,
                    label="Cutter"
                )
                ax.add_patch(circle)

        ax.set_aspect("equal")
        if self._blank:
            ax.set_xlim(-50, self._blank.length + 50)
            ax.set_ylim(-self._blank.width / 2 - 50, self._blank.width / 2 + 50)
        ax.set_xlabel("Along blank (mm)")
        ax.set_ylabel("Across blank (mm)")
        if self._moves:
            progress_pct = int(100 * min(self._playback_idx, len(self._moves)) / len(self._moves)) if len(self._moves) > 0 else 0
            status = "Playing" if self._is_playing else "Paused"
            ax.set_title(f"Toolpath Playback — {status} ({progress_pct}%) Move {min(self._playback_idx + 1, len(self._moves))}/{len(self._moves)}")
        else:
            ax.set_title("Toolpath Playback — No G-code generated")
        ax.grid(True, alpha=0.2, color="#555555")

        self.fig.tight_layout(pad=2.0)
        self.draw()

    def set_3d_view(self, enable_3d=True):
        """Toggle between 2D and 3D visualization modes."""
        self._use_3d = enable_3d
        if enable_3d and self._moves:
            self.plot_toolpaths_3d(self._moves)
        elif self._moves:
            self.plot_toolpaths(self._moves)

    def plot_toolpaths_3d(self, moves):
        """Plot toolpaths in 3D with rotation and zoom controls."""
        if not moves:
            return

        self.fig.clear()
        ax = self.fig.add_subplot(111, projection="3d")
        ax.set_facecolor("#2b2b2b")
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("#555555")
        ax.yaxis.pane.set_edgecolor("#555555")
        ax.zaxis.pane.set_edgecolor("#555555")
        ax.tick_params(colors="#cccccc", labelsize=8)
        ax.xaxis.label.set_color("#cccccc")
        ax.yaxis.label.set_color("#cccccc")
        ax.zaxis.label.set_color("#cccccc")

        # Group moves by Z depth for coloring
        z_values = [m.z for m in moves]
        z_min, z_max = min(z_values), max(z_values)
        z_range = z_max - z_min if z_max != z_min else 1.0

        # Plot moves as line segments
        for i in range(1, len(moves)):
            prev_move = moves[i - 1]
            curr_move = moves[i]
            z_norm = (curr_move.z - z_min) / z_range if z_range > 0 else 0.5

            color = cm.coolwarm(z_norm)
            linestyle = "--" if curr_move.is_rapid else "-"
            linewidth = 0.5 if curr_move.is_rapid else 1.5

            ax.plot(
                [prev_move.x, curr_move.x],
                [prev_move.y, curr_move.y],
                [prev_move.z, curr_move.z],
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=0.6
            )

        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")
        ax.set_title("3D Toolpath (Right-click to rotate, Scroll to zoom)")

        # True aspect ratio
        x_vals = [m.x for m in moves]
        y_vals = [m.y for m in moves]
        z_vals = [m.z for m in moves]
        x_rng = max(x_vals) - min(x_vals) or 1.0
        y_rng = max(y_vals) - min(y_vals) or 1.0
        z_rng = max(z_vals) - min(z_vals) or 1.0
        ax.set_box_aspect([x_rng, y_rng, z_rng])

        self.fig.tight_layout(pad=2.0)
        self.draw()

        # Enable mouse controls
        self._enable_3d_mouse_controls(ax)

    def _enable_3d_mouse_controls(self, ax):
        """Add mouse controls to 3D plot for rotation and zoom."""
        self._3d_press = None
        self._3d_xpress = None
        self._3d_ypress = None

        def on_press(event):
            if event.inaxes != ax:
                return
            self._3d_press = (event.button, event.xdata, event.ydata)
            self._3d_xpress = ax.view_init()[1]
            self._3d_ypress = ax.view_init()[0]

        def on_release(event):
            self._3d_press = None

        def on_motion(event):
            if self._3d_press is None or event.inaxes != ax:
                return
            button, xpress, ypress = self._3d_press
            if button == 3:  # Right mouse button for rotation
                dx = event.xdata - xpress if event.xdata else 0
                dy = event.ydata - ypress if event.ydata else 0
                ax.view_init(elev=self._3d_ypress + dy, azim=self._3d_xpress + dx)
                self.draw()

        def on_scroll(event):
            if event.inaxes != ax:
                return
            cur_xlim = ax.get_xlim()
            cur_ylim = ax.get_ylim()
            cur_zlim = ax.get_zlim()

            xdata = event.xdata
            ydata = event.ydata

            if event.button == "up":
                scale_factor = 0.8
            elif event.button == "down":
                scale_factor = 1.2
            else:
                return

            new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
            new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
            new_depth = (cur_zlim[1] - cur_zlim[0]) * scale_factor

            relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
            rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

            ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * relx])
            ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * rely])
            self.draw()

        self.mpl_connect("button_press_event", on_press)
        self.mpl_connect("button_release_event", on_release)
        self.mpl_connect("motion_notify_event", on_motion)
        self.mpl_connect("scroll_event", on_scroll)


# ── Parameter input panel ─────────────────────────────────────────────────────

class _FloatField(QLineEdit):
    def __init__(self, default: float):
        super().__init__(str(default))
        self.setFixedWidth(90)

    def value(self) -> float:
        return float(self.text())


class SlotParameterPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(6)

        # ── Cutting parameters ────────────────────────────────────────────────
        cut_group = QGroupBox("Cutting Parameters")
        cut_lay = QFormLayout(cut_group)

        self.f_tool_diameter = _FloatField(6.0)
        self.f_spindle_speed = _FloatField(18000.0)
        self.f_cutting_feed = _FloatField(1000.0)
        self.f_plunge_feed = _FloatField(300.0)
        self.f_rapid_feed = _FloatField(2000.0)
        self.f_depth_per_pass = _FloatField(3.0)
        self.f_stepover = _FloatField(2.0)
        self.f_spoilboard_skim = _FloatField(0.5)
        self.f_clearance_height = _FloatField(10.0)
        self.combo_stepover_dir = QComboBox()
        self.combo_stepover_dir.addItems(["Conventional", "Climb"])

        cut_lay.addRow("Tool diameter (mm):", self.f_tool_diameter)
        cut_lay.addRow("Spindle speed (RPM):", self.f_spindle_speed)
        cut_lay.addRow("Cutting feed (mm/min):", self.f_cutting_feed)
        cut_lay.addRow("Plunge feed (mm/min):", self.f_plunge_feed)
        cut_lay.addRow("Rapid feed (mm/min):", self.f_rapid_feed)
        cut_lay.addRow("Depth per pass (mm):", self.f_depth_per_pass)
        cut_lay.addRow("Stepover (mm):", self.f_stepover)
        cut_lay.addRow("Stepover direction:", self.combo_stepover_dir)
        cut_lay.addRow("Spoilboard skim (mm):", self.f_spoilboard_skim)
        cut_lay.addRow("Clearance height (mm):", self.f_clearance_height)
        root.addWidget(cut_group)

        # ── Tab parameters ────────────────────────────────────────────────────
        tab_group = QGroupBox("Tab Parameters")
        tab_lay = QFormLayout(tab_group)

        self.f_tab_spacing = _FloatField(100.0)
        self.f_tab_length = _FloatField(10.0)
        self.f_tab_thickness = _FloatField(1.0)

        tab_lay.addRow("Tab spacing (mm):", self.f_tab_spacing)
        tab_lay.addRow("Tab length (mm):", self.f_tab_length)
        tab_lay.addRow("Tab thickness (mm):", self.f_tab_thickness)
        root.addWidget(tab_group)

        # ── Buttons ───────────────────────────────────────────────────────────
        button_lay = QHBoxLayout()
        self.btn_load_params = QPushButton("Load params…")
        self.btn_save_params = QPushButton("Save params…")
        self.btn_generate = QPushButton("Generate G-code")
        self.btn_save_gcode = QPushButton("Save G-code…")

        button_lay.addWidget(self.btn_load_params)
        button_lay.addWidget(self.btn_save_params)
        button_lay.addStretch()
        button_lay.addWidget(self.btn_generate)
        button_lay.addWidget(self.btn_save_gcode)
        root.addLayout(button_lay)

        # ── Status / progress ─────────────────────────────────────────────────
        self.lbl_validation = QLabel("✓ Ready")
        self.lbl_validation.setStyleSheet("color: #60cc60;")
        root.addWidget(self.lbl_validation)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)   # indeterminate
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(12)
        root.addWidget(self.progress_bar)

        root.addStretch()

    def get_params(self) -> SlotParams:
        return SlotParams(
            tool_diameter=self.f_tool_diameter.value(),
            spindle_speed=int(self.f_spindle_speed.value()),
            cutting_feed=self.f_cutting_feed.value(),
            plunge_feed=self.f_plunge_feed.value(),
            rapid_feed=self.f_rapid_feed.value(),
            depth_per_pass=self.f_depth_per_pass.value(),
            stepover=self.f_stepover.value(),
            spoilboard_skim=self.f_spoilboard_skim.value(),
            clearance_height=self.f_clearance_height.value(),
            stepover_direction=self.combo_stepover_dir.currentText().lower(),
            tab_spacing=self.f_tab_spacing.value(),
            tab_length=self.f_tab_length.value(),
            tab_thickness=self.f_tab_thickness.value(),
        )

    def set_params(self, p: SlotParams):
        self.f_tool_diameter.setText(str(p.tool_diameter))
        self.f_spindle_speed.setText(str(p.spindle_speed))
        self.f_cutting_feed.setText(str(p.cutting_feed))
        self.f_plunge_feed.setText(str(p.plunge_feed))
        self.f_rapid_feed.setText(str(p.rapid_feed))
        self.f_depth_per_pass.setText(str(p.depth_per_pass))
        self.f_stepover.setText(str(p.stepover))
        self.f_spoilboard_skim.setText(str(p.spoilboard_skim))
        self.f_clearance_height.setText(str(p.clearance_height))
        self.combo_stepover_dir.setCurrentText(p.stepover_direction.capitalize())
        self.f_tab_spacing.setText(str(p.tab_spacing))
        self.f_tab_length.setText(str(p.tab_length))
        self.f_tab_thickness.setText(str(p.tab_thickness))


# ── Tab widget ────────────────────────────────────────────────────────────────

class GcodeTab(QWidget):
    def __init__(self, geom: SkiGeometry, params, blank_tab):
        super().__init__()
        self.geom = geom
        self.params = params
        self.blank_tab = blank_tab
        self._gcode_string = None
        self._moves = None
        self._use_3d = False
        self._build_ui()
        self._connect_signals()
        self._update_preview()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.splitter = QSplitter(Qt.Vertical)

        # Top: canvas with playback controls
        canvas_layout = QVBoxLayout()
        self.canvas = GcodeCanvas()
        canvas_layout.addWidget(self.canvas)

        # Playback control buttons
        playback_lay = QHBoxLayout()
        self.btn_play_pause = QPushButton("▶ Play")
        self.btn_reset = QPushButton("⏮ Reset")
        self.btn_speed_down = QPushButton("◀ Speed")
        self.lbl_speed = QLabel("1.0×")
        self.lbl_speed.setFixedWidth(50)
        self.btn_speed_up = QPushButton("Speed ▶")
        self.btn_toggle_3d = QPushButton("3D View")

        playback_lay.addWidget(self.btn_play_pause)
        playback_lay.addWidget(self.btn_reset)
        playback_lay.addWidget(self.btn_speed_down)
        playback_lay.addWidget(self.lbl_speed)
        playback_lay.addWidget(self.btn_speed_up)
        playback_lay.addStretch()
        playback_lay.addWidget(self.btn_toggle_3d)
        canvas_layout.addLayout(playback_lay)

        canvas_widget = QWidget()
        canvas_widget.setLayout(canvas_layout)
        self.splitter.addWidget(canvas_widget)

        # Bottom: parameter panel
        self.panel = SlotParameterPanel()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.panel)
        self.splitter.addWidget(scroll)

        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 1)

        layout.addWidget(self.splitter)

    def _connect_signals(self):
        # Parameters → preview
        self.panel.f_tool_diameter.textChanged.connect(self._update_preview)
        self.panel.f_spoilboard_skim.textChanged.connect(self._update_preview)
        self.panel.f_tab_spacing.textChanged.connect(self._update_preview)
        self.panel.f_tab_length.textChanged.connect(self._update_preview)
        self.panel.f_tab_thickness.textChanged.connect(self._update_preview)
        self.panel.combo_stepover_dir.currentTextChanged.connect(self._update_preview)

        # Buttons
        self.panel.btn_generate.clicked.connect(self._generate_gcode)
        self.panel.btn_save_gcode.clicked.connect(self._save_gcode)
        self.panel.btn_load_params.clicked.connect(self._load_params)
        self.panel.btn_save_params.clicked.connect(self._save_params)

        # Playback controls
        self.btn_play_pause.clicked.connect(self._playback_toggle)
        self.btn_reset.clicked.connect(self._playback_reset)
        self.btn_speed_down.clicked.connect(self._playback_speed_down)
        self.btn_speed_up.clicked.connect(self._playback_speed_up)
        self.btn_toggle_3d.clicked.connect(self._toggle_3d_view)

    def _update_preview(self):
        """Redraw slot preview (not expensive)."""
        try:
            blank = self.blank_tab.panel.get_blank()
            slot_params = self.panel.get_params()
            x_offset = (blank.length - (self.geom.core_tail_x - self.geom.core_tip_x + 50)) / 2 - (self.geom.core_tip_x - 25)
            self.canvas.plot_slot_preview(blank, self.geom, self.params, slot_params, x_offset)

            # Check for warnings
            warnings = []
            if slot_params.tool_diameter > self.params.sidewall_width:
                warnings.append(f"⚠ Tool diameter ({slot_params.tool_diameter:.1f}mm) > slot width ({self.params.sidewall_width:.1f}mm)")

            if warnings:
                msg = "\n".join(warnings)
                self.panel.lbl_validation.setText(msg)
                self.panel.lbl_validation.setStyleSheet("color: #ffa040;")
            else:
                self.panel.lbl_validation.setText("✓ Ready to generate")
                self.panel.lbl_validation.setStyleSheet("color: #60cc60;")
        except Exception as e:
            # Show blank outline with proper axis limits before g-code generation
            try:
                blank = self.blank_tab.panel.get_blank()
                self.canvas._setup_axes()
                ax = self.canvas.ax

                # Draw blank outline
                rect_blank = patches.Rectangle(
                    (0, -blank.width / 2), blank.length, blank.width,
                    linewidth=2, edgecolor="#80c0ff", facecolor="none", label="Blank"
                )
                ax.add_patch(rect_blank)

                ax.set_xlim(-50, blank.length + 50)
                ax.set_ylim(-blank.width / 2 - 50, blank.width / 2 + 50)
                ax.set_aspect("equal")
                ax.set_xlabel("Along blank (mm)")
                ax.set_ylabel("Across blank (mm)")
                ax.set_title("Configure parameters and click 'Generate G-code'")
                ax.grid(True, alpha=0.2, color="#555555")
                ax.legend(fontsize=8, facecolor="#333333", labelcolor="#dddddd")
                self.canvas.fig.tight_layout(pad=2.0)
                self.canvas.draw()
                self.panel.lbl_validation.setText("✓ Ready to generate")
                self.panel.lbl_validation.setStyleSheet("color: #60cc60;")
            except Exception as e2:
                self.panel.lbl_validation.setText(f"✗ Error: {str(e2)}")
                self.panel.lbl_validation.setStyleSheet("color: #ff6060;")

    def _generate_gcode(self):
        """Start background G-code generation."""
        try:
            blank = self.blank_tab.panel.get_blank()
            slot_params = self.panel.get_params()
            x_offset = (blank.length - (self.geom.core_tail_x - self.geom.core_tip_x + 50)) / 2 - (self.geom.core_tip_x - 25)
        except Exception as e:
            self.panel.lbl_validation.setText(f"✗ Error: {str(e)}")
            self.panel.lbl_validation.setStyleSheet("color: #ff6060;")
            return

        self._pending_blank = blank
        self._pending_slot_params = slot_params
        self.panel.btn_generate.setEnabled(False)
        self.panel.lbl_validation.setText("Generating G-code…")
        self.panel.lbl_validation.setStyleSheet("color: #aaaaaa;")
        self.panel.progress_bar.setVisible(True)

        self._worker = _GcodeWorker(
            generate_slot_gcode,
            self.geom, self.params, blank, slot_params, x_offset
        )
        self._worker.finished.connect(self._on_gcode_ready)
        self._worker.error.connect(self._on_gcode_error)
        self._worker.start()

    def _on_gcode_ready(self, gcode_string: str, moves: list):
        self._gcode_string = gcode_string
        self._moves = moves
        blank = self._pending_blank
        slot_params = self._pending_slot_params

        self.canvas._moves = moves
        self.canvas._blank = blank
        self.canvas._tool_diameter = slot_params.tool_diameter
        self.canvas._playback_idx = 0
        self.canvas._playback_speed_multiplier = 1
        self.canvas._step_playback_frame()
        self.lbl_speed.setText("1×")

        self.panel.btn_generate.setEnabled(True)
        self.panel.progress_bar.setVisible(False)
        self.panel.lbl_validation.setText(f"✓ G-code generated ({len(moves)} moves)")
        self.panel.lbl_validation.setStyleSheet("color: #60cc60;")

    def _on_gcode_error(self, msg: str):
        self.panel.btn_generate.setEnabled(True)
        self.panel.progress_bar.setVisible(False)
        self.panel.lbl_validation.setText(f"✗ Error: {msg}")
        self.panel.lbl_validation.setStyleSheet("color: #ff6060;")

    def _save_gcode(self):
        """Save G-code to file."""
        if not self._gcode_string:
            QMessageBox.warning(self, "No G-code", "Generate G-code first")
            return

        from PyQt5.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Save G-code", "slots.nc",
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
        """Load slot params from JSON."""
        from PyQt5.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Parameters", "", "JSON Files (*.json);;All Files (*)"
        )
        if not path:
            return

        try:
            params = SlotParams.from_json(path)
            self.panel.set_params(params)
            self._update_preview()
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))

    def _save_params(self):
        """Save slot params to JSON."""
        from PyQt5.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Parameters", "slot_params.json",
            "JSON Files (*.json);;All Files (*)"
        )
        if not path:
            return

        try:
            self.panel.get_params().to_json(path)
            QMessageBox.information(self, "Saved", f"Parameters saved to {Path(path).name}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    # ── Playback controls ─────────────────────────────────────────────────────

    def _playback_toggle(self):
        """Toggle between play and pause."""
        if not self._moves:
            QMessageBox.warning(self, "No G-code", "Generate G-code first")
            return

        if self.canvas._is_playing:
            # Currently playing, so pause
            self.canvas.pause_playback()
            self.btn_play_pause.setText("▶ Play")
        else:
            # Currently paused, so resume (or start if at beginning)
            blank = self.blank_tab.panel.get_blank()
            slot_params = self.panel.get_params()
            # start_playback will not reset index if already playing
            self.canvas.start_playback(self._moves, blank, slot_params.tool_diameter, self.canvas._playback_speed_multiplier)
            self.btn_play_pause.setText("⏸ Pause")

    def _playback_reset(self):
        """Reset playback to beginning."""
        self.canvas.reset_playback()
        self.canvas._playback_speed_multiplier = 1
        self.lbl_speed.setText("1×")
        self.btn_play_pause.setText("▶ Play")

    def _playback_speed_down(self):
        """Decrease playback speed to next lower step (1, 2, 4, 8, 16x)."""
        speeds = [1, 2, 4, 8, 16]
        current = self.canvas._playback_speed_multiplier
        idx = speeds.index(current) if current in speeds else len(speeds) - 1
        new_speed = speeds[max(0, idx - 1)]
        self.canvas.set_playback_speed(new_speed)
        self.lbl_speed.setText(f"{new_speed}×")

    def _playback_speed_up(self):
        """Increase playback speed to next higher step (1, 2, 4, 8, 16x)."""
        speeds = [1, 2, 4, 8, 16]
        current = self.canvas._playback_speed_multiplier
        idx = speeds.index(current) if current in speeds else 0
        new_speed = speeds[min(len(speeds) - 1, idx + 1)]
        self.canvas.set_playback_speed(new_speed)
        self.lbl_speed.setText(f"{new_speed}×")

    def _toggle_3d_view(self):
        """Toggle between 2D and 3D visualization."""
        if not self._moves:
            QMessageBox.warning(self, "No G-code", "Generate G-code first")
            return

        self._use_3d = not self._use_3d
        self.btn_toggle_3d.setText("2D View" if self._use_3d else "3D View")

        if self._use_3d:
            self.canvas.plot_toolpaths_3d(self._moves)
        else:
            self.canvas.plot_toolpaths(self._moves)
