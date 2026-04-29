"""Tab — Core Profiling: thickness profile machining."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QPushButton, QLabel, QLineEdit, QGroupBox, QFormLayout,
    QMessageBox, QScrollArea, QSizePolicy, QFileDialog, QComboBox, QProgressBar,
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal


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
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as patches
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

from core_carve.profile_generator import ProfileParams, generate_profile_gcode
from core_carve.ski_geometry import SkiGeometry


# ── Canvas ────────────────────────────────────────────────────────────────────

class ProfileCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(facecolor="#1e1e1e", figsize=(12, 6))
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ax = None
        self._setup_axes()
        self._moves = None
        self._blank = None
        self._playback_idx = 0
        self._playback_speed_multiplier = 1
        self._is_playing = False
        self._playback_timer = QTimer()
        self._playback_timer.timeout.connect(self._step_playback)
        self._use_3d = False
        self._tool_diameter = 12.0

    def _setup_axes(self):
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#2b2b2b")
        self.ax.tick_params(colors="#cccccc", labelsize=8)
        for spine in self.ax.spines.values():
            spine.set_edgecolor("#555555")
        self.fig.tight_layout(pad=2.0)
        self.draw()

    def plot_toolpaths(self, moves, blank=None):
        """Plot profiling toolpaths in 2D (top view)."""
        if not moves:
            return

        self._setup_axes()
        ax = self.ax

        # Draw blank if provided
        if blank:
            rect_blank = patches.Rectangle(
                (0, -blank.width / 2), blank.length, blank.width,
                linewidth=2, edgecolor="#80c0ff", facecolor="none", label="Blank"
            )
            ax.add_patch(rect_blank)

        z_values = [m.z for m in moves]
        z_min, z_max = min(z_values), max(z_values)
        z_range = z_max - z_min if z_max != z_min else 1.0

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
                color=color, linestyle=linestyle, linewidth=linewidth, alpha=0.6
            )

        if blank:
            ax.set_xlim(-50, blank.length + 50)
            ax.set_ylim(-blank.width / 2 - 50, blank.width / 2 + 50)
        ax.set_aspect("equal")
        ax.set_xlabel("Along blank (mm)")
        ax.set_ylabel("Across blank (mm)")
        ax.set_title("Profiling Toolpath (Top View)")
        ax.grid(True, alpha=0.2, color="#555555")
        if blank:
            ax.legend(fontsize=8, facecolor="#333333", labelcolor="#dddddd")

        self.fig.tight_layout(pad=2.0)
        self.draw()

    def plot_toolpaths_3d(self, moves):
        """Plot profiling toolpaths in 3D using Line3DCollection for fast rendering."""
        if not moves:
            return

        from mpl_toolkits.mplot3d.art3d import Line3DCollection

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

        # Downsample to max 4000 displayed segments
        MAX_SEGS = 4000
        step = max(1, (len(moves) - 1) // MAX_SEGS)
        sampled = moves[::step]
        if sampled[-1] is not moves[-1]:
            sampled = sampled + [moves[-1]]

        z_values = [m.z for m in sampled]
        z_min, z_max = min(z_values), max(z_values)
        z_range = z_max - z_min if z_max != z_min else 1.0

        cut_segs, cut_colors = [], []
        rapid_segs, rapid_colors = [], []

        for i in range(1, len(sampled)):
            p, c = sampled[i - 1], sampled[i]
            seg = [[p.x, p.y, p.z], [c.x, c.y, c.z]]
            z_norm = (c.z - z_min) / z_range if z_range > 0 else 0.5
            color = cm.coolwarm(z_norm)
            if c.is_rapid:
                rapid_segs.append(seg)
                rapid_colors.append(color)
            else:
                cut_segs.append(seg)
                cut_colors.append(color)

        if cut_segs:
            lc = Line3DCollection(cut_segs, colors=cut_colors, linewidths=1.5, alpha=0.7)
            ax.add_collection3d(lc)
        if rapid_segs:
            rl = Line3DCollection(rapid_segs, colors=rapid_colors, linewidths=0.5, alpha=0.3)
            ax.add_collection3d(rl)

        xs = [m.x for m in sampled]
        ys = [m.y for m in sampled]
        zs = [m.z for m in sampled]
        ax.set_xlim(min(xs), max(xs))
        ax.set_ylim(min(ys), max(ys))
        ax.set_zlim(min(zs), max(zs))

        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")
        n_shown = len(sampled) - 1
        ax.set_title(f"3D Profiling Toolpath — {n_shown} segments shown (of {len(moves)-1})")

        x_rng = (max(xs) - min(xs)) or 1.0
        y_rng = (max(ys) - min(ys)) or 1.0
        z_rng = (max(zs) - min(zs)) or 1.0
        ax.set_box_aspect([x_rng, y_rng, z_rng])

        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.draw()
        self._enable_3d_mouse_controls(ax)

    def _enable_3d_mouse_controls(self, ax):
        """Add mouse controls to 3D plot for rotation, zoom, and pan."""
        self._3d_press = None
        self._3d_xpress = None
        self._3d_ypress = None
        self._3d_pan_xlim = None
        self._3d_pan_ylim = None
        self._3d_pan_px = None
        self._3d_pan_py = None

        def on_press(event):
            if event.inaxes != ax:
                return
            self._3d_press = (event.button, event.xdata, event.ydata)
            if event.button == 3:
                self._3d_xpress = ax.azim
                self._3d_ypress = ax.elev
            elif event.button == 2:
                self._3d_pan_xlim = ax.get_xlim()
                self._3d_pan_ylim = ax.get_ylim()
                self._3d_pan_px = event.x
                self._3d_pan_py = event.y

        def on_release(event):
            self._3d_press = None

        def on_motion(event):
            if self._3d_press is None or event.inaxes != ax:
                return
            button, xpress, ypress = self._3d_press
            if button == 3:  # Right mouse: rotate
                dx = event.xdata - xpress if event.xdata else 0
                dy = event.ydata - ypress if event.ydata else 0
                ax.view_init(elev=self._3d_ypress + dy, azim=self._3d_xpress + dx)
                self.draw()
            elif button == 2 and self._3d_pan_xlim is not None:  # Middle mouse: pan
                dx = event.x - self._3d_pan_px
                dy = event.y - self._3d_pan_py
                xlim = self._3d_pan_xlim
                ylim = self._3d_pan_ylim
                xrng = xlim[1] - xlim[0]
                yrng = ylim[1] - ylim[0]
                w, h = self.get_width_height()
                scale_x = xrng / w if w > 0 else 1.0
                scale_y = yrng / h if h > 0 else 1.0
                ax.set_xlim([xlim[0] - dx * scale_x, xlim[1] - dx * scale_x])
                ax.set_ylim([ylim[0] + dy * scale_y, ylim[1] + dy * scale_y])
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
            relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0]) if xdata and (cur_xlim[1] - cur_xlim[0]) != 0 else 0.5
            rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0]) if ydata and (cur_ylim[1] - cur_ylim[0]) != 0 else 0.5
            ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * relx] if xdata else
                        [(cur_xlim[0] + cur_xlim[1]) / 2 - new_width / 2, (cur_xlim[0] + cur_xlim[1]) / 2 + new_width / 2])
            ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * rely] if ydata else
                        [(cur_ylim[0] + cur_ylim[1]) / 2 - new_height / 2, (cur_ylim[0] + cur_ylim[1]) / 2 + new_height / 2])
            ax.set_zlim([(cur_zlim[0] + cur_zlim[1]) / 2 - new_depth / 2,
                         (cur_zlim[0] + cur_zlim[1]) / 2 + new_depth / 2])
            self.draw()

        self.mpl_connect("button_press_event", on_press)
        self.mpl_connect("button_release_event", on_release)
        self.mpl_connect("motion_notify_event", on_motion)
        self.mpl_connect("scroll_event", on_scroll)

    def start_playback(self, moves, speed_multiplier=1):
        """Start playback animation."""
        self._moves = moves
        self._playback_idx = 0
        self._playback_speed_multiplier = speed_multiplier
        self._is_playing = True
        self._playback_timer.start(50)

    def pause_playback(self):
        self._is_playing = False
        self._playback_timer.stop()

    def reset_playback(self):
        self._playback_timer.stop()
        self._playback_idx = 0
        self._is_playing = False
        self._step_playback_frame()

    def set_playback_speed(self, speed_multiplier):
        self._playback_speed_multiplier = speed_multiplier

    def _step_playback(self):
        if not self._moves or self._playback_idx >= len(self._moves):
            self._playback_timer.stop()
            self._is_playing = False
            return
        self._step_playback_frame()
        self._playback_idx += self._playback_speed_multiplier
        if self._playback_idx >= len(self._moves):
            self._playback_idx = len(self._moves) - 1
            self._playback_timer.stop()
            self._is_playing = False

    def _step_playback_frame(self):
        """Draw current playback frame."""
        self._setup_axes()
        ax = self.ax

        # Draw blank outline
        if self._blank:
            rect_blank = patches.Rectangle(
                (0, -self._blank.width / 2), self._blank.length, self._blank.width,
                linewidth=2, edgecolor="#80c0ff", facecolor="none", label="Blank"
            )
            ax.add_patch(rect_blank)

        # Draw path taken so far
        if self._moves and len(self._moves) > 0:
            cutting_x, cutting_y = [], []
            rapid_x, rapid_y = [], []

            for i in range(1, min(self._playback_idx, len(self._moves))):
                prev = self._moves[i - 1]
                curr = self._moves[i]
                if curr.is_rapid:
                    rapid_x.extend([prev.x, curr.x, None])
                    rapid_y.extend([prev.y, curr.y, None])
                else:
                    cutting_x.extend([prev.x, curr.x, None])
                    cutting_y.extend([prev.y, curr.y, None])

            if cutting_x:
                ax.plot(cutting_x, cutting_y, color="#60cc60", linewidth=1.5, linestyle="-", alpha=0.7)
            if rapid_x:
                ax.plot(rapid_x, rapid_y, color="#ff8844", linewidth=1.5, linestyle="--", alpha=0.7)

            if self._playback_idx < len(self._moves):
                curr_move = self._moves[self._playback_idx]
                circle = patches.Circle((curr_move.x, curr_move.y), self._tool_diameter / 2, edgecolor="#ff0000", facecolor="none", linewidth=2)
                ax.add_patch(circle)

        ax.set_aspect("equal")
        if self._blank:
            ax.set_xlim(-50, self._blank.length + 50)
            ax.set_ylim(-self._blank.width / 2 - 50, self._blank.width / 2 + 50)
        ax.set_xlabel("Along blank (mm)")
        ax.set_ylabel("Across blank (mm)")
        if self._moves:
            progress = int(100 * min(self._playback_idx, len(self._moves)) / len(self._moves))
            status = "Playing" if self._is_playing else "Paused"
            ax.set_title(f"Profiling Playback — {status} ({progress}%) Move {min(self._playback_idx + 1, len(self._moves))}/{len(self._moves)}")
        ax.grid(True, alpha=0.2, color="#555555")

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


class ProfileParameterPanel(QWidget):
    def __init__(self):
        super().__init__()
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(6)

        # ── Cutting parameters ────────────────────────────────────────────────
        cut_group = QGroupBox("Cutting Parameters")
        cut_lay = QFormLayout(cut_group)

        self.f_tool_diameter = _FloatField(12.0)
        self.f_spindle_speed = _FloatField(12000.0)
        self.f_cutting_feed = _FloatField(800.0)
        self.f_plunge_feed = _FloatField(200.0)
        self.f_roughing_depth = _FloatField(2.0)
        self.f_finishing_depth = _FloatField(0.5)
        self.f_stepover = _FloatField(3.0)
        self.f_clearance = _FloatField(10.0)
        self.combo_direction = QComboBox()
        self.combo_direction.addItems(["Along ski", "Across ski"])

        cut_lay.addRow("Tool diameter (mm):", self.f_tool_diameter)
        cut_lay.addRow("Spindle speed (RPM):", self.f_spindle_speed)
        cut_lay.addRow("Cutting feed (mm/min):", self.f_cutting_feed)
        cut_lay.addRow("Plunge feed (mm/min):", self.f_plunge_feed)
        cut_lay.addRow("Roughing depth (mm):", self.f_roughing_depth)
        cut_lay.addRow("Finishing depth (mm):", self.f_finishing_depth)
        cut_lay.addRow("Stepover (mm):", self.f_stepover)
        cut_lay.addRow("Clearance height (mm):", self.f_clearance)
        cut_lay.addRow("Cut direction:", self.combo_direction)
        root.addWidget(cut_group)

        # ── Buttons ───────────────────────────────────────────────────────────
        button_lay = QHBoxLayout()
        self.btn_generate = QPushButton("Generate G-code")
        self.btn_save_gcode = QPushButton("Save G-code…")
        self.btn_load_params = QPushButton("Load params…")
        self.btn_save_params = QPushButton("Save params…")

        button_lay.addWidget(self.btn_load_params)
        button_lay.addWidget(self.btn_save_params)
        button_lay.addStretch()
        button_lay.addWidget(self.btn_generate)
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

    def get_params(self) -> ProfileParams:
        return ProfileParams(
            tool_diameter=self.f_tool_diameter.value(),
            spindle_speed=int(self.f_spindle_speed.value()),
            cutting_feed=self.f_cutting_feed.value(),
            plunge_feed=self.f_plunge_feed.value(),
            roughing_depth_per_pass=self.f_roughing_depth.value(),
            finishing_depth_of_cut=self.f_finishing_depth.value(),
            stepover=self.f_stepover.value(),
            clearance_height=self.f_clearance.value(),
            direction=self.combo_direction.currentText().lower().split()[0],
        )

    def set_params(self, p: ProfileParams):
        self.f_tool_diameter.setText(str(p.tool_diameter))
        self.f_spindle_speed.setText(str(p.spindle_speed))
        self.f_cutting_feed.setText(str(p.cutting_feed))
        self.f_plunge_feed.setText(str(p.plunge_feed))
        self.f_roughing_depth.setText(str(p.roughing_depth_per_pass))
        self.f_finishing_depth.setText(str(p.finishing_depth_of_cut))
        self.f_stepover.setText(str(p.stepover))
        self.f_clearance.setText(str(p.clearance_height))


# ── Tab widget ────────────────────────────────────────────────────────────────

class ProfileTab(QWidget):
    def __init__(self, geom: SkiGeometry, params, blank_tab):
        super().__init__()
        self.geom = geom
        self.params = params
        self.blank_tab = blank_tab
        self._gcode_string = None
        self._moves = None
        self._build_ui()
        self._connect_signals()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.splitter = QSplitter(Qt.Vertical)

        # Top: canvas with playback controls
        canvas_layout = QVBoxLayout()
        self.canvas = ProfileCanvas()
        canvas_layout.addWidget(self.canvas)

        playback_lay = QHBoxLayout()
        self.btn_play_pause = QPushButton("▶ Play")
        self.btn_reset = QPushButton("⏮ Reset")
        self.btn_speed_down = QPushButton("◀ Speed")
        self.lbl_speed = QLabel("1×")
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
        self.panel = ProfileParameterPanel()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.panel)
        self.splitter.addWidget(scroll)

        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 1)

        layout.addWidget(self.splitter)

    def _connect_signals(self):
        self.panel.btn_generate.clicked.connect(self._generate_gcode)
        self.panel.btn_save_gcode.clicked.connect(self._save_gcode)
        self.panel.btn_load_params.clicked.connect(self._load_params)
        self.panel.btn_save_params.clicked.connect(self._save_params)

        self.btn_play_pause.clicked.connect(self._playback_toggle)
        self.btn_reset.clicked.connect(self._playback_reset)
        self.btn_speed_down.clicked.connect(self._playback_speed_down)
        self.btn_speed_up.clicked.connect(self._playback_speed_up)
        self.btn_toggle_3d.clicked.connect(self._toggle_3d_view)

    def _generate_gcode(self):
        """Start background G-code generation."""
        try:
            blank = self.blank_tab.panel.get_blank()
            profile_params = self.panel.get_params()
        except Exception as e:
            self.panel.lbl_status.setText(f"✗ Error: {str(e)}")
            self.panel.lbl_status.setStyleSheet("color: #ff6060;")
            return

        self._pending_blank = blank
        self._pending_profile_params = profile_params
        self.panel.btn_generate.setEnabled(False)
        self.panel.lbl_status.setText("Generating G-code…")
        self.panel.lbl_status.setStyleSheet("color: #aaaaaa;")
        self.panel.progress_bar.setVisible(True)

        self._worker = _GcodeWorker(
            generate_profile_gcode,
            self.geom, self.params, blank, profile_params
        )
        self._worker.finished.connect(self._on_gcode_ready)
        self._worker.error.connect(self._on_gcode_error)
        self._worker.start()

    def _on_gcode_ready(self, gcode_string: str, moves: list):
        self._gcode_string = gcode_string
        self._moves = moves
        blank = self._pending_blank
        profile_params = self._pending_profile_params

        self.canvas._moves = moves
        self.canvas._blank = blank
        self.canvas._tool_diameter = profile_params.tool_diameter
        self.canvas._playback_idx = 0
        self.canvas.plot_toolpaths(moves, blank)
        self.lbl_speed.setText("1×")

        self.panel.btn_generate.setEnabled(True)
        self.panel.progress_bar.setVisible(False)
        self.panel.lbl_status.setText(f"✓ G-code generated ({len(moves)} moves)")
        self.panel.lbl_status.setStyleSheet("color: #60cc60;")

    def _on_gcode_error(self, msg: str):
        self.panel.btn_generate.setEnabled(True)
        self.panel.progress_bar.setVisible(False)
        self.panel.lbl_status.setText(f"✗ Error: {msg}")
        self.panel.lbl_status.setStyleSheet("color: #ff6060;")

    def _save_gcode(self):
        """Save G-code to file."""
        if not self._gcode_string:
            QMessageBox.warning(self, "No G-code", "Generate G-code first")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save G-code", "profile.nc",
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
        """Load params from JSON."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Parameters", "", "JSON Files (*.json);;All Files (*)"
        )
        if not path:
            return

        try:
            params = ProfileParams.from_json(path)
            self.panel.set_params(params)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))

    def _save_params(self):
        """Save params to JSON."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Parameters", "profile_params.json",
            "JSON Files (*.json);;All Files (*)"
        )
        if not path:
            return

        try:
            self.panel.get_params().to_json(path)
            QMessageBox.information(self, "Saved", f"Parameters saved to {Path(path).name}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def _playback_toggle(self):
        if not self._moves:
            QMessageBox.warning(self, "No G-code", "Generate G-code first")
            return

        if self.canvas._is_playing:
            self.canvas.pause_playback()
            self.btn_play_pause.setText("▶ Play")
        else:
            self.canvas.start_playback(self._moves, self.canvas._playback_speed_multiplier)
            self.btn_play_pause.setText("⏸ Pause")

    def _playback_reset(self):
        self.canvas.reset_playback()
        self.canvas._playback_speed_multiplier = 1
        self.lbl_speed.setText("1×")
        self.btn_play_pause.setText("▶ Play")

    def _playback_speed_down(self):
        speeds = [1, 2, 4, 8, 16]
        current = self.canvas._playback_speed_multiplier
        idx = speeds.index(current) if current in speeds else len(speeds) - 1
        new_speed = speeds[max(0, idx - 1)]
        self.canvas.set_playback_speed(new_speed)
        self.lbl_speed.setText(f"{new_speed}×")

    def _playback_speed_up(self):
        speeds = [1, 2, 4, 8, 16]
        current = self.canvas._playback_speed_multiplier
        idx = speeds.index(current) if current in speeds else 0
        new_speed = speeds[min(len(speeds) - 1, idx + 1)]
        self.canvas.set_playback_speed(new_speed)
        self.lbl_speed.setText(f"{new_speed}×")

    def _toggle_3d_view(self):
        if not self._moves:
            QMessageBox.warning(self, "No G-code", "Generate G-code first")
            return

        self.canvas._use_3d = not self.canvas._use_3d
        self.btn_toggle_3d.setText("2D View" if self.canvas._use_3d else "3D View")

        if self.canvas._use_3d:
            self.canvas.plot_toolpaths_3d(self._moves)
        else:
            self.canvas.plot_toolpaths(self._moves)
