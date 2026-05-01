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

    def _transform_to_machine(self, x, y):
        """Transform blank-space coords to machine coords based on blank settings."""
        if not self._blank:
            return x, y

        from core_carve.core_blank import MachineOrientation, OriginCorner

        # Swap axes if ski runs along Y
        if self._blank.machine_orientation == MachineOrientation.Y_AXIS:
            x, y = y, x

        # Shift based on origin corner
        if self._blank.origin_corner == OriginCorner.TOP_LEFT:
            y = y - self._blank.width
        elif self._blank.origin_corner == OriginCorner.TOP_RIGHT:
            x = x - self._blank.length
            y = y - self._blank.width
        elif self._blank.origin_corner == OriginCorner.BOTTOM_RIGHT:
            x = x - self._blank.length

        return x, y

    def plot_slot_preview(self, blank, geom, params, slot_params, x_offset=0.0):
        """Show blank and slot outlines in machine coordinates."""
        self._setup_axes()
        self._blank = blank
        ax = self.ax

        # Draw blank outline aligned with toolpath coordinates (absolute 0 to width)
        # The blank should match where the toolpaths are drawn
        corners = [
            (0, 0), (blank.length, 0),
            (blank.length, blank.width), (0, blank.width), (0, 0)
        ]

        # Transform to machine space
        transformed = [self._transform_to_machine(x, y) for x, y in corners]
        xs = [pt[0] for pt in transformed]
        ys = [pt[1] for pt in transformed]
        ax.plot(xs, ys, color="#80c0ff", linewidth=2, label="Blank")

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

        # Draw slots and tab markers (in machine coordinates)
        for core_idx, (core_x, core_y) in enumerate(core_positions):
            # Left slot edges (in absolute blank coordinates)
            slot_left_outer = left_outline + core_y + blank.width / 2.0
            slot_left_inner = left_outline + core_offset + core_y + blank.width / 2.0

            # Right slot edges (in absolute blank coordinates)
            slot_right_inner = right_outline - core_offset + core_y + blank.width / 2.0
            slot_right_outer = right_outline + core_y + blank.width / 2.0

            # Transform and draw slot outline (left)
            machine_left_outer = [self._transform_to_machine(x, y) for x, y in zip(y_samples_offset, slot_left_outer)]
            machine_left_inner = [self._transform_to_machine(x, y) for x, y in zip(y_samples_offset, slot_left_inner)]
            if machine_left_outer and not any(np.isnan(p).any() for p in machine_left_outer):
                mx_outer, my_outer = zip(*machine_left_outer)
                ax.plot(mx_outer, my_outer, color="#ffa040", linewidth=1, linestyle="-")
            if machine_left_inner and not any(np.isnan(p).any() for p in machine_left_inner):
                mx_inner, my_inner = zip(*machine_left_inner)
                ax.plot(mx_inner, my_inner, color="#ffa040", linewidth=1, linestyle="-")

            # Transform and draw slot outline (right)
            machine_right_inner = [self._transform_to_machine(x, y) for x, y in zip(y_samples_offset, slot_right_inner)]
            machine_right_outer = [self._transform_to_machine(x, y) for x, y in zip(y_samples_offset, slot_right_outer)]
            if machine_right_inner and not any(np.isnan(p).any() for p in machine_right_inner):
                mx_inner, my_inner = zip(*machine_right_inner)
                ax.plot(mx_inner, my_inner, color="#ffa040", linewidth=1, linestyle="-")
            if machine_right_outer and not any(np.isnan(p).any() for p in machine_right_outer):
                mx_outer, my_outer = zip(*machine_right_outer)
                ax.plot(mx_outer, my_outer, color="#ffa040", linewidth=1, linestyle="-")

            # Tab markers (transform to machine space)
            slot_centers = [(y_samples_offset, slot_left_inner), (y_samples_offset, slot_right_inner)]
            for slot_x_samples, slot_y_vals in slot_centers:
                # Skip if any NaN values
                if np.isnan(slot_y_vals).any():
                    continue
                arc_length = np.cumsum(np.concatenate([[0], np.linalg.norm(np.diff(np.column_stack([slot_x_samples, slot_y_vals])), axis=1)]))
                total_arc = arc_length[-1]
                if total_arc > 0:
                    tab_centers = np.arange(0, total_arc, slot_params.tab_spacing)
                    for tc in tab_centers:
                        idx = np.argmin(np.abs(arc_length - tc))
                        mx, my = self._transform_to_machine(slot_x_samples[idx], slot_y_vals[idx])
                        ax.plot(mx, my, "r.", markersize=4)

        # Set axis limits based on transformed blank dimensions and origin corner
        from core_carve.core_blank import MachineOrientation, OriginCorner

        # Compute bounds in machine space
        corners_machine = [(x, y) for x, y in transformed]
        xs = [pt[0] for pt in corners_machine]
        ys = [pt[1] for pt in corners_machine]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        x_margin = max(blank.length, blank.width) * 0.1
        y_margin = max(blank.length, blank.width) * 0.1
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)

        ax.set_aspect("equal")
        ax.set_xlabel("Machine X (mm)")
        ax.set_ylabel("Machine Y (mm)")
        ax.set_title("Slot Layout (Top View)")
        ax.grid(True, alpha=0.2, color="#555555")
        ax.legend(fontsize=8, facecolor="#333333", labelcolor="#dddddd")

        self.fig.tight_layout(pad=2.0)
        self.draw()

    def plot_toolpaths(self, moves):
        """Overlay toolpaths colour-coded by Z depth in machine coordinates."""
        if not moves or not self._blank:
            return

        self._setup_axes()
        ax = self.ax

        # Group moves by Z depth
        z_values = [m.z for m in moves]
        z_min, z_max = min(z_values), max(z_values)
        z_range = z_max - z_min if z_max != z_min else 1.0

        # Plot moves (transformed to machine space)
        for i in range(1, len(moves)):
            prev_move = moves[i - 1]
            curr_move = moves[i]
            z_norm = (curr_move.z - z_min) / z_range if z_range > 0 else 0.5

            # Colour gradient: blue (z=z_min, top) to red (z=z_max, bottom)
            color = cm.coolwarm(z_norm)

            linestyle = "--" if curr_move.is_rapid else "-"
            linewidth = 0.5 if curr_move.is_rapid else 1.5

            # Transform to machine space
            prev_mx, prev_my = self._transform_to_machine(prev_move.x, prev_move.y)
            curr_mx, curr_my = self._transform_to_machine(curr_move.x, curr_move.y)

            ax.plot(
                [prev_mx, curr_mx],
                [prev_my, curr_my],
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=0.6
            )

        # Set axis limits based on blank dimensions and origin corner
        blank = self._blank
        corners = [
            (0, 0), (blank.length, 0),
            (blank.length, blank.width), (0, blank.width)
        ]
        transformed = [self._transform_to_machine(x, y) for x, y in corners]
        xs = [pt[0] for pt in transformed]
        ys = [pt[1] for pt in transformed]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        x_margin = max(blank.length, blank.width) * 0.1
        y_margin = max(blank.length, blank.width) * 0.1
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)

        ax.set_aspect("equal")
        ax.set_xlabel("Machine X (mm)")
        ax.set_ylabel("Machine Y (mm)")
        ax.set_title("Toolpath Visualization (Machine Coordinates)")
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

        # Draw blank outline in machine coordinates
        if self._blank:
            # Transform blank corners to machine space
            corners = [
                (0, 0), (self._blank.length, 0),
                (self._blank.length, self._blank.width), (0, self._blank.width), (0, 0)
            ]
            transformed = [self._transform_to_machine(x, y) for x, y in corners]
            xs = [pt[0] for pt in transformed]
            ys = [pt[1] for pt in transformed]
            ax.plot(xs, ys, color="#80c0ff", linewidth=2, label="Blank")

        # Draw path taken so far in machine coordinates (rapid moves dashed, cutting moves solid)
        # Optimize by batching moves of the same type instead of drawing each segment individually
        if self._moves and len(self._moves) > 0:
            cutting_x, cutting_y = [], []
            rapid_x, rapid_y = [], []

            # Collect path segments up to current index (transformed to machine space)
            for i in range(1, min(self._playback_idx, len(self._moves))):
                prev = self._moves[i - 1]
                curr = self._moves[i]
                prev_mx, prev_my = self._transform_to_machine(prev.x, prev.y)
                curr_mx, curr_my = self._transform_to_machine(curr.x, curr.y)
                if curr.is_rapid:
                    rapid_x.extend([prev_mx, curr_mx, None])
                    rapid_y.extend([prev_my, curr_my, None])
                else:
                    cutting_x.extend([prev_mx, curr_mx, None])
                    cutting_y.extend([prev_my, curr_my, None])

            # Draw batched paths (single plot call per type is much faster)
            if cutting_x:
                ax.plot(cutting_x, cutting_y, color="#60cc60", linewidth=1.5, linestyle="-", alpha=0.7)
            if rapid_x:
                ax.plot(rapid_x, rapid_y, color="#ff8844", linewidth=1.5, linestyle="--", alpha=0.7)

            # Draw current cutter position as red circle in machine coordinates (only if valid index)
            if self._playback_idx < len(self._moves):
                curr_move = self._moves[self._playback_idx]
                curr_mx, curr_my = self._transform_to_machine(curr_move.x, curr_move.y)
                circle = patches.Circle(
                    (curr_mx, curr_my),
                    self._tool_diameter / 2,
                    edgecolor="#ff0000",
                    facecolor="none",
                    linewidth=2,
                    label="Cutter"
                )
                ax.add_patch(circle)

        ax.set_aspect("equal")
        if self._blank:
            # Set axis limits based on blank dimensions and origin corner
            corners = [
                (0, 0), (self._blank.length, 0),
                (self._blank.length, self._blank.width), (0, self._blank.width)
            ]
            transformed = [self._transform_to_machine(x, y) for x, y in corners]
            xs = [pt[0] for pt in transformed]
            ys = [pt[1] for pt in transformed]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            x_margin = max(self._blank.length, self._blank.width) * 0.1
            y_margin = max(self._blank.length, self._blank.width) * 0.1
            ax.set_xlim(x_min - x_margin, x_max + x_margin)
            ax.set_ylim(y_min - y_margin, y_max + y_margin)
        ax.set_xlabel("Machine X (mm)")
        ax.set_ylabel("Machine Y (mm)")
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
        """Plot toolpaths in 3D using Line3DCollection for fast rendering in machine coordinates."""
        if not moves or not self._blank:
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

        # Build two collections: cutting moves and rapid moves (transformed to machine space)
        cut_segs, cut_colors = [], []
        rapid_segs, rapid_colors = [], []

        for i in range(1, len(sampled)):
            p, c = sampled[i - 1], sampled[i]
            # Transform to machine space
            p_mx, p_my = self._transform_to_machine(p.x, p.y)
            c_mx, c_my = self._transform_to_machine(c.x, c.y)
            seg = [[p_mx, p_my, p.z], [c_mx, c_my, c.z]]
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

        # Set axis limits explicitly (required after add_collection3d) in machine space
        machine_coords = [self._transform_to_machine(m.x, m.y) for m in sampled]
        xs = [mc[0] for mc in machine_coords]
        ys = [mc[1] for mc in machine_coords]
        zs = [m.z for m in sampled]
        ax.set_xlim(min(xs), max(xs))
        ax.set_ylim(min(ys), max(ys))
        ax.set_zlim(min(zs), max(zs))

        ax.set_xlabel("Machine X (mm)")
        ax.set_ylabel("Machine Y (mm)")
        ax.set_zlabel("Machine Z (mm)")
        n_shown = len(sampled) - 1
        ax.set_title(f"3D Toolpath — {n_shown} segments shown (of {len(moves)-1})")

        # True aspect ratio
        x_rng = (max(xs) - min(xs)) or 1.0
        y_rng = (max(ys) - min(ys)) or 1.0
        z_rng = (max(zs) - min(zs)) or 1.0
        ax.set_box_aspect([x_rng, y_rng, z_rng])

        # Fill the full pane width — skip tight_layout which shrinks 3D axes
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.draw()

        # Enable mouse controls
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
        except Exception:
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
