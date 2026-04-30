"""Tab 0 — Ski Design: parametric planform design with interactive Bezier control."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QPushButton, QLineEdit, QGroupBox,
    QFormLayout, QFileDialog, QScrollArea, QSizePolicy,
    QFrame,
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from core_carve.ski_design import (
    SkiPlanformParams, SkiOutlineResult, build_ski_outline,
)


# ── tiny helpers ──────────────────────────────────────────────────────────────

class _FloatField(QLineEdit):
    def __init__(self, default: float, width: int = 70):
        super().__init__(f"{default:.1f}")
        self.setFixedWidth(width)

    def value(self) -> float:
        try:
            return float(self.text())
        except ValueError:
            return 0.0

    def set_value(self, v: float):
        self.blockSignals(True)
        self.setText(f"{v:.1f}")
        self.blockSignals(False)


def _divider() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.HLine)
    f.setStyleSheet("color: #444;")
    return f


# ── Drawing constants ─────────────────────────────────────────────────────────

_CTRL_R   = 7    # control-point circle radius (px)
_PICK_R   = 14   # click pick radius (px)
_LINE_PICK = 8   # pick radius for parameter lines (px)

_COL_OUTLINE = "#80c0ff"
_COL_CTRL    = "#ff6633"
_COL_ARM     = "#ff6633"
_COL_TIP_L   = "#f0a040"
_COL_TRANS   = "#c080ff"
_COL_WAIST   = "#60e0a0"
_COL_TAIL_L  = "#80d080"


def _arm(ax, a, b):
    ax.plot([a[1], b[1]], [a[0], b[0]],
            color=_COL_ARM, lw=0.8, ls="--", alpha=0.7)


def _ctrl_pt(ax, pt, alpha=1.0):
    ax.plot(pt[1], pt[0], "o", color=_COL_CTRL,
            markersize=_CTRL_R, zorder=5, alpha=alpha)


def _style_ax(ax, fontsize=7):
    ax.set_facecolor("#2b2b2b")
    ax.tick_params(colors="#cccccc", labelsize=fontsize)
    for sp in ax.spines.values():
        sp.set_edgecolor("#555")
    ax.xaxis.label.set_color("#cccccc")
    ax.yaxis.label.set_color("#cccccc")
    ax.title.set_color("#eeeeee")


# ── Canvas ────────────────────────────────────────────────────────────────────

# Drag item types
_DRAG_CTRL = "ctrl"
_DRAG_LINE = "line"

# Control point indices (into the 8 draggable points)
_I_TIP_P1, _I_TIP_P2 = 0, 1
_I_TT_P1,  _I_TT_P2  = 2, 3
_I_TR_P1,  _I_TR_P2  = 4, 5
_I_TAIL_P1, _I_TAIL_P2 = 6, 7

# Draggable parameter lines (name → attr on SkiPlanformParams to update)
_PARAM_LINES = ["tip_l", "tip_trans", "waist", "tail_trans", "tail_contact"]


class DesignCanvas(FigureCanvas):
    """Three-subplot canvas: tip zoom (top-left), tail zoom (top-right), full (bottom)."""

    def __init__(self):
        self.fig = Figure(facecolor="#1e1e1e", figsize=(14, 9))
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ax = self.ax_tip = self.ax_tail = None
        self._setup_axes()

    def _setup_axes(self):
        self.fig.clear()
        gs = GridSpec(2, 2, figure=self.fig,
                      height_ratios=[1, 1.5], hspace=0.32, wspace=0.22)
        self.ax_tip  = self.fig.add_subplot(gs[0, 0])
        self.ax_tail = self.fig.add_subplot(gs[0, 1])
        self.ax      = self.fig.add_subplot(gs[1, :])
        for ax in (self.ax_tip, self.ax_tail, self.ax):
            _style_ax(ax)
        self.fig.tight_layout(pad=0.5)
        self.draw()

    def _clear_axes(self):
        """Fast clear: just cla() each axis and reapply style, preserving layout."""
        for ax in (self.ax_tip, self.ax_tail, self.ax):
            ax.cla()
            _style_ax(ax)

    def connect_events(self, on_press, on_motion, on_release):
        self.mpl_connect("button_press_event",   on_press)
        self.mpl_connect("motion_notify_event",  on_motion)
        self.mpl_connect("button_release_event", on_release)

    # ── Drawing helpers ───────────────────────────────────────────────────────

    def _draw_ctrl_set(self, ax, result: SkiOutlineResult, which: str):
        """Draw bezier arms and control points for 'tip', 'tip_trans', 'tail_trans', or 'tail'."""
        r = result
        if which == "tip":
            _arm(ax, r.tip_P0, r.tip_P1); _arm(ax, r.tip_P3, r.tip_P2)
            _arm(ax, r.tip_P1, r.tip_P2)
            _ctrl_pt(ax, r.tip_P1); _ctrl_pt(ax, r.tip_P2)
            # mirror
            _arm(ax, _mx(r.tip_P0), _mx(r.tip_P1))
            _arm(ax, _mx(r.tip_P3), _mx(r.tip_P2))
            _arm(ax, _mx(r.tip_P1), _mx(r.tip_P2))
            _ctrl_pt(ax, _mx(r.tip_P1), 0.4); _ctrl_pt(ax, _mx(r.tip_P2), 0.4)
        elif which == "tip_trans":
            _arm(ax, r.tip_P3, r.tt_P1); _arm(ax, r.tt_P3, r.tt_P2)
            _arm(ax, r.tt_P1, r.tt_P2)
            _ctrl_pt(ax, r.tt_P1); _ctrl_pt(ax, r.tt_P2)
            _arm(ax, _mx(r.tip_P3), _mx(r.tt_P1))
            _arm(ax, _mx(r.tt_P3), _mx(r.tt_P2))
            _arm(ax, _mx(r.tt_P1), _mx(r.tt_P2))
            _ctrl_pt(ax, _mx(r.tt_P1), 0.4); _ctrl_pt(ax, _mx(r.tt_P2), 0.4)
        elif which == "tail_trans":
            _arm(ax, r.tr_P0, r.tr_P1); _arm(ax, r.tail_P0, r.tr_P2)
            _arm(ax, r.tr_P1, r.tr_P2)
            _ctrl_pt(ax, r.tr_P1); _ctrl_pt(ax, r.tr_P2)
            _arm(ax, _mx(r.tr_P0), _mx(r.tr_P1))
            _arm(ax, _mx(r.tail_P0), _mx(r.tr_P2))
            _arm(ax, _mx(r.tr_P1), _mx(r.tr_P2))
            _ctrl_pt(ax, _mx(r.tr_P1), 0.4); _ctrl_pt(ax, _mx(r.tr_P2), 0.4)
        elif which == "tail":
            _arm(ax, r.tail_P0, r.tail_P1); _arm(ax, r.tail_P3, r.tail_P2)
            _arm(ax, r.tail_P1, r.tail_P2)
            _ctrl_pt(ax, r.tail_P1); _ctrl_pt(ax, r.tail_P2)
            _arm(ax, _mx(r.tail_P0), _mx(r.tail_P1))
            _arm(ax, _mx(r.tail_P3), _mx(r.tail_P2))
            _arm(ax, _mx(r.tail_P1), _mx(r.tail_P2))
            _ctrl_pt(ax, _mx(r.tail_P1), 0.4); _ctrl_pt(ax, _mx(r.tail_P2), 0.4)

    # ── Main plot entry ───────────────────────────────────────────────────────

    def plot_design(self, result: SkiOutlineResult, params: SkiPlanformParams):
        self._clear_axes()   # fast: cla() each axis, no GridSpec rebuild
        r = result
        outline = r.outline

        # ── full planform ─────────────────────────────────────────────────────
        ax = self.ax
        ax.plot(outline[:, 1], outline[:, 0], color=_COL_OUTLINE, lw=1.5)

        ax.axvline(r.tip_l,        color=_COL_TIP_L,  lw=1.2, ls="--")
        ax.axvline(r.tip_trans_y,  color=_COL_TRANS,  lw=1.0, ls="--")
        ax.axvline(r.waist_y,      color=_COL_WAIST,  lw=1.2, ls="-")
        ax.axvline(r.tail_trans_y, color=_COL_TRANS,  lw=1.0, ls="--")
        ax.axvline(r.tail_contact, color=_COL_TAIL_L, lw=1.2, ls="--")

        ax.text(0.01, 0.97, f"R = {r.R/1000:.2f} m", transform=ax.transAxes,
                color="#cccccc", fontsize=8, va="top")
        ax.set_aspect("equal")
        ax.set_xlabel("Along ski (mm)", fontsize=8)
        ax.set_ylabel("Across ski (mm)", fontsize=8)
        ax.set_title("Ski Planform", fontsize=9)
        ax.grid(True, alpha=0.15, color="#555")

        # ── tip zoom ──────────────────────────────────────────────────────────
        ax_t = self.ax_tip
        ax_t.plot(outline[:, 1], outline[:, 0], color=_COL_OUTLINE, lw=1.5)
        self._draw_ctrl_set(ax_t, r, "tip")
        self._draw_ctrl_set(ax_t, r, "tip_trans")
        ax_t.axvline(r.tip_l,       color=_COL_TIP_L, lw=1.0, ls="--")
        ax_t.axvline(r.tip_trans_y, color=_COL_TRANS,  lw=1.0, ls="--")
        pad = max(params.tip_w * 0.4, 30.0)
        ax_t.set_xlim(-pad * 0.2, r.tip_trans_y + pad * 0.8)
        ax_t.set_ylim(-(params.tip_w / 2 + pad * 0.6), params.tip_w / 2 + pad * 0.6)
        ax_t.set_aspect("equal", adjustable="datalim")
        ax_t.set_title("Tip", fontsize=9)
        ax_t.set_xlabel("Along (mm)", fontsize=7)
        ax_t.set_ylabel("Across (mm)", fontsize=7)
        ax_t.grid(True, alpha=0.15, color="#555")

        # ── tail zoom ─────────────────────────────────────────────────────────
        ax_tl = self.ax_tail
        ax_tl.plot(outline[:, 1], outline[:, 0], color=_COL_OUTLINE, lw=1.5)
        self._draw_ctrl_set(ax_tl, r, "tail_trans")
        self._draw_ctrl_set(ax_tl, r, "tail")
        ax_tl.axvline(r.tail_trans_y, color=_COL_TRANS,  lw=1.0, ls="--")
        ax_tl.axvline(r.tail_contact, color=_COL_TAIL_L, lw=1.0, ls="--")
        pad = max(params.tail_w * 0.4, 30.0)
        ax_tl.set_xlim(r.tail_trans_y - pad * 0.8, params.length + pad * 0.2)
        ax_tl.set_ylim(-(params.tail_w / 2 + pad * 0.6), params.tail_w / 2 + pad * 0.6)
        ax_tl.set_aspect("equal", adjustable="datalim")
        ax_tl.set_title("Tail", fontsize=9)
        ax_tl.set_xlabel("Along (mm)", fontsize=7)
        ax_tl.set_ylabel("Across (mm)", fontsize=7)
        ax_tl.grid(True, alpha=0.15, color="#555")

        self.draw_idle()   # non-blocking; batches rapid drag events

    # ── Coordinate helpers ────────────────────────────────────────────────────

    def disp(self, ax, x_data, y_data):
        return ax.transData.transform((x_data, y_data))

    def all_axes(self):
        return [self.ax, self.ax_tip, self.ax_tail]


def _mx(pt):
    """Mirror a control point (negate X/across)."""
    m = pt.copy()
    m[0] = -m[0]
    return m


# ── Parameter panel ───────────────────────────────────────────────────────────

class DesignPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._load_defaults_from_file()
        self._build_ui()

    def _load_defaults_from_file(self):
        """Load default values from data/ski.json if available."""
        ski_file = Path(__file__).parent.parent / "data" / "ski.json"
        # Use ski.json values as defaults (from ski.json hardcoded)
        self._defaults = {
            'length': 1800.0,
            'waist_w': 98.0,
            'sidecut_radius': 16.0,  # Already in meters
            'tip_w': 125.0,
            'tail_w': 115.0,
            'tip_l': 134.0,
            'tip_trans': 201.0,
            'setback': 61.0,
            'tail_trans': 210.0,
            'tail_l': 105.0,
            'tip_apex_arm': 30.0,
            'tip_junc_arm': 130.0,
            'tip_trans_junc_arm': 30.0,
            'tip_trans_arc_arm': 30.0,
            'tail_trans_arc_arm': 30.0,
            'tail_trans_junc_arm': 30.0,
            'tail_junc_arm': 100.0,
            'tail_apex_arm': 40.0,
        }

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(5)

        # Ski dimensions
        dim = QGroupBox("Ski Dimensions")
        dl = QFormLayout(dim)
        self.f_length      = _FloatField(self._defaults.get('length', 1800.0))
        self.f_waist_w     = _FloatField(self._defaults.get('waist_w', 96.0))
        self.f_sidecut_r   = _FloatField(self._defaults.get('sidecut_radius', 16.0))   # m
        dl.addRow("Length (mm):", self.f_length)
        dl.addRow("Waist width (mm):", self.f_waist_w)
        dl.addRow("Sidecut radius (m):", self.f_sidecut_r)
        root.addWidget(dim)

        # Tip & Tail widths
        tt = QGroupBox("Tip & Tail Widths")
        tl = QFormLayout(tt)
        self.f_tip_w  = _FloatField(self._defaults.get('tip_w', 125.0))
        self.f_tail_w = _FloatField(self._defaults.get('tail_w', 115.0))
        tl.addRow("Tip width (mm):", self.f_tip_w)
        tl.addRow("Tail width (mm):", self.f_tail_w)
        root.addWidget(tt)

        # Positions (drag lines on main view, or type here)
        pos = QGroupBox("Key Positions (drag lines or type)")
        pl = QFormLayout(pos)
        self.f_tip_l        = _FloatField(self._defaults.get('tip_l', 134.0))
        self.f_tip_trans    = _FloatField(self._defaults.get('tip_trans', 201.0), width=55)
        self.f_setback      = _FloatField(self._defaults.get('setback', 61.0),  width=55)
        self.f_tail_trans   = _FloatField(self._defaults.get('tail_trans', 210.0), width=55)
        self.f_tail_l       = _FloatField(self._defaults.get('tail_l', 105.0))
        pl.addRow("Tip junction (mm):", self.f_tip_l)
        pl.addRow("Tip transition length:", self.f_tip_trans)
        pl.addRow("Waist setback:", self.f_setback)
        pl.addRow("Tail transition length:", self.f_tail_trans)
        pl.addRow("Tail junction (mm):", self.f_tail_l)
        root.addWidget(pos)

        # Tip control arms
        tip_arms = QGroupBox("Tip Control Arms")
        tal = QFormLayout(tip_arms)
        self.f_tip_apex_arm       = _FloatField(self._defaults.get('tip_apex_arm', 30.0))
        self.f_tip_junc_arm       = _FloatField(self._defaults.get('tip_junc_arm', 130.0))
        self.f_tip_trans_junc_arm = _FloatField(self._defaults.get('tip_trans_junc_arm', 30.0))
        self.f_tip_trans_arc_arm  = _FloatField(self._defaults.get('tip_trans_arc_arm', 30.0))
        tal.addRow("Apex arm (mm):",          self.f_tip_apex_arm)
        tal.addRow("Junction arm (mm):",      self.f_tip_junc_arm)
        tal.addRow("Trans junction arm (mm):", self.f_tip_trans_junc_arm)
        tal.addRow("Trans arc arm (mm):",      self.f_tip_trans_arc_arm)
        root.addWidget(tip_arms)

        # Tail control arms
        tail_arms = QGroupBox("Tail Control Arms")
        ttal = QFormLayout(tail_arms)
        self.f_tail_trans_arc_arm  = _FloatField(self._defaults.get('tail_trans_arc_arm', 30.0))
        self.f_tail_trans_junc_arm = _FloatField(self._defaults.get('tail_trans_junc_arm', 30.0))
        self.f_tail_junc_arm       = _FloatField(self._defaults.get('tail_junc_arm', 100.0))
        self.f_tail_apex_arm       = _FloatField(self._defaults.get('tail_apex_arm', 40.0))
        ttal.addRow("Trans arc arm (mm):",      self.f_tail_trans_arc_arm)
        ttal.addRow("Trans junction arm (mm):", self.f_tail_trans_junc_arm)
        ttal.addRow("Junction arm (mm):",      self.f_tail_junc_arm)
        ttal.addRow("Apex arm (mm):",          self.f_tail_apex_arm)
        root.addWidget(tail_arms)

        # Actions
        btn = QHBoxLayout()
        self.btn_load = QPushButton("Load ski")
        self.btn_save = QPushButton("Save ski")
        self.btn_use  = QPushButton("Use as Planform →")
        self.btn_use.setStyleSheet("font-weight:bold;")
        btn.addWidget(self.btn_load)
        btn.addWidget(self.btn_save)
        root.addLayout(btn)
        root.addWidget(self.btn_use)
        root.addStretch()

    # ── params → UI ──────────────────────────────────────────────────────────

    def load_params(self, p: SkiPlanformParams):
        for field, val in [
            (self.f_length,    p.length),  (self.f_waist_w, p.waist_w),
            (self.f_sidecut_r, p.sidecut_radius / 1000.0),  # mm → m for display
            (self.f_tip_w,     p.tip_w),   (self.f_tail_w,  p.tail_w),
            (self.f_tip_l,     p.tip_l),   (self.f_tail_l,  p.tail_l),
            (self.f_setback,   p.setback),
            (self.f_tip_trans,  p.tip_trans_len),
            (self.f_tail_trans, p.tail_trans_len),
            (self.f_tip_apex_arm,       p.tip_apex_arm),
            (self.f_tip_junc_arm,       p.tip_junc_arm),
            (self.f_tip_trans_junc_arm, p.tip_trans_junc_arm),
            (self.f_tip_trans_arc_arm,  p.tip_trans_arc_arm),
            (self.f_tail_trans_arc_arm,  p.tail_trans_arc_arm),
            (self.f_tail_trans_junc_arm, p.tail_trans_junc_arm),
            (self.f_tail_junc_arm,       p.tail_junc_arm),
            (self.f_tail_apex_arm,       p.tail_apex_arm),
        ]:
            field.set_value(val)

    # ── UI → params ──────────────────────────────────────────────────────────

    def get_params(self) -> SkiPlanformParams:
        return SkiPlanformParams(
            length=self.f_length.value(),
            waist_w=self.f_waist_w.value(),
            sidecut_radius=max(1000.0, self.f_sidecut_r.value() * 1000.0),  # m → mm
            tip_l=self.f_tip_l.value(),
            tip_w=self.f_tip_w.value(),
            tail_l=self.f_tail_l.value(),
            tail_w=self.f_tail_w.value(),
            setback=self.f_setback.value(),
            tip_trans_len=max(1.0, self.f_tip_trans.value()),
            tail_trans_len=max(1.0, self.f_tail_trans.value()),
            tip_apex_arm=max(1.0, self.f_tip_apex_arm.value()),
            tip_junc_arm=max(0.0, self.f_tip_junc_arm.value()),
            tip_trans_junc_arm=max(0.0, self.f_tip_trans_junc_arm.value()),
            tip_trans_arc_arm=max(1.0, self.f_tip_trans_arc_arm.value()),
            tail_trans_arc_arm=max(1.0, self.f_tail_trans_arc_arm.value()),
            tail_trans_junc_arm=max(0.0, self.f_tail_trans_junc_arm.value()),
            tail_junc_arm=max(0.0, self.f_tail_junc_arm.value()),
            tail_apex_arm=max(1.0, self.f_tail_apex_arm.value()),
        )

    def all_fields(self):
        return [
            self.f_length, self.f_waist_w, self.f_sidecut_r,
            self.f_tip_w, self.f_tail_w,
            self.f_tip_l, self.f_tail_l, self.f_setback,
            self.f_tip_trans, self.f_tail_trans,
            self.f_tip_apex_arm, self.f_tip_junc_arm,
            self.f_tip_trans_junc_arm, self.f_tip_trans_arc_arm,
            self.f_tail_trans_arc_arm, self.f_tail_trans_junc_arm,
            self.f_tail_junc_arm, self.f_tail_apex_arm,
        ]


# ── Main tab ──────────────────────────────────────────────────────────────────

class DesignTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._result: Optional[SkiOutlineResult] = None
        self._on_outline_ready: Optional[Callable] = None
        self._on_ski_definition_loaded: Optional[Callable] = None

        # Drag state
        self._drag_type:    Optional[str] = None  # _DRAG_CTRL or _DRAG_LINE
        self._drag_ctrl_idx: Optional[int] = None
        self._drag_line:    Optional[str]  = None
        self._drag_ax = None

        self._build_ui()
        self._connect_signals()
        self._update()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Horizontal)

        self.canvas = DesignCanvas()
        splitter.addWidget(self.canvas)

        self.panel = DesignPanel()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.panel)
        scroll.setFixedWidth(260)
        splitter.addWidget(scroll)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        layout.addWidget(splitter)

    def _connect_signals(self):
        for f in self.panel.all_fields():
            f.textChanged.connect(self._update)
        self.panel.btn_load.clicked.connect(self._load_json)
        self.panel.btn_save.clicked.connect(self._save_json)
        self.panel.btn_use.clicked.connect(self._use_as_planform)
        self.canvas.connect_events(self._on_press, self._on_motion, self._on_release)

    # ── Update ────────────────────────────────────────────────────────────────

    def _update(self, quick: bool = False):
        p = self.panel.get_params()
        kw = dict(n_bez=25, n_arc=60, n_trans=15) if quick else {}
        try:
            result = build_ski_outline(p, **kw)
        except (ValueError, ZeroDivisionError):
            return
        self._result = result
        self.canvas.plot_design(result, p)

    def _update_from_params(self, p: SkiPlanformParams):
        """Update the panel fields from a params object, then redraw."""
        self.panel.load_params(p)
        # load_params sets values without signals; trigger manually
        self._update()

    # ── Drag: control points ──────────────────────────────────────────────────

    def _ctrl_pts_display(self, ax):
        """Return list of (idx, disp_x, disp_y) for all 8 control points in given ax."""
        r = self._result
        pts = [r.tip_P1, r.tip_P2, r.tt_P1, r.tt_P2,
               r.tr_P1,  r.tr_P2,  r.tail_P1, r.tail_P2]
        out = []
        for i, pt in enumerate(pts):
            dx, dy = self.canvas.disp(ax, pt[1], pt[0])
            out.append((i, dx, dy))
        return out

    def _line_x_display(self, r: SkiOutlineResult):
        """Return dict of line_name → display x position in main ax."""
        vals = {
            "tip_l":        r.tip_l,
            "tip_trans":    r.tip_trans_y,
            "waist":        r.waist_y,
            "tail_trans":   r.tail_trans_y,
            "tail_contact": r.tail_contact,
        }
        out = {}
        for k, v in vals.items():
            # lines in main ax: x=along, so xdata=v, ydata=0
            dx, _ = self.canvas.disp(self.canvas.ax, v, 0)
            out[k] = dx
        return out

    def _on_press(self, event):
        if event.button != 1 or self._result is None:
            return

        r = self._result
        click_dx = event.x
        click_dy = event.y

        # Which axes did the click land in?
        clicked_ax = event.inaxes

        # Try control point pick (all 3 axes)
        if clicked_ax in self.canvas.all_axes():
            for idx, pdx, pdy in self._ctrl_pts_display(clicked_ax):
                dist = np.hypot(click_dx - pdx, click_dy - pdy)
                if dist < _PICK_R:
                    self._drag_type = _DRAG_CTRL
                    self._drag_ctrl_idx = idx
                    self._drag_ax = clicked_ax
                    return

        # Try parameter-line pick (main ax only)
        if clicked_ax is self.canvas.ax:
            line_disp = self._line_x_display(r)
            best_name, best_dist = None, _LINE_PICK
            for name, lx in line_disp.items():
                d = abs(click_dx - lx)
                if d < best_dist:
                    best_dist = d
                    best_name = name
            if best_name:
                self._drag_type = _DRAG_LINE
                self._drag_line = best_name
                self._drag_ax = self.canvas.ax
                return

    def _on_motion(self, event):
        if self._drag_type is None or event.inaxes is None:
            return
        if event.inaxes is not self._drag_ax:
            return
        if self._result is None:
            return

        if self._drag_type == _DRAG_CTRL:
            self._handle_ctrl_drag(event)
        elif self._drag_type == _DRAG_LINE:
            self._handle_line_drag(event)

    def _handle_ctrl_drag(self, event):
        """Compute new arm length from drag position and update panel field."""
        idx = self._drag_ctrl_idx
        r   = self._result
        p   = self.panel.get_params()
        # Transposed axes: xdata=along-ski, ydata=across-ski
        drag_along  = event.xdata  # Y_along
        drag_across = event.ydata  # X_across

        if idx == _I_TIP_P1:
            # Locked Y=0 (same along as apex); arm = across-ski distance
            arm = max(1.0, drag_across)
            self.panel.f_tip_apex_arm.set_value(arm)

        elif idx == _I_TIP_P2:
            # Locked X=tip_w/2 (same across as junction); arm = along-ski distance below junction
            arm = max(0.0, p.tip_l - drag_along)
            self.panel.f_tip_junc_arm.set_value(arm)

        elif idx == _I_TT_P1:
            # Locked X=tip_w/2; arm = along-ski distance above junction
            arm = max(0.0, drag_along - p.tip_l)
            self.panel.f_tip_trans_junc_arm.set_value(arm)

        elif idx == _I_TT_P2:
            # Locked along arc tangent — project drag onto tangent (backward from arc end)
            drag_ski = np.array([drag_across, drag_along])
            arm = max(1.0, float(-np.dot(drag_ski - r.tt_P3, r.tang_tip)))
            self.panel.f_tip_trans_arc_arm.set_value(arm)

        elif idx == _I_TR_P1:
            # Locked along arc tangent — project drag onto tangent (forward from arc end)
            drag_ski = np.array([drag_across, drag_along])
            arm = max(1.0, float(np.dot(drag_ski - r.tr_P0, r.tang_tail)))
            self.panel.f_tail_trans_arc_arm.set_value(arm)

        elif idx == _I_TR_P2:
            # Locked X=tail_w/2; arm = along-ski distance below tail junction
            arm = max(0.0, p.length - p.tail_l - drag_along)
            self.panel.f_tail_trans_junc_arm.set_value(arm)

        elif idx == _I_TAIL_P1:
            # Locked X=tail_w/2; arm = along-ski distance above tail junction
            arm = max(0.0, drag_along - (p.length - p.tail_l))
            self.panel.f_tail_junc_arm.set_value(arm)

        elif idx == _I_TAIL_P2:
            # Locked Y=length (same along as apex); arm = across-ski distance
            arm = max(1.0, drag_across)
            self.panel.f_tail_apex_arm.set_value(arm)

        self._update(quick=True)

    def _handle_line_drag(self, event):
        p   = self.panel.get_params()
        pos = event.xdata  # along-ski position

        if self._drag_line == "tip_l":
            new_tip_l = max(10.0, min(pos, p.tip_l + p.tip_trans_len - 5.0))
            self.panel.f_tip_l.set_value(new_tip_l)

        elif self._drag_line == "tip_trans":
            new_len = max(5.0, pos - p.tip_l)
            max_len = (p.length - p.tail_l - p.tail_trans_len - p.tip_l) / 2.0
            self.panel.f_tip_trans.set_value(min(new_len, max(5.0, max_len)))

        elif self._drag_line == "waist":
            tail_contact = p.length - p.tail_l
            mid = (p.tip_l + tail_contact) / 2.0
            self.panel.f_setback.set_value(pos - mid)

        elif self._drag_line == "tail_trans":
            tail_contact = p.length - p.tail_l
            new_len = max(5.0, tail_contact - pos)
            max_len = (p.length - p.tail_l - p.tip_trans_len - p.tip_l) / 2.0
            self.panel.f_tail_trans.set_value(min(new_len, max(5.0, max_len)))

        elif self._drag_line == "tail_contact":
            new_tail_l = max(10.0, min(p.length - pos, p.length - p.tail_trans_len - p.tip_l - 5.0))
            self.panel.f_tail_l.set_value(new_tail_l)

        self._update(quick=True)

    def _on_release(self, event):
        was_dragging = self._drag_type is not None
        self._drag_type     = None
        self._drag_ctrl_idx = None
        self._drag_line     = None
        self._drag_ax       = None
        if was_dragging:
            # Redraw at full resolution and update derived labels
            self._update(quick=False)

    # ── JSON I/O ──────────────────────────────────────────────────────────────

    def _load_json(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Ski Definition JSON", "", "JSON Files (*.json)"
        )
        if not path:
            return
        try:
            import json
            with open(path) as f:
                data = json.load(f)

            # Check if this is a full ski definition
            is_full_definition = any(k in data for k in ["outline", "core", "base", "camber"])

            if is_full_definition and self._on_ski_definition_loaded is not None:
                # Let main window handle full definition
                self._on_ski_definition_loaded(path)
            else:
                # Just load outline params
                p = SkiPlanformParams.from_json(path)
                self._update_from_params(p)
        except Exception as e:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Load Failed", str(e))

    def _save_json(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Ski Definition JSON", "", "JSON Files (*.json)"
        )
        if not path:
            return
        try:
            self.panel.get_params().save_to_json(path)
        except Exception as e:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Save Failed", str(e))

    # ── DXF export ────────────────────────────────────────────────────────────

    def _export_dxf(self):
        if self._result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Ski Planform DXF", "", "DXF Files (*.dxf)"
        )
        if not path:
            return
        try:
            import ezdxf
            doc = ezdxf.new("R2010")
            msp = doc.modelspace()
            pts = [(float(r[1]), float(r[0])) for r in self._result.outline]
            msp.add_lwpolyline(pts, close=False, dxfattribs={"layer": "SKI_OUTLINE"})
            doc.saveas(path)
        except Exception as e:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Export Failed", str(e))

    # ── Callback wiring ───────────────────────────────────────────────────────

    def set_outline_callback(self, fn: Callable) -> None:
        self._on_outline_ready = fn

    def set_ski_definition_loaded_callback(self, fn: Callable) -> None:
        self._on_ski_definition_loaded = fn

    def _use_as_planform(self):
        if self._result is not None and self._on_outline_ready is not None:
            self._on_outline_ready(self._result.outline.copy())
