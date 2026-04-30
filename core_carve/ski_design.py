"""Parametric ski planform geometry for the Design tab."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


# ── Bezier / circle primitives ────────────────────────────────────────────────

def bezier_cubic(p0, p1, p2, p3, n: int = 80) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)[:, np.newaxis]
    return (1-t)**3*p0 + 3*(1-t)**2*t*p1 + 3*(1-t)*t**2*p2 + t**3*p3


def fit_sidecut_circle(p1, p2, p3):
    """Circumscribed circle through three (x, y) points → (cx, cy, R)."""
    ax, ay = p1; bx, by = p2; cx2, cy2 = p3
    d = 2*(ax*(by-cy2) + bx*(cy2-ay) + cx2*(ay-by))
    if abs(d) < 1e-10:
        raise ValueError("Sidecut points are collinear.")
    ux = ((ax**2+ay**2)*(by-cy2) + (bx**2+by**2)*(cy2-ay) + (cx2**2+cy2**2)*(ay-by)) / d
    uy = ((ax**2+ay**2)*(cx2-bx) + (bx**2+by**2)*(ax-cx2) + (cx2**2+cy2**2)*(bx-ax)) / d
    return ux, uy, float(np.hypot(ax-ux, ay-uy))


def sample_arc(cx, cy, R, y_start, y_end, n: int = 200) -> np.ndarray:
    """Right-edge arc points (x, y) from y_start to y_end."""
    y = np.linspace(y_start, y_end, n)
    x = cx - np.sqrt(np.maximum(R**2 - (y - cy)**2, 0.0))
    return np.column_stack([x, y])


def arc_tangent_unit(cx: float, cy: float, x_arc: float, y_arc: float) -> np.ndarray:
    """
    Unit tangent of the right-edge arc at (x_arc, y_arc), pointing in +Y direction.
    Returns [dX, dY] in ski space (X=across, Y=along).
    """
    dxdy = (y_arc - cy) / (cx - x_arc) if abs(cx - x_arc) > 1e-9 else 0.0
    t = np.array([dxdy, 1.0])
    return t / np.linalg.norm(t)


# ── Planform parameter dataclass ──────────────────────────────────────────────

@dataclass
class SkiPlanformParams:
    # Main dimensions
    length: float = 1800.0
    waist_w: float = 96.0
    sidecut_radius: float = 16000.0  # mm — circle radius; cx = waist_w/2 + R
    tip_l: float = 300.0
    tip_w: float = 130.0
    tail_l: float = 200.0
    tail_w: float = 115.0
    setback: float = 0.0

    # Transition region lengths (along-ski from junction to arc)
    tip_trans_len: float = 50.0
    tail_trans_len: float = 50.0

    # Control arm lengths (Euclidean distances, all positive)
    # Tip bezier (apex → junction):
    tip_apex_arm: float = 100.0   # along-ski from tip apex
    tip_junc_arm: float = 32.5    # across-ski from tip junction (inward)
    # Tip transition bezier (junction → arc):
    tip_trans_junc_arm: float = 32.5  # across-ski from junction (outward)
    tip_trans_arc_arm: float = 30.0   # along arc tangent from arc end
    # Tail transition bezier (arc → junction):
    tail_trans_arc_arm: float = 30.0
    tail_trans_junc_arm: float = 28.75
    # Tail bezier (junction → apex):
    tail_junc_arm: float = 28.75
    tail_apex_arm: float = 66.7

    # ── JSON persistence ──────────────────────────────────────────────────────

    _KEYS = None  # populated lazily

    _DIMENSION_KEYS = {
        "length", "waist_w", "sidecut_radius", "tip_w", "tail_w", "tip_l", "tail_l",
        "setback", "tip_trans_len", "tail_trans_len",
    }
    _ARM_KEYS = {
        "tip_apex_arm", "tip_junc_arm", "tip_trans_junc_arm", "tip_trans_arc_arm",
        "tail_trans_arc_arm", "tail_trans_junc_arm", "tail_junc_arm", "tail_apex_arm",
    }

    def to_dict(self) -> dict:
        flat = asdict(self)
        return {
            "dimensions":   {k: flat[k] for k in self._DIMENSION_KEYS if k in flat},
            "control_arms": {k: flat[k] for k in self._ARM_KEYS if k in flat},
        }

    @classmethod
    def _planform_keys(cls):
        import dataclasses
        return {f.name for f in dataclasses.fields(cls)}

    @classmethod
    def from_dict(cls, d: dict) -> "SkiPlanformParams":
        if "dimensions" in d or "control_arms" in d:
            flat: dict = {}
            flat.update(d.get("dimensions", {}))
            flat.update(d.get("control_arms", {}))
        else:
            flat = d  # backward-compat with old flat format
        keys = cls._planform_keys()
        return cls(**{k: float(v) for k, v in flat.items() if k in keys})

    @classmethod
    def from_json(cls, path) -> "SkiPlanformParams":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def save_to_json(self, path) -> None:
        p = Path(path)
        existing = {}
        if p.exists() and p.stat().st_size > 0:
            with open(p) as f:
                existing = json.load(f)
        grouped = self.to_dict()
        existing.setdefault("dimensions", {}).update(grouped["dimensions"])
        existing.setdefault("control_arms", {}).update(grouped["control_arms"])
        with open(p, "w") as f:
            json.dump(existing, f, indent=2)


# ── Outline result dataclass ──────────────────────────────────────────────────

@dataclass
class SkiOutlineResult:
    outline: np.ndarray
    R: float
    cx: float
    cy: float

    # Key Y (along-ski) positions
    tip_l: float
    tip_trans_y: float
    tail_trans_y: float
    tail_contact: float
    waist_y: float

    # Fixed anchor points [X_across, Y_along] (right side)
    tip_P0: np.ndarray   # tip apex (0, 0)
    tip_P3: np.ndarray   # tip junction = tt_P0
    tt_P3: np.ndarray    # arc start
    tr_P0: np.ndarray    # arc end = tt_P3 for tail
    tail_P0: np.ndarray  # tail junction = tr_P3
    tail_P3: np.ndarray  # tail apex (0, length)

    # Draggable control points [X_across, Y_along] (right side)
    tip_P1: np.ndarray   # tip apex arm
    tip_P2: np.ndarray   # tip junction arm
    tt_P1: np.ndarray    # tip transition junction arm
    tt_P2: np.ndarray    # tip transition arc arm
    tr_P1: np.ndarray    # tail transition arc arm
    tr_P2: np.ndarray    # tail transition junction arm
    tail_P1: np.ndarray  # tail junction arm
    tail_P2: np.ndarray  # tail apex arm

    # Arc tangent unit vectors at transition ends
    tang_tip: np.ndarray   # at arc start, pointing +Y
    tang_tail: np.ndarray  # at arc end, pointing +Y


# ── Main outline builder ──────────────────────────────────────────────────────

def build_ski_outline(
    params: SkiPlanformParams,
    n_bez: int = 80,
    n_arc: int = 200,
    n_trans: int = 50,
) -> SkiOutlineResult:
    p = params
    tail_contact = p.length - p.tail_l
    y_mid = (p.tip_l + tail_contact) / 2.0 + p.setback
    tip_trans_y = p.tip_l + p.tip_trans_len
    tail_trans_y = tail_contact - p.tail_trans_len

    # Sidecut circle: defined directly by waist_w and sidecut_radius.
    # Centre is (sidecut_radius + waist_w/2) from centreline at the waist position,
    # so the right edge of the arc at the waist equals exactly waist_w/2.
    R  = p.sidecut_radius
    cy = y_mid
    cx = p.waist_w / 2.0 + R

    # Arc endpoints
    x_tip = cx - np.sqrt(max(R**2 - (tip_trans_y - cy)**2, 0.0))
    x_tail = cx - np.sqrt(max(R**2 - (tail_trans_y - cy)**2, 0.0))

    tang_tip  = arc_tangent_unit(cx, cy, x_tip,  tip_trans_y)
    tang_tail = arc_tangent_unit(cx, cy, x_tail, tail_trans_y)

    # Fixed anchors
    tip_P0  = np.array([0.0,         0.0])
    tip_P3  = np.array([p.tip_w/2,   p.tip_l])
    tt_P3   = np.array([x_tip,       tip_trans_y])
    tr_P0   = np.array([x_tail,      tail_trans_y])
    tail_P0 = np.array([p.tail_w/2,  tail_contact])
    tail_P3 = np.array([0.0,         p.length])

    # Tip bezier control points
    # P1: apex arm — across-ski from apex: locked Y=0, arm is across-ski distance
    tip_P1 = np.array([p.tip_apex_arm, 0.0])
    # P2: junction arm — along-ski from junction: locked X=tip_w/2, arm is along-ski distance
    tip_P2 = np.array([p.tip_w/2, p.tip_l - p.tip_junc_arm])

    # Tip transition control points
    # Q1: junction arm along-ski outward: locked X=tip_w/2, arm is along-ski distance after junction
    tt_P1 = np.array([p.tip_w/2, p.tip_l + p.tip_trans_junc_arm])
    # Q2: arc arm along arc tangent from arc end toward junction
    tt_P2 = tt_P3 - p.tip_trans_arc_arm * tang_tip

    # Tail transition control points
    # R1: arc arm along arc tangent from arc end toward tail
    tr_P1 = tr_P0 + p.tail_trans_arc_arm * tang_tail
    # R2: junction arm along-ski inward: locked X=tail_w/2, arm is along-ski distance before junction
    tr_P2 = np.array([p.tail_w/2, tail_contact - p.tail_trans_junc_arm])

    # Tail bezier control points
    # S1: junction arm along-ski outward: locked X=tail_w/2, arm is along-ski distance after junction
    tail_P1 = np.array([p.tail_w/2, tail_contact + p.tail_junc_arm])
    # S2: apex arm — across-ski from apex: locked Y=length, arm is across-ski distance
    tail_P2 = np.array([p.tail_apex_arm, p.length])

    # Sample curves
    tip_curve   = bezier_cubic(tip_P0, tip_P1, tip_P2, tip_P3, n=n_bez)
    tip_trans   = bezier_cubic(tip_P3, tt_P1, tt_P2, tt_P3, n=n_trans)
    arc_pts     = sample_arc(cx, cy, R, tip_trans_y, tail_trans_y, n=n_arc)
    tail_trans  = bezier_cubic(tr_P0, tr_P1, tr_P2, tail_P0, n=n_trans)
    tail_curve  = bezier_cubic(tail_P0, tail_P1, tail_P2, tail_P3, n=n_bez)

    # Assemble right edge (Y increases tip→tail)
    right = np.vstack([
        tip_curve,
        tip_trans[1:],
        arc_pts[1:],
        tail_trans[1:],
        tail_curve[1:],
    ])

    left = right[::-1].copy()
    left[:, 0] = -left[:, 0]
    outline = np.vstack([right, left[1:], right[:1]]).astype(np.float64)

    return SkiOutlineResult(
        outline=outline, R=R, cx=cx, cy=cy,
        tip_l=p.tip_l, tip_trans_y=tip_trans_y,
        tail_trans_y=tail_trans_y, tail_contact=tail_contact,
        waist_y=y_mid,
        tip_P0=tip_P0, tip_P3=tip_P3, tt_P3=tt_P3,
        tr_P0=tr_P0, tail_P0=tail_P0, tail_P3=tail_P3,
        tip_P1=tip_P1, tip_P2=tip_P2,
        tt_P1=tt_P1, tt_P2=tt_P2,
        tr_P1=tr_P1, tr_P2=tr_P2,
        tail_P1=tail_P1, tail_P2=tail_P2,
        tang_tip=tang_tip, tang_tail=tang_tail,
    )
