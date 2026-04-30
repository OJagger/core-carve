"""G-code generation for core thickness profiling."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


@dataclass
class ProfileParams:
    """CNC machine and tool parameters for thickness profiling."""
    tool_diameter: float = 12.0
    spindle_speed: int = 12000
    cutting_feed: float = 800.0
    plunge_feed: float = 200.0
    rapid_feed: float = 1500.0
    roughing_depth_per_pass: float = 2.0
    finishing_depth_of_cut: float = 0.5
    stepover: float = 6.0
    clearance_height: float = 10.0
    direction: str = "along"  # "along" (ski length) or "across" (width)

    def to_json(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> "ProfileParams":
        with open(path) as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Move:
    """A single CNC move."""
    x: float
    y: float
    z: float
    is_rapid: bool = False
    feed: float = 0.0


def generate_profile_gcode(
    geom,
    params,
    blank,
    profile_params: ProfileParams,
) -> tuple[str, list[Move]]:
    """
    Generate G-code for core thickness profiling in blank-space coordinates.

    Strategy: Multi-pass cutting that follows the core thickness contour.
    - For "along" direction: tool traverses along ski Y, Z varies with core thickness
    - Each pass removes a layer, with final pass achieving exact contour
    - Tool extends past sidewalls by approximately one tool diameter

    Blank space:
      X = along blank = ski_Y + x_offset  (ski runs along blank X)
      Y = across blank = ski_X + core_y_offset

    Returns:
        (gcode_string, moves_list)  — moves are in blank space for visualization.
    """
    from core_carve.core_blank import MachineOrientation, OriginCorner

    core_positions = blank.get_core_positions(geom, params)
    if not core_positions:
        return "", []

    core_start = geom.core_tip_x - 25.0
    core_end = geom.core_tail_x + 25.0
    core_extent = core_end - core_start

    # Centering offset: same calculation as tab_gcode.py
    x_offset = (blank.length - core_extent) / 2.0 - core_start

    def ski_to_blank(y_ski: float, x_ski: float) -> tuple[float, float]:
        """Convert ski-geometry coords to blank-space coords."""
        return y_ski + x_offset, x_ski

    def transform_to_machine_space(bx: float, by: float) -> tuple[float, float]:
        """Apply machine orientation and origin corner transforms."""
        if blank.machine_orientation == MachineOrientation.Y_AXIS:
            bx, by = by, bx
        if blank.origin_corner == OriginCorner.TOP_LEFT:
            by = blank.width - by
        elif blank.origin_corner == OriginCorner.TOP_RIGHT:
            bx = blank.length - bx
            by = blank.width - by
        elif blank.origin_corner == OriginCorner.BOTTOM_RIGHT:
            bx = blank.length - bx
        return bx, by

    moves: list[Move] = []
    gcode = [
        "G21 G17 G90 G94",
        f"G00 Z{profile_params.clearance_height:.3f}",
        f"S{profile_params.spindle_speed} M03",
    ]

    def rapid(bx, by, bz):
        moves.append(Move(bx, by, bz, is_rapid=True))
        mx, my = transform_to_machine_space(bx, by)
        gcode.append(f"G00 X{mx:.3f} Y{my:.3f} Z{bz:.3f}")

    def feed_move(bx, by, bz, f):
        moves.append(Move(bx, by, bz, is_rapid=False, feed=f))
        mx, my = transform_to_machine_space(bx, by)
        gcode.append(f"G01 X{mx:.3f} Y{my:.3f} Z{bz:.3f} F{f:.1f}")

    # Dense sampling of core thickness along ski length
    y_thickness_samples = np.linspace(core_start, core_end, 500)
    h_core_samples = geom.thickness_at(y_thickness_samples)

    # Blank bottom is at Z=0, top surface at Z=blank.thickness
    # Core bottom surface is at Z=blank.thickness - h_core(y)
    # We cut from blank top down to the core surface

    # Generate pass depths: rough passes + finishing pass
    passes = []  # list of (depth_per_pass, is_finish)
    total_depth = blank.thickness
    z_roughed = 0.0  # current Z after roughing (from top, positive down)

    while z_roughed + profile_params.roughing_depth_per_pass < total_depth - profile_params.finishing_depth_of_cut:
        passes.append((profile_params.roughing_depth_per_pass, False))
        z_roughed += profile_params.roughing_depth_per_pass

    # Finishing pass: remove last finishing_depth_of_cut to surface
    remaining = total_depth - z_roughed
    if remaining > 0.01:
        passes.append((remaining - profile_params.finishing_depth_of_cut, False))
    passes.append((profile_params.finishing_depth_of_cut, True))

    for core_x_along, core_y_across in core_positions:
        # Width of core to cover: extend across full ski width plus tool diameter on each side
        half_w = params.sidewall_width + getattr(params, "sidewall_overlap", 0.0)
        tool_radius = profile_params.tool_diameter / 2.0
        x_ski_min = -half_w - tool_radius
        x_ski_max = half_w + tool_radius

        if profile_params.direction == "along":
            # Passes run along the ski (blank X axis), spaced across width (blank Y axis)
            x_ski_passes = np.arange(x_ski_min, x_ski_max + profile_params.stepover, profile_params.stepover)

            for pass_depth, is_finish in passes:
                for i, x_ski in enumerate(x_ski_passes):
                    bx_start, by = ski_to_blank(core_start, x_ski + core_y_across)
                    bx_end, _ = ski_to_blank(core_end, x_ski + core_y_across)

                    # Plunge at start
                    rapid(bx_start, by, profile_params.clearance_height)
                    z_start = -blank.thickness + h_core_samples[0] + (pass_depth if not is_finish else 0)
                    feed_move(bx_start, by, z_start, profile_params.plunge_feed)

                    # Traverse along Y, Z following core contour
                    ys = y_thickness_samples if i % 2 == 0 else y_thickness_samples[::-1]
                    for j, y_ski in enumerate(ys):
                        bx, _ = ski_to_blank(y_ski, x_ski + core_y_across)
                        # Core surface Z = blank.thickness - h_core(y)
                        # Tool position: go down pass_depth below surface (roughing) or to surface (finish)
                        h_idx = np.searchsorted(y_thickness_samples, y_ski)
                        h_idx = np.clip(h_idx, 0, len(h_core_samples) - 1)
                        h_core = h_core_samples[h_idx]
                        z_surface = -blank.thickness + h_core
                        z_target = z_surface + (pass_depth if not is_finish else 0)
                        feed_move(bx, by, z_target, profile_params.cutting_feed)

                    rapid(bx_end if i % 2 == 0 else bx_start, by, profile_params.clearance_height)

        else:
            # Passes run across the ski (blank Y axis), spaced along length (blank X axis)
            y_ski_passes = np.arange(core_start, core_end + profile_params.stepover, profile_params.stepover)
            x_ski_samples = np.linspace(x_ski_min, x_ski_max, max(10, int((x_ski_max - x_ski_min) / (profile_params.stepover/2))))

            for pass_depth, is_finish in passes:
                for i, y_ski in enumerate(y_ski_passes):
                    bx, by_start = ski_to_blank(y_ski, x_ski_min + core_y_across)
                    _, by_end = ski_to_blank(y_ski, x_ski_max + core_y_across)

                    # Plunge at start
                    rapid(bx, by_start, profile_params.clearance_height)
                    h_idx = np.searchsorted(y_thickness_samples, y_ski)
                    h_idx = np.clip(h_idx, 0, len(h_core_samples) - 1)
                    h_core = h_core_samples[h_idx]
                    z_surface = -blank.thickness + h_core
                    z_start = z_surface + (pass_depth if not is_finish else 0)
                    feed_move(bx, by_start, z_start, profile_params.plunge_feed)

                    # Traverse across Y, Z constant for this Y position
                    xs = x_ski_samples if i % 2 == 0 else x_ski_samples[::-1]
                    for x_ski in xs:
                        _, by = ski_to_blank(y_ski, x_ski + core_y_across)
                        z_target = z_surface + (pass_depth if not is_finish else 0)
                        feed_move(bx, by, z_target, profile_params.cutting_feed)

                    rapid(bx, by_end if i % 2 == 0 else by_start, profile_params.clearance_height)

    gcode += [f"G00 Z{profile_params.clearance_height:.3f}", "M05", "M30"]
    return "\n".join(gcode), moves
