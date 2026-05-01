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
    cutting_direction: str = "both"  # "conventional", "climb", or "both"

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

    Strategy:
    - Roughing passes cut at a fixed flat depth (from blank top), rising only where
      the core surface protrudes above that depth. Leaves finishing_depth_of_cut
      above the final surface.
    - Finishing pass follows the exact core surface contour.
    - "Along" direction passes split in two halves: tip→center and tail→center.
    - Width extends past the outer ski outline edge by one tool radius each side.

    Blank space:
      X = along blank = ski_Y + x_offset  (ski runs along blank X)
      Y = across blank = ski_X + blank.width/2

    Returns:
        (gcode_string, moves_list)  — moves are in blank-space absolute coords.
    """
    from core_carve.core_blank import MachineOrientation, OriginCorner
    from core_carve.ski_geometry import half_widths_at_y

    core_positions = blank.get_core_positions(geom, params)
    if not core_positions:
        return "", []

    core_start = geom.core_tip_x - 25.0
    core_end = geom.core_tail_x + 25.0
    core_extent = core_end - core_start
    x_offset = (blank.length - core_extent) / 2.0 - core_start

    def ski_to_blank(y_ski: float, x_ski: float) -> tuple[float, float]:
        return y_ski + x_offset, x_ski + blank.width / 2.0

    def transform_to_machine_space(bx: float, by: float) -> tuple[float, float]:
        if blank.machine_orientation == MachineOrientation.Y_AXIS:
            bx, by = by, bx
        if blank.origin_corner == OriginCorner.TOP_LEFT:
            by = by - blank.width
        elif blank.origin_corner == OriginCorner.TOP_RIGHT:
            bx = bx - blank.length
            by = by - blank.width
        elif blank.origin_corner == OriginCorner.BOTTOM_RIGHT:
            bx = bx - blank.length
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

    # Dense sampling along ski length
    y_samps = np.linspace(core_start, core_end, 500)

    # Clamp to core extent for thickness evaluation so extensions use tip/tail thickness
    y_for_h = np.clip(y_samps, geom.core_tip_x, geom.core_tail_x)
    h_samps = geom.thickness_at(y_for_h)

    # Precompute ski outline half-widths at each sample (clamp to outline bounds)
    outline_y_min = geom.outline[:, 1].min()
    outline_y_max = geom.outline[:, 1].max()
    y_for_outline = np.clip(y_samps, outline_y_min, outline_y_max)
    left_samps, right_samps = half_widths_at_y(geom.outline, y_for_outline)

    def z_target_at(idx: int, z_flat, is_finish: bool) -> float:
        """Z cutting target: flat roughing depth clamped by surface + finishing allowance."""
        z_surface = -blank.thickness + h_samps[idx]
        if is_finish:
            return z_surface
        # Leave finishing_depth_of_cut above the surface during roughing
        z_limit = z_surface + profile_params.finishing_depth_of_cut
        return max(z_flat, z_limit)

    # Build pass list: (z_flat, is_finish, z_flat_prev)
    # z_flat_prev is the previous pass depth — used to know when to stop following the surface
    # (once the surface+allowance reaches the previous pass level, that region was already cleared)
    passes: list[tuple] = []
    cumulative = 0.0
    z_prev = 0.0  # blank top surface = no previous cut
    while cumulative + profile_params.roughing_depth_per_pass < blank.thickness:
        cumulative += profile_params.roughing_depth_per_pass
        passes.append((-cumulative, False, z_prev))
        z_prev = -cumulative
    passes.append((None, True, z_prev))

    # Split point: center of underfoot region (where core surface is highest)
    y_split = (geom.underfoot_start_x + geom.underfoot_end_x) / 2.0
    split_idx = int(np.clip(np.searchsorted(y_samps, y_split), 1, len(y_samps) - 2))

    def useful_indices(natural_idx, z_flat, is_finish, z_flat_prev):
        """Return the prefix of natural_idx for one roughing pass half.

        The pass cuts at z_flat (flat depth) until the surface+allowance rises above z_flat,
        then follows the surface uphill until it reaches z_flat_prev (the previous pass depth).
        Everything above z_flat_prev was already removed by the prior pass.
        For finishing: return all indices (full contour).
        """
        if is_finish:
            return list(natural_idx)
        result = []
        for idx in natural_idx:
            z = z_target_at(idx, z_flat, is_finish)
            if z >= z_flat_prev - 0.01:
                break
            result.append(idx)
        return result

    for core_x_along, core_y_across in core_positions:
        tool_radius = profile_params.tool_diameter / 2.0

        if profile_params.direction == "along":
            # Width: use max ski outline extent + tool_radius on each side
            valid_left = left_samps[~np.isnan(left_samps)]
            valid_right = right_samps[~np.isnan(right_samps)]
            if len(valid_left) == 0 or len(valid_right) == 0:
                continue
            x_ski_min = float(np.min(valid_left)) - tool_radius
            x_ski_max = float(np.max(valid_right)) + tool_radius
            x_ski_passes = np.arange(x_ski_min, x_ski_max + profile_params.stepover, profile_params.stepover)

            # Natural traversal order for each half — always working inward
            tip_natural = np.arange(0, split_idx + 1)           # tip → center
            tail_natural = np.arange(len(y_samps) - 1, split_idx - 1, -1)  # tail → center

            for z_flat, is_finish, z_flat_prev in passes:
                for x_ski in x_ski_passes:
                    _, by = ski_to_blank(y_samps[0], x_ski + core_y_across)

                    for half_natural in (tip_natural, tail_natural):
                        half = useful_indices(half_natural, z_flat, is_finish, z_flat_prev)
                        if len(half) < 2:
                            continue
                        bx0, _ = ski_to_blank(y_samps[half[0]], x_ski + core_y_across)
                        rapid(bx0, by, profile_params.clearance_height)
                        feed_move(bx0, by, z_target_at(half[0], z_flat, is_finish), profile_params.plunge_feed)
                        for idx in half[1:]:
                            bx, _ = ski_to_blank(y_samps[idx], x_ski + core_y_across)
                            feed_move(bx, by, z_target_at(idx, z_flat, is_finish), profile_params.cutting_feed)
                        bx_end, _ = ski_to_blank(y_samps[half[-1]], x_ski + core_y_across)
                        rapid(bx_end, by, profile_params.clearance_height)

        else:
            # "across" direction: passes run across the ski width at each Y position.
            # Split into tip half (core_start → center) and tail half (core_end → center)
            # so the tool always machines uphill toward the thicker center.
            # Roughing passes stop when the core surface protrudes above the flat cut depth.
            # For "both" cutting direction, adjacent passes share a turnaround point so the
            # tool links directly without retracting between passes within each half.
            tip_y_passes = np.arange(core_start, y_split + profile_params.stepover, profile_params.stepover)
            tail_y_passes = np.arange(core_end, y_split - profile_params.stepover, -profile_params.stepover)

            for z_flat, is_finish, z_flat_prev in passes:
                for half_y_passes in (tip_y_passes, tail_y_passes):
                    prev_end_bx: float | None = None
                    prev_end_by: float | None = None

                    for i, y_ski in enumerate(half_y_passes):
                        h_idx = int(np.clip(np.searchsorted(y_samps, y_ski), 0, len(h_samps) - 1))

                        # Width from actual outline at this Y position
                        y_clamped = float(np.clip(y_ski, outline_y_min, outline_y_max))
                        l_arr, r_arr = half_widths_at_y(geom.outline, np.array([y_clamped]))
                        if np.isnan(l_arr[0]) or np.isnan(r_arr[0]):
                            prev_end_bx = prev_end_by = None
                            continue
                        x_ski_min = float(l_arr[0]) - tool_radius
                        x_ski_max = float(r_arr[0]) + tool_radius
                        n_pts = max(10, int((x_ski_max - x_ski_min) / (profile_params.stepover / 2)))
                        x_ski_samples = np.linspace(x_ski_min, x_ski_max, n_pts)

                        z_cut = z_target_at(h_idx, z_flat, is_finish)

                        # Stop when the surface+allowance has risen to the previous pass depth —
                        # that region was already cleared. All remaining positions are thicker.
                        if not is_finish and z_cut >= z_flat_prev - 0.01:
                            break

                        should_reverse = i % 2 == 1
                        if profile_params.cutting_direction == "conventional":
                            should_reverse = False
                        elif profile_params.cutting_direction == "climb":
                            should_reverse = True

                        xs = x_ski_samples[::-1] if should_reverse else x_ski_samples
                        bx, by_start = ski_to_blank(y_ski, float(xs[0]) + core_y_across)

                        # For "both" mode: adjacent passes share a turnaround point — link directly
                        use_link = (
                            profile_params.cutting_direction == "both"
                            and prev_end_bx is not None
                        )
                        if use_link:
                            feed_move(bx, by_start, z_cut, profile_params.cutting_feed)
                        else:
                            if prev_end_bx is not None:
                                rapid(prev_end_bx, prev_end_by, profile_params.clearance_height)
                            rapid(bx, by_start, profile_params.clearance_height)
                            feed_move(bx, by_start, z_cut, profile_params.plunge_feed)

                        for x_ski in xs[1:]:
                            _, by = ski_to_blank(y_ski, float(x_ski) + core_y_across)
                            feed_move(bx, by, z_cut, profile_params.cutting_feed)

                        prev_end_bx = bx
                        _, prev_end_by = ski_to_blank(y_ski, float(xs[-1]) + core_y_across)

                    # Retract after each half
                    if prev_end_bx is not None:
                        rapid(prev_end_bx, prev_end_by, profile_params.clearance_height)

    gcode += [f"G00 Z{profile_params.clearance_height:.3f}", "M05", "M30"]
    return "\n".join(gcode), moves
