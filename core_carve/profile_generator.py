"""G-code generation for core thickness profiling."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


@dataclass
class ProfileParams:
    """CNC machine and tool parameters for thickness profiling."""
    # Tool parameters
    tool_diameter: float = 12.0         # mm
    spindle_speed: int = 12000          # RPM

    # Feeds
    cutting_feed: float = 800.0         # mm/min
    plunge_feed: float = 200.0          # mm/min
    rapid_feed: float = 1500.0          # mm/min

    # Cut strategy
    roughing_depth_per_pass: float = 2.0   # mm per pass
    finishing_depth_of_cut: float = 0.5    # mm final pass
    stepover: float = 3.0                  # mm lateral step
    clearance_height: float = 10.0         # mm above blank
    direction: str = "along"               # "along" (ski length) or "across" (width)

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
    """A single CNC move for visualization and G-code output."""
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
    Generate G-code for core thickness profiling.

    Args:
        geom: SkiGeometry
        params: SkiParams
        blank: CoreBlank
        profile_params: ProfileParams

    Returns:
        (gcode_string, moves_list)
    """
    from core_carve.ski_geometry import half_widths_at_y

    core_positions = blank.get_core_positions(geom, params)
    if not core_positions:
        return "", []

    # Determine cutting path direction
    if profile_params.direction == "along":
        # Cuts run along ski length (Y axis)
        core_start = geom.core_tip_x - 25.0
        core_end = geom.core_tail_x + 25.0
        n_cuts = max(2, int(np.ceil(params.sidewall_width * 2 / profile_params.stepover)))

        # Position cuts across core width
        x_positions = np.linspace(
            -params.sidewall_width,
            params.sidewall_width,
            n_cuts
        )
    else:
        # Cuts run across ski width (X axis)
        core_start = geom.core_tip_x - 25.0
        core_end = geom.core_tail_x + 25.0

        # Create regular cuts along ski length
        y_samples = np.linspace(core_start, core_end, int(np.ceil((core_end - core_start) / profile_params.stepover)))
        x_positions = y_samples

    # Thickness profile simulation (simplified parabolic profile)
    def get_profile_depth(x, y, core_x, core_y):
        """Simplified profile: deepest at center, feather to edges."""
        dist_from_center = np.sqrt((x - core_x) ** 2 + (y - core_y) ** 2)
        max_dist = params.sidewall_width
        if dist_from_center >= max_dist:
            return 0.0
        # Parabolic taper
        return (params.sidewall_width * 0.5) * (1.0 - (dist_from_center / max_dist) ** 2)

    moves = []
    gcode_lines = ["G21 G17 G90 G94", f"G00 Z{profile_params.clearance_height:.3f}", f"S{profile_params.spindle_speed} M03"]

    # Generate toolpaths for each core
    for core_idx, (core_x, core_y) in enumerate(core_positions):
        if profile_params.direction == "along":
            # Cuts along ski length
            y_samples = np.linspace(core_start, core_end, 200)

            for x_offset in x_positions:
                x = core_x + x_offset

                # Roughing pass
                z_roughing = -profile_params.roughing_depth_per_pass
                moves.append(Move(x, y_samples[0] + core_y, profile_params.clearance_height, is_rapid=True))
                gcode_lines.append(f"G00 X{x:.3f} Y{y_samples[0] + core_y:.3f}")
                gcode_lines.append(f"G00 Z{profile_params.clearance_height:.3f}")

                moves.append(Move(x, y_samples[0] + core_y, z_roughing, is_rapid=False, feed=profile_params.plunge_feed))
                gcode_lines.append(f"G01 Z{z_roughing:.3f} F{profile_params.plunge_feed:.1f}")

                for y in y_samples:
                    moves.append(Move(x, y + core_y, z_roughing, is_rapid=False, feed=profile_params.cutting_feed))
                    gcode_lines.append(f"G01 X{x:.3f} Y{y + core_y:.3f} F{profile_params.cutting_feed:.1f}")

                moves.append(Move(x, y_samples[-1] + core_y, profile_params.clearance_height, is_rapid=True))
                gcode_lines.append(f"G00 Z{profile_params.clearance_height:.3f}")

        else:
            # Cuts across ski width
            for y_pos in x_positions:
                y = core_start + y_pos if profile_params.direction == "across" else y_pos

                # Roughing pass across width
                x_range = np.linspace(-params.sidewall_width, params.sidewall_width, 100)
                z_roughing = -profile_params.roughing_depth_per_pass

                moves.append(Move(core_x + x_range[0], y + core_y, profile_params.clearance_height, is_rapid=True))
                gcode_lines.append(f"G00 X{core_x + x_range[0]:.3f} Y{y + core_y:.3f}")
                gcode_lines.append(f"G00 Z{profile_params.clearance_height:.3f}")

                moves.append(Move(core_x + x_range[0], y + core_y, z_roughing, is_rapid=False, feed=profile_params.plunge_feed))
                gcode_lines.append(f"G01 Z{z_roughing:.3f} F{profile_params.plunge_feed:.1f}")

                for x_offset in x_range:
                    moves.append(Move(core_x + x_offset, y + core_y, z_roughing, is_rapid=False, feed=profile_params.cutting_feed))
                    gcode_lines.append(f"G01 X{core_x + x_offset:.3f} Y{y + core_y:.3f} F{profile_params.cutting_feed:.1f}")

                moves.append(Move(core_x + x_range[-1], y + core_y, profile_params.clearance_height, is_rapid=True))
                gcode_lines.append(f"G00 Z{profile_params.clearance_height:.3f}")

    # Postamble
    gcode_lines.append(f"G00 Z{profile_params.clearance_height:.3f}")
    gcode_lines.append("M05")
    gcode_lines.append("M30")

    gcode_string = "\n".join(gcode_lines)
    return gcode_string, moves
