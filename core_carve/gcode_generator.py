"""G-code generation for sidewall slot cutting."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


@dataclass
class SlotParams:
    """CNC machine and tool parameters for slot cutting."""
    # Tool parameters
    tool_diameter: float = 6.0          # mm
    spindle_speed: int = 18000          # RPM

    # Feeds
    cutting_feed: float = 1000.0        # mm/min
    plunge_feed: float = 300.0          # mm/min
    rapid_feed: float = 2000.0          # mm/min (used for G01 rapid xy)

    # Cut strategy
    depth_per_pass: float = 3.0         # mm per pass
    stepover: float = 2.0               # mm lateral step
    spoilboard_skim: float = 0.5        # mm below blank
    clearance_height: float = 10.0      # mm above blank
    stepover_direction: str = "conventional"  # "conventional" or "climb"

    # Tab parameters
    tab_spacing: float = 100.0          # mm between tab centres
    tab_length: float = 10.0            # mm along-slot
    tab_thickness: float = 1.0          # mm height

    def to_json(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> "SlotParams":
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


def generate_slot_gcode(
    geom,
    params,
    blank,
    slot_params: SlotParams,
    x_offset: float = 0.0,
) -> tuple[str, list[Move]]:
    """
    Generate G-code for sidewall slots and return moves list for visualization.

    Args:
        geom: SkiGeometry
        params: SkiParams
        blank: CoreBlank
        slot_params: SlotParams
        x_offset: blank centering offset (from tab_blank calculation)

    Returns:
        (gcode_string, moves_list)
    """
    from core_carve.ski_geometry import half_widths_at_y
    from core_carve.core_blank import MachineOrientation, OriginCorner

    # Slot geometry: left and right edges of each core
    core_positions = blank.get_core_positions(geom, params)
    core_start = geom.core_tip_x - 25.0
    core_end = geom.core_tail_x + 25.0
    y_samples = np.linspace(core_start, core_end, 500)
    left_outline, right_outline = half_widths_at_y(geom.outline, y_samples)

    # Function to transform coordinates based on machine origin and orientation
    def transform_to_machine_space(x, y):
        """Transform blank-space coords to machine coords based on origin and orientation."""
        # Swap axes if ski runs along Y
        if blank.machine_orientation == MachineOrientation.Y_AXIS:
            x, y = y, x

        # Flip based on origin corner
        if blank.origin_corner == OriginCorner.TOP_LEFT:
            y = blank.width - y
        elif blank.origin_corner == OriginCorner.TOP_RIGHT:
            x = blank.length - x
            y = blank.width - y
        elif blank.origin_corner == OriginCorner.BOTTOM_RIGHT:
            x = blank.length - x

        return x, y

    # Slot dimensions
    slot_width = params.sidewall_width
    slot_depth = blank.thickness + slot_params.spoilboard_skim
    core_offset = params.sidewall_width - params.sidewall_overlap

    # Collect all slot centerlines
    slots = []  # list of (y_samples_offset, slot_centerline_y) tuples
    for core_x, core_y in core_positions:
        # Left slot
        slot_left_center = left_outline + core_offset + core_y
        slots.append((y_samples_offset := y_samples + x_offset, slot_left_center))

        # Right slot
        slot_right_center = right_outline - core_offset + core_y
        slots.append((y_samples_offset, slot_right_center))

    # Build moves and G-code
    moves = []
    gcode_lines = ["G21 G17 G90 G94", f"G00 Z{slot_params.clearance_height:.3f}", f"S{slot_params.spindle_speed} M03"]

    # Multi-pass cutting: depth passes × width passes per slot
    n_depth_passes = int(np.ceil(slot_depth / slot_params.depth_per_pass))
    n_width_passes = max(1, int(np.ceil((slot_width - slot_params.tool_diameter) / slot_params.stepover + 1)))
    width_offsets = np.linspace(
        -slot_width / 2 + slot_params.tool_diameter / 2,
        slot_width / 2 - slot_params.tool_diameter / 2,
        n_width_passes,
    )

    # Apply stepover direction (climb vs conventional)
    if slot_params.stepover_direction == "climb":
        width_offsets = width_offsets[::-1]

    # Pre-calculate tab zones for each slot
    tab_zones = []
    for slot_y_samples, slot_centerline in slots:
        # Arc-length parameterization to find tab positions
        arc_length = np.cumsum(np.concatenate([[0], np.linalg.norm(np.diff(np.column_stack([slot_y_samples, slot_centerline])), axis=1)]))
        total_arc = arc_length[-1]
        tab_centers_arc = np.arange(0, total_arc, slot_params.tab_spacing)
        tab_start_arc = tab_centers_arc - slot_params.tab_length / 2
        tab_end_arc = tab_centers_arc + slot_params.tab_length / 2
        tab_zones.append((arc_length, tab_start_arc, tab_end_arc))

    # Generate toolpaths, alternating direction to minimize rapids between slots
    for depth_pass in range(n_depth_passes):
        z_target = -min((depth_pass + 1) * slot_params.depth_per_pass, slot_depth)

        for width_offset in width_offsets:
            for slot_idx, (slot_y_samples, slot_centerline) in enumerate(slots):
                arc_length, tab_start_arc, tab_end_arc = tab_zones[slot_idx]

                # Alternate direction: even indices forward, odd indices backward
                reverse_cut = slot_idx % 2 == 1
                if reverse_cut:
                    cut_indices = np.arange(len(slot_y_samples) - 1, -1, -1)
                else:
                    cut_indices = np.arange(len(slot_y_samples))

                # Rapid to slot start (or end if reversing) at clearance height
                start_idx = cut_indices[0]
                start_x = slot_y_samples[start_idx]
                start_y = slot_centerline[start_idx] + width_offset
                moves.append(Move(start_x, start_y, slot_params.clearance_height, is_rapid=True, feed=slot_params.rapid_feed))
                mach_x, mach_y = transform_to_machine_space(start_x, start_y)
                gcode_lines.append(f"G00 X{mach_x:.3f} Y{mach_y:.3f}")
                gcode_lines.append(f"G00 Z{slot_params.clearance_height:.3f}")

                # Plunge
                moves.append(Move(start_x, start_y, z_target, is_rapid=False, feed=slot_params.plunge_feed))
                gcode_lines.append(f"G01 Z{z_target:.3f} F{slot_params.plunge_feed:.1f}")

                # Cut along slot in chosen direction, respecting tabs
                for i in cut_indices:
                    x = slot_y_samples[i]
                    y = slot_centerline[i] + width_offset
                    arc = arc_length[i]
                    # Check if in a tab zone
                    in_tab = any(start <= arc <= end for start, end in zip(tab_start_arc, tab_end_arc))
                    z = max(z_target, -(slot_depth - slot_params.tab_thickness)) if in_tab else z_target
                    feed = slot_params.cutting_feed
                    moves.append(Move(x, y, z, is_rapid=False, feed=feed))
                    mach_x, mach_y = transform_to_machine_space(x, y)
                    gcode_lines.append(f"G01 X{mach_x:.3f} Y{mach_y:.3f} Z{z:.3f} F{feed:.1f}")

                # Rapid to clearance
                end_idx = cut_indices[-1]
                end_x, end_y = slot_y_samples[end_idx], slot_centerline[end_idx] + width_offset
                moves.append(Move(end_x, end_y, slot_params.clearance_height, is_rapid=True, feed=slot_params.rapid_feed))
                end_mach_x, end_mach_y = transform_to_machine_space(end_x, end_y)
                gcode_lines.append(f"G00 X{end_mach_x:.3f} Y{end_mach_y:.3f}")
                gcode_lines.append(f"G00 Z{slot_params.clearance_height:.3f}")

    # Postamble
    gcode_lines.append(f"G00 Z{slot_params.clearance_height:.3f}")
    gcode_lines.append("M05")
    gcode_lines.append("M30")

    gcode_string = "\n".join(gcode_lines)
    return gcode_string, moves
