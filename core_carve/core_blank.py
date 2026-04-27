"""Core blank design: size, positioning, and CNC layout."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import numpy as np


class MachineOrientation(Enum):
    """Ski orientation in CNC machine."""
    X_AXIS = "X"  # Ski length along machine X axis
    Y_AXIS = "Y"  # Ski length along machine Y axis


class OriginCorner(Enum):
    """Which corner of the blank holds the machine origin (0, 0)."""
    TOP_LEFT = "TL"
    TOP_RIGHT = "TR"
    BOTTOM_LEFT = "BL"
    BOTTOM_RIGHT = "BR"


@dataclass
class CoreBlank:
    """Blank stock dimensions and core positioning."""
    # Blank dimensions
    length: float = 2000.0      # mm, along ski length
    width: float = 200.0        # mm, across ski width
    thickness: float = 15.0     # mm

    # Core positioning
    num_cores: int = 1          # 1 or 2
    machine_orientation: MachineOrientation = MachineOrientation.X_AXIS
    origin_corner: OriginCorner = OriginCorner.BOTTOM_LEFT

    # Position adjustments from default (mm)
    position_offset_x: float = 0.0  # along ski length
    position_offset_y: float = 0.0  # across ski width

    # Spacing between core centerlines (for 2 cores)
    core_spacing: float | None = None  # mm (only used when num_cores == 2); None = auto-calculated

    def get_core_positions(self, geom, params) -> list[tuple[float, float]]:
        """
        Compute the (x, y) positions of core(s) within the blank.

        Returns list of (x_center, y_center) tuples for each core.
        x = position along ski length (0 = tip infill start)
        y = position across ski width (0 = centerline, ± = left/right)
        """
        # Default X position: centered along the extended core length
        # The core extends 25mm beyond infill regions on each end
        core_extended_length = geom.core_tail_x - geom.core_tip_x + 50.0
        default_x = geom.core_tip_x - 25.0 + core_extended_length / 2.0

        if self.num_cores == 1:
            # Single core: centered in blank
            default_y = 0.0
            positions = [(default_x, default_y)]
        else:  # 2 cores
            # Calculate spacing to maintain ~15mm gap at widest point
            spacing = self.core_spacing
            if spacing is None:
                # Find max ski width in the core region
                from core_carve.ski_geometry import half_widths_at_y
                core_start = geom.core_tip_x - 25.0
                core_end = geom.core_tail_x + 25.0
                y_samples = np.linspace(core_start, core_end, 100)
                left_w, right_w = half_widths_at_y(geom.outline, y_samples)
                max_half_width = max(np.nanmax(np.abs(left_w)), np.nanmax(np.abs(right_w)))
                spacing = 15.0 + 2.0 * max_half_width

            positions = [
                (default_x, -spacing / 2.0),
                (default_x, spacing / 2.0),
            ]

        # Apply position adjustments
        adjusted = []
        for x, y in positions:
            adjusted.append((x + self.position_offset_x, y + self.position_offset_y))

        return adjusted

    def validate(self, geom, params) -> tuple[bool, list[str]]:
        """
        Validate blank size and positioning.

        Returns (is_valid, list_of_warnings).
        """
        warnings = []

        # Check if blank is large enough for ski + extensions
        ski_length = geom.ski_length + 50.0  # 25mm on each end
        if self.length < ski_length:
            warnings.append(
                f"Blank length ({self.length:.0f} mm) is less than required ({ski_length:.0f} mm)"
            )

        # Check if cores fit within blank after positioning
        positions = self.get_core_positions(geom, params)
        core_length = geom.core_tail_x - geom.core_tip_x + 50.0  # 25mm extension on each end
        core_width_half = 70.0  # rough estimate

        for i, (x, y) in enumerate(positions):
            # Check along length
            if x - core_length / 2.0 < 0 or x + core_length / 2.0 > self.length:
                warnings.append(
                    f"Core {i + 1} extends outside blank length bounds"
                )

            # Check across width
            if abs(y) + core_width_half > self.width / 2.0:
                warnings.append(
                    f"Core {i + 1} extends outside blank width bounds"
                )

        return len(warnings) == 0, warnings
