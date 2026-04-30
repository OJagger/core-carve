"""Mould cross-section generation and G-code output."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import ezdxf


@dataclass
class LayerInfo:
    """Info about a single layer in the cross-section."""
    name: str
    thickness: float  # mm
    color: str
    z_bottom: float  # mm from base


def compute_mould_section(
    y_pos: float,
    geom,
    core_params,
    materials_db,
    layup,
    topsheet_mass_per_area: float = 50.0,
) -> tuple[list[tuple[float, float]], list[LayerInfo]]:
    """
    Compute 2D cross-section profile at position y along ski.

    Args:
        y_pos: position along ski (0 = tip, L = tail)
        geom: SkiGeometry object
        core_params: SkiParams object
        materials_db: MaterialDatabase object
        layup: LayupConfig object
        topsheet_mass_per_area: topsheet areal density in g/m²

    Returns:
        (outline_points, layers_info)
        outline_points: list of (x, z) tuples forming closed polygon
        layers_info: list of LayerInfo for visualization
    """
    from core_carve.ski_geometry import half_widths_at_y

    # Get material objects
    core_mat = materials_db.get(layup.core_material)
    base_mat = materials_db.get(layup.base_material)
    sw_mat = materials_db.get(layup.sidewall_material)

    # Get dimensions at this Y position
    left_w, right_w = half_widths_at_y(geom.outline, np.array([y_pos]))
    width = float(right_w[0] - left_w[0]) if len(right_w) > 0 else 0.0

    # Core thickness
    h_core = 0.0
    if geom.core_tip_x <= y_pos <= geom.core_tail_x:
        h_core = float(geom.thickness_at(np.array([y_pos]))[0])

    # Base thickness
    t_base = base_mat.thickness if base_mat else 0.0

    # Laminate thicknesses
    from core_carve.ski_mechanics import _laminate_props
    t_top, _, _, _ = _laminate_props(layup.top_layers, materials_db)
    bot_layers = layup.effective_bottom()
    t_bot, _, _, _ = _laminate_props(bot_layers, materials_db)

    # Topsheet thickness (approximate from areal density)
    # density ≈ 1200 kg/m³ for typical topsheet
    topsheet_density = 1200  # kg/m³
    t_topsheet = (topsheet_mass_per_area / 1e6) / (topsheet_density * 1e-6) if topsheet_mass_per_area > 0 else 0.5

    # Sidewall thickness
    t_sw = sw_mat.thickness if sw_mat else 0.0

    # Build outline: closed polygon with base at z=0, centered at x=0
    # Coordinates: x = horizontal (width), z = vertical (stacked)
    outline = []

    # Build layer stack (z positions)
    z_base_top = t_base
    z_bot_top = z_base_top + t_bot
    z_core_top = z_bot_top + h_core
    z_top_top = z_core_top + t_top
    z_topsheet_top = z_top_top + t_topsheet

    # Cross-section profile (side view)
    # Starting at bottom-left, go clockwise

    # Bottom-left corner
    x_left = -width / 2.0
    x_right = width / 2.0

    outline.append((x_left, 0.0))  # base bottom-left
    outline.append((x_right, 0.0))  # base bottom-right
    outline.append((x_right, z_base_top))  # base top-right
    outline.append((x_right, z_topsheet_top))  # topsheet top-right

    # Top surface (topsheet)
    outline.append((x_right, z_topsheet_top))
    outline.append((x_left, z_topsheet_top))

    # Left edge down
    outline.append((x_left, z_base_top))
    outline.append((x_left, 0.0))

    # Close path
    outline = [(x, z) for x, z in outline]

    # Remove duplicates while preserving order
    seen = set()
    outline_clean = []
    for pt in outline:
        if pt not in seen:
            outline_clean.append(pt)
            seen.add(pt)

    # Build layer info for visualization
    layers = []
    if t_base > 0 and base_mat:
        layers.append(LayerInfo(
            name=f"Base ({base_mat.name})",
            thickness=t_base,
            color=base_mat.color,
            z_bottom=0.0,
        ))
    if t_bot > 0:
        layers.append(LayerInfo(
            name="Bottom laminate",
            thickness=t_bot,
            color="#80c0ff",
            z_bottom=z_base_top,
        ))
    if h_core > 0 and core_mat:
        layers.append(LayerInfo(
            name=f"Core ({core_mat.name})",
            thickness=h_core,
            color=core_mat.color,
            z_bottom=z_bot_top,
        ))
    if t_top > 0:
        layers.append(LayerInfo(
            name="Top laminate",
            thickness=t_top,
            color="#80c0ff",
            z_bottom=z_core_top,
        ))
    if t_topsheet > 0:
        layers.append(LayerInfo(
            name="Topsheet",
            thickness=t_topsheet,
            color="#e0e080",
            z_bottom=z_top_top,
        ))

    return outline_clean, layers


def write_mould_dxf(outline: list[tuple[float, float]], filename: str):
    """
    Write mould cross-section as DXF polyline.

    Args:
        outline: list of (x, z) points forming closed polygon
        filename: output DXF file path
    """
    doc = ezdxf.new()
    msp = doc.modelspace()

    # Add polyline
    if outline:
        msp.add_lwpolyline(outline)

    # Save
    doc.saveas(filename)


def generate_mould_gcode(
    outline: list[tuple[float, float]],
    slot_params,
    position_name: str = "section",
) -> str:
    """
    Generate G-code to cut the mould profile.

    Args:
        outline: list of (x, z) points forming closed profile
        slot_params: SlotParams for feed/speed/depth
        position_name: descriptive name for the cut

    Returns:
        G-code string
    """
    gcode_lines = [
        f"( Mould profile: {position_name} )",
        "G21 G17 G90 G94",
        f"G00 Z{slot_params.clearance_height:.3f}",
        f"S{slot_params.spindle_speed} M03",
    ]

    if not outline:
        gcode_lines.extend(["M05", "M30"])
        return "\n".join(gcode_lines)

    # Move to first point (in blank space, Y is fixed at 0)
    x0, z0 = outline[0]
    gcode_lines.append(f"G00 X{x0:.3f} Y0.0")
    gcode_lines.append(f"G00 Z{slot_params.clearance_height:.3f}")

    # Plunge to cut depth
    gcode_lines.append(f"G01 Z{-slot_params.depth_per_pass:.3f} F{slot_params.plunge_feed:.1f}")

    # Trace outline (note: Z coordinate is used for the vertical dimension)
    for x, z in outline[1:]:
        # In the mould context, z from the cross-section becomes the actual Z (depth)
        actual_z = -z / 100.0  # Scale and invert for machine depth (rough conversion)
        gcode_lines.append(f"G01 X{x:.3f} Y0.0 Z{actual_z:.3f} F{slot_params.cutting_feed:.1f}")

    # Close and retract
    gcode_lines.append(f"G00 Z{slot_params.clearance_height:.3f}")
    gcode_lines.append("M05")
    gcode_lines.append("M30")

    return "\n".join(gcode_lines)
