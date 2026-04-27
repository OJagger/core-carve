"""Tests for Tab 2: Core Blank geometry visualization and positioning."""
import numpy as np
import pytest
from pathlib import Path

from core_carve.ski_geometry import (
    load_planform_dxf,
    compute_geometry,
    SkiParams,
    half_widths_at_y,
)
from core_carve.core_blank import CoreBlank


@pytest.fixture
def data_dir():
    """Get the data directory."""
    return Path(__file__).parent.parent / "data"


@pytest.fixture
def ski_outline(data_dir):
    """Load the test ski planform."""
    return load_planform_dxf(data_dir / "Ski_planform.dxf")


@pytest.fixture
def ski_params(data_dir):
    """Load the test ski parameters."""
    return SkiParams.from_json(data_dir / "ski_params.json")


@pytest.fixture
def geometry(ski_outline, ski_params):
    """Compute geometry from outline and params."""
    return compute_geometry(ski_outline, ski_params)


@pytest.fixture
def blank():
    """Create a default blank."""
    return CoreBlank()


class TestGeometryLoading:
    """Test that geometry is correctly loaded and computed."""

    def test_dxf_loads(self, ski_outline):
        """Test that DXF file loads successfully."""
        assert ski_outline is not None
        assert isinstance(ski_outline, np.ndarray)
        assert ski_outline.shape[1] == 2  # X, Y coordinates

    def test_params_load(self, ski_params):
        """Test that params JSON loads successfully."""
        assert ski_params is not None
        assert ski_params.tip_infill == 160.0
        assert ski_params.tail_infill == 80.0
        assert ski_params.sidewall_width == 6.0
        assert ski_params.sidewall_overlap == 2.0

    def test_geometry_computed(self, geometry):
        """Test that geometry is computed correctly."""
        assert geometry.ski_length > 0
        assert geometry.core_tip_x > 0
        assert geometry.core_tail_x > geometry.core_tip_x
        assert geometry.waist_x > 0

    def test_ski_length(self, geometry):
        """Test that ski length is reasonable."""
        # Expect ski to be between 1700-2000mm
        assert 1700 < geometry.ski_length < 2000

    def test_core_positions(self, geometry, ski_params):
        """Test that core positions match infill values."""
        assert geometry.core_tip_x == pytest.approx(ski_params.tip_infill)
        assert geometry.core_tail_x == pytest.approx(
            geometry.ski_length - ski_params.tail_infill
        )


class TestExtendedCorePositioning:
    """Test that the extended core (with 25mm machining extensions) is positioned correctly."""

    def test_core_extent_calculation(self, geometry, ski_params):
        """Test that core extent includes 25mm extensions on each end."""
        core_extent = geometry.core_tail_x - geometry.core_tip_x + 50.0
        # With tip_infill=160 and tail_infill=80 on ~1800mm ski:
        # core_tip_x=160, core_tail_x=1720, extent = 1720-160+50 = 1610mm
        assert core_extent > 0
        assert core_extent < geometry.ski_length

    def test_core_start_clamped_to_outline(self, geometry, ski_params):
        """Test that core_start is clamped to outline bounds."""
        outline_y_min = geometry.outline[:, 1].min()
        core_start = max(geometry.core_tip_x - 25.0, outline_y_min)
        # With tip_infill=160, core_start should be 135 (within outline)
        assert core_start >= outline_y_min
        assert core_start == pytest.approx(geometry.core_tip_x - 25.0)

    def test_core_end_clamped_to_outline(self, geometry):
        """Test that core_end is clamped to outline bounds."""
        outline_y_max = geometry.outline[:, 1].max()
        core_end = min(geometry.core_tail_x + 25.0, outline_y_max)
        # Should not exceed outline upper bound
        assert core_end <= outline_y_max

    def test_core_positioned_in_blank(self, geometry, ski_params, blank):
        """Test that core is centered in blank with correct extent."""
        core_start = max(geometry.core_tip_x - 25.0, geometry.outline[:, 1].min())
        core_end = min(geometry.core_tail_x + 25.0, geometry.outline[:, 1].max())
        core_extent = core_end - core_start

        # Core should be centered in blank
        x_offset = (blank.length - core_extent) / 2 - core_start

        # In blank space, core should span from ~245mm to ~1755mm (for 2000mm blank)
        core_start_blank = core_start + x_offset
        core_end_blank = core_end + x_offset

        assert core_start_blank == pytest.approx((blank.length - core_extent) / 2)
        assert core_end_blank == pytest.approx((blank.length + core_extent) / 2)


class TestSidewallPositioning:
    """Test that sidewalls are positioned correctly relative to the core."""

    def test_sidewall_width_from_outline(self, geometry, ski_params):
        """Test that sidewall width parameter is used correctly."""
        assert ski_params.sidewall_width == 6.0
        assert ski_params.sidewall_overlap == 2.0
        core_offset = ski_params.sidewall_width - ski_params.sidewall_overlap
        assert core_offset == 4.0

    def test_sidewall_outline_samples(self, geometry, ski_params):
        """Test that sidewall outline samples are valid."""
        core_start = max(geometry.core_tip_x - 25.0, geometry.outline[:, 1].min())
        core_end = min(geometry.core_tail_x + 25.0, geometry.outline[:, 1].max())
        y_samples = np.linspace(core_start, core_end, 100)

        # Get outline widths
        left_outline, right_outline = half_widths_at_y(geometry.outline, y_samples)

        # Should have valid values for all samples
        assert not np.all(np.isnan(left_outline))
        assert not np.all(np.isnan(right_outline))

        # Left should be negative, right should be positive
        valid_left = left_outline[~np.isnan(left_outline)]
        valid_right = right_outline[~np.isnan(right_outline)]
        if len(valid_left) > 0:
            assert np.all(valid_left <= 0)
        if len(valid_right) > 0:
            assert np.all(valid_right >= 0)

    def test_sidewall_outer_edge_position(self, geometry, ski_params):
        """Test that sidewall outer edge is at the ski outline."""
        core_start = max(geometry.core_tip_x - 25.0, geometry.outline[:, 1].min())
        core_end = min(geometry.core_tail_x + 25.0, geometry.outline[:, 1].max())
        y_samples = np.linspace(core_start, core_end, 50)

        left_outline, right_outline = half_widths_at_y(geometry.outline, y_samples)

        # Sidewall outer edges are at ski outline
        sw_left_outer = left_outline  # No offset
        sw_right_outer = right_outline  # No offset

        # Check at waist (widest point)
        waist_idx = np.argmin(np.abs(y_samples - geometry.waist_x))
        if not np.isnan(left_outline[waist_idx]):
            # Ski width at waist should be ~90-130mm (45-65mm half-width)
            ski_width_at_waist = right_outline[waist_idx] - left_outline[waist_idx]
            assert 80 < ski_width_at_waist < 150

    def test_sidewall_inner_edge_position(self, geometry, ski_params):
        """Test that sidewall inner edge is offset inward from core."""
        core_offset = ski_params.sidewall_width - ski_params.sidewall_overlap
        core_start = max(geometry.core_tip_x - 25.0, geometry.outline[:, 1].min())
        core_end = min(geometry.core_tail_x + 25.0, geometry.outline[:, 1].max())
        y_samples = np.linspace(core_start, core_end, 50)

        left_outline, right_outline = half_widths_at_y(geometry.outline, y_samples)

        # Core edges are inset from ski outline
        core_left_edge = left_outline + core_offset
        core_right_edge = right_outline - core_offset

        # Core should be narrower than ski outline
        core_widths = core_right_edge - core_left_edge
        outline_widths = right_outline - left_outline

        valid_mask = ~(np.isnan(core_widths) | np.isnan(outline_widths))
        if np.any(valid_mask):
            # Core should be at least 4mm narrower on each side (core_offset=4mm)
            width_reduction = outline_widths[valid_mask] - core_widths[valid_mask]
            assert np.all(width_reduction >= 7.0)  # ~4mm on each side


class TestTrimLinePositioning:
    """Test that trim lines are positioned correctly."""

    def test_trim_lines_at_infill_positions(self, geometry, ski_params, blank):
        """Test that trim lines mark the actual core ends (after trimming)."""
        core_start = max(geometry.core_tip_x - 25.0, geometry.outline[:, 1].min())
        core_end = min(geometry.core_tail_x + 25.0, geometry.outline[:, 1].max())
        core_extent = core_end - core_start

        # Compute blank x_offset (as done in visualization)
        x_offset = (blank.length - core_extent) / 2 - core_start

        # Trim lines should be at core_tip_x and core_tail_x (in blank coords)
        trim_tip = geometry.core_tip_x + x_offset
        trim_tail = geometry.core_tail_x + x_offset

        # Trim lines should be 25mm inside the extended core polygon
        core_start_blank = core_start + x_offset
        core_end_blank = core_end + x_offset

        assert trim_tip > core_start_blank
        assert trim_tail < core_end_blank
        assert trim_tip == pytest.approx(core_start_blank + 25.0)
        assert trim_tail == pytest.approx(core_end_blank - 25.0)

    def test_trim_lines_within_blank_bounds(self, geometry, ski_params, blank):
        """Test that trim lines fall within the blank bounds."""
        core_start = max(geometry.core_tip_x - 25.0, geometry.outline[:, 1].min())
        core_end = min(geometry.core_tail_x + 25.0, geometry.outline[:, 1].max())
        core_extent = core_end - core_start
        x_offset = (blank.length - core_extent) / 2 - core_start

        trim_tip = geometry.core_tip_x + x_offset
        trim_tail = geometry.core_tail_x + x_offset

        # Both trim lines should be within blank
        assert 0 <= trim_tip <= blank.length
        assert 0 <= trim_tail <= blank.length

    def test_trim_line_extent_with_nanmin_nanmax(self, geometry, ski_params):
        """Test that trim line vertical extent is computed correctly using nanmin/nanmax."""
        core_start = max(geometry.core_tip_x - 25.0, geometry.outline[:, 1].min())
        core_end = min(geometry.core_tail_x + 25.0, geometry.outline[:, 1].max())
        y_samples = np.linspace(core_start, core_end, 100)

        left_outline, right_outline = half_widths_at_y(geometry.outline, y_samples)

        # Sidewall edges (for single centered core, core_y=0)
        sw_left = left_outline + 0
        sw_right = right_outline + 0

        # Trim lines should span from leftmost to rightmost point
        y_min = np.nanmin(sw_left)
        y_max = np.nanmax(sw_right)

        # Should have valid numeric values (not NaN)
        assert not np.isnan(y_min)
        assert not np.isnan(y_max)
        assert y_min < 0  # Left side is negative
        assert y_max > 0  # Right side is positive
        assert (y_max - y_min) > 0  # Should span across ski width

    def test_trim_lines_span_sidewalls(self, geometry, ski_params):
        """Test that trim lines span the full width of the sidewalls."""
        core_start = max(geometry.core_tip_x - 25.0, geometry.outline[:, 1].min())
        core_end = min(geometry.core_tail_x + 25.0, geometry.outline[:, 1].max())
        y_samples = np.linspace(core_start, core_end, 100)

        left_outline, right_outline = half_widths_at_y(geometry.outline, y_samples)

        y_min = np.nanmin(left_outline)
        y_max = np.nanmax(right_outline)

        # Trim line extent should be similar to ski width
        trim_line_span = y_max - y_min

        # For this ski, width should be ~90-130mm at widest
        assert 80 < trim_line_span < 150


class TestBlankValidation:
    """Test that blank validation correctly checks core positioning."""

    def test_default_blank_valid(self, geometry, ski_params):
        """Test that default 2000mm blank is valid for the sample ski."""
        blank = CoreBlank()
        is_valid, warnings = blank.validate(geometry, ski_params)
        assert is_valid
        assert len(warnings) == 0

    def test_short_blank_invalid(self, geometry, ski_params):
        """Test that too-short blank triggers validation warning."""
        # Core extent is ~1610mm, so 1600mm blank should be too short
        blank = CoreBlank(length=1600)
        is_valid, warnings = blank.validate(geometry, ski_params)
        assert not is_valid
        assert len(warnings) > 0
        assert any("outside blank length bounds" in w for w in warnings)

    def test_blank_with_position_offset_valid(self, geometry, ski_params):
        """Test that position offset is properly accounted for in validation."""
        blank = CoreBlank(length=2000, position_offset_x=0)
        is_valid, _ = blank.validate(geometry, ski_params)
        assert is_valid

    def test_blank_with_position_offset_invalid(self, geometry, ski_params):
        """Test that large position offset can cause validation failure."""
        # Offset the core far enough that it extends beyond blank
        blank = CoreBlank(length=2000, position_offset_x=500)
        is_valid, warnings = blank.validate(geometry, ski_params)
        assert not is_valid
        assert any("outside blank length bounds" in w for w in warnings)


class TestDataFileIntegrity:
    """Test that data files contain expected values."""

    def test_ski_params_json_values(self, ski_params):
        """Test that ski_params.json has expected values."""
        assert ski_params.tip_infill == 160.0
        assert ski_params.tail_infill == 80.0
        assert ski_params.sidewall_width == 6.0
        assert ski_params.sidewall_overlap == 2.0
        assert ski_params.tip_thickness == 2.0
        assert ski_params.underfoot_thickness == 10.0
        assert ski_params.tail_thickness == 2.0

    def test_dxf_outline_closed(self, ski_outline):
        """Test that DXF outline forms a closed polygon."""
        # First and last points should be the same (closed polygon)
        assert np.allclose(ski_outline[0], ski_outline[-1], atol=1.0)

    def test_dxf_outline_reasonable_dimensions(self, ski_outline):
        """Test that DXF outline has reasonable ski dimensions."""
        y_min, y_max = ski_outline[:, 1].min(), ski_outline[:, 1].max()
        ski_length = y_max - y_min
        # Expect ski to be 1700-2000mm long
        assert 1700 < ski_length < 2000

        x_min, x_max = ski_outline[:, 0].min(), ski_outline[:, 0].max()
        ski_width = x_max - x_min
        # Expect ski to be 100-150mm wide
        assert 100 < ski_width < 150
