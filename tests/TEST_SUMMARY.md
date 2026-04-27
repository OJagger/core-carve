# Tab 2 Geometry Tests Summary

## Overview

Comprehensive test suite for Tab 2 (Core Blank) geometry visualization, covering the loading of DXF and parameter files, core positioning, sidewall placement, and trim line positioning.

**Total Tests:** 24  
**Status:** ✅ All Passing

---

## Test Categories

### 1. Geometry Loading (5 tests)
Verifies that the DXF file and JSON parameters load correctly and that geometry is computed accurately.

- ✅ `test_dxf_loads` — DXF file loads as numpy array with X,Y coordinates
- ✅ `test_params_load` — JSON parameters load with correct values
- ✅ `test_geometry_computed` — Geometry is computed with valid dimensions
- ✅ `test_ski_length` — Ski length is within expected range (1700-2000mm)
- ✅ `test_core_positions` — Core tip and tail positions match infill parameters

**Key Values Verified:**
- `tip_infill`: 160.0 mm
- `tail_infill`: 80.0 mm
- `sidewall_width`: 6.0 mm
- `sidewall_overlap`: 2.0 mm
- Ski length: ~1800 mm

### 2. Extended Core Positioning (4 tests)
Verifies that the extended core (with 25mm machining extensions on each end) is positioned correctly in the blank.

- ✅ `test_core_extent_calculation` — Core extent includes 25mm extensions
- ✅ `test_core_start_clamped_to_outline` — Core start is clamped to outline minimum
- ✅ `test_core_end_clamped_to_outline` — Core end is clamped to outline maximum
- ✅ `test_core_positioned_in_blank` — Core is properly centered in blank space

**Key Calculations Verified:**
- Extended core length: 1610 mm (from tip_infill 160 to 1720, ±25mm)
- Core centered in blank: spans from ~245mm to ~1755mm in 2000mm blank
- Y-samples clamped to valid outline range to prevent NaN values

### 3. Sidewall Positioning (4 tests)
Verifies that sidewall outlines are positioned correctly relative to the core.

- ✅ `test_sidewall_width_from_outline` — Sidewall width parameter is correct (6mm)
- ✅ `test_sidewall_outline_samples` — Outline samples are valid (no all-NaN regions)
- ✅ `test_sidewall_outer_edge_position` — Sidewall outer edge matches ski outline
- ✅ `test_sidewall_inner_edge_position` — Core inner edge is offset correctly

**Key Measurements:**
- Sidewall offset: 4.0 mm (width 6.0 - overlap 2.0)
- Ski width at waist: ~96 mm
- Core width reduction: ≥7.0 mm on each side

### 4. Trim Line Positioning (4 tests)
Verifies that dashed trim lines marking the final core ends (after 25mm trimming) are positioned correctly.

- ✅ `test_trim_lines_at_infill_positions` — Trim lines are 25mm inside core polygon
- ✅ `test_trim_lines_within_blank_bounds` — Trim lines fall within blank bounds
- ✅ `test_trim_line_extent_with_nanmin_nanmax` — Y-extent computed correctly with nanmin/nanmax
- ✅ `test_trim_lines_span_sidewalls` — Trim lines span full width of sidewalls

**Key Positions:**
- Trim tip (at core_tip_x): 270 mm in blank space
- Trim tail (at core_tail_x): 1730 mm in blank space
- Y-extent: ~96 mm (full ski width at that position)

### 5. Blank Validation (4 tests)
Verifies that blank validation correctly checks if the core fits within blank dimensions.

- ✅ `test_default_blank_valid` — Default 2000mm blank is valid
- ✅ `test_short_blank_invalid` — 1600mm blank triggers validation warning
- ✅ `test_blank_with_position_offset_valid` — Zero offset is valid
- ✅ `test_blank_with_position_offset_invalid` — Large offset causes validation failure

**Validation Logic:**
- Blank must be large enough for extended core (1610mm)
- Position offset is properly accounted for
- Clear error messages show why validation fails

### 6. Data File Integrity (3 tests)
Verifies that input data files contain expected values and are correctly formatted.

- ✅ `test_ski_params_json_values` — JSON contains expected infill and thickness values
- ✅ `test_dxf_outline_closed` — DXF outline forms a closed polygon
- ✅ `test_dxf_outline_reasonable_dimensions` — Ski dimensions are within expected ranges

---

## Key Fixes Tested

1. **Y-Sample Clamping** — Core sampling is clamped to outline bounds to prevent NaN values
2. **NaN-Safe Trim Lines** — Uses `np.nanmin`/`np.nanmax` for safe extent calculation
3. **Coordinate System Correction** — Proper conversion between ski-geometry and blank-space coordinates
4. **Validation Logic** — Correctly checks if core fits in blank accounting for positioning

---

## Running the Tests

```bash
# Run all Tab 2 geometry tests
pytest tests/test_tab2_geometry.py -v

# Run a specific test class
pytest tests/test_tab2_geometry.py::TestExtendedCorePositioning -v

# Run a specific test
pytest tests/test_tab2_geometry.py::TestTrimLinePositioning::test_trim_lines_at_infill_positions -v

# Run with detailed output
pytest tests/test_tab2_geometry.py -vv --tb=long
```

---

## Expected Behavior

With `data/Ski_planform.dxf` and `data/ski_params.json`:

| Dimension | Value |
|-----------|-------|
| Ski Length | ~1800 mm |
| Tip Infill | 160 mm |
| Tail Infill | 80 mm |
| Core Length (extended) | 1610 mm |
| Blank Length (default) | 2000 mm |
| Blank Width (default) | 200 mm |
| Core Position in Blank | Centered (245-1755mm) |
| Sidewall Width | 6 mm |
| Core Offset from Outline | 4 mm (each side) |
| Ski Width at Waist | ~96 mm |

All tests verify these relationships are maintained correctly.
