# CLAUDE.md

Guidance for Claude Code working with the core-carve repository.

## Project Overview

**core-carve** is a PyQt5 desktop application for CNC ski core manufacturing. It generates G-code for cutting sidewall slots from 2D ski geometry.

- **Language:** Python 3.8+
- **License:** AGPL-3.0-or-later
- **Status:** Early development (active)

---

## Development Setup

### Initial Setup (One-time)

```bash
cd /Users/oliver/Documents/Ski\ making/core-carve

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies with dev tools
pip install -e ".[dev]"
```

### Every Session

```bash
cd /Users/oliver/Documents/Ski\ making/core-carve
source .venv/bin/activate
```

---

## Running the Application

```bash
# Ensure venv is activated first
source .venv/bin/activate

# Start the GUI application
python -m core_carve.main
```

The application opens with Tab 1 (Ski Geometry). Use the menu to:
1. Load a DXF planform file (e.g., `data/Ski_planform.dxf`)
2. Load parameters JSON (e.g., `data/ski_params.json`)
3. Click "Update Geometry" to compute derived values
4. Switch between tabs to design and generate G-code

---

## Project Structure

```
core_carve/
├── __init__.py                 # Package marker
├── main.py                     # Application entry point, QMainWindow with tabs
├── ski_geometry.py             # DXF loading, ski outline, geometry computation
├── core_blank.py               # Blank design, core positioning, validation
├── gcode_generator.py          # Toolpath generation, G-code output
├── tab_geometry.py             # Tab 1 UI: geometry editor with Matplotlib canvas
├── tab_blank.py                # Tab 2 UI: blank layout visualization
└── tab_gcode.py                # Tab 3 UI: G-code generation and playback

tests/
├── test_tab2_geometry.py       # 24 comprehensive geometry tests
└── TEST_SUMMARY.md             # Test documentation

data/
├── Ski_planform.dxf            # Example ski outline (1800mm × 96mm)
└── ski_params.json             # Example core parameters
```

---

## Key Files and Responsibilities

### `ski_geometry.py` — Core Geometry Engine
- **Classes:** `SkiParams`, `SkiGeometry`
- **Functions:** `load_planform_dxf()`, `compute_geometry()`, `half_widths_at_y()`
- **Purpose:** Loads DXF files, computes ski-space geometry (waist, thickness profiles, core positions)
- **Key Data:** Outline as Nx2 numpy array (X=across, Y=along ski length)

### `core_blank.py` — Blank Design
- **Classes:** `CoreBlank`, `MachineOrientation`, `OriginCorner`
- **Methods:** `get_core_positions()`, `validate()`
- **Purpose:** Manages blank dimensions and core positioning; validates fit
- **Returns Positions:** In ski-geometry coordinates (not blank-space)

### `tab_blank.py` — Tab 2 Visualization
- **Canvas:** `BlankCanvas` — Top-view blank layout with cores, sidewalls, trim lines
- **Panel:** `BlankParameterPanel` — Forms for blank and positioning inputs
- **Key Calculation:** `x_offset = (blank.length - core_extent) / 2 - core_start`
  - Converts ski-geometry coords to blank-space coords
  - Centers core in blank with equal margins

### `gcode_generator.py` — Toolpath Generation
- **Classes:** `SlotParams`, `Move`
- **Function:** `generate_slot_gcode()` — Main G-code generation algorithm
- **Features:** Multi-pass cutting, tab placement (arc-length parameterization), climb/conventional milling
- **Output:** G-code string + list of Move objects for visualization

### `tab_gcode.py` — Tab 3 UI and Playback
- **Canvas:** `GcodeCanvas` — Slot preview + animated toolpath playback
- **Methods:** `plot_slot_preview()`, `plot_toolpaths()`, `start_playback()`, `_step_playback_frame()`
- **Playback:** QTimer-driven animation with frame-skipping for speed control (1×, 2×, 4×, 8×, 16×)

---

## Recent Fixes (April 2026)

### Bug: Core Outline Not Displaying (Tab 2 & 3)
- **Root Cause:** Y-samples extended beyond ski outline bounds → NaN values
- **Fix:** Clamp y_samples to `[outline.min(), outline.max()]`
- **Files:** `tab_blank.py`, `tab_gcode.py`

### Bug: Trim Lines Not Rendering (Tab 2)
- **Root Cause:** Python's `min()`/`max()` propagate NaN from array start
- **Fix:** Use `np.nanmin()` / `np.nanmax()` instead
- **File:** `tab_blank.py` (lines 144-145, 159-160)

### Bug: Stale Geometry in Tabs 2 & 3
- **Root Cause:** `BlankTab` and `GcodeTab` created once; geometry updates not propagated
- **Fix:** `_check_geometry_loaded()` now updates existing tabs when geometry changes
- **File:** `main.py` (lines 40-44)

### Bug: Invalid Core Extent Validation
- **Root Cause:** Validation mixed ski-geometry coords with blank-space coords
- **Fix:** Rewrite `validate()` to properly compute blank-space extents
- **File:** `core_blank.py` (lines 100-110)

---

## Important Coordinate Systems

### Ski-Geometry Space (ski_geometry.py, core_blank.py)
- Origin: Ski tip at (0, 0)
- X-axis: Across ski (negative=left, positive=right)
- Y-axis: Along ski length (0=tip, ski_length=tail)
- **Data:** Outline, core_tip_x, core_tail_x, waist_x all in this space

### Blank Space (tab_blank.py visualization)
- Origin: Blank corner (0, 0)
- X-axis: Along blank length (0 to blank.length)
- Y-axis: Across blank width (centered, -width/2 to +width/2)
- **Conversion:** `blank_coord = ski_coord + x_offset` where:
  - `x_offset = (blank.length - core_extent) / 2 - core_start`

### CNC Machine Space (tab_gcode.py)
- X, Y: Machine table coordinates
- Z: Vertical (-depth = down into material)

---

## Common Development Tasks

### Run All Tests
```bash
source .venv/bin/activate
pytest tests/ -v
```

### Run Specific Test
```bash
source .venv/bin/activate
pytest tests/test_tab2_geometry.py::TestTrimLinePositioning -v
```

### Format and Lint
```bash
source .venv/bin/activate
ruff format core_carve tests
ruff check core_carve tests
```

### Test with Sample Data
```bash
source .venv/bin/activate
python << 'EOF'
from core_carve.ski_geometry import load_planform_dxf, compute_geometry, SkiParams
outline = load_planform_dxf("data/Ski_planform.dxf")
params = SkiParams.from_json("data/ski_params.json")
geom = compute_geometry(outline, params)
print(f"Ski length: {geom.ski_length} mm")
print(f"Core: {geom.core_tip_x} → {geom.core_tail_x} mm")
EOF
```

---

## Testing

**24 tests** in `tests/test_tab2_geometry.py` covering:
- DXF loading and DXF outline validation
- Geometry computation (ski length, core positions)
- Extended core positioning (with 25mm machining extensions)
- Sidewall geometry and outline sampling
- Trim line placement and Y-extent calculation
- Blank validation logic with position offsets
- Data file integrity checks

**All tests pass** and use the example data files (`data/Ski_planform.dxf`, `data/ski_params.json`).

Run tests:
```bash
source .venv/bin/activate
pytest tests/test_tab2_geometry.py -v
```

---

## Example Data Files

### `data/Ski_planform.dxf`
- **Format:** AutoCAD DXF
- **Geometry:** Closed polygon (ski outline)
- **Dimensions:** ~1800 mm length, ~96 mm waist width
- **Usage:** Load in Tab 1 with "Load Planform DXF…"

### `data/ski_params.json`
- **Format:** JSON
- **Parameters:** tip_infill=160, tail_infill=80, sidewall_width=6, etc.
- **Usage:** Load in Tab 1 with "Load Params JSON…"

Both files are used together in the example workflow and in all tests.

---

## Notes for Future Development

- **Geometry updates:** If you modify `compute_geometry()`, ensure `SkiGeometry` dataclass fields are consistent
- **UI responsiveness:** Canvas redraws happen in `plot_*()` methods; keep heavy computation out
- **NaN handling:** Always clamp sample ranges to outline bounds; use `np.nanmin/nanmax` for aggregations
- **Coordinate conversion:** Remember `x_offset` converts ski-space to blank-space; don't mix them
- **Tab synchronization:** Updates to `geometry_tab._geom` trigger `_check_geometry_loaded()` on all tab switches

---

## Dependencies

- **PyQt5** — GUI widgets, signals, QTimer
- **matplotlib** — Canvas visualization (FigureCanvasQTAgg)
- **numpy** — Array operations, sampling
- **scipy** — CubicSpline (thickness profiles), BSpline (DXF splines)
- **ezdxf** — DXF file parsing

Development tools:
- **ruff** — Linting and formatting
- **pytest** — Test framework

---

## Quick Reference

| Command | Purpose |
|---------|---------|
| `source .venv/bin/activate` | Activate virtual environment |
| `python -m core_carve.main` | Run the GUI application |
| `pytest tests/test_tab2_geometry.py -v` | Run all geometry tests |
| `ruff check core_carve` | Lint the code |
| `ruff format core_carve` | Auto-format code |

---

**Last Updated:** April 27, 2026
