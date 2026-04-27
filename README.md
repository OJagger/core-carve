# Core Carve — SkiNC G-code Generator

A Python desktop application for designing and generating G-code for CNC ski core manufacturing. Design ski core geometry, position blanks, and generate toolpaths with interactive visualization.

**Status:** Early development  
**License:** AGPL-3.0-or-later

---

## System Requirements

- **Python:** 3.8 or later
- **OS:** macOS, Linux, or Windows
- **Display:** Minimum 1400×900 resolution recommended

---

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd core-carve
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -e ".[dev]"
```

This installs:
- PyQt5 — GUI framework
- matplotlib — Visualization
- numpy — Numerical computation
- scipy — Spline interpolation
- ezdxf — DXF file support
- ruff — Code linting (dev)
- pytest — Testing (dev)

---

## Running the Application

### Start the GUI
```bash
source .venv/bin/activate
python -m core_carve.main
```

The application window opens showing Tab 1 (Ski Geometry).

### Example Workflow

1. **Load geometry:** Click "Load Planform DXF…" and select `data/Ski_planform.dxf`
2. **Load parameters:** Click "Load Params JSON…" and select `data/ski_params.json`
3. **Update geometry:** Click "Update Geometry" to compute derived values
4. **Switch to Tab 2:** Click the "2 · Core Blank" tab to view blank layout
5. **Adjust blank:** Modify blank dimensions or core positioning as needed
6. **Switch to Tab 3:** Click "3 · G-code" to generate cutting toolpaths
7. **Configure cutting:** Adjust tool diameter, feeds, speeds, and tab spacing
8. **Generate code:** Click "Generate G-code" to create the machining program
9. **Preview:** Watch the animated toolpath playback
10. **Export:** Click "Save G-code…" to write the `.nc` file for your CNC machine

---

## Example Data Files
### Located in `./data/`


---

**Version:** 0.1.0  
**Last Updated:** April 2026
