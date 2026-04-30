"""Ski mass and stiffness distributions along ski length."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from core_carve.materials import MaterialDatabase, Material


@dataclass
class PlyCfg:
    material_name: str = ""
    fiber_angle: float = 0.0  # degrees
    n_plies: int = 1
    enabled: bool = True


@dataclass
class LayupConfig:
    top_layers: list = field(default_factory=lambda: [PlyCfg("E-glass UD 300", 0.0, 1)])
    bottom_layers: list = field(default_factory=lambda: [PlyCfg("E-glass UD 300", 0.0, 1)])
    mirror_bottom: bool = True
    core_material: str = "Paulownia"
    base_material: str = "PTEX 2000"
    edge_material: str = "Steel edge"
    sidewall_material: str = "UHMWPE sidewall"

    def effective_bottom(self) -> list:
        return self.top_layers if self.mirror_bottom else self.bottom_layers


@dataclass
class MechanicsResult:
    y: np.ndarray               # positions along ski (mm)
    mass_per_mm: np.ndarray     # g/mm
    EI: np.ndarray              # bending stiffness N·mm²
    GJ: np.ndarray              # torsional stiffness N·mm²
    total_mass_g: float = 0.0


def _laminate_props(layers: list[PlyCfg], db: MaterialDatabase):
    """Return (total_thickness_mm, effective_Ex_MPa, effective_Gxy_MPa, density_g_per_mm3)."""
    if not layers:
        return 0.0, 0.0, 0.0, 0.0
    t_total = 0.0
    Ex_sum = 0.0
    Gxy_sum = 0.0
    mass_sum = 0.0  # g/mm²
    for ply in layers:
        if not ply.enabled or not ply.material_name:
            continue
        mat = db.get(ply.material_name)
        if mat is None:
            continue
        t_ply = mat.ply_thickness_mm() * ply.n_plies
        if t_ply <= 0:
            continue
        Ex = mat.effective_Ex(ply.fiber_angle)
        Gxy = mat.G12  # in-plane shear modulus of ply
        t_total += t_ply
        Ex_sum += Ex * t_ply
        Gxy_sum += Gxy * t_ply
        # mass g/mm² = areal_weight[g/m²] * n_plies / 1e6
        mass_sum += mat.areal_weight * ply.n_plies / 1e6
    if t_total <= 0:
        return 0.0, 0.0, 0.0, 0.0
    return t_total, Ex_sum / t_total, Gxy_sum / t_total, mass_sum


def compute_mechanics(geom, ski_params, layup: LayupConfig, db: MaterialDatabase,
                      n_pts: int = 200) -> MechanicsResult:
    """
    Compute mass/length, bending stiffness EI, and torsional stiffness GJ along the ski.

    Uses a simplified sandwich-beam model:
    - Core: rectangular cross-section, isotropic in bending
    - Top/bottom laminates: thin shells, contribution via parallel-axis theorem
    - Bending about the transverse axis (ski bending), torsion of a thin rectangular box
    """
    from core_carve.ski_geometry import half_widths_at_y

    L = geom.ski_length
    y_arr = np.linspace(0.0, L, n_pts)

    # Core properties
    core_mat = db.get(layup.core_material) or Material("fallback", "wood_core", E=5000, G=500, density=400)
    E_core = core_mat.E  # MPa
    G_core = core_mat.G  # MPa
    rho_core = core_mat.density  # kg/m³ = g/dm³ = 1e-6 g/mm³

    # Base
    base_mat = db.get(layup.base_material)
    t_base = base_mat.thickness if base_mat else 0.0
    E_base = base_mat.E if base_mat else 0.0
    rho_base = base_mat.density if base_mat else 0.0  # kg/m³

    # Edge
    edge_mat = db.get(layup.edge_material)
    m_edge = edge_mat.mass_per_length if edge_mat else 0.0  # g/m per edge → g/mm = /1000

    # Sidewall
    sw_mat = db.get(layup.sidewall_material)
    t_sw = sw_mat.thickness if sw_mat else 0.0
    rho_sw = sw_mat.density if sw_mat else 0.0

    # Laminate (top)
    t_top, Ex_top, Gxy_top, m_top_per_mm2 = _laminate_props(layup.top_layers, db)
    # Laminate (bottom)
    bot_layers = layup.effective_bottom()
    t_bot, Ex_bot, Gxy_bot, m_bot_per_mm2 = _laminate_props(bot_layers, db)

    # Width at each position
    left_w, right_w = half_widths_at_y(geom.outline, y_arr)
    w_arr = np.abs(right_w - left_w)  # full width mm

    # Core thickness at each position (clamp to core extent)
    y_core = np.clip(y_arr, geom.core_tip_x, geom.core_tail_x)
    h_core = geom.thickness_at(y_core)
    h_core = np.where((y_arr >= geom.core_tip_x) & (y_arr <= geom.core_tail_x), h_core, 0.0)

    mass_per_mm = np.zeros(n_pts)
    EI = np.zeros(n_pts)
    GJ = np.zeros(n_pts)

    for i in range(n_pts):
        w = w_arr[i]
        h_c = h_core[i]

        # ── Mass per mm ────────────────────────────────────────────────────────
        # Core (g/mm)
        m = rho_core * 1e-6 * w * h_c  # kg/m³ * 1e-6 mm³→g * mm² = g/mm
        # Laminates
        m += m_top_per_mm2 * w + m_bot_per_mm2 * w
        # Base
        m += rho_base * 1e-6 * w * t_base
        # Edges (two edges, g/m per edge → g/mm = /1000)
        m += 2.0 * m_edge / 1000.0
        # Sidewalls (two sides, h_c height × t_sw thickness)
        m += 2.0 * rho_sw * 1e-6 * h_c * t_sw
        mass_per_mm[i] = m

        if h_c <= 0 or w <= 0:
            continue

        # ── Bending stiffness EI (N·mm²) ──────────────────────────────────────
        # Core contribution: E_core * w * h_c³ / 12
        EI_core = E_core * w * h_c**3 / 12.0
        # Top laminate: distance of its centroid from neutral axis ≈ (h_c + t_top)/2
        z_top = (h_c + t_top) / 2.0
        EI_top = Ex_top * t_top * w * z_top**2
        # Bottom laminate
        z_bot = (h_c + t_bot) / 2.0
        EI_bot = Ex_bot * t_bot * w * z_bot**2
        # Base
        z_base = (h_c / 2.0 + t_bot + t_base / 2.0)
        EI_base = E_base * t_base * w * z_base**2
        EI[i] = EI_core + EI_top + EI_bot + EI_base

        # ── Torsional stiffness GJ (N·mm²) ────────────────────────────────────
        # Thin-walled closed box (Bredt-Batho): J = 4*A²*t_eff / perimeter
        # Effective wall: top+bot laminates + base
        t_face = (t_top + t_bot + t_base) / 2.0  # average face thickness
        t_web = t_sw if t_sw > 0 else t_face      # sidewall thickness
        if t_face > 0 and t_web > 0:
            A_enclosed = w * h_c
            # J = 4*A² / (2 * Σ(s/t)) but for two faces+two webs:
            ds_over_t = 2.0 * (w / t_face + h_c / t_web)
            J = 4.0 * A_enclosed**2 / ds_over_t
            G_lam = (Gxy_top + Gxy_bot) / 2.0 if (Gxy_top + Gxy_bot) > 0 else G_core
            GJ[i] = G_lam * J
        else:
            GJ[i] = 0.0

    total_mass = float(np.trapz(mass_per_mm, y_arr))
    return MechanicsResult(y=y_arr, mass_per_mm=mass_per_mm, EI=EI, GJ=GJ, total_mass_g=total_mass)
