"""Material properties database for ski construction."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import math


@dataclass
class Material:
    """Generic ski construction material. Fields unused by a type default to 0."""
    name: str
    type: str  # "composite", "wood_core", "base", "edge", "sidewall"

    # Composite ply (unidirectional or woven)
    E1: float = 0.0       # Along-fibre modulus MPa
    E2: float = 0.0       # Transverse modulus MPa
    G12: float = 0.0      # In-plane shear modulus MPa
    nu12: float = 0.3     # Major Poisson ratio
    areal_weight: float = 0.0  # g/m²

    # Isotropic / orthotropic (wood, base, sidewall)
    E: float = 0.0        # Young's modulus MPa (along grain for wood)
    G: float = 0.0        # Shear modulus MPa
    nu: float = 0.3       # Poisson ratio
    density: float = 0.0  # kg/m³

    # Geometry embedded in material
    thickness: float = 0.0       # mm — for base and sidewall
    mass_per_length: float = 0.0  # g/m — for edge (per edge)

    color: str = "#888888"
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Material":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def effective_Ex(self, theta_deg: float = 0.0) -> float:
        """Effective Young's modulus along ski axis for composite ply at fibre angle theta."""
        if self.type != "composite" or self.E1 <= 0:
            return self.E
        th = math.radians(theta_deg)
        c2, s2 = math.cos(th)**2, math.sin(th)**2
        # Tsai-Hill transformation (Jones 1975)
        denom = c2**2 / self.E1 + s2**2 / self.E2 + c2 * s2 * (1.0/self.G12 - 2.0*self.nu12/self.E1)
        return 1.0 / denom if denom > 0 else 0.0

    def ply_thickness_mm(self) -> float:
        """Estimate ply thickness from areal weight and density."""
        if self.density > 0 and self.areal_weight > 0:
            return self.areal_weight / self.density  # mm
        return self.thickness


_DEFAULTS: list[Material] = [
    Material("E-glass UD 300", "composite",
             E1=38000, E2=8600, G12=3800, nu12=0.27, areal_weight=300, density=2100,
             color="#b0c8e8", notes="Unidirectional e-glass 300 g/m², 0°=along ski"),
    Material("E-glass ±45 200", "composite",
             E1=15000, E2=15000, G12=14000, nu12=0.05, areal_weight=200, density=2100,
             color="#90b0d0", notes="±45° biaxial e-glass 200 g/m² total"),
    Material("E-glass 0/90 300", "composite",
             E1=22000, E2=22000, G12=4000, nu12=0.12, areal_weight=300, density=2100,
             color="#a0bce0", notes="0/90 woven e-glass 300 g/m²"),
    Material("Carbon UD 200", "composite",
             E1=135000, E2=7500, G12=4000, nu12=0.28, areal_weight=200, density=1550,
             color="#444444", notes="Unidirectional carbon 200 g/m², 0°=along ski"),
    Material("Carbon ±45 200", "composite",
             E1=35000, E2=35000, G12=34000, nu12=0.05, areal_weight=200, density=1550,
             color="#333333", notes="±45° biaxial carbon 200 g/m²"),
    Material("Paulownia", "wood_core",
             E=5000, G=500, nu=0.3, density=270,
             color="#e8d090", notes="Very light core wood"),
    Material("Poplar", "wood_core",
             E=8000, G=800, nu=0.3, density=450,
             color="#d4b870", notes="Medium weight/stiffness"),
    Material("Ash", "wood_core",
             E=13000, G=1200, nu=0.3, density=700,
             color="#b89040", notes="Heavy, high stiffness"),
    Material("Bamboo", "wood_core",
             E=15000, G=1000, nu=0.3, density=700,
             color="#c8c050", notes="High stiffness-to-weight"),
    Material("PTEX 2000", "base",
             E=700, nu=0.4, density=940, thickness=1.2,
             color="#f0f0f0", notes="Sintered PTEX 1.2 mm"),
    Material("PTEX 4000", "base",
             E=700, nu=0.4, density=940, thickness=1.8,
             color="#ffffff", notes="HD sintered PTEX 1.8 mm"),
    Material("Steel edge", "edge",
             mass_per_length=55.0, color="#a0a0a0", notes="Standard ski edge ~55 g/m each side"),
    Material("Race edge", "edge",
             mass_per_length=70.0, color="#c0c0c0", notes="Race edge ~70 g/m each side"),
    Material("UHMWPE sidewall", "sidewall",
             E=700, G=250, nu=0.4, density=940, thickness=2.0,
             color="#e0e0f0", notes="Ultra-HMWPE 2 mm"),
    Material("ABS sidewall", "sidewall",
             E=2300, G=850, nu=0.35, density=1050, thickness=2.0,
             color="#d0d0d0", notes="ABS 2 mm"),
]


class MaterialDatabase:
    _DB_PATH = Path(__file__).parent.parent / "data" / "materials.json"

    def __init__(self):
        self._mats: dict[str, Material] = {m.name: m for m in _DEFAULTS}
        if self._DB_PATH.exists():
            self._load_custom()

    def _load_custom(self):
        try:
            with open(self._DB_PATH) as f:
                data = json.load(f)
            for d in data.get("materials", []):
                m = Material.from_dict(d)
                self._mats[m.name] = m
        except Exception:
            pass

    def save_custom(self):
        builtin = {m.name for m in _DEFAULTS}
        custom = [m.to_dict() for m in self._mats.values() if m.name not in builtin]
        self._DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(self._DB_PATH, "w") as f:
            json.dump({"materials": custom}, f, indent=2)

    def all(self) -> list[Material]:
        return list(self._mats.values())

    def by_type(self, t: str) -> list[Material]:
        return [m for m in self._mats.values() if m.type == t]

    def names(self, t: str | None = None) -> list[str]:
        if t is None:
            return [m.name for m in self._mats.values()]
        return [m.name for m in self.by_type(t)]

    def get(self, name: str) -> Material | None:
        return self._mats.get(name)

    def add_or_replace(self, m: Material):
        self._mats[m.name] = m

    def remove(self, name: str):
        self._mats.pop(name, None)
