"""
Shared payload schemas for GUI bridge and API service.

Defines the data structures that both the direct bridge and the FastAPI
transport layer consume and produce. Keeping them in one place prevents
drift between the two call paths.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ------------------------------------------------------------------ #
#  Input schemas
# ------------------------------------------------------------------ #


@dataclass
class BuildConfig:
    """Everything needed to build a System + ForceCalculator."""

    units: str = "LJ"
    boundary: Dict[str, Any] = field(default_factory=lambda: {"type": "periodic"})
    system: Dict[str, Any] = field(default_factory=dict)
    potential: Dict[str, Any] = field(default_factory=lambda: {"type": "lj"})
    backend: str = "numerical"
    integrator: Dict[str, Any] = field(default_factory=lambda: {"dt": 0.005})
    thermostat: Dict[str, Any] = field(default_factory=lambda: {"type": "nve"})

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BuildConfig":
        return cls(
            units=d.get("units", "LJ"),
            boundary=d.get("boundary", {"type": "periodic"}),
            system=d.get("system", {}),
            potential=d.get("potential", {"type": "lj"}),
            backend=d.get("backend", "numerical"),
            integrator=d.get("integrator", {"dt": 0.005}),
            thermostat=d.get("thermostat", {"type": "nve"}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "units": self.units,
            "boundary": self.boundary,
            "system": self.system,
            "potential": self.potential,
            "backend": self.backend,
            "integrator": self.integrator,
            "thermostat": self.thermostat,
        }


@dataclass
class SimulationParams:
    """Parameters for running a simulation."""

    num_steps: int = 1000
    viz_interval: int = 50
    init_temp: float = 1.0


@dataclass
class MinimizationParams:
    """Parameters for running energy minimization."""

    algorithm: str = "conjugate_gradient"
    force_tol: float = 1e-4
    energy_tol: float = 1e-8
    max_steps: int = 10000

    @classmethod
    def from_dict(cls, algorithm: str, d: Dict[str, Any]) -> "MinimizationParams":
        return cls(
            algorithm=algorithm.lower(),
            force_tol=float(d.get("force_tol", 1e-4)),
            energy_tol=float(d.get("energy_tol", 1e-8)),
            max_steps=int(d.get("max_steps", 10000)),
        )


# ------------------------------------------------------------------ #
#  Output schemas
# ------------------------------------------------------------------ #


@dataclass
class SystemSummary:
    """Summary returned after a successful build."""

    n_atoms: int
    element: str
    box: List[float]
    units: str
    potential: str
    boundary: str
    dt: float
    thermostat: str
    xyz: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_atoms": self.n_atoms,
            "element": self.element,
            "box": self.box,
            "units": self.units,
            "potential": self.potential,
            "boundary": self.boundary,
            "dt": self.dt,
            "thermostat": self.thermostat,
            "xyz": self.xyz,
        }


@dataclass
class SimulationUpdate:
    """Single progress update during simulation."""

    step: int
    total: int
    pe: float
    ke: float
    total_e: float
    temperature: float
    xyz: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "total": self.total,
            "pe": self.pe,
            "ke": self.ke,
            "total_e": self.total_e,
            "temperature": self.temperature,
            "xyz": self.xyz,
        }


@dataclass
class EnergyData:
    """Full energy time-series from a completed simulation."""

    steps: List[int]
    pe: List[float]
    ke: List[float]
    total: List[float]
    temperature: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": self.steps,
            "pe": self.pe,
            "ke": self.ke,
            "total": self.total,
            "temperature": self.temperature,
        }


@dataclass
class MinimizationResult:
    """Result returned after energy minimization completes."""

    converged: bool
    n_steps: int
    initial_energy: float
    final_energy: float
    max_force: float
    message: str
    energy_history: List[float]
    xyz: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "converged": self.converged,
            "n_steps": self.n_steps,
            "initial_energy": self.initial_energy,
            "final_energy": self.final_energy,
            "max_force": self.max_force,
            "message": self.message,
            "energy_history": self.energy_history,
            "xyz": self.xyz,
        }
