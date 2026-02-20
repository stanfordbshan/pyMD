"""
Pydantic request / response models for the pymd REST API.

All validation, field constraints, and serialisation logic lives here.
Routes import these models — they never define their own.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ------------------------------------------------------------------ #
#  Request models
# ------------------------------------------------------------------ #


class BuildRequest(BaseModel):
    """Payload for ``POST /build``."""

    units: str = Field("LJ", description="Unit system (LJ, REAL, METAL, SI)")
    boundary: Dict[str, Any] = Field(
        default_factory=lambda: {"type": "periodic"},
        description="Boundary condition configuration",
    )
    system: Dict[str, Any] = Field(
        default_factory=dict,
        description="System spec (element, mass, lattice, …)",
    )
    potential: Dict[str, Any] = Field(
        default_factory=lambda: {"type": "lj"},
        description="Potential configuration",
    )
    backend: str = Field(
        "numerical",
        description="Autodiff backend (numerical, jax, torch, autograd)",
    )
    integrator: Dict[str, Any] = Field(
        default_factory=lambda: {"dt": 0.005},
        description="Integrator settings",
    )
    thermostat: Dict[str, Any] = Field(
        default_factory=lambda: {"type": "nve"},
        description="Thermostat settings",
    )


class SimulationRequest(BaseModel):
    """Payload for ``POST /simulation/start``."""

    num_steps: int = Field(1000, gt=0, description="Total integration steps")
    viz_interval: int = Field(50, gt=0, description="Visualization update interval")
    init_temp: float = Field(1.0, ge=0.0, description="Initial temperature")

    @field_validator("viz_interval")
    @classmethod
    def viz_interval_le_steps(cls, v: int, info) -> int:
        num_steps = info.data.get("num_steps")
        if num_steps is not None and v > num_steps:
            raise ValueError("viz_interval must be ≤ num_steps")
        return v


class MinimizationRequest(BaseModel):
    """Payload for ``POST /minimize``."""

    algorithm: str = Field(
        "conjugate_gradient",
        description="Algorithm (steepest_descent, conjugate_gradient, lbfgs)",
    )
    force_tol: float = Field(1e-4, gt=0, description="Force convergence tolerance")
    energy_tol: float = Field(1e-8, gt=0, description="Energy convergence tolerance")
    max_steps: int = Field(10000, gt=0, description="Maximum optimisation steps")

    @field_validator("algorithm")
    @classmethod
    def algorithm_must_be_valid(cls, v: str) -> str:
        allowed = {"steepest_descent", "conjugate_gradient", "lbfgs"}
        v_lower = v.lower()
        if v_lower not in allowed:
            raise ValueError(
                f"Unknown algorithm '{v}'. Choose from: {', '.join(sorted(allowed))}"
            )
        return v_lower


# ------------------------------------------------------------------ #
#  Response models
# ------------------------------------------------------------------ #


class SystemSummaryResponse(BaseModel):
    """System summary returned after a successful build."""

    n_atoms: int
    element: str
    box: List[float]
    units: str
    potential: str
    boundary: str
    dt: float
    thermostat: str
    xyz: str


class BuildResponse(BaseModel):
    """Response for ``POST /build``."""

    ok: bool = True
    summary: SystemSummaryResponse


class XyzResponse(BaseModel):
    """Response for ``GET /xyz``."""

    ok: bool = True
    xyz: str


class SimulationUpdateResponse(BaseModel):
    """Single progress snapshot during simulation."""

    step: int
    total: int
    pe: float
    ke: float
    total_e: float
    temperature: float
    xyz: str


class SimulationStartResponse(BaseModel):
    """Response for ``POST /simulation/start``."""

    ok: bool = True
    updates: List[SimulationUpdateResponse]


class StopResponse(BaseModel):
    """Response for ``POST /simulation/stop``."""

    ok: bool = True


class EnergyDataPayload(BaseModel):
    """Energy time-series data."""

    steps: List[int]
    pe: List[float]
    ke: List[float]
    total: List[float]
    temperature: List[float]


class EnergyResponse(BaseModel):
    """Response for ``GET /energy``."""

    ok: bool = True
    data: EnergyDataPayload


class MinimizationResultPayload(BaseModel):
    """Minimization result data."""

    converged: bool
    n_steps: int
    initial_energy: float
    final_energy: float
    max_force: float
    message: str
    energy_history: List[float]
    xyz: str


class MinimizationResponse(BaseModel):
    """Response for ``POST /minimize``."""

    ok: bool = True
    result: MinimizationResultPayload


class HealthResponse(BaseModel):
    """Response for ``GET /health``."""

    status: str = "ok"
    version: str
