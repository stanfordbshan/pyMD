"""
API routes — thin adapters that delegate to :class:`MDService`.
"""
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from pymd.core.schemas import BuildConfig, MinimizationParams, SimulationParams
from pymd.core.service import MDService

router = APIRouter()

# One service instance per process (matches the GUI's single-session model).
_service = MDService()


def get_service() -> MDService:
    return _service


# ------------------------------------------------------------------ #
#  Pydantic request / response models
# ------------------------------------------------------------------ #


class BuildRequest(BaseModel):
    units: str = "LJ"
    boundary: Dict[str, Any] = Field(default_factory=lambda: {"type": "periodic"})
    system: Dict[str, Any] = Field(default_factory=dict)
    potential: Dict[str, Any] = Field(default_factory=lambda: {"type": "lj"})
    backend: str = "numerical"
    integrator: Dict[str, Any] = Field(default_factory=lambda: {"dt": 0.005})
    thermostat: Dict[str, Any] = Field(default_factory=lambda: {"type": "nve"})


class SimulationRequest(BaseModel):
    num_steps: int = 1000
    viz_interval: int = 50
    init_temp: float = 1.0


class MinimizationRequest(BaseModel):
    algorithm: str = "conjugate_gradient"
    force_tol: float = 1e-4
    energy_tol: float = 1e-8
    max_steps: int = 10000


# ------------------------------------------------------------------ #
#  Endpoints
# ------------------------------------------------------------------ #


@router.post("/build")
def build_system(req: BuildRequest):
    try:
        cfg = BuildConfig(
            units=req.units,
            boundary=req.boundary,
            system=req.system,
            potential=req.potential,
            backend=req.backend,
            integrator=req.integrator,
            thermostat=req.thermostat,
        )
        summary = get_service().build_system(cfg)
        return {"ok": True, "summary": summary.to_dict()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/xyz")
def get_xyz():
    try:
        xyz = get_service().get_xyz()
        return {"ok": True, "xyz": xyz}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/simulation/start")
def start_simulation(req: SimulationRequest):
    svc = get_service()
    try:
        params = SimulationParams(
            num_steps=req.num_steps,
            viz_interval=req.viz_interval,
            init_temp=req.init_temp,
        )
        svc.prepare_simulation(params)
        # Run synchronously (HTTP request blocks until done).
        updates = []
        svc.run_steps(
            num_steps=params.num_steps,
            viz_interval=params.viz_interval,
            on_update=lambda u: updates.append(u.to_dict()),
        )
        return {"ok": True, "updates": updates}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/simulation/stop")
def stop_simulation():
    get_service().stop()
    return {"ok": True}


@router.get("/energy")
def get_energy_data():
    try:
        data = get_service().get_energy_data()
        return {"ok": True, "data": data.to_dict()}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/minimize")
def run_minimization(req: MinimizationRequest):
    try:
        params = MinimizationParams(
            algorithm=req.algorithm,
            force_tol=req.force_tol,
            energy_tol=req.energy_tol,
            max_steps=req.max_steps,
        )
        result = get_service().run_minimization(params)
        return {"ok": True, "result": result.to_dict()}
    except (RuntimeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
