"""
API routes — thin adapters that delegate to :class:`MDService`.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from pymd.api.models import (
    BuildRequest,
    BuildResponse,
    EnergyDataPayload,
    EnergyResponse,
    MinimizationRequest,
    MinimizationResponse,
    MinimizationResultPayload,
    SimulationRequest,
    SimulationStartResponse,
    SimulationUpdateResponse,
    StopResponse,
    SystemSummaryResponse,
    XyzResponse,
)
from pymd.core.schemas import BuildConfig, MinimizationParams, SimulationParams
from pymd.core.service import MDService

router = APIRouter()

# One service instance per process (matches the GUI's single-session model).
_service = MDService()


def get_service() -> MDService:
    return _service


# ------------------------------------------------------------------ #
#  Endpoints
# ------------------------------------------------------------------ #


@router.post("/build", response_model=BuildResponse)
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
        return BuildResponse(
            summary=SystemSummaryResponse(**summary.to_dict()),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/xyz", response_model=XyzResponse)
def get_xyz():
    try:
        xyz = get_service().get_xyz()
        return XyzResponse(xyz=xyz)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/simulation/start", response_model=SimulationStartResponse)
def start_simulation(req: SimulationRequest):
    svc = get_service()
    try:
        params = SimulationParams(
            num_steps=req.num_steps,
            viz_interval=req.viz_interval,
            init_temp=req.init_temp,
        )
        svc.prepare_simulation(params)
        updates: list[SimulationUpdateResponse] = []
        svc.run_steps(
            num_steps=params.num_steps,
            viz_interval=params.viz_interval,
            on_update=lambda u: updates.append(
                SimulationUpdateResponse(**u.to_dict())
            ),
        )
        return SimulationStartResponse(updates=updates)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/simulation/stop", response_model=StopResponse)
def stop_simulation():
    get_service().stop()
    return StopResponse()


@router.get("/energy", response_model=EnergyResponse)
def get_energy_data():
    try:
        data = get_service().get_energy_data()
        return EnergyResponse(
            data=EnergyDataPayload(**data.to_dict()),
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/minimize", response_model=MinimizationResponse)
def run_minimization(req: MinimizationRequest):
    try:
        params = MinimizationParams(
            algorithm=req.algorithm,
            force_tol=req.force_tol,
            energy_tol=req.energy_tol,
            max_steps=req.max_steps,
        )
        result = get_service().run_minimization(params)
        return MinimizationResponse(
            result=MinimizationResultPayload(**result.to_dict()),
        )
    except (RuntimeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
