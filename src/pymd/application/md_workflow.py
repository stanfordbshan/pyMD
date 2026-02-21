"""Transport-agnostic MD workflow.

Wraps :class:`MDService` so that callers (API routes, GUI bridge) pass
plain Python primitives and receive plain dicts/strings — no schema
objects cross the boundary.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from pymd.core.schemas import (
    BuildConfig,
    MinimizationParams,
    SimulationParams,
)
from pymd.core.service import MDService


class MDWorkflow:
    """Stateful orchestrator — one instance per session."""

    def __init__(self, service: Optional[MDService] = None):
        self._svc = service or MDService()

    # ------------------------------------------------------------------ #
    #  Properties
    # ------------------------------------------------------------------ #

    @property
    def is_running(self) -> bool:
        return self._svc.is_running

    @property
    def has_system(self) -> bool:
        return self._svc.has_system

    # ------------------------------------------------------------------ #
    #  Build
    # ------------------------------------------------------------------ #

    def build_system(
        self,
        *,
        units: str = "LJ",
        boundary: Optional[Dict[str, Any]] = None,
        system: Optional[Dict[str, Any]] = None,
        potential: Optional[Dict[str, Any]] = None,
        backend: str = "numerical",
        integrator: Optional[Dict[str, Any]] = None,
        thermostat: Optional[Dict[str, Any]] = None,
    ) -> dict:
        cfg = BuildConfig(
            units=units,
            boundary=boundary or {"type": "periodic"},
            system=system or {},
            potential=potential or {"type": "lj"},
            backend=backend,
            integrator=integrator or {"dt": 0.005},
            thermostat=thermostat or {"type": "nve"},
        )
        summary = self._svc.build_system(cfg)
        return summary.to_dict()

    # ------------------------------------------------------------------ #
    #  XYZ
    # ------------------------------------------------------------------ #

    def get_xyz(self) -> str:
        return self._svc.get_xyz()

    # ------------------------------------------------------------------ #
    #  Simulation
    # ------------------------------------------------------------------ #

    def start_simulation(
        self,
        *,
        num_steps: int = 1000,
        viz_interval: int = 50,
        init_temp: float = 1.0,
        on_update: Optional[Callable[[dict], None]] = None,
    ) -> list[dict]:
        params = SimulationParams(
            num_steps=num_steps,
            viz_interval=viz_interval,
            init_temp=init_temp,
        )
        self._svc.prepare_simulation(params)
        updates: list[dict] = []

        def _collect(u):
            d = u.to_dict()
            updates.append(d)
            if on_update is not None:
                on_update(d)

        self._svc.run_steps(
            num_steps=params.num_steps,
            viz_interval=params.viz_interval,
            on_update=_collect,
        )
        return updates

    def stop(self) -> None:
        self._svc.stop()

    # ------------------------------------------------------------------ #
    #  Energy data
    # ------------------------------------------------------------------ #

    def get_energy_data(self) -> dict:
        return self._svc.get_energy_data().to_dict()

    # ------------------------------------------------------------------ #
    #  Minimization
    # ------------------------------------------------------------------ #

    def run_minimization(
        self,
        *,
        algorithm: str = "conjugate_gradient",
        force_tol: float = 1e-4,
        energy_tol: float = 1e-8,
        max_steps: int = 10000,
    ) -> dict:
        params = MinimizationParams(
            algorithm=algorithm,
            force_tol=force_tol,
            energy_tol=energy_tol,
            max_steps=max_steps,
        )
        result = self._svc.run_minimization(params)
        return result.to_dict()
