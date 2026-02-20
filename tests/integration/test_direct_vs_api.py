"""
Integration tests: verify that the direct bridge path and the FastAPI
path produce consistent outputs for the same inputs.
"""
import json
import pytest

from pymd.core.schemas import BuildConfig, MinimizationParams, SimulationParams
from pymd.core.service import MDService

# Skip API tests if fastapi/httpx are not installed.
fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient
from pymd.api.app import create_app


# ------------------------------------------------------------------ #
#  Fixtures
# ------------------------------------------------------------------ #

LJ_CONFIG = {
    "units": "LJ",
    "boundary": {"type": "periodic"},
    "system": {
        "element": "Ar",
        "mass": 1.0,
        "lattice": {"type": "sc", "nx": 2, "ny": 2, "nz": 2, "a": 1.5},
    },
    "potential": {"type": "lj", "epsilon": 1.0, "sigma": 1.0, "cutoff": 2.5},
    "backend": "numerical",
    "integrator": {"dt": 0.005},
    "thermostat": {"type": "nve"},
}


@pytest.fixture
def direct_service():
    return MDService()


@pytest.fixture
def api_client():
    app = create_app()
    return TestClient(app)


# ------------------------------------------------------------------ #
#  Tests
# ------------------------------------------------------------------ #


class TestBuildConsistency:
    """build_system must produce the same summary via both paths."""

    def test_build_direct(self, direct_service):
        cfg = BuildConfig.from_dict(LJ_CONFIG)
        summary = direct_service.build_system(cfg)
        assert summary.n_atoms == 8
        assert summary.element == "Ar"
        assert len(summary.xyz) > 0

    def test_build_api(self, api_client):
        resp = api_client.post("/api/build", json=LJ_CONFIG)
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["summary"]["n_atoms"] == 8
        assert body["summary"]["element"] == "Ar"

    def test_build_same_n_atoms(self, direct_service, api_client):
        cfg = BuildConfig.from_dict(LJ_CONFIG)
        direct_summary = direct_service.build_system(cfg)

        resp = api_client.post("/api/build", json=LJ_CONFIG)
        api_summary = resp.json()["summary"]

        assert direct_summary.n_atoms == api_summary["n_atoms"]
        assert direct_summary.element == api_summary["element"]
        assert direct_summary.units == api_summary["units"]


class TestMinimizationConsistency:
    """Minimization must converge via both paths."""

    def _build(self, svc):
        cfg = BuildConfig.from_dict(LJ_CONFIG)
        svc.build_system(cfg)

    def test_minimize_direct(self, direct_service):
        self._build(direct_service)
        params = MinimizationParams(
            algorithm="steepest_descent",
            force_tol=1e-2,
            energy_tol=1e-6,
            max_steps=200,
        )
        result = direct_service.run_minimization(params)
        assert result.converged
        assert result.final_energy <= result.initial_energy

    def test_minimize_api(self, api_client):
        api_client.post("/api/build", json=LJ_CONFIG)
        resp = api_client.post("/api/minimize", json={
            "algorithm": "steepest_descent",
            "force_tol": 1e-2,
            "energy_tol": 1e-6,
            "max_steps": 200,
        })
        assert resp.status_code == 200
        result = resp.json()["result"]
        assert result["converged"]
        assert result["final_energy"] <= result["initial_energy"]


class TestSimulationConsistency:
    """Simulation must run and return energy data via both paths."""

    def test_simulation_direct(self, direct_service):
        cfg = BuildConfig.from_dict(LJ_CONFIG)
        direct_service.build_system(cfg)
        params = SimulationParams(num_steps=10, viz_interval=5, init_temp=0.5)
        direct_service.prepare_simulation(params)
        updates = []
        direct_service.run_steps(10, viz_interval=5, on_update=updates.append)
        assert len(updates) == 2  # step 5 and step 10
        data = direct_service.get_energy_data()
        assert len(data.steps) == 10

    def test_simulation_api(self, api_client):
        api_client.post("/api/build", json=LJ_CONFIG)
        resp = api_client.post("/api/simulation/start", json={
            "num_steps": 10,
            "viz_interval": 5,
            "init_temp": 0.5,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert len(body["updates"]) == 2

        resp = api_client.get("/api/energy")
        assert resp.status_code == 200
        assert len(resp.json()["data"]["steps"]) == 10
