"""
Integration tests for the application layer (MDWorkflow).

Verifies that MDWorkflow correctly wraps MDService and that the
API routes produce consistent results when called through MDWorkflow.
"""
import pytest

from pymd.application import MDWorkflow

# Skip API tests if fastapi/httpx are not installed.
fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient
from pymd.api.app import create_app


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
def workflow():
    return MDWorkflow()


@pytest.fixture
def api_client():
    app = create_app()
    return TestClient(app)


class TestWorkflowBuild:
    """MDWorkflow.build_system returns plain dict with expected keys."""

    def test_build_returns_dict(self, workflow):
        summary = workflow.build_system(**LJ_CONFIG)
        assert isinstance(summary, dict)
        assert summary["n_atoms"] == 8
        assert summary["element"] == "Ar"
        assert len(summary["xyz"]) > 0

    def test_has_system_after_build(self, workflow):
        assert not workflow.has_system
        workflow.build_system(**LJ_CONFIG)
        assert workflow.has_system


class TestWorkflowSimulation:
    """MDWorkflow simulation returns list of update dicts."""

    def test_simulation_returns_updates(self, workflow):
        workflow.build_system(**LJ_CONFIG)
        updates = workflow.start_simulation(
            num_steps=10, viz_interval=5, init_temp=0.5,
        )
        assert isinstance(updates, list)
        assert len(updates) == 2  # step 5 and step 10
        assert updates[-1]["step"] == 10

    def test_energy_data_after_simulation(self, workflow):
        workflow.build_system(**LJ_CONFIG)
        workflow.start_simulation(num_steps=10, viz_interval=5, init_temp=0.5)
        data = workflow.get_energy_data()
        assert isinstance(data, dict)
        assert len(data["steps"]) == 10


class TestWorkflowMinimization:
    """MDWorkflow minimization returns plain dict."""

    def test_minimization_returns_dict(self, workflow):
        workflow.build_system(**LJ_CONFIG)
        result = workflow.run_minimization(
            algorithm="steepest_descent",
            force_tol=1e-2,
            energy_tol=1e-6,
            max_steps=200,
        )
        assert isinstance(result, dict)
        assert result["converged"]
        assert result["final_energy"] <= result["initial_energy"]


class TestWorkflowVsApi:
    """MDWorkflow and API routes produce consistent results."""

    def test_build_consistency(self, workflow, api_client):
        wf_summary = workflow.build_system(**LJ_CONFIG)

        resp = api_client.post("/api/build", json=LJ_CONFIG)
        api_summary = resp.json()["summary"]

        assert wf_summary["n_atoms"] == api_summary["n_atoms"]
        assert wf_summary["element"] == api_summary["element"]
        assert wf_summary["units"] == api_summary["units"]
