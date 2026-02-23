# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

pymd is a Python molecular dynamics simulation framework where users write **only energy functions** `E(positions)` and forces `F = -∇E` are computed automatically via autodiff. This is the core design principle — no manual force derivations.

The project follows a **backend-first** architecture: domain logic is framework-independent, an application layer (`application/`) provides transport-agnostic orchestration, and both the GUI and API are thin transport layers over the shared `MDWorkflow` class.

## Setup

```bash
conda env create -f environment.yml
conda activate pymd
pip install -e ".[test,api]"
```

Or manually: `pip install numpy pyyaml pytest` (Python >= 3.10 required).

## Commands

```bash
# Run all tests (unit + integration)
python -m pytest tests/ -q

# Run unit tests only (192 tests)
python -m pytest tests/unit/ -q

# Run integration tests (direct vs API consistency)
python -m pytest tests/integration/ -v

# Run with coverage
python -m pytest tests/ --cov=pymd

# Run an example
python examples/quick_test.py

# Start the API server
uvicorn pymd.api.app:app --reload

# Launch the GUI
python -m pymd.gui                          # auto mode
python -m pymd.gui --compute-mode direct    # in-process only
python -m pymd.gui --compute-mode api       # HTTP API only
```

The package uses a `src/` layout with `pyproject.toml` — install in editable mode with `pip install -e .`.

## Architecture

### Simulation Flow

```
SystemBuilder.build() → System (atoms + state + boundary + units)
ForceCalculator(potential, autodiff_backend)
Simulator(system, integrator, force_calculator, thermostat, observers)
Simulator.run(num_steps) → loop: forces → integrate → thermostat → observe
```

### Module Layout

- **`core/`** — `System`, `State`, `Atom`, `Units`, `ElementRegistry`, plus shared `schemas.py` (payload dataclasses) and `service.py` (backend business logic)
- **`application/`** — `MDWorkflow` — transport-agnostic orchestration (primitives in, dicts out)
- **`potential/`** — ABC `PotentialEnergy`. Implementations: `LennardJonesPotential`, `MorsePotential`, `SuttonChenEAM`, `CompositePotential`
- **`force/`** — `ForceCalculator` + autodiff backends: `JAXBackend`, `PyTorchBackend`, `AutogradBackend`, `NumericalBackend`
- **`boundary/`** — `PeriodicBoundaryCondition`, `OpenBoundaryCondition`, `MixedBoundaryCondition`
- **`neighbor/`** — `BruteForceNeighborList`, `VerletList`, `CellList`
- **`integrator/`** — `VelocityVerlet`
- **`minimizer/`** — `SteepestDescent`, `ConjugateGradient`, `LBFGS`
- **`thermostat/`** — `NoThermostat` (NVE), `BerendsenThermostat`, `NoseHooverThermostat`
- **`observer/`** — `EnergyObserver`, `PrintObserver`, `CompositeObserver`
- **`builder/`** — `SystemBuilder` (fluent API), `ConfigLoader` (YAML)
- **`simulator/`** — `Simulator` orchestrates the integration loop
- **`api/`** — FastAPI transport layer (routes + Pydantic models → MDWorkflow)
- **`gui/`** — Desktop GUI: `app.py` (launcher + embedded API mgmt), `bridge.py` (direct calls → MDWorkflow), `assets/` (HTML/CSS/JS with BackendDispatcher)

### Design Patterns

Every swappable component follows the **Strategy pattern**: ABC + concrete classes. Other patterns: **Builder** (`SystemBuilder`), **Observer** (simulation monitoring), **Factory** (`BackendFactory`), **Composite** (`CompositePotential`, `CompositeObserver`).

### GUI Compute Modes

The GUI supports `--compute-mode direct|api|auto`:
- **direct**: JS → pywebview → `DirectBridge` → `MDWorkflow` → `MDService` (in-process)
- **api**: JS → fetch() → FastAPI → `MDWorkflow` → `MDService` (HTTP, with embedded API subprocess)
- **auto**: tries direct first, falls back to API fetch; embedded API subprocess started for fallback

### Extending the Framework

To add a new potential: subclass `PotentialEnergy`, implement `compute_energy()`. Forces are derived automatically.

To add a new API endpoint: add schema in `core/schemas.py`, method in `core/service.py`, workflow method in `application/md_workflow.py`, route in `api/routes.py`, bridge method in `gui/bridge.py`, and frontend dispatch in `gui/assets/app.js`.

See `docs/architecture.md` for detailed call flow diagrams and extension procedures.

### Key Files

- `src/pymd/core/service.py` — backend service layer (all business logic)
- `src/pymd/core/schemas.py` — shared payload schemas
- `src/pymd/application/md_workflow.py` — transport-agnostic orchestration
- `src/pymd/api/routes.py` — FastAPI REST endpoints
- `src/pymd/gui/app.py` — GUI launcher + embedded API management
- `src/pymd/gui/bridge.py` — direct-call GUI bridge
- `src/pymd/gui/assets/app.js` — frontend with BackendDispatcher
- `src/pymd/simulator/simulator.py` — main simulation loop
- `src/pymd/force/force_calculator.py` — force computation via autodiff
- `src/pymd/builder/config_loader.py` — YAML configuration loading
- `docs/architecture.md` — architecture documentation
- `docs/开发者指南.md` — developer guide (Chinese)
