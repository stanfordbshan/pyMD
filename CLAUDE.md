# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

pymd is a Python molecular dynamics simulation framework where users write **only energy functions** `E(positions)` and forces `F = -∇E` are computed automatically via autodiff. This is the core design principle — no manual force derivations.

The project follows a **backend-first** architecture: domain logic is framework-independent, and both the GUI and API are thin transport layers over a shared service (`core/service.py`) and shared schemas (`core/schemas.py`).

## Setup

```bash
conda env create -f environment.yml
conda activate pymd
pip install -e ".[test,api]"
```

Or manually: `pip install numpy pyyaml pytest` (Python >= 3.10 required).

## Commands

```bash
# Run all tests (unit + integration, 199 tests)
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
python -m pymd.gui                 # auto mode
python -m pymd.gui --mode direct   # in-process only
python -m pymd.gui --mode api      # HTTP API only
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
- **`api/`** — FastAPI transport layer (routes + Pydantic models, no domain logic)
- **`gui/`** — Desktop GUI: `bridge.py` (direct calls), `api_client.py` (HTTP), `runtime.py` (mode dispatcher), `assets/` (HTML/CSS/JS)

### Design Patterns

Every swappable component follows the **Strategy pattern**: ABC + concrete classes. Other patterns: **Builder** (`SystemBuilder`), **Observer** (simulation monitoring), **Factory** (`BackendFactory`), **Composite** (`CompositePotential`, `CompositeObserver`).

### GUI Compute Modes

The GUI supports `--mode direct|api|auto`:
- **direct**: JS → pywebview → `DirectBridge` → `MDService` (in-process)
- **api**: JS → pywebview → `APIClient` → FastAPI → `MDService` (HTTP)
- **auto**: tries direct, falls back to api

### Extending the Framework

To add a new potential: subclass `PotentialEnergy`, implement `compute_energy()`. Forces are derived automatically.

To add a new API endpoint: add schema in `core/schemas.py`, method in `core/service.py`, route in `api/routes.py`, methods in `gui/bridge.py` and `gui/api_client.py`.

See `docs/architecture.md` for detailed call flow diagrams and extension procedures.

### Key Files

- `src/pymd/core/service.py` — backend service layer (all business logic)
- `src/pymd/core/schemas.py` — shared payload schemas
- `src/pymd/api/routes.py` — FastAPI REST endpoints
- `src/pymd/gui/bridge.py` — direct-call GUI bridge
- `src/pymd/gui/runtime.py` — compute mode dispatcher
- `src/pymd/simulator/simulator.py` — main simulation loop
- `src/pymd/force/force_calculator.py` — force computation via autodiff
- `src/pymd/builder/config_loader.py` — YAML configuration loading
- `docs/architecture.md` — architecture documentation
- `docs/开发者指南.md` — developer guide (Chinese)
