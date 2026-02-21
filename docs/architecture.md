# Architecture

## Design Philosophy

pymd follows a **backend-first** architecture with strict separation of concerns:

1. **Domain logic lives in `core/` and method modules** — no framework imports.
2. **An application layer (`application/`) provides transport-agnostic orchestration** — accepts plain primitives, returns plain dicts.
3. **Transport layers are thin adapters** — `api/` (FastAPI) and `gui/bridge.py` both delegate to the application layer.
4. **Shared schemas prevent drift** — `core/schemas.py` defines every payload consumed or produced internally.
5. **The GUI supports three compute modes** — `direct` (in-process), `api` (HTTP), `auto` (direct with API fallback via embedded subprocess).

## File Tree

```
src/pymd/
├── core/                        # Domain logic + shared contracts
│   ├── atom.py
│   ├── state.py
│   ├── system.py
│   ├── units.py
│   ├── element_registry.py
│   ├── constants.py
│   ├── schemas.py               # Shared payload dataclasses
│   └── service.py               # MDService — backend business logic
│
├── application/                 # Transport-agnostic orchestration
│   ├── __init__.py              # Exports MDWorkflow
│   └── md_workflow.py           # MDWorkflow (primitives in, dicts out)
│
├── boundary/                    # Boundary condition strategies
├── builder/                     # SystemBuilder + ConfigLoader
├── force/                       # ForceCalculator + autodiff backends
├── integrator/                  # VelocityVerlet
├── minimizer/                   # SD, CG, L-BFGS
├── neighbor/                    # Neighbor list algorithms
├── observer/                    # Energy/Print/Trajectory observers
├── potential/                   # LJ, Morse, EAM potentials
├── simulator/                   # Simulator orchestrator
├── thermostat/                  # NVE, Berendsen, Nose-Hoover
│
├── api/                         # FastAPI transport (no domain logic)
│   ├── app.py                   # App factory
│   ├── models.py                # Pydantic request/response models
│   └── routes.py                # REST endpoints → MDWorkflow
│
└── gui/                         # Desktop GUI (pywebview)
    ├── app.py                   # Launcher + embedded API management
    ├── bridge.py                # Direct-call js_api → MDWorkflow
    ├── __init__.py              # Exports launch_gui
    ├── __main__.py              # python -m pymd.gui entry point
    └── assets/
        ├── index.html
        ├── styles.css
        ├── app.js               # Frontend with BackendDispatcher
        └── 3Dmol-min.js

tests/
├── unit/                        # Unit tests
└── integration/                 # Direct vs API consistency tests

benchmarks/                      # Performance benchmarks (pytest-benchmark)
```

## Call Flow

### Direct mode (default)

```
HTML  →  JS (app.js)
          │
          ▼
     pywebview js_api
          │
          ▼
     DirectBridge  (gui/bridge.py)
          │
          ▼
     MDWorkflow    (application/md_workflow.py)
          │
          ▼
     MDService     (core/service.py)
          │
          ▼
     pymd domain   (builder, simulator, minimizer, …)
```

### API mode

```
HTML  →  JS (app.js)
          │
          ▼  fetch()
     FastAPI        (api/routes.py)
          │
          ▼
     MDWorkflow     (application/md_workflow.py)
          │
          ▼
     MDService      (core/service.py)
          │
          ▼
     pymd domain
```

### Auto mode

The GUI starts an embedded API subprocess, passes its URL to the
frontend via query params, and the JS `BackendDispatcher` tries
direct pywebview calls first, falling back to HTTP fetch on failure.
The embedded subprocess is cleaned up when the window closes.

## Shared Schemas (`core/schemas.py`)

Internal schemas used by `MDService` and `MDWorkflow`:

| Schema              | Direction | Description                         |
|---------------------|-----------|-------------------------------------|
| `BuildConfig`       | Input     | System + potential + integrator cfg |
| `SimulationParams`  | Input     | Steps, interval, temperature        |
| `MinimizationParams`| Input     | Algorithm + tolerances              |
| `SystemSummary`     | Output    | Atom count, box, XYZ after build    |
| `SimulationUpdate`  | Output    | Per-interval progress snapshot      |
| `EnergyData`        | Output    | Full energy time-series             |
| `MinimizationResult`| Output    | Convergence info + final XYZ        |

## Extending pymd

### Adding a new potential

1. Create `src/pymd/potential/<name>.py`, subclass `PotentialEnergy`.
2. Implement `compute_energy(positions, box, atom_types) -> float`.
3. Register in `builder/config_loader.py :: _parse_potential()`.
4. Add unit tests in `tests/unit/test_potential.py`.
5. No force code needed — autodiff handles it.

### Adding a new API endpoint

1. Define input/output schemas in `core/schemas.py`.
2. Add the service method in `core/service.py`.
3. Add a workflow method in `application/md_workflow.py` (primitives → schema → service → dict).
4. Add a Pydantic model + route in `api/routes.py` (delegates to workflow).
5. Add a bridge method in `gui/bridge.py` (delegates to workflow).
6. Add frontend dispatch in `gui/assets/app.js` (direct + API paths).
7. Add integration tests in `tests/integration/`.

### Adding a new GUI feature

1. Add backend logic in `core/service.py` first.
2. Wrap it in `application/md_workflow.py`.
3. Wire through `gui/bridge.py` (direct calls).
4. Add both direct and API call paths in `assets/app.js` via `dispatch()`.
