# Architecture

## Design Philosophy

pymd follows a **backend-first** architecture with strict separation of concerns:

1. **Domain logic lives in `core/` and method modules** — no framework imports.
2. **Transport layers are thin adapters** — `api/` (FastAPI) and `gui/bridge.py` both delegate to the same `core.service.MDService`.
3. **Shared schemas prevent drift** — `core/schemas.py` defines every payload consumed or produced by both call paths.
4. **The GUI supports three compute modes** — `direct` (in-process), `api` (HTTP), `auto` (direct with API fallback).

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
│   └── routes.py                # REST endpoints + Pydantic models
│
└── gui/                         # Desktop GUI (pywebview)
    ├── main.py                  # Entry point (--mode direct|api|auto)
    ├── bridge.py                # Direct-call js_api (compute_mode=direct)
    ├── api_client.py            # HTTP-client js_api (compute_mode=api)
    ├── runtime.py               # Mode dispatcher
    └── assets/
        ├── index.html
        ├── styles.css
        ├── app.js
        └── 3Dmol-min.js

tests/
├── unit/                        # 192 unit tests
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
     MDService     (core/service.py)
          │
          ▼
     pymd domain   (builder, simulator, minimizer, …)
```

### API mode

```
HTML  →  JS (app.js)
          │
          ▼
     pywebview js_api
          │
          ▼
     APIClient     (gui/api_client.py)
          │
          ▼  HTTP
     FastAPI        (api/routes.py)
          │
          ▼
     MDService      (core/service.py)
          │
          ▼
     pymd domain
```

### Auto mode

Tries the direct path first. If the import or initialization fails
(e.g., missing native dependency), falls back to the API path
transparently.

## Shared Schemas (`core/schemas.py`)

Both `gui/bridge.py` and `api/routes.py` consume and produce the same
dataclasses defined in `core/schemas.py`:

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
3. Add a Pydantic model + route in `api/routes.py`.
4. Add a method with the same name in `gui/bridge.py` and `gui/api_client.py`.
5. Add integration tests in `tests/integration/`.

### Adding a new GUI feature

1. Add backend logic in `core/service.py` first.
2. Wire it through `gui/bridge.py` (direct) and `gui/api_client.py` (HTTP).
3. Call from `assets/app.js` via `window.pywebview.api.<method>()`.
