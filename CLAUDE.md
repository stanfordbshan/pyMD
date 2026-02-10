# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

pyMD is a Python molecular dynamics simulation framework where users write **only energy functions** `E(positions)` and forces `F = -∇E` are computed automatically via autodiff. This is the core design principle — no manual force derivations.

## Setup

```bash
conda env create -f environment.yml
conda activate pymd
```

Or manually: `pip install numpy pyyaml pytest` (Python >= 3.10 required).

## Commands

```bash
# Run all unit tests (192 tests)
python -m pytest tests/unit/ -q

# Run a specific test file
python -m pytest tests/unit/test_potential.py -v

# Run with coverage
python -m pytest tests/unit/ --cov=md_simulator

# Run an example
python examples/quick_test.py
```

No linter or formatter is configured. No `pyproject.toml` or `setup.py` exists — the package is imported directly from the `md_simulator/` directory.

## Architecture

### Simulation Flow

```
SystemBuilder.build() → System (atoms + state + boundary + units)
ForceCalculator(potential, autodiff_backend)
Simulator(system, integrator, force_calculator, thermostat, observers)
Simulator.run(num_steps) → loop: forces → integrate → thermostat → observe
```

### Module Layout

- **`core/`** — `System` (central container), `State` (positions/velocities/forces arrays), `Atom`, `Units` (LAMMPS-style: LJ, REAL, METAL, SI), `ElementRegistry`
- **`potential/`** — ABC `PotentialEnergy` with `compute_energy(positions, box, atom_types) → float`. Implementations: `LennardJonesPotential`, `MorsePotential`, `SuttonChenEAM`, `CompositePotential`
- **`force/`** — `ForceCalculator` wraps a potential + an autodiff backend. `AutodiffBackend` implementations: `JAXBackend`, `PyTorchBackend`, `AutogradBackend`, `NumericalBackend` (fallback, no extra deps)
- **`boundary/`** — `PeriodicBoundaryCondition`, `OpenBoundaryCondition`, `MixedBoundaryCondition`
- **`neighbor/`** — `BruteForceNeighborList`, `VerletList`, `CellList`
- **`integrator/`** — `VelocityVerlet`
- **`minimizer/`** — ABC `Minimizer` (template method) + `MinimizationResult`. Implementations: `SteepestDescent`, `ConjugateGradient` (Polak-Ribiere), `LBFGS` (two-loop recursion)
- **`thermostat/`** — `NoThermostat` (NVE), `BerendsenThermostat`, `NoseHooverThermostat`
- **`observer/`** — `EnergyObserver`, `PrintObserver`, `CompositeObserver`
- **`builder/`** — `SystemBuilder` (fluent API), `ConfigLoader` (YAML-based simulation setup)
- **`simulator/`** — `Simulator` orchestrates the integration loop

### Design Patterns

Every swappable component (boundary conditions, neighbor lists, potentials, integrators, thermostats, observers) follows the **Strategy pattern**: an ABC defines the interface, concrete classes implement it. Components are interchangeable at construction time.

Other patterns: **Builder** (`SystemBuilder`), **Observer** (simulation monitoring), **Factory** (`BackendFactory` for autodiff backends), **Composite** (`CompositePotential`, `CompositeObserver`).

### Extending the Framework

To add a new potential: subclass `PotentialEnergy`, implement `compute_energy(positions, box, atom_types) -> float` and `get_name() -> str`. Forces are derived automatically — never implement force computation manually.

To add a new minimizer: subclass `Minimizer`, implement `_step(system, force_calculator, forces, energy) -> (new_forces, new_energy)` and `get_name() -> str`. Optionally override `_initialize()` for setup. The `minimize()` template method handles the convergence loop.

To add a new integrator/thermostat/boundary condition: subclass the corresponding ABC and implement its abstract methods. See `docs/开发者指南.md` for full examples.

### Key Files

- `md_simulator/simulator/simulator.py` — main simulation loop
- `md_simulator/force/force_calculator.py` — force computation via autodiff
- `md_simulator/force/autodiff_backend.py` — all autodiff backend implementations
- `md_simulator/builder/system_builder.py` — fluent system construction API
- `md_simulator/builder/config_loader.py` — YAML configuration loading
- `md_simulator/minimizer/minimizer.py` — Minimizer ABC + MinimizationResult
- `md_simulator/core/system.py` — central System container
- `MD_SIMULATOR_DESIGN.md` — comprehensive design document
- `docs/开发者指南.md` — developer guide with extension examples (Chinese)
- `examples/lj_argon.yaml` — reference YAML configuration
