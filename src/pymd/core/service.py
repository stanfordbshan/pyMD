"""
Backend service layer for pymd.

Framework-independent business logic consumed by both the GUI bridge
and the FastAPI transport layer.  No references to webview, FastAPI,
or any transport concern belong here.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np

from pymd.builder.config_loader import (
    _parse_boundary,
    _parse_potential,
    _parse_thermostat,
    _parse_units,
)
from pymd.builder.system_builder import SystemBuilder
from pymd.core.schemas import (
    BuildConfig,
    EnergyData,
    MinimizationParams,
    MinimizationResult,
    SimulationParams,
    SimulationUpdate,
    SystemSummary,
)
from pymd.force import BackendFactory, ForceCalculator
from pymd.integrator import VelocityVerlet
from pymd.minimizer import ConjugateGradient, LBFGS, SteepestDescent
from pymd.observer import EnergyObserver
from pymd.simulator import Simulator


class MDService:
    """Stateful MD backend.  One instance per session."""

    def __init__(self):
        self._system = None
        self._force_calc = None
        self._simulator = None
        self._integrator = None
        self._thermostat = None
        self._energy_observer = None
        self._running = False
        self._config: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------ #
    #  Properties
    # ------------------------------------------------------------------ #

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def has_system(self) -> bool:
        return self._system is not None

    # ------------------------------------------------------------------ #
    #  Build
    # ------------------------------------------------------------------ #

    def build_system(self, config: BuildConfig) -> SystemSummary:
        """Build System + ForceCalculator from *config*.

        Returns a :class:`SystemSummary` (no JSON, no HTTP).
        """
        raw = config.to_dict()

        units = _parse_units(raw)
        bc = _parse_boundary(raw)

        sys_config = raw.get("system", {})
        builder = SystemBuilder()
        builder.units(units)
        builder.boundary_condition(bc)
        builder.element(
            atom_type=sys_config.get("element", "X"),
            mass=float(sys_config.get("mass", 1.0)),
        )

        lattice_config = sys_config.get("lattice", {})
        lattice_type = lattice_config.get("type", "fcc").lower()

        if lattice_type == "fcc":
            builder.fcc_lattice(
                nx=int(lattice_config.get("nx", 4)),
                ny=int(lattice_config.get("ny", 4)),
                nz=int(lattice_config.get("nz", 4)),
                a=float(lattice_config.get("a", 1.0)),
            )
        elif lattice_type == "bcc":
            builder.bcc_lattice(
                nx=int(lattice_config.get("nx", 4)),
                ny=int(lattice_config.get("ny", 4)),
                nz=int(lattice_config.get("nz", 4)),
                a=float(lattice_config.get("a", 1.0)),
            )
        elif lattice_type == "sc":
            builder.sc_lattice(
                nx=int(lattice_config.get("nx", 4)),
                ny=int(lattice_config.get("ny", 4)),
                nz=int(lattice_config.get("nz", 4)),
                a=float(lattice_config.get("a", 1.0)),
            )
        elif lattice_type == "random":
            builder.box(
                float(lattice_config.get("lx", 10.0)),
                float(lattice_config.get("ly", 10.0)),
                float(lattice_config.get("lz", 10.0)),
            )
            builder.random_positions(
                n_atoms=int(lattice_config.get("n_atoms", 100)),
            )
        elif lattice_type == "positions":
            coords = lattice_config.get("coordinates", [])
            builder.box(
                float(lattice_config.get("lx", 100.0)),
                float(lattice_config.get("ly", 100.0)),
                float(lattice_config.get("lz", 100.0)),
            )
            builder.positions(np.array(coords, dtype=float))

        if "temperature" in sys_config:
            builder.temperature(float(sys_config["temperature"]))

        self._system = builder.build()

        potential = _parse_potential(raw)
        backend = BackendFactory.create(raw.get("backend", "numerical"))
        self._force_calc = ForceCalculator(potential=potential, backend=backend)

        int_config = raw.get("integrator", {})
        dt = float(int_config.get("dt", 0.005))
        self._integrator = VelocityVerlet(dt=dt)
        self._thermostat = _parse_thermostat(raw, dt)
        self._config = raw

        box = self._system.state.box
        return SystemSummary(
            n_atoms=self._system.get_num_atoms(),
            element=sys_config.get("element", "X"),
            box=[float(box[0]), float(box[1]), float(box[2])],
            units=raw.get("units", "LJ"),
            potential=raw.get("potential", {}).get("type", "lj"),
            boundary=raw.get("boundary", {}).get("type", "periodic"),
            dt=dt,
            thermostat=raw.get("thermostat", {}).get("type", "nve"),
            xyz=self._system_to_xyz(),
        )

    # ------------------------------------------------------------------ #
    #  XYZ
    # ------------------------------------------------------------------ #

    def get_xyz(self) -> str:
        """Return current positions as XYZ-format string."""
        if self._system is None:
            raise RuntimeError("No system built")
        return self._system_to_xyz()

    # ------------------------------------------------------------------ #
    #  Simulation (synchronous, one step at a time)
    # ------------------------------------------------------------------ #

    def prepare_simulation(self, params: SimulationParams) -> None:
        """Initialise velocities, observer, and simulator for a run."""
        if self._system is None:
            raise RuntimeError("No system built. Use build_system first.")
        if self._running:
            raise RuntimeError("Simulation already running.")

        self._running = True
        self._initialize_velocities(params.init_temp)
        self._energy_observer = EnergyObserver(interval=1)
        self._simulator = Simulator(
            system=self._system,
            integrator=self._integrator,
            force_calculator=self._force_calc,
            thermostat=self._thermostat,
            observers=[self._energy_observer],
        )
        self._simulator.initialize()

    def run_steps(
        self,
        num_steps: int,
        viz_interval: int = 50,
        on_update: Optional[Callable[[SimulationUpdate], None]] = None,
    ) -> None:
        """Run *num_steps* synchronously, calling *on_update* at intervals."""
        try:
            for i in range(num_steps):
                if not self._running:
                    break
                self._simulator.run(1)
                if (i + 1) % viz_interval == 0 or i == num_steps - 1:
                    update = self._make_sim_update(i + 1, num_steps)
                    if on_update is not None:
                        on_update(update)
        finally:
            self._running = False

    def stop(self) -> None:
        self._running = False

    def get_energy_data(self) -> EnergyData:
        if self._energy_observer is None:
            raise RuntimeError("No energy data available")
        obs = self._energy_observer
        return EnergyData(
            steps=list(obs.steps),
            pe=[float(x) for x in obs.potential_energies],
            ke=[float(x) for x in obs.kinetic_energies],
            total=[float(x) for x in obs.total_energies],
            temperature=[float(x) for x in obs.temperatures],
        )

    # ------------------------------------------------------------------ #
    #  Minimization (synchronous)
    # ------------------------------------------------------------------ #

    def run_minimization(self, params: MinimizationParams) -> MinimizationResult:
        if self._system is None:
            raise RuntimeError("No system built. Use build_system first.")
        if self._running:
            raise RuntimeError("A task is already running.")

        self._running = True
        self._system.state.velocities[:] = 0.0

        try:
            algo = params.algorithm
            kwargs = dict(
                force_tol=params.force_tol,
                energy_tol=params.energy_tol,
                max_steps=params.max_steps,
            )
            if algo == "steepest_descent":
                minimizer = SteepestDescent(**kwargs)
            elif algo == "conjugate_gradient":
                minimizer = ConjugateGradient(**kwargs)
            elif algo == "lbfgs":
                minimizer = LBFGS(**kwargs)
            else:
                raise ValueError(f"Unknown algorithm: {algo}")

            result = minimizer.minimize(self._system, self._force_calc)

            return MinimizationResult(
                converged=result.converged,
                n_steps=result.n_steps,
                initial_energy=float(result.initial_energy),
                final_energy=float(result.final_energy),
                max_force=float(result.max_force),
                message=result.message,
                energy_history=[float(e) for e in result.energy_history],
                xyz=self._system_to_xyz(),
            )
        finally:
            self._running = False

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _make_sim_update(self, current_step: int, total_steps: int) -> SimulationUpdate:
        obs = self._energy_observer
        return SimulationUpdate(
            step=current_step,
            total=total_steps,
            pe=float(obs.potential_energies[-1]) if obs.potential_energies else 0,
            ke=float(obs.kinetic_energies[-1]) if obs.kinetic_energies else 0,
            total_e=float(obs.total_energies[-1]) if obs.total_energies else 0,
            temperature=float(obs.temperatures[-1]) if obs.temperatures else 0,
            xyz=self._system_to_xyz(),
        )

    def _initialize_velocities(self, temperature: float) -> None:
        system = self._system
        n_atoms = system.get_num_atoms()
        if temperature <= 0 or n_atoms < 2:
            system.state.velocities[:] = 0.0
            return

        kB = system.units.boltzmann
        masses = system.get_masses()

        velocities = np.random.randn(n_atoms, 3)
        for i in range(n_atoms):
            sigma = np.sqrt(kB * temperature / masses[i])
            velocities[i] *= sigma

        total_mass = np.sum(masses)
        com_velocity = np.sum(masses[:, np.newaxis] * velocities, axis=0) / total_mass
        velocities -= com_velocity

        ke = 0.5 * np.sum(masses[:, np.newaxis] * velocities**2)
        n_dof = 3 * n_atoms - 3
        current_temp = 2.0 * ke / (n_dof * kB)
        if current_temp > 0:
            velocities *= np.sqrt(temperature / current_temp)

        system.state.velocities[:] = velocities

    def _system_to_xyz(self) -> str:
        positions = self._system.state.positions
        n = len(positions)
        step = self._system.state.step
        lines = [str(n), f"Step {step}"]
        for i, pos in enumerate(positions):
            atom_type = self._system.atoms[i].atom_type
            lines.append(f"{atom_type}  {pos[0]:.6f}  {pos[1]:.6f}  {pos[2]:.6f}")
        return "\n".join(lines)
