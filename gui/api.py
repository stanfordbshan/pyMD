"""
SimulatorAPI: Python backend exposed to JavaScript via pywebview js_api.

Wraps pyMD library calls for the desktop GUI.
"""
import json
import threading
import traceback

import numpy as np

from pyMD.builder.config_loader import load_yaml as _load_yaml
from pyMD.builder.config_loader import (
    _parse_units,
    _parse_boundary,
    _parse_potential,
    _parse_thermostat,
    build_simulation_from_config,
)
from pyMD.builder.system_builder import SystemBuilder
from pyMD.core import Units
from pyMD.force import BackendFactory, ForceCalculator
from pyMD.integrator import VelocityVerlet
from pyMD.thermostat import NoThermostat, BerendsenThermostat, NoseHooverThermostat
from pyMD.observer import EnergyObserver
from pyMD.simulator import Simulator
from pyMD.minimizer import SteepestDescent, ConjugateGradient, LBFGS


class SimulatorAPI:
    """Exposed to JS via pywebview js_api. Wraps pyMD."""

    def __init__(self):
        self._window = None
        self._system = None
        self._force_calc = None
        self._simulator = None
        self._integrator = None
        self._thermostat = None
        self._energy_observer = None
        self._running = False
        self._sim_thread = None
        self._config = None

    def set_window(self, window):
        """Called by main.py to provide the pywebview window reference."""
        self._window = window

    # ------------------------------------------------------------------ #
    #  Setup
    # ------------------------------------------------------------------ #

    def load_yaml(self):
        """Open native file dialog, parse YAML, return config dict to JS."""
        try:
            import webview
            result = self._window.create_file_dialog(
                dialog_type=webview.FileDialog.OPEN,
                file_types=("YAML Files (*.yaml;*.yml)", "All files (*.*)"),
            )
            if not result:
                return json.dumps({"error": "No file selected"})

            path = result[0] if isinstance(result, (list, tuple)) else result
            config = _load_yaml(path)
            self._config = config
            return json.dumps({"ok": True, "config": config, "path": str(path)})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def build_system(self, config_json):
        """Build System + ForceCalculator from config dict.

        Returns summary dict + XYZ string for 3Dmol.js viewer.
        """
        try:
            config = json.loads(config_json) if isinstance(config_json, str) else config_json

            # Parse units
            units = _parse_units(config)

            # Parse boundary
            bc = _parse_boundary(config)

            # Build system via SystemBuilder
            sys_config = config.get("system", {})
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

            # Parse potential + force calculator
            potential = _parse_potential(config)
            backend_name = config.get("backend", "numerical")
            backend = BackendFactory.create(backend_name)
            self._force_calc = ForceCalculator(potential=potential, backend=backend)

            # Parse integrator
            int_config = config.get("integrator", {})
            dt = float(int_config.get("dt", 0.005))
            self._integrator = VelocityVerlet(dt=dt)

            # Parse thermostat
            self._thermostat = _parse_thermostat(config, dt)

            # Store config for later use
            self._config = config

            # Build summary
            box = self._system.state.box
            summary = {
                "n_atoms": self._system.get_num_atoms(),
                "element": sys_config.get("element", "X"),
                "box": [float(box[0]), float(box[1]), float(box[2])],
                "units": config.get("units", "LJ"),
                "potential": config.get("potential", {}).get("type", "lj"),
                "boundary": config.get("boundary", {}).get("type", "periodic"),
                "dt": dt,
                "thermostat": config.get("thermostat", {}).get("type", "nve"),
            }

            xyz = self._system_to_xyz()
            return json.dumps({"ok": True, "summary": summary, "xyz": xyz})

        except Exception as e:
            traceback.print_exc()
            return json.dumps({"error": str(e)})

    def get_xyz(self):
        """Generate XYZ-format string from current system positions."""
        if self._system is None:
            return json.dumps({"error": "No system built"})
        try:
            xyz = self._system_to_xyz()
            return json.dumps({"ok": True, "xyz": xyz})
        except Exception as e:
            return json.dumps({"error": str(e)})

    # ------------------------------------------------------------------ #
    #  Simulation
    # ------------------------------------------------------------------ #

    def start_simulation(self, num_steps, viz_interval=50):
        """Spawn daemon thread to run simulation step by step."""
        if self._system is None:
            return json.dumps({"error": "No system built. Use Setup tab first."})
        if self._running:
            return json.dumps({"error": "Simulation already running."})

        num_steps = int(num_steps)
        viz_interval = int(viz_interval)
        self._running = True

        # Create energy observer
        self._energy_observer = EnergyObserver(interval=1)

        # Create simulator
        self._simulator = Simulator(
            system=self._system,
            integrator=self._integrator,
            force_calculator=self._force_calc,
            thermostat=self._thermostat,
            observers=[self._energy_observer],
        )
        self._simulator.initialize()

        def run_loop():
            try:
                for i in range(num_steps):
                    if not self._running:
                        break

                    # Run one step
                    self._simulator.run(1)

                    # Push updates at viz_interval
                    if (i + 1) % viz_interval == 0 or i == num_steps - 1:
                        self._push_sim_update(i + 1, num_steps)

            except Exception as e:
                traceback.print_exc()
                self._eval_js(f"onSimulationError({json.dumps(str(e))})")
            finally:
                self._running = False
                self._eval_js("onSimulationDone()")

        self._sim_thread = threading.Thread(target=run_loop, daemon=True)
        self._sim_thread.start()
        return json.dumps({"ok": True, "message": f"Started {num_steps} steps"})

    def stop_simulation(self):
        """Set running flag to False to stop simulation loop."""
        self._running = False
        return json.dumps({"ok": True})

    def get_energy_data(self):
        """Return EnergyObserver data as dict of lists."""
        if self._energy_observer is None:
            return json.dumps({"error": "No energy data available"})
        try:
            data = {
                "steps": list(self._energy_observer.steps),
                "pe": [float(x) for x in self._energy_observer.potential_energies],
                "ke": [float(x) for x in self._energy_observer.kinetic_energies],
                "total": [float(x) for x in self._energy_observer.total_energies],
                "temperature": [float(x) for x in self._energy_observer.temperatures],
            }
            return json.dumps({"ok": True, "data": data})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _push_sim_update(self, current_step, total_steps):
        """Push simulation progress to JS via evaluate_js."""
        if self._energy_observer is None:
            return

        obs = self._energy_observer
        latest = {
            "step": current_step,
            "total": total_steps,
            "pe": float(obs.potential_energies[-1]) if obs.potential_energies else 0,
            "ke": float(obs.kinetic_energies[-1]) if obs.kinetic_energies else 0,
            "total_e": float(obs.total_energies[-1]) if obs.total_energies else 0,
            "temperature": float(obs.temperatures[-1]) if obs.temperatures else 0,
        }
        xyz = self._system_to_xyz()
        self._eval_js(f"onSimulationUpdate({json.dumps(latest)}, {json.dumps(xyz)})")

    # ------------------------------------------------------------------ #
    #  Minimization
    # ------------------------------------------------------------------ #

    def run_minimization(self, algorithm, params_json):
        """Run energy minimization in daemon thread."""
        if self._system is None:
            return json.dumps({"error": "No system built. Use Setup tab first."})
        if self._running:
            return json.dumps({"error": "A task is already running."})

        params = json.loads(params_json) if isinstance(params_json, str) else params_json
        self._running = True

        def minimize_thread():
            try:
                force_tol = float(params.get("force_tol", 1e-4))
                energy_tol = float(params.get("energy_tol", 1e-8))
                max_steps = int(params.get("max_steps", 10000))

                algo = algorithm.lower()
                if algo == "steepest_descent":
                    minimizer = SteepestDescent(
                        force_tol=force_tol, energy_tol=energy_tol, max_steps=max_steps
                    )
                elif algo == "conjugate_gradient":
                    minimizer = ConjugateGradient(
                        force_tol=force_tol, energy_tol=energy_tol, max_steps=max_steps
                    )
                elif algo == "lbfgs":
                    minimizer = LBFGS(
                        force_tol=force_tol, energy_tol=energy_tol, max_steps=max_steps
                    )
                else:
                    self._eval_js(f"onMinimizationError({json.dumps(f'Unknown algorithm: {algorithm}')})")
                    return

                result = minimizer.minimize(self._system, self._force_calc)

                result_dict = {
                    "converged": result.converged,
                    "n_steps": result.n_steps,
                    "initial_energy": float(result.initial_energy),
                    "final_energy": float(result.final_energy),
                    "max_force": float(result.max_force),
                    "message": result.message,
                    "energy_history": [float(e) for e in result.energy_history],
                }
                xyz = self._system_to_xyz()
                self._eval_js(
                    f"onMinimizationDone({json.dumps(result_dict)}, {json.dumps(xyz)})"
                )
            except Exception as e:
                traceback.print_exc()
                self._eval_js(f"onMinimizationError({json.dumps(str(e))})")
            finally:
                self._running = False

        self._sim_thread = threading.Thread(target=minimize_thread, daemon=True)
        self._sim_thread.start()
        return json.dumps({"ok": True, "message": "Minimization started"})

    # ------------------------------------------------------------------ #
    #  Utility
    # ------------------------------------------------------------------ #

    def _system_to_xyz(self):
        """Convert system positions + atom types to XYZ string."""
        positions = self._system.state.positions
        n = len(positions)
        step = self._system.state.step
        lines = [str(n), f"Step {step}"]
        for i, pos in enumerate(positions):
            atom_type = self._system.atoms[i].atom_type
            lines.append(f"{atom_type}  {pos[0]:.6f}  {pos[1]:.6f}  {pos[2]:.6f}")
        return "\n".join(lines)

    def _eval_js(self, js_code):
        """Safely call window.evaluate_js from any thread."""
        try:
            if self._window:
                self._window.evaluate_js(js_code)
        except Exception:
            pass
