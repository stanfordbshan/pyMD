"""
Configuration loader for YAML-based simulation setup.

Provides functions to load simulation configuration from YAML files.
"""
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from pyMD.core import Units, UnitSystem
from pyMD.boundary import (
    BoundaryCondition,
    MixedBoundaryCondition,
    OpenBoundaryCondition,
    PeriodicBoundaryCondition,
)
from pyMD.force import BackendFactory, ForceCalculator
from pyMD.integrator import VelocityVerlet
from pyMD.potential import LennardJonesPotential, MorsePotential
from pyMD.thermostat import (
    BerendsenThermostat,
    NoThermostat,
    NoseHooverThermostat,
)
from pyMD.observer import EnergyObserver, PrintObserver, TrajectoryObserver
from pyMD.simulator import Simulator

from .system_builder import SystemBuilder


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        path: Path to YAML file.

    Returns:
        Dictionary with configuration.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required: pip install pyyaml")

    with open(path, "r") as f:
        return yaml.safe_load(f)


def _parse_units(config: Dict[str, Any]) -> UnitSystem:
    """Parse unit system from config."""
    units_name = config.get("units", "LJ").upper()
    units_map = {
        "LJ": Units.LJ,
        "REAL": Units.REAL,
        "METAL": Units.METAL,
        "SI": Units.SI,
    }
    if units_name not in units_map:
        raise ValueError(f"Unknown units: {units_name}")
    return units_map[units_name]()


def _parse_boundary(config: Dict[str, Any]) -> BoundaryCondition:
    """Parse boundary condition from config."""
    bc_config = config.get("boundary", {"type": "periodic"})
    bc_type = bc_config.get("type", "periodic").lower()

    if bc_type == "periodic":
        return PeriodicBoundaryCondition()
    elif bc_type == "open":
        return OpenBoundaryCondition()
    elif bc_type == "mixed":
        periodic_dims = tuple(bc_config.get("periodic_dims", [True, True, True]))
        return MixedBoundaryCondition(periodic_dims=periodic_dims)
    else:
        raise ValueError(f"Unknown boundary type: {bc_type}")


def _parse_potential(config: Dict[str, Any]):
    """Parse potential from config."""
    pot_config = config.get("potential", {})
    pot_type = pot_config.get("type", "lj").lower()

    if pot_type == "lj" or pot_type == "lennard_jones":
        return LennardJonesPotential(
            epsilon=pot_config.get("epsilon", 1.0),
            sigma=pot_config.get("sigma", 1.0),
            cutoff=pot_config.get("cutoff", 2.5),
        )
    elif pot_type == "morse":
        return MorsePotential(
            D=pot_config.get("D", 1.0),
            a=pot_config.get("a", 1.0),
            r0=pot_config.get("r0", 1.0),
            cutoff=pot_config.get("cutoff", 5.0),
        )
    else:
        raise ValueError(f"Unknown potential type: {pot_type}")


def _parse_thermostat(config: Dict[str, Any], dt: float):
    """Parse thermostat from config."""
    thermo_config = config.get("thermostat", {"type": "nve"})
    thermo_type = thermo_config.get("type", "nve").lower()

    if thermo_type == "nve" or thermo_type == "none":
        return NoThermostat()
    elif thermo_type == "berendsen":
        return BerendsenThermostat(
            target_temperature=thermo_config.get("temperature", 1.0),
            tau=thermo_config.get("tau", 100 * dt),
        )
    elif thermo_type == "nose_hoover" or thermo_type == "nvt":
        return NoseHooverThermostat(
            target_temperature=thermo_config.get("temperature", 1.0),
            tau=thermo_config.get("tau", 20 * dt),
        )
    else:
        raise ValueError(f"Unknown thermostat type: {thermo_type}")


def build_simulation_from_config(
    config: Dict[str, Any],
) -> Simulator:
    """
    Build a complete Simulator from configuration dictionary.

    Args:
        config: Configuration dictionary (typically from YAML).

    Returns:
        Configured Simulator ready to run.

    Example config:
        units: LJ
        system:
          element: Ar
          mass: 1.0
          lattice:
            type: fcc
            nx: 4
            ny: 4
            nz: 4
            a: 1.0
          temperature: 0.8
        potential:
          type: lj
          epsilon: 1.0
          sigma: 1.0
          cutoff: 2.5
        integrator:
          dt: 0.005
        thermostat:
          type: berendsen
          temperature: 0.8
          tau: 0.5
        run:
          steps: 10000
    """
    # Parse units
    units = _parse_units(config)

    # Parse boundary
    bc = _parse_boundary(config)

    # Build system
    sys_config = config.get("system", {})
    builder = SystemBuilder()
    builder.units(units)
    builder.boundary_condition(bc)

    # Element
    builder.element(
        atom_type=sys_config.get("element", "X"),
        mass=sys_config.get("mass", 1.0),
    )

    # Lattice or positions
    lattice_config = sys_config.get("lattice", {})
    lattice_type = lattice_config.get("type", "fcc").lower()

    if lattice_type == "fcc":
        builder.fcc_lattice(
            nx=lattice_config.get("nx", 4),
            ny=lattice_config.get("ny", 4),
            nz=lattice_config.get("nz", 4),
            a=lattice_config.get("a", 1.0),
        )
    elif lattice_type == "bcc":
        builder.bcc_lattice(
            nx=lattice_config.get("nx", 4),
            ny=lattice_config.get("ny", 4),
            nz=lattice_config.get("nz", 4),
            a=lattice_config.get("a", 1.0),
        )
    elif lattice_type == "sc":
        builder.sc_lattice(
            nx=lattice_config.get("nx", 4),
            ny=lattice_config.get("ny", 4),
            nz=lattice_config.get("nz", 4),
            a=lattice_config.get("a", 1.0),
        )
    elif lattice_type == "random":
        builder.box(
            lattice_config.get("lx", 10.0),
            lattice_config.get("ly", 10.0),
            lattice_config.get("lz", 10.0),
        )
        builder.random_positions(
            n_atoms=lattice_config.get("n_atoms", 100),
        )
    elif lattice_type == "positions":
        coords = lattice_config.get("coordinates", [])
        builder.box(
            lattice_config.get("lx", 100.0),
            lattice_config.get("ly", 100.0),
            lattice_config.get("lz", 100.0),
        )
        builder.positions(np.array(coords, dtype=float))

    # Temperature for velocity initialization
    if "temperature" in sys_config:
        builder.temperature(sys_config["temperature"])

    system = builder.build()

    # Parse potential
    potential = _parse_potential(config)

    # Create force calculator
    backend_name = config.get("backend", "numerical")
    backend = BackendFactory.create(backend_name)
    force_calc = ForceCalculator(potential=potential, backend=backend)

    # Parse integrator
    int_config = config.get("integrator", {})
    dt = int_config.get("dt", 0.005)
    integrator = VelocityVerlet(dt=dt)

    # Parse thermostat
    thermostat = _parse_thermostat(config, dt)

    # Create observers
    observers = []
    obs_config = config.get("observers", {})
    if obs_config.get("energy", True):
        observers.append(EnergyObserver(interval=obs_config.get("energy_interval", 100)))
    if obs_config.get("print", True):
        observers.append(PrintObserver(interval=obs_config.get("print_interval", 1000)))
    if obs_config.get("trajectory", False):
        observers.append(
            TrajectoryObserver(interval=obs_config.get("trajectory_interval", 100))
        )

    # Build simulator
    return Simulator(
        system=system,
        integrator=integrator,
        force_calculator=force_calc,
        thermostat=thermostat,
        observers=observers,
    )


def load_and_run(path: Union[str, Path]) -> Simulator:
    """
    Load configuration from YAML and run simulation.

    Args:
        path: Path to YAML configuration file.

    Returns:
        Simulator after run completes.
    """
    config = load_yaml(path)
    sim = build_simulation_from_config(config)

    run_config = config.get("run", {})
    num_steps = run_config.get("steps", 1000)

    sim.run(num_steps=num_steps)
    return sim
