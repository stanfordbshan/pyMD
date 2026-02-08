#!/usr/bin/env python3
"""
Quick Test: Small 2-atom LJ simulation

A minimal example for fast testing of the MD framework.
Uses only 2 atoms so numerical differentiation is fast.

Usage:
    python examples/quick_test.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from md_simulator.core import Atom, State, System, Units
from md_simulator.boundary import PeriodicBoundaryCondition
from md_simulator.force import ForceCalculator, NumericalBackend
from md_simulator.integrator import VelocityVerlet
from md_simulator.observer import EnergyObserver
from md_simulator.potential import LennardJonesPotential
from md_simulator.simulator import Simulator
from md_simulator.thermostat import NoThermostat


def main():
    """Run quick 2-atom test."""
    print("=" * 50)
    print("  QUICK TEST: 2-Atom LJ Simulation")
    print("=" * 50)

    # Create a simple 2-atom system
    atoms = [Atom(mass=1.0, atom_type="Ar"), Atom(mass=1.0, atom_type="Ar")]

    # Place atoms at ~equilibrium distance (r_min = 2^(1/6) * sigma ≈ 1.12)
    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.2, 0.0, 0.0],  # Slightly stretched
    ])
    velocities = np.array([
        [-0.1, 0.0, 0.0],  # Moving apart slightly
        [0.1, 0.0, 0.0],
    ])
    box = np.array([10.0, 10.0, 10.0])

    state = State(
        positions=positions,
        velocities=velocities,
        forces=np.zeros((2, 3)),
        box=box,
    )

    system = System(
        atoms=atoms,
        state=state,
        boundary_condition=PeriodicBoundaryCondition(),
        units=Units.LJ(),
    )

    print(f"\nSystem: {system.get_num_atoms()} atoms")
    print(f"Initial distance: {np.linalg.norm(positions[1] - positions[0]):.4f}")

    # LJ potential
    potential = LennardJonesPotential(epsilon=1.0, sigma=1.0, cutoff=2.5)
    backend = NumericalBackend()
    force_calc = ForceCalculator(potential=potential, backend=backend)

    # NVE dynamics
    integrator = VelocityVerlet(dt=0.005)
    thermostat = NoThermostat()
    energy_obs = EnergyObserver(interval=10)

    sim = Simulator(
        system=system,
        integrator=integrator,
        force_calculator=force_calc,
        thermostat=thermostat,
        observers=[energy_obs],
    )

    print("\nRunning 100 NVE steps...")
    sim.run(num_steps=100)

    # Results
    E_total = np.array(energy_obs.total_energies)
    E_mean = np.mean(E_total)
    E_std = np.std(E_total)
    drift = energy_obs.get_energy_drift()

    print(f"\n{'='*40}")
    print("RESULTS")
    print(f"{'='*40}")
    print(f"Steps run:       {sim.get_total_steps()}")
    print(f"<E_total>:       {E_mean:.6f}")
    print(f"std(E_total):    {E_std:.6f}")
    print(f"Energy drift:    {drift:.2e}")

    final_dist = np.linalg.norm(
        system.state.positions[1] - system.state.positions[0]
    )
    print(f"Final distance:  {final_dist:.4f}")

    if abs(drift) < 0.01:
        print("\n[PASS] Energy conservation OK!")
    else:
        print("\n[WARN] Energy drift detected")

    print("=" * 50)


if __name__ == "__main__":
    main()
