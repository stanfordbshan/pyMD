#!/usr/bin/env python3
"""
Example: LJ Argon Simulation

Simulates liquid argon using Lennard-Jones potential in reduced units.
Demonstrates basic usage of the pyMD framework.

Usage:
    python examples/run_lj_argon.py
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from pyMD.builder import SystemBuilder
from pyMD.boundary import PeriodicBoundaryCondition
from pyMD.core import Units
from pyMD.force import ForceCalculator, NumericalBackend
from pyMD.integrator import VelocityVerlet
from pyMD.observer import EnergyObserver, PrintObserver
from pyMD.potential import LennardJonesPotential
from pyMD.simulator import Simulator
from pyMD.thermostat import BerendsenThermostat, NoThermostat


def main():
    """Run LJ Argon simulation."""
    print("=" * 60)
    print("  LJ ARGON SIMULATION")
    print("  pyMD Molecular Dynamics Framework")
    print("=" * 60)

    # =========================================================================
    # 1. Build the system using SystemBuilder
    # =========================================================================
    print("\n[1] Building FCC lattice...")

    system = (
        SystemBuilder()
        .element("Ar", mass=1.0)  # LJ reduced units
        .fcc_lattice(nx=4, ny=4, nz=4, a=1.5)  # 256 atoms
        .temperature(0.8)  # T* = 0.8
        .units(Units.LJ())
        .periodic_boundary()
        .build()
    )

    print(f"    Created {system.get_num_atoms()} atoms")
    print(f"    Box: {system.get_box()}")
    print(f"    Initial T: {system.compute_temperature():.3f}")

    # =========================================================================
    # 2. Set up potential and force calculator
    # =========================================================================
    print("\n[2] Setting up LJ potential...")

    potential = LennardJonesPotential(
        epsilon=1.0,  # Energy unit
        sigma=1.0,    # Length unit
        cutoff=2.5,   # 2.5 sigma
    )

    backend = NumericalBackend()
    print(f"    Using backend: {backend.get_name()}")

    force_calc = ForceCalculator(potential=potential, backend=backend)

    # =========================================================================
    # 3. Configure integrator and thermostat
    # =========================================================================
    print("\n[3] Configuring dynamics...")

    dt = 0.005  # Reduced time units
    integrator = VelocityVerlet(dt=dt)

    # Use Berendsen for equilibration
    thermostat = BerendsenThermostat(
        target_temperature=0.8,
        tau=0.5,  # 100 * dt
    )

    print(f"    Integrator: {integrator.get_name()}")
    print(f"    Thermostat: {thermostat.get_name()}")

    # =========================================================================
    # 4. Set up observers
    # =========================================================================
    energy_obs = EnergyObserver(interval=100)
    print_obs = PrintObserver(interval=500)

    # =========================================================================
    # 5. Create and run simulator
    # =========================================================================
    print("\n[4] Running simulation (5000 steps)...")
    print("-" * 60)

    sim = Simulator(
        system=system,
        integrator=integrator,
        force_calculator=force_calc,
        thermostat=thermostat,
        observers=[energy_obs, print_obs],
    )

    sim.run(num_steps=5000)

    # =========================================================================
    # 6. Analyze results
    # =========================================================================
    print("-" * 60)
    print("\n[5] Analysis:")

    # Energy conservation
    drift = energy_obs.get_energy_drift()
    print(f"    Energy drift: {drift:.2e}")

    # Final temperature
    final_T = system.compute_temperature()
    print(f"    Final temperature: {final_T:.4f}")

    # Average temperature over last 1000 steps
    temps = np.array(energy_obs.temperatures[-10:])
    avg_T = np.mean(temps)
    std_T = np.std(temps)
    print(f"    Average T (last 1000): {avg_T:.4f} +/- {std_T:.4f}")

    # Average energies
    PE = np.mean(energy_obs.potential_energies[-10:])
    KE = np.mean(energy_obs.kinetic_energies[-10:])
    print(f"    <PE>: {PE:.4f}")
    print(f"    <KE>: {KE:.4f}")
    print(f"    <E_total>: {PE + KE:.4f}")

    print("\n" + "=" * 60)
    print("  Simulation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
