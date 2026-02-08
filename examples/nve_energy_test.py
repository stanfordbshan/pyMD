#!/usr/bin/env python3
"""
Example: Energy Conservation Test (NVE)

Demonstrates energy conservation in the microcanonical ensemble.
No thermostat is applied - total energy should be conserved.

Usage:
    python examples/nve_energy_test.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from md_simulator.builder import SystemBuilder
from md_simulator.core import Units
from md_simulator.force import ForceCalculator, NumericalBackend
from md_simulator.integrator import VelocityVerlet
from md_simulator.observer import EnergyObserver
from md_simulator.potential import LennardJonesPotential
from md_simulator.simulator import Simulator
from md_simulator.thermostat import NoThermostat


def main():
    """Test energy conservation in NVE ensemble."""
    print("=" * 60)
    print("  NVE ENERGY CONSERVATION TEST")
    print("=" * 60)

    # Build a small system
    system = (
        SystemBuilder()
        .element("Ar", mass=1.0)
        .fcc_lattice(nx=3, ny=3, nz=3, a=1.5)  # 108 atoms
        .temperature(0.5)  # Start with some kinetic energy
        .units(Units.LJ())
        .build()
    )

    print(f"\nSystem: {system.get_num_atoms()} atoms")
    print(f"Initial T: {system.compute_temperature():.4f}")

    # LJ potential
    potential = LennardJonesPotential(epsilon=1.0, sigma=1.0, cutoff=2.5)
    backend = NumericalBackend()
    force_calc = ForceCalculator(potential=potential, backend=backend)

    # NVE: No thermostat
    thermostat = NoThermostat()
    integrator = VelocityVerlet(dt=0.002)  # Smaller dt for better conservation

    # Track energy
    energy_obs = EnergyObserver(interval=50)

    sim = Simulator(
        system=system,
        integrator=integrator,
        force_calculator=force_calc,
        thermostat=thermostat,
        observers=[energy_obs],
    )

    print("\nRunning 2000 NVE steps...")
    sim.run(num_steps=2000)

    # Analyze energy conservation
    E_total = np.array(energy_obs.total_energies)
    E_mean = np.mean(E_total)
    E_std = np.std(E_total)
    drift = energy_obs.get_energy_drift()

    print(f"\n{'='*40}")
    print("RESULTS")
    print(f"{'='*40}")
    print(f"<E_total>:       {E_mean:.6f}")
    print(f"std(E_total):    {E_std:.6f}")
    print(f"Energy drift:    {drift:.2e}")
    print(f"Relative fluct:  {E_std/abs(E_mean):.2e}")

    # Check if energy is well conserved
    if abs(drift) < 0.01:
        print("\n[PASS] Energy conservation is good!")
    else:
        print("\n[WARN] Energy drift may be significant. Consider smaller dt.")

    print("=" * 60)


if __name__ == "__main__":
    main()
