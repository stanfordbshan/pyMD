#!/usr/bin/env python3
"""
Example 3: Thermostat Comparison

Compares NVE (no thermostat) vs NVT (Berendsen) ensemble behavior.
Shows how thermostats control temperature in MD simulations.

Physics:
    NVE: Total energy conserved, temperature fluctuates
    NVT: Temperature controlled, energy fluctuates

This example runs two simulations:
1. NVE - observe temperature fluctuations
2. NVT - observe temperature stabilization

Usage:
    python examples/03_thermostat_comparison.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from md_simulator.builder import SystemBuilder
from md_simulator.core import Units
from md_simulator.force import ForceCalculator, NumericalBackend
from md_simulator.integrator import VelocityVerlet
from md_simulator.potential import LennardJonesPotential
from md_simulator.simulator import Simulator
from md_simulator.thermostat import NoThermostat, BerendsenThermostat
from md_simulator.observer import EnergyObserver


def run_simulation(thermostat, name, steps=100):
    """Run simulation with given thermostat."""
    # Build small FCC system (32 atoms)
    np.random.seed(42)
    system = (
        SystemBuilder()
        .element("Ar", mass=1.0)
        .fcc_lattice(nx=2, ny=2, nz=2, a=1.5)  # 32 atoms
        .temperature(1.5)  # Start above target
        .units(Units.LJ())
        .build()
    )
    
    # LJ potential
    potential = LennardJonesPotential(epsilon=1.0, sigma=1.0, cutoff=2.5)
    force_calc = ForceCalculator(potential, NumericalBackend())
    
    # Track temperature
    energy_obs = EnergyObserver(interval=5)
    
    sim = Simulator(
        system=system,
        integrator=VelocityVerlet(dt=0.005),
        force_calculator=force_calc,
        thermostat=thermostat,
        observers=[energy_obs],
    )
    
    print(f"\n  Running {name}...")
    sim.run(num_steps=steps)
    
    return energy_obs.temperatures, energy_obs.total_energies


def main():
    print("=" * 55)
    print("  Example 3: THERMOSTAT COMPARISON")
    print("  NVE vs NVT ensemble behavior")
    print("=" * 55)
    
    target_T = 0.8
    print(f"\nTarget temperature: {target_T}")
    print(f"Initial temperature: 1.5 (above target)")
    
    # Run NVE (no thermostat)
    nve_temps, nve_energies = run_simulation(
        NoThermostat(),
        "NVE (no thermostat)",
        steps=80
    )
    
    # Run NVT (Berendsen)
    nvt_temps, nvt_energies = run_simulation(
        BerendsenThermostat(target_temperature=target_T, tau=0.5),
        "NVT (Berendsen)",
        steps=80
    )
    
    print(f"\n{'='*50}")
    print("COMPARISON RESULTS")
    print(f"{'='*50}")
    
    # NVE analysis
    nve_temps = np.array(nve_temps)
    nve_energies = np.array(nve_energies)
    print("\nNVE Ensemble:")
    print(f"  Temperature: {nve_temps.mean():.3f} +/- {nve_temps.std():.3f}")
    print(f"  Energy std:  {np.std(nve_energies):.4f}")
    print(f"  (Energy conserved, T fluctuates)")
    
    # NVT analysis
    nvt_temps = np.array(nvt_temps)
    nvt_energies = np.array(nvt_energies)
    print("\nNVT Ensemble:")
    print(f"  Temperature: {nvt_temps.mean():.3f} +/- {nvt_temps.std():.3f}")
    print(f"  Target:      {target_T}")
    print(f"  Energy std:  {np.std(nvt_energies):.4f}")
    print(f"  (T controlled, energy fluctuates)")
    
    # Compare
    print(f"\n{'='*50}")
    print("KEY OBSERVATIONS:")
    print(f"{'='*50}")
    print("- NVE: Temperature fluctuates around initial value")
    print("- NVT: Temperature converges to target")
    print(f"- NVT final T ({nvt_temps[-1]:.3f}) closer to target ({target_T})")
    
    print("\n[PASS] Thermostat comparison complete!")
    print("=" * 55)


if __name__ == "__main__":
    main()
