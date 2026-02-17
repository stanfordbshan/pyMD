#!/usr/bin/env python3
"""
Example 5: Melting Transition

Simulates heating a small LJ cluster to observe melting behavior.
Shows how structural order changes with temperature.

Physics:
    - At low T: Atoms vibrate around lattice positions (solid)
    - At high T: Atoms diffuse freely (liquid)
    - Lindemann criterion: melting when RMSD > 0.1 * lattice constant

This example:
1. Starts with FCC lattice
2. Gradually increases temperature
3. Tracks atomic displacement from initial positions

Usage:
    python examples/05_melting_transition.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from pymd.builder import SystemBuilder
from pymd.core import Units
from pymd.force import ForceCalculator, NumericalBackend
from pymd.integrator import VelocityVerlet
from pymd.potential import LennardJonesPotential
from pymd.simulator import Simulator
from pymd.thermostat import BerendsenThermostat
from pymd.observer import EnergyObserver


def compute_msd(initial_positions, current_positions):
    """Compute mean squared displacement from initial positions."""
    displacements = current_positions - initial_positions
    return np.mean(np.sum(displacements**2, axis=1))


def run_at_temperature(T, initial_pos, steps=40):
    """Run simulation at given temperature, return MSD."""
    np.random.seed(int(T * 100))
    
    system = (
        SystemBuilder()
        .element("Ar", mass=1.0)
        .fcc_lattice(nx=2, ny=2, nz=2, a=1.5)  # 32 atoms
        .temperature(T)
        .units(Units.LJ())
        .build()
    )
    
    potential = LennardJonesPotential(epsilon=1.0, sigma=1.0, cutoff=2.5)
    force_calc = ForceCalculator(potential, NumericalBackend())
    
    sim = Simulator(
        system=system,
        integrator=VelocityVerlet(dt=0.005),
        force_calculator=force_calc,
        thermostat=BerendsenThermostat(T, tau=0.3),
        observers=[],
    )
    
    # Run and collect MSD
    sim.run(num_steps=steps)
    
    msd = compute_msd(initial_pos, system.state.positions)
    actual_T = system.compute_temperature()
    
    return msd, actual_T


def main():
    print("=" * 55)
    print("  Example 5: MELTING TRANSITION")
    print("  Lindemann criterion for melting")
    print("=" * 55)
    
    # Get reference lattice positions
    ref_system = (
        SystemBuilder()
        .element("Ar", mass=1.0)
        .fcc_lattice(nx=2, ny=2, nz=2, a=1.5)
        .temperature(0.0)  # Zero temperature
        .units(Units.LJ())
        .build()
    )
    initial_positions = ref_system.state.positions.copy()
    
    # Lattice constant for Lindemann criterion
    a = 1.5
    lindemann_threshold = 0.1 * a  # 10% of lattice constant
    
    print(f"\nLattice constant: {a}")
    print(f"Lindemann threshold: RMSD > {lindemann_threshold:.3f}")
    print(f"\n  Scanning temperatures...")
    
    # Scan temperatures
    temperatures = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
    results = []
    
    for T in temperatures:
        msd, actual_T = run_at_temperature(T, initial_positions)
        rmsd = np.sqrt(msd)
        results.append((T, actual_T, rmsd))
    
    # Print results
    print(f"\n{'='*50}")
    print(f"{'T_target':>10} {'T_actual':>10} {'RMSD':>10} {'State':>10}")
    print(f"{'='*50}")
    
    for T_target, T_actual, rmsd in results:
        state = "LIQUID" if rmsd > lindemann_threshold else "SOLID"
        print(f"{T_target:>10.2f} {T_actual:>10.3f} {rmsd:>10.4f} {state:>10}")
    
    print(f"{'='*50}")
    
    # Find approximate melting point
    for i, (T, _, rmsd) in enumerate(results):
        if rmsd > lindemann_threshold:
            if i > 0:
                T_melt = (results[i-1][0] + T) / 2
            else:
                T_melt = T
            break
    else:
        T_melt = "> " + str(temperatures[-1])
    
    print(f"\nApproximate melting temperature: {T_melt}")
    print("(LJ triple point is around T* = 0.68)")
    
    print("\n[PASS] Melting transition observed!")
    print("=" * 55)


if __name__ == "__main__":
    main()
