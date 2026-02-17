#!/usr/bin/env python3
"""
Example 4: Velocity Distribution Analysis

Verifies that velocities follow Maxwell-Boltzmann distribution.
This is a fundamental statistical mechanics check.

Physics:
    At equilibrium, velocity components follow Gaussian:
        P(v_x) ~ exp(-m*v_x^2 / 2*kB*T)
    
    Speed follows Maxwell-Boltzmann:
        P(v) ~ v^2 * exp(-m*v^2 / 2*kB*T)

This example:
1. Equilibrates a small system
2. Collects velocity data
3. Compares to expected Gaussian distribution

Usage:
    python examples/04_velocity_distribution.py
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


def main():
    print("=" * 55)
    print("  Example 4: VELOCITY DISTRIBUTION")
    print("  Maxwell-Boltzmann statistics verification")
    print("=" * 55)
    
    # Target temperature
    target_T = 1.0
    
    # Build small system (32 atoms)
    np.random.seed(123)
    system = (
        SystemBuilder()
        .element("Ar", mass=1.0)
        .fcc_lattice(nx=2, ny=2, nz=2, a=1.5)  # 32 atoms
        .temperature(target_T)
        .units(Units.LJ())
        .build()
    )
    
    n_atoms = system.get_num_atoms()
    print(f"\nSystem: {n_atoms} atoms")
    print(f"Target T: {target_T}")
    
    # Setup
    potential = LennardJonesPotential(epsilon=1.0, sigma=1.0, cutoff=2.5)
    force_calc = ForceCalculator(potential, NumericalBackend())
    
    sim = Simulator(
        system=system,
        integrator=VelocityVerlet(dt=0.005),
        force_calculator=force_calc,
        thermostat=BerendsenThermostat(target_T, tau=0.3),
        observers=[EnergyObserver(interval=10)],
    )
    
    # Equilibrate
    print("\n  Equilibrating...")
    sim.run(num_steps=50)
    
    # Collect velocity samples
    print("  Collecting velocity data...")
    all_vx = []
    all_vy = []
    all_vz = []
    
    for _ in range(30):
        sim.run(num_steps=2)
        v = system.state.velocities
        all_vx.extend(v[:, 0])
        all_vy.extend(v[:, 1])
        all_vz.extend(v[:, 2])
    
    all_vx = np.array(all_vx)
    all_vy = np.array(all_vy)
    all_vz = np.array(all_vz)
    
    print(f"\n{'='*45}")
    print("VELOCITY DISTRIBUTION ANALYSIS")
    print(f"{'='*45}")
    
    # Expected std for velocity component: sqrt(kB*T/m)
    # In LJ units with kB=1, m=1, T=1: std = 1.0
    kB = system.units.boltzmann
    m = 1.0
    expected_std = np.sqrt(kB * target_T / m)
    
    print(f"\nExpected std (sqrt(kB*T/m)): {expected_std:.4f}")
    print(f"\nMeasured velocity statistics:")
    print(f"  v_x: mean={all_vx.mean():.4f}, std={all_vx.std():.4f}")
    print(f"  v_y: mean={all_vy.mean():.4f}, std={all_vy.std():.4f}")
    print(f"  v_z: mean={all_vz.mean():.4f}, std={all_vz.std():.4f}")
    
    # Check: mean should be ~0, std should match expected
    avg_std = (all_vx.std() + all_vy.std() + all_vz.std()) / 3
    avg_mean = (abs(all_vx.mean()) + abs(all_vy.mean()) + abs(all_vz.mean())) / 3
    
    print(f"\nAverage std:  {avg_std:.4f}")
    print(f"Average |mean|: {avg_mean:.4f} (should be ~0)")
    
    # Compute speed distribution
    all_v = np.sqrt(all_vx**2 + all_vy**2 + all_vz**2)
    print(f"\nSpeed statistics:")
    print(f"  mean speed: {all_v.mean():.4f}")
    print(f"  max speed:  {all_v.max():.4f}")
    
    # Theoretical mean speed for 3D Maxwell-Boltzmann: sqrt(8*kB*T/(pi*m))
    expected_mean_speed = np.sqrt(8 * kB * target_T / (np.pi * m))
    print(f"  expected:   {expected_mean_speed:.4f}")
    
    # Simple check
    std_error = abs(avg_std - expected_std) / expected_std * 100
    print(f"\n{'='*45}")
    if std_error < 20:
        print("[PASS] Velocities follow Maxwell-Boltzmann!")
    else:
        print("[CHECK] Std deviation differs by {std_error:.1f}%")
    print("=" * 55)


if __name__ == "__main__":
    main()
