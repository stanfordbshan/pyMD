#!/usr/bin/env python3
"""
Example 6: Equipartition Theorem

Demonstrates that kinetic energy is equally distributed among
degrees of freedom, a fundamental result of statistical mechanics.

Physics:
    Equipartition theorem states:
        <KE per DOF> = (1/2) * kB * T
    
    For N atoms in 3D:
        <KE_total> = (3/2) * N * kB * T
        
    Or per dimension:
        <KE_x> = <KE_y> = <KE_z> = (1/2) * N * kB * T

Usage:
    python examples/06_equipartition.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from pyMD.builder import SystemBuilder
from pyMD.core import Units
from pyMD.force import ForceCalculator, NumericalBackend
from pyMD.integrator import VelocityVerlet
from pyMD.potential import LennardJonesPotential
from pyMD.simulator import Simulator
from pyMD.thermostat import BerendsenThermostat


def compute_ke_components(system):
    """Compute kinetic energy for each dimension."""
    masses = system.get_masses()
    velocities = system.state.velocities
    
    ke_x = 0.5 * np.sum(masses * velocities[:, 0]**2)
    ke_y = 0.5 * np.sum(masses * velocities[:, 1]**2)
    ke_z = 0.5 * np.sum(masses * velocities[:, 2]**2)
    
    return ke_x, ke_y, ke_z


def main():
    print("=" * 55)
    print("  Example 6: EQUIPARTITION THEOREM")
    print("  Equal energy distribution among DOFs")
    print("=" * 55)
    
    target_T = 1.0
    
    # Build system
    np.random.seed(456)
    system = (
        SystemBuilder()
        .element("Ar", mass=1.0)
        .fcc_lattice(nx=2, ny=2, nz=2, a=1.5)  # 32 atoms
        .temperature(target_T)
        .units(Units.LJ())
        .build()
    )
    
    n_atoms = system.get_num_atoms()
    kB = system.units.boltzmann
    
    print(f"\nSystem: {n_atoms} atoms")
    print(f"Target T: {target_T}")
    print(f"kB: {kB}")
    
    # Expected KE per dimension: (1/2) * N * kB * T
    expected_ke_per_dim = 0.5 * n_atoms * kB * target_T
    print(f"\nExpected <KE> per dimension: {expected_ke_per_dim:.4f}")
    
    # Setup simulation
    potential = LennardJonesPotential(epsilon=1.0, sigma=1.0, cutoff=2.5)
    force_calc = ForceCalculator(potential, NumericalBackend())
    
    sim = Simulator(
        system=system,
        integrator=VelocityVerlet(dt=0.005),
        force_calculator=force_calc,
        thermostat=BerendsenThermostat(target_T, tau=0.3),
        observers=[],
    )
    
    # Equilibrate
    print("\n  Equilibrating...")
    sim.run(num_steps=50)
    
    # Collect KE samples
    print("  Collecting KE samples...")
    ke_x_samples = []
    ke_y_samples = []
    ke_z_samples = []
    
    for _ in range(30):
        sim.run(num_steps=2)
        ke_x, ke_y, ke_z = compute_ke_components(system)
        ke_x_samples.append(ke_x)
        ke_y_samples.append(ke_y)
        ke_z_samples.append(ke_z)
    
    ke_x_samples = np.array(ke_x_samples)
    ke_y_samples = np.array(ke_y_samples)
    ke_z_samples = np.array(ke_z_samples)
    
    print(f"\n{'='*50}")
    print("EQUIPARTITION RESULTS")
    print(f"{'='*50}")
    
    print(f"\nKinetic energy per dimension:")
    print(f"  <KE_x>: {ke_x_samples.mean():.4f} +/- {ke_x_samples.std():.4f}")
    print(f"  <KE_y>: {ke_y_samples.mean():.4f} +/- {ke_y_samples.std():.4f}")
    print(f"  <KE_z>: {ke_z_samples.mean():.4f} +/- {ke_z_samples.std():.4f}")
    print(f"  Expected: {expected_ke_per_dim:.4f}")
    
    # Total
    ke_total = ke_x_samples + ke_y_samples + ke_z_samples
    expected_total = 3 * expected_ke_per_dim
    
    print(f"\nTotal kinetic energy:")
    print(f"  <KE_total>: {ke_total.mean():.4f}")
    print(f"  Expected:   {expected_total:.4f}")
    
    # Check equipartition
    avg_ke = (ke_x_samples.mean() + ke_y_samples.mean() + ke_z_samples.mean()) / 3
    deviation = abs(ke_x_samples.mean() - ke_y_samples.mean())
    max_deviation = max(
        abs(ke_x_samples.mean() - avg_ke),
        abs(ke_y_samples.mean() - avg_ke),
        abs(ke_z_samples.mean() - avg_ke),
    )
    
    print(f"\nEquipartition check:")
    print(f"  Max deviation from average: {max_deviation:.4f}")
    print(f"  Relative: {max_deviation/avg_ke*100:.1f}%")
    
    print(f"\n{'='*50}")
    if max_deviation / avg_ke < 0.2:
        print("[PASS] Equipartition theorem verified!")
    else:
        print("[CHECK] KE not equally distributed - need more samples")
    print("=" * 55)


if __name__ == "__main__":
    main()
