#!/usr/bin/env python3
"""
Example 8: Five-Atom LJ Cluster Optimization

Optimizes a 5-atom cluster using the Lennard-Jones potential in reduced
units (ε=1, σ=1). Starts from a planar initial configuration and
minimizes with L-BFGS to find the optimal 3D geometry.

Usage:
    python examples/08_five_atom_optimization.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from pymd.core import Atom, State, System, Units
from pymd.boundary import OpenBoundaryCondition
from pymd.force import ForceCalculator, NumericalBackend
from pymd.potential import LennardJonesPotential
from pymd.minimizer import LBFGS


def main():
    print("=" * 60)
    print("  5-Atom LJ Cluster Optimization (Reduced Units)")
    print("=" * 60)

    # Initial positions (planar, z=0)
    initial_positions = np.array([
        [-1.208,  1.042, 0.0],
        [ 0.534,  0.995, 0.0],
        [-0.006, -1.101, 0.0],
        [-1.324, -1.647, 0.0],
        [ 1.085, -1.917, 0.0],
    ])

    n = len(initial_positions)

    # Build system
    atoms = [Atom(atom_type="Ar", mass=1.0, index=i) for i in range(n)]
    state = State(
        positions=initial_positions.copy(),
        velocities=np.zeros((n, 3)),
        forces=np.zeros((n, 3)),
        box=np.array([100.0, 100.0, 100.0]),
    )
    system = System(atoms, state, OpenBoundaryCondition(), Units.LJ())

    # LJ potential (reduced units)
    potential = LennardJonesPotential(epsilon=1.0, sigma=1.0, cutoff=10.0)
    force_calc = ForceCalculator(potential, NumericalBackend())

    # Print initial geometry
    print("\nInitial geometry:")
    print(f"  {'Atom':<6} {'x':>10} {'y':>10} {'z':>10}")
    print(f"  {'-'*38}")
    for i, pos in enumerate(initial_positions):
        print(f"  {i+1:<6} {pos[0]:>10.6f} {pos[1]:>10.6f} {pos[2]:>10.6f}")

    initial_energy = force_calc.compute_energy(system)
    print(f"\nInitial energy: {initial_energy:.6f}")

    # Minimize with L-BFGS
    minimizer = LBFGS(force_tol=1e-6, energy_tol=1e-10, max_steps=10000)
    result = minimizer.minimize(system, force_calc)

    # Print results
    print(f"\n{'='*60}")
    print(f"  Optimization Results")
    print(f"{'='*60}")
    print(f"  Converged:      {result.converged}")
    print(f"  Steps:          {result.n_steps}")
    print(f"  Max force:      {result.max_force:.2e}")

    print(f"\nOptimized geometry:")
    print(f"  {'Atom':<6} {'x':>10} {'y':>10} {'z':>10}")
    print(f"  {'-'*38}")
    final_positions = system.state.positions
    for i, pos in enumerate(final_positions):
        print(f"  {i+1:<6} {pos[0]:>10.6f} {pos[1]:>10.6f} {pos[2]:>10.6f}")

    print(f"\nOptimized energy: {result.final_energy:.6f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
