#!/usr/bin/env python3
"""
Example 7: Geometry Optimization

Minimize the energy of a 5-atom Lennard-Jones cluster using three
different minimization algorithms: Steepest Descent, Conjugate Gradient,
and L-BFGS. Then independently verify the final energy by computing
the LJ pair sum directly (no pyMD code).

Physics:
    U(r) = 4ε[(σ/r)¹² - (σ/r)⁶]

    The global minimum of a 5-atom LJ cluster is a triangular bipyramid.
    Starting from a non-equilibrium configuration, all three minimizers
    should converge to the same local minimum.

Usage:
    python examples/07_geometry_optimization.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from pyMD.core import Atom, State, System, Units
from pyMD.boundary import OpenBoundaryCondition
from pyMD.force import ForceCalculator, NumericalBackend
from pyMD.potential import LennardJonesPotential
from pyMD.minimizer import SteepestDescent, ConjugateGradient, LBFGS


def compute_lj_energy_manual(positions, epsilon, sigma):
    """Direct pair summation — no pyMD code used."""
    energy = 0.0
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(positions[i] - positions[j])
            sr6 = (sigma / r) ** 6
            energy += 4.0 * epsilon * (sr6**2 - sr6)
    return energy


def make_system(initial_positions):
    """Build a fresh 5-atom LJ system (reduced units)."""
    n = len(initial_positions)
    atoms = [Atom(atom_type="Ar", mass=1.0, index=i) for i in range(n)]
    state = State(
        positions=initial_positions.copy(),
        velocities=np.zeros((n, 3)),
        forces=np.zeros((n, 3)),
        box=np.array([100.0, 100.0, 100.0]),
    )
    return System(atoms, state, OpenBoundaryCondition(), Units.LJ())


def main():
    print("=" * 60)
    print("  Example 7: GEOMETRY OPTIMIZATION")
    print("  Minimizing a 5-atom Lennard-Jones cluster")
    print("=" * 60)

    # LJ parameters (reduced units)
    epsilon = 1.0
    sigma = 1.0
    cutoff = 10.0

    # Initial positions: slightly perturbed / non-equilibrium cluster
    # r_min = 2^(1/6) * sigma ≈ 1.122, so we use spacings around 1.3-1.5
    initial_positions = np.array([
        [0.0, 0.0, 0.0],
        [1.4, 0.0, 0.0],
        [0.7, 1.3, 0.0],
        [0.7, 0.5, 1.4],
        [0.7, 0.5, -1.4],
    ])

    initial_energy = compute_lj_energy_manual(initial_positions, epsilon, sigma)
    print(f"\nInitial energy (manual):  {initial_energy:.6f}")
    print(f"Number of atoms:         {len(initial_positions)}")
    print(f"LJ parameters:           eps={epsilon}, sig={sigma}, rc={cutoff}")

    # Potential and backend (shared, stateless)
    potential = LennardJonesPotential(epsilon=epsilon, sigma=sigma, cutoff=cutoff)
    backend = NumericalBackend()
    force_calc = ForceCalculator(potential, backend)

    # --- Run all three minimizers on independent copies ---
    minimizers = [
        SteepestDescent(force_tol=1e-6, energy_tol=1e-10, max_steps=10000),
        ConjugateGradient(force_tol=1e-6, energy_tol=1e-10, max_steps=10000),
        LBFGS(force_tol=1e-6, energy_tol=1e-10, max_steps=10000),
    ]

    results = []
    systems = []
    for minimizer in minimizers:
        system = make_system(initial_positions)
        result = minimizer.minimize(system, force_calc)
        results.append(result)
        systems.append(system)

    # --- Comparison table ---
    print(f"\n{'='*60}")
    print(f"{'Method':<20} {'Steps':>6} {'Energy':>14} {'Max Force':>12} {'Conv':>6}")
    print(f"{'-'*60}")
    short_names = ["Steepest Descent", "Conjugate Gradient", "L-BFGS"]
    for name, result in zip(short_names, results):
        print(f"{name:<20} {result.n_steps:>6} {result.final_energy:>14.6f} "
              f"{result.max_force:>12.2e} {'Yes' if result.converged else 'No':>6}")
    print(f"{'='*60}")

    # --- Independent verification using L-BFGS result ---
    best_idx = 2  # L-BFGS
    best_result = results[best_idx]
    best_system = systems[best_idx]
    final_positions = best_system.state.positions

    manual_energy = compute_lj_energy_manual(final_positions, epsilon, sigma)
    pymd_energy = best_result.final_energy

    print(f"\n--- Independent Energy Verification (L-BFGS result) ---")
    print(f"pyMD energy:    {pymd_energy:.10f}")
    print(f"Manual energy:  {manual_energy:.10f}")
    print(f"Difference:     {abs(pymd_energy - manual_energy):.2e}")

    tol = 1e-6
    if abs(pymd_energy - manual_energy) < tol:
        print(f"\n[PASS] Energies agree within {tol}")
    else:
        print(f"\n[FAIL] Energies differ by more than {tol}")

    # Check that all minimizers converged
    all_converged = all(r.converged for r in results)
    if all_converged:
        print("[PASS] All three minimizers converged")
    else:
        failed = [m.get_name() for m, r in zip(minimizers, results) if not r.converged]
        print(f"[FAIL] Did not converge: {', '.join(failed)}")

    # Check that all minimizers found similar energies
    energies = [r.final_energy for r in results]
    energy_spread = max(energies) - min(energies)
    if energy_spread < 1e-4:
        print(f"[PASS] All minimizers agree on energy (spread: {energy_spread:.2e})")
    else:
        print(f"[FAIL] Energy spread too large: {energy_spread:.2e}")

    print("=" * 60)


if __name__ == "__main__":
    main()
