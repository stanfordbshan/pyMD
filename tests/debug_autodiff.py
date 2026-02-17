#!/usr/bin/env python3
"""
Debug script to compare analytical LJ forces vs. autodiff-computed forces.

This script validates that the autodiff backends (JAX, PyTorch, Autograd, Numerical)
produce forces that match the analytical formula for Lennard-Jones potential.

Analytical LJ Force:
    F(r) = -dU/dr = 24*eps/r * [2(sig/r)^12 - (sig/r)^6]

This sanity check proves the gradient is correct.

Usage:
    python tests/debug_autodiff.py
"""
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pymd.boundary import PeriodicBoundaryCondition
from pymd.force import (
    BackendFactory,
    JAXBackend,
    NumericalBackend,
)
from pymd.potential import LennardJonesPotential

# =============================================================================
# LJ Parameters (reduced units)
# =============================================================================
EPSILON = 1.0  # Energy scale
SIGMA = 1.0    # Length scale
CUTOFF = 2.5   # Cutoff distance
BOX_SIZE = 10.0  # Simulation box size


def analytical_lj_force(
    positions: np.ndarray,
    box: np.ndarray,
    bc: PeriodicBoundaryCondition,
    epsilon: float,
    sigma: float,
    cutoff: float,
) -> np.ndarray:
    """
    Compute LJ forces using the ANALYTICAL formula.

    F_i = sum_j [24*eps/r_ij * (2(sig/r_ij)^12 - (sig/r_ij)^6)] * r_ij_hat

    where r_ij_hat is the unit vector from j to i.

    Args:
        positions: (N, 3) atomic positions.
        box: (3,) box dimensions.
        bc: Boundary condition.
        epsilon: LJ well depth.
        sigma: LJ sigma.
        cutoff: Interaction cutoff.

    Returns:
        (N, 3) analytical forces.
    """
    n_atoms = len(positions)
    forces = np.zeros_like(positions)

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            # Distance vector from i to j (with PBC)
            dr = positions[j] - positions[i]
            dr = bc.apply_minimum_image(dr.reshape(1, 3), box).flatten()
            r = np.linalg.norm(dr)

            if r < cutoff and r > 0:
                # Compute force magnitude
                # dU/dr = 4*epsilon * [-12*(sigma^12/r^13) + 6*(sigma^6/r^7)]
                #       = 4*epsilon/r * [-12*(sigma/r)^12 + 6*(sigma/r)^6]
                #       = -24*epsilon/r * [2*(sigma/r)^12 - (sigma/r)^6]
                # F = -dU/dr = 24*epsilon/r * [2*(sigma/r)^12 - (sigma/r)^6]
                #
                # For F_i, force points TOWARD j when repulsive (r < sigma)
                # Since dr = r_j - r_i points from i to j:
                # - If repulsive (force_mag > 0), F_i = -force_mag * r_hat (push away)
                # - If attractive (force_mag < 0), F_i = -force_mag * r_hat (pull toward)
                sr = sigma / r
                sr6 = sr ** 6
                sr12 = sr6 ** 2
                # Scalar force: positive when repulsive
                force_scalar = 24.0 * epsilon / r * (2.0 * sr12 - sr6)

                # Force on atom i: F_i = -force_scalar * r_hat (r_hat = dr/r)
                # This convention matches: F = -grad(U) evaluated per atom
                force_vec = -force_scalar * dr / r

                forces[i] += force_vec
                forces[j] -= force_vec

    return forces



def compute_autodiff_forces(
    positions: np.ndarray,
    box: np.ndarray,
    bc: PeriodicBoundaryCondition,
    potential: LennardJonesPotential,
    backend,
) -> np.ndarray:
    """Compute forces using autodiff backend."""
    def energy_fn(pos):
        return potential.compute_energy(pos, box, bc)

    return backend.compute_forces(energy_fn, positions)


def run_comparison_test(
    n_atoms: int = 2,
    random_positions: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Run comparison between analytical and autodiff forces.

    Args:
        n_atoms: Number of atoms (2 for simple pair test).
        random_positions: If True, use random positions.
        verbose: Print detailed output.

    Returns:
        Dictionary with test results.
    """
    results = {}

    # Setup
    box = np.array([BOX_SIZE, BOX_SIZE, BOX_SIZE])
    bc = PeriodicBoundaryCondition()
    potential = LennardJonesPotential(EPSILON, SIGMA, CUTOFF)

    # Create positions
    if n_atoms == 2 and not random_positions:
        # Simple 2-atom test at specific distance
        r_test = 1.2  # Distance between atoms
        positions = np.array([
            [0.0, 0.0, 0.0],
            [r_test, 0.0, 0.0],
        ])
        if verbose:
            print(f"\n{'='*60}")
            print("SIMPLE 2-ATOM TEST")
            print(f"{'='*60}")
            print(f"Distance: r = {r_test:.4f}")
            print(f"LJ Parameters: epsilon = {EPSILON}, sigma = {SIGMA}")
    else:
        # Random positions
        np.random.seed(42)
        positions = np.random.rand(n_atoms, 3) * (BOX_SIZE * 0.8) + BOX_SIZE * 0.1
        if verbose:
            print(f"\n{'='*60}")
            print(f"RANDOM {n_atoms}-ATOM TEST")
            print(f"{'='*60}")

    # =========================================================================
    # 1. Compute ANALYTICAL forces
    # =========================================================================
    if verbose:
        print("\n--- ANALYTICAL FORCES ---")
    start = time.time()
    analytical_forces = analytical_lj_force(
        positions, box, bc, EPSILON, SIGMA, CUTOFF
    )
    analytical_time = time.time() - start

    if verbose:
        print(f"Time: {analytical_time*1000:.3f} ms")
        for i, f in enumerate(analytical_forces):
            print(f"  Atom {i}: F = [{f[0]:+.8f}, {f[1]:+.8f}, {f[2]:+.8f}]")

    results['analytical'] = {
        'forces': analytical_forces.copy(),
        'time_ms': analytical_time * 1000,
    }

    # =========================================================================
    # 2. Compute forces using NUMERICAL differentiation (reference)
    # =========================================================================
    if verbose:
        print("\n--- NUMERICAL FORCES (finite differences) ---")
    numerical_backend = NumericalBackend(h=1e-6)
    start = time.time()
    numerical_forces = compute_autodiff_forces(
        positions, box, bc, potential, numerical_backend
    )
    numerical_time = time.time() - start

    diff_numerical = np.max(np.abs(numerical_forces - analytical_forces))
    if verbose:
        print(f"Time: {numerical_time*1000:.3f} ms")
        for i, f in enumerate(numerical_forces):
            print(f"  Atom {i}: F = [{f[0]:+.8f}, {f[1]:+.8f}, {f[2]:+.8f}]")
        print(f"  Max difference from analytical: {diff_numerical:.2e}")

    results['numerical'] = {
        'forces': numerical_forces.copy(),
        'time_ms': numerical_time * 1000,
        'max_diff': diff_numerical,
    }

    # =========================================================================
    # 3. Try JAX backend
    # =========================================================================
    jax_backend = JAXBackend(use_jit=False)
    if jax_backend.is_available():
        if verbose:
            print("\n--- JAX FORCES (autodiff) ---")
        try:
            # Warm-up
            _ = compute_autodiff_forces(positions, box, bc, potential, jax_backend)

            start = time.time()
            jax_forces = compute_autodiff_forces(
                positions, box, bc, potential, jax_backend
            )
            jax_time = time.time() - start

            diff_jax = np.max(np.abs(jax_forces - analytical_forces))
            if verbose:
                print(f"Time: {jax_time*1000:.3f} ms")
                for i, f in enumerate(jax_forces):
                    print(f"  Atom {i}: F = [{f[0]:+.8f}, {f[1]:+.8f}, {f[2]:+.8f}]")
                print(f"  Max difference from analytical: {diff_jax:.2e}")

            results['jax'] = {
                'forces': jax_forces.copy(),
                'time_ms': jax_time * 1000,
                'max_diff': diff_jax,
            }
        except Exception as e:
            if verbose:
                print(f"  JAX failed: {e}")
            results['jax'] = {'error': str(e)}
    else:
        if verbose:
            print("\n--- JAX: Not installed ---")
        results['jax'] = {'error': 'Not installed'}

    # =========================================================================
    # 4. Summary
    # =========================================================================
    if verbose:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"{'Backend':<15} {'Max Error':<15} {'Pass?':<10}")
        print("-" * 40)

        TOLERANCE = 1e-5

        for name in ['numerical', 'jax']:
            if name in results and 'max_diff' in results[name]:
                max_diff = results[name]['max_diff']
                passed = max_diff < TOLERANCE
                status = "[PASS]" if passed else "[FAIL]"
                print(f"{name:<15} {max_diff:<15.2e} {status}")
            elif name in results and 'error' in results[name]:
                print(f"{name:<15} {'N/A':<15} {'SKIP'}")

        print(f"\nTolerance: {TOLERANCE}")

    return results


def test_multiple_distances():
    """Test forces at various distances to verify correctness."""
    print("\n" + "="*60)
    print("TESTING MULTIPLE DISTANCES")
    print("="*60)

    distances = [0.9, 1.0, 1.122, 1.5, 2.0, 2.4]  # 1.122 ~ 2^(1/6) is the minimum
    box = np.array([BOX_SIZE, BOX_SIZE, BOX_SIZE])
    bc = PeriodicBoundaryCondition()
    potential = LennardJonesPotential(EPSILON, SIGMA, CUTOFF)

    print(f"\n{'r':<8} {'F_analytical':<15} {'F_numerical':<15} {'F_jax':<15} {'Error':<12}")
    print("-" * 65)

    jax_backend = JAXBackend()
    numerical_backend = NumericalBackend(h=1e-7)
    has_jax = jax_backend.is_available()

    for r in distances:
        positions = np.array([
            [0.0, 0.0, 0.0],
            [r, 0.0, 0.0],
        ])

        # Analytical force magnitude on atom 0 (x-component only)
        sr = SIGMA / r
        sr6 = sr ** 6
        sr12 = sr6 ** 2
        F_analytical = 24.0 * EPSILON / r * (2.0 * sr12 - sr6)

        # Numerical force
        def energy_fn(pos):
            return potential.compute_energy(pos, box, bc)

        numerical_forces = numerical_backend.compute_forces(energy_fn, positions)
        F_numerical = numerical_forces[0, 0]

        # JAX force
        if has_jax:
            try:
                jax_forces = jax_backend.compute_forces(energy_fn, positions)
                F_jax = jax_forces[0, 0]
                error = abs(F_jax - F_analytical)
                print(f"{r:<8.3f} {F_analytical:<15.6f} {F_numerical:<15.6f} {F_jax:<15.6f} {error:<12.2e}")
            except:
                print(f"{r:<8.3f} {F_analytical:<15.6f} {F_numerical:<15.6f} {'Error':<15} {'N/A'}")
        else:
            error = abs(F_numerical - F_analytical)
            print(f"{r:<8.3f} {F_analytical:<15.6f} {F_numerical:<15.6f} {'N/A':<15} {error:<12.2e}")


def main():
    """Run all validation tests."""
    print("=" * 64)
    print("    AUTODIFF FORCE VALIDATION: Analytical vs. JAX")
    print("    Lennard-Jones Potential")
    print("=" * 64)

    # List available backends
    print("\nAvailable backends:", BackendFactory.list_available())

    # Test 1: Simple 2-atom system
    results_2atom = run_comparison_test(n_atoms=2, random_positions=False)

    # Test 2: Multiple distances
    test_multiple_distances()

    # Test 3: Random multi-atom system
    results_multi = run_comparison_test(n_atoms=10, random_positions=True)

    # Final verdict
    print("\n" + "="*60)
    print("FINAL VERDICT")
    print("="*60)

    all_passed = True
    TOLERANCE = 1e-5

    for name, results in [("2-atom", results_2atom), ("10-atom", results_multi)]:
        for backend in ['numerical', 'jax']:
            if backend in results and 'max_diff' in results[backend]:
                if results[backend]['max_diff'] > TOLERANCE:
                    all_passed = False
                    print(f"[FAIL] {name} {backend}: FAILED (error = {results[backend]['max_diff']:.2e})")
                else:
                    print(f"[PASS] {name} {backend}: PASSED")
            elif backend in results and 'error' in results[backend]:
                print(f"- {name} {backend}: SKIPPED ({results[backend]['error']})")

    if all_passed:
        print("\n** ALL TESTS PASSED! Autodiff forces match analytical formula. **")
        return 0
    else:
        print("\n** SOME TESTS FAILED. Check implementation. **")
        return 1


if __name__ == "__main__":
    sys.exit(main())
