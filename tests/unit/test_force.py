"""
Unit tests for force module.

Tests for autodiff backends and force calculator.
"""
import numpy as np
import pytest

from pymd.boundary import PeriodicBoundaryCondition
from pymd.force import (
    AutoDiffBackend,
    BackendFactory,
    NumericalBackend,
)
from pymd.potential import LennardJonesPotential


class TestNumericalBackend:
    """Tests for NumericalBackend."""

    def test_is_always_available(self) -> None:
        """Numerical backend should always be available."""
        backend = NumericalBackend()
        assert backend.is_available() is True

    def test_compute_forces_simple(self) -> None:
        """Test force computation on simple quadratic energy."""
        backend = NumericalBackend(h=1e-5)

        # E = sum(x^2) -> F = -2x
        def energy_fn(positions: np.ndarray) -> float:
            return np.sum(positions ** 2)

        positions = np.array([[1.0, 2.0, 3.0]])
        forces = backend.compute_forces(energy_fn, positions)

        expected = -2 * positions
        np.testing.assert_array_almost_equal(forces, expected, decimal=4)

    def test_lj_forces_match_analytical(self) -> None:
        """Test numerical forces match LJ analytical formula."""
        backend = NumericalBackend(h=1e-7)
        lj = LennardJonesPotential(epsilon=1.0, sigma=1.0, cutoff=2.5)
        box = np.array([10.0, 10.0, 10.0])
        bc = PeriodicBoundaryCondition()

        # Two atoms at distance 1.2
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.2, 0.0, 0.0],
        ])

        def energy_fn(pos: np.ndarray) -> float:
            return lj.compute_energy(pos, box, bc)

        forces = backend.compute_forces(energy_fn, positions)

        # Analytical force on atom 0 in x-direction
        r = 1.2
        sr = 1.0 / r
        sr6 = sr ** 6
        sr12 = sr6 ** 2
        # F = -dU/dr * dr/dx, dr/dx = 1 for x-component
        # dU/dr = 24 * epsilon / r * (2*sr12 - sr6) with sign convention
        # Force on atom 0 points in -x direction (repulsive, pushing away from atom 1)
        expected_force_x = -24.0 / r * (2.0 * sr12 - sr6)

        # Numerical should match analytical  
        assert pytest.approx(forces[0, 0], rel=1e-4) == expected_force_x

    def test_get_name(self) -> None:
        """Test backend name."""
        backend = NumericalBackend(h=1e-5)
        assert "Numerical" in backend.get_name()


class TestBackendFactory:
    """Tests for BackendFactory."""

    def test_create_numerical(self) -> None:
        """Test creating numerical backend."""
        backend = BackendFactory.create("numerical", h=1e-6)
        assert isinstance(backend, NumericalBackend)

    def test_create_unknown_raises(self) -> None:
        """Test that unknown backend raises ValueError."""
        with pytest.raises(ValueError, match="Unknown backend"):
            BackendFactory.create("unknown_backend")

    def test_list_available_includes_numerical(self) -> None:
        """Numerical should always be in available list."""
        available = BackendFactory.list_available()
        assert "numerical" in available

    def test_get_best_available(self) -> None:
        """Test getting best available backend."""
        backend = BackendFactory.get_best_available()
        assert isinstance(backend, AutoDiffBackend)
        assert backend.is_available()


class TestAutoDiffBackendInterface:
    """Test backend interface compliance."""

    def test_numerical_is_autodiff_backend(self) -> None:
        """Test inheritance."""
        backend = NumericalBackend()
        assert isinstance(backend, AutoDiffBackend)

    def test_has_required_methods(self) -> None:
        """Test required methods exist."""
        backend = NumericalBackend()
        assert hasattr(backend, "compute_forces")
        assert hasattr(backend, "is_available")
        assert hasattr(backend, "get_name")
