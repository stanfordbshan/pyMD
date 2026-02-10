"""
Unit tests for minimizer module.
"""
import numpy as np
import pytest

from md_simulator.core import Atom, State, System, Units
from md_simulator.boundary import PeriodicBoundaryCondition
from md_simulator.minimizer import (
    Minimizer,
    MinimizationResult,
    SteepestDescent,
    ConjugateGradient,
    LBFGS,
)


class MockHarmonicForceCalculator:
    """
    Mock force calculator with harmonic potential.

    E = 0.5 * k * sum((x - x_eq)^2)
    F = -k * (x - x_eq)

    Analytically solvable: minimum at x_eq.
    """

    def __init__(
        self,
        k: float = 1.0,
        x_eq: np.ndarray | None = None,
    ):
        self.k = k
        self.x_eq = x_eq  # Equilibrium positions (N, 3)

    def compute_forces(self, system: System) -> np.ndarray:
        x_eq = self._get_x_eq(system)
        return -self.k * (system.state.positions - x_eq)

    def compute_forces_and_energy(self, system: System) -> tuple:
        x_eq = self._get_x_eq(system)
        diff = system.state.positions - x_eq
        energy = 0.5 * self.k * np.sum(diff**2)
        forces = -self.k * diff
        return forces, energy

    def _get_x_eq(self, system: System) -> np.ndarray:
        if self.x_eq is not None:
            return self.x_eq
        return np.zeros_like(system.state.positions)


def make_system(
    n_atoms: int = 1,
    positions: np.ndarray | None = None,
    box_size: float = 20.0,
) -> System:
    """Helper to create a simple test system."""
    atoms = [Atom(mass=1.0, atom_type="Ar") for _ in range(n_atoms)]
    if positions is None:
        positions = np.random.RandomState(42).rand(n_atoms, 3) * 2.0
    velocities = np.ones((n_atoms, 3))  # Non-zero to verify zeroing
    state = State(
        positions=positions,
        velocities=velocities,
        forces=np.zeros((n_atoms, 3)),
        box=np.array([box_size, box_size, box_size]),
    )
    bc = PeriodicBoundaryCondition()
    units = Units.LJ()
    return System(atoms, state, bc, units)


# =============================================================================
# Steepest Descent Tests
# =============================================================================


class TestSteepestDescent:
    """Tests for Steepest Descent minimizer."""

    def test_converges_to_minimum(self) -> None:
        """SD should converge to the harmonic minimum."""
        system = make_system(n_atoms=1, positions=np.array([[1.0, 2.0, 3.0]]))
        fc = MockHarmonicForceCalculator(k=1.0)
        minimizer = SteepestDescent(force_tol=1e-6, max_steps=5000)

        result = minimizer.minimize(system, fc)

        assert result.converged
        np.testing.assert_array_almost_equal(
            system.state.positions, np.zeros((1, 3)), decimal=4
        )

    def test_energy_decreases(self) -> None:
        """Energy should monotonically decrease (or stay flat)."""
        system = make_system(n_atoms=1, positions=np.array([[3.0, 0.0, 0.0]]))
        fc = MockHarmonicForceCalculator(k=1.0)
        minimizer = SteepestDescent(force_tol=1e-6, max_steps=1000)

        result = minimizer.minimize(system, fc)

        for i in range(1, len(result.energy_history)):
            assert result.energy_history[i] <= result.energy_history[i - 1] + 1e-12

    def test_velocities_zeroed(self) -> None:
        """Velocities should be zeroed after minimization."""
        system = make_system(n_atoms=1, positions=np.array([[1.0, 0.0, 0.0]]))
        fc = MockHarmonicForceCalculator(k=1.0)
        minimizer = SteepestDescent(force_tol=1e-4, max_steps=1000)

        minimizer.minimize(system, fc)

        np.testing.assert_array_equal(
            system.state.velocities, np.zeros((1, 3))
        )

    def test_result_fields(self) -> None:
        """MinimizationResult should have correct fields."""
        system = make_system(n_atoms=1, positions=np.array([[1.0, 0.0, 0.0]]))
        fc = MockHarmonicForceCalculator(k=1.0)
        minimizer = SteepestDescent(force_tol=1e-4, max_steps=1000)

        result = minimizer.minimize(system, fc)

        assert isinstance(result, MinimizationResult)
        assert isinstance(result.converged, bool)
        assert isinstance(result.n_steps, int)
        assert isinstance(result.initial_energy, float)
        assert isinstance(result.final_energy, float)
        assert isinstance(result.max_force, float)
        assert isinstance(result.energy_history, list)
        assert isinstance(result.message, str)
        assert result.n_steps > 0
        assert result.final_energy <= result.initial_energy
        assert len(result.energy_history) == result.n_steps + 1

    def test_already_converged(self) -> None:
        """If already at minimum, should return immediately with 0 steps."""
        system = make_system(n_atoms=1, positions=np.array([[0.0, 0.0, 0.0]]))
        fc = MockHarmonicForceCalculator(k=1.0)
        minimizer = SteepestDescent(force_tol=1e-4)

        result = minimizer.minimize(system, fc)

        assert result.converged
        assert result.n_steps == 0

    def test_max_steps_reached(self) -> None:
        """Should return converged=False when max_steps is hit."""
        system = make_system(n_atoms=1, positions=np.array([[10.0, 10.0, 10.0]]))
        fc = MockHarmonicForceCalculator(k=1.0)
        minimizer = SteepestDescent(
            force_tol=1e-12, max_steps=5, initial_step_size=0.001
        )

        result = minimizer.minimize(system, fc)

        assert not result.converged
        assert result.n_steps == 5

    def test_invalid_step_size(self) -> None:
        """Negative step size should raise ValueError."""
        with pytest.raises(ValueError):
            SteepestDescent(initial_step_size=-0.01)

    def test_invalid_max_step_size(self) -> None:
        """Negative max step size should raise ValueError."""
        with pytest.raises(ValueError):
            SteepestDescent(max_step_size=-0.1)

    def test_get_name(self) -> None:
        """get_name should contain class name."""
        minimizer = SteepestDescent()
        assert "SteepestDescent" in minimizer.get_name()

    def test_multi_atom_convergence(self) -> None:
        """SD should converge for multi-atom systems."""
        n_atoms = 5
        x_eq = np.array([[i, 0.0, 0.0] for i in range(n_atoms)], dtype=float)
        positions = x_eq + 0.5 * np.random.RandomState(123).randn(n_atoms, 3)
        system = make_system(n_atoms=n_atoms, positions=positions)
        fc = MockHarmonicForceCalculator(k=2.0, x_eq=x_eq)
        minimizer = SteepestDescent(force_tol=1e-5, max_steps=5000)

        result = minimizer.minimize(system, fc)

        assert result.converged
        np.testing.assert_array_almost_equal(
            system.state.positions, x_eq, decimal=3
        )


# =============================================================================
# Conjugate Gradient Tests
# =============================================================================


class TestConjugateGradient:
    """Tests for Conjugate Gradient minimizer."""

    def test_converges_to_minimum(self) -> None:
        """CG should converge to the harmonic minimum."""
        system = make_system(n_atoms=1, positions=np.array([[1.0, 2.0, 3.0]]))
        fc = MockHarmonicForceCalculator(k=1.0)
        minimizer = ConjugateGradient(force_tol=1e-6, max_steps=5000)

        result = minimizer.minimize(system, fc)

        assert result.converged
        np.testing.assert_array_almost_equal(
            system.state.positions, np.zeros((1, 3)), decimal=4
        )

    def test_energy_decreases(self) -> None:
        """Energy should monotonically decrease."""
        system = make_system(n_atoms=1, positions=np.array([[3.0, 0.0, 0.0]]))
        fc = MockHarmonicForceCalculator(k=1.0)
        minimizer = ConjugateGradient(force_tol=1e-6, max_steps=1000)

        result = minimizer.minimize(system, fc)

        for i in range(1, len(result.energy_history)):
            assert result.energy_history[i] <= result.energy_history[i - 1] + 1e-12

    def test_velocities_zeroed(self) -> None:
        """Velocities should be zeroed after minimization."""
        system = make_system(n_atoms=1, positions=np.array([[1.0, 0.0, 0.0]]))
        fc = MockHarmonicForceCalculator(k=1.0)
        minimizer = ConjugateGradient(force_tol=1e-4, max_steps=1000)

        minimizer.minimize(system, fc)

        np.testing.assert_array_equal(
            system.state.velocities, np.zeros((1, 3))
        )

    def test_already_converged(self) -> None:
        """If already at minimum, should return immediately."""
        system = make_system(n_atoms=1, positions=np.array([[0.0, 0.0, 0.0]]))
        fc = MockHarmonicForceCalculator(k=1.0)
        minimizer = ConjugateGradient(force_tol=1e-4)

        result = minimizer.minimize(system, fc)

        assert result.converged
        assert result.n_steps == 0

    def test_max_steps_reached(self) -> None:
        """Should return converged=False when max_steps is hit."""
        system = make_system(n_atoms=1, positions=np.array([[10.0, 10.0, 10.0]]))
        fc = MockHarmonicForceCalculator(k=1.0)
        minimizer = ConjugateGradient(
            force_tol=1e-12, max_steps=5, initial_step_size=0.001
        )

        result = minimizer.minimize(system, fc)

        assert not result.converged
        assert result.n_steps == 5

    def test_invalid_step_size(self) -> None:
        """Negative step size should raise ValueError."""
        with pytest.raises(ValueError):
            ConjugateGradient(initial_step_size=-0.01)

    def test_invalid_restart_interval(self) -> None:
        """Negative restart interval should raise ValueError."""
        with pytest.raises(ValueError):
            ConjugateGradient(restart_interval=-1)

    def test_get_name(self) -> None:
        """get_name should contain class name."""
        minimizer = ConjugateGradient()
        assert "ConjugateGradient" in minimizer.get_name()

    def test_multi_atom_convergence(self) -> None:
        """CG should converge for multi-atom systems."""
        n_atoms = 5
        x_eq = np.array([[i, 0.0, 0.0] for i in range(n_atoms)], dtype=float)
        positions = x_eq + 0.5 * np.random.RandomState(123).randn(n_atoms, 3)
        system = make_system(n_atoms=n_atoms, positions=positions)
        fc = MockHarmonicForceCalculator(k=2.0, x_eq=x_eq)
        minimizer = ConjugateGradient(force_tol=1e-5, max_steps=5000)

        result = minimizer.minimize(system, fc)

        assert result.converged
        np.testing.assert_array_almost_equal(
            system.state.positions, x_eq, decimal=3
        )

    def test_fewer_steps_than_sd(self) -> None:
        """CG should converge in fewer or equal steps than SD for harmonic potential."""
        positions = np.array([[2.0, 3.0, 1.0]])
        fc = MockHarmonicForceCalculator(k=1.0)
        tol = 1e-6

        system_sd = make_system(n_atoms=1, positions=positions.copy())
        sd = SteepestDescent(force_tol=tol, max_steps=10000)
        result_sd = sd.minimize(system_sd, fc)

        system_cg = make_system(n_atoms=1, positions=positions.copy())
        cg = ConjugateGradient(force_tol=tol, max_steps=10000)
        result_cg = cg.minimize(system_cg, fc)

        assert result_cg.converged
        assert result_sd.converged
        assert result_cg.n_steps <= result_sd.n_steps


# =============================================================================
# L-BFGS Tests
# =============================================================================


class TestLBFGS:
    """Tests for L-BFGS minimizer."""

    def test_converges_to_minimum(self) -> None:
        """L-BFGS should converge to the harmonic minimum."""
        system = make_system(n_atoms=1, positions=np.array([[1.0, 2.0, 3.0]]))
        fc = MockHarmonicForceCalculator(k=1.0)
        minimizer = LBFGS(force_tol=1e-6, max_steps=5000)

        result = minimizer.minimize(system, fc)

        assert result.converged
        np.testing.assert_array_almost_equal(
            system.state.positions, np.zeros((1, 3)), decimal=4
        )

    def test_energy_decreases(self) -> None:
        """Energy should monotonically decrease."""
        system = make_system(n_atoms=1, positions=np.array([[3.0, 0.0, 0.0]]))
        fc = MockHarmonicForceCalculator(k=1.0)
        minimizer = LBFGS(force_tol=1e-6, max_steps=1000)

        result = minimizer.minimize(system, fc)

        for i in range(1, len(result.energy_history)):
            assert result.energy_history[i] <= result.energy_history[i - 1] + 1e-12

    def test_velocities_zeroed(self) -> None:
        """Velocities should be zeroed after minimization."""
        system = make_system(n_atoms=1, positions=np.array([[1.0, 0.0, 0.0]]))
        fc = MockHarmonicForceCalculator(k=1.0)
        minimizer = LBFGS(force_tol=1e-4, max_steps=1000)

        minimizer.minimize(system, fc)

        np.testing.assert_array_equal(
            system.state.velocities, np.zeros((1, 3))
        )

    def test_already_converged(self) -> None:
        """If already at minimum, should return immediately."""
        system = make_system(n_atoms=1, positions=np.array([[0.0, 0.0, 0.0]]))
        fc = MockHarmonicForceCalculator(k=1.0)
        minimizer = LBFGS(force_tol=1e-4)

        result = minimizer.minimize(system, fc)

        assert result.converged
        assert result.n_steps == 0

    def test_max_steps_reached(self) -> None:
        """Should return converged=False when max_steps is hit."""
        system = make_system(n_atoms=1, positions=np.array([[10.0, 10.0, 10.0]]))
        fc = MockHarmonicForceCalculator(k=1.0)
        minimizer = LBFGS(force_tol=1e-12, max_steps=5, initial_step_size=0.001)

        result = minimizer.minimize(system, fc)

        assert not result.converged
        assert result.n_steps == 5

    def test_invalid_memory(self) -> None:
        """Non-positive memory should raise ValueError."""
        with pytest.raises(ValueError):
            LBFGS(memory=0)

    def test_invalid_step_size(self) -> None:
        """Negative step size should raise ValueError."""
        with pytest.raises(ValueError):
            LBFGS(initial_step_size=-0.01)

    def test_get_name(self) -> None:
        """get_name should contain class name."""
        minimizer = LBFGS()
        assert "LBFGS" in minimizer.get_name()

    def test_multi_atom_convergence(self) -> None:
        """L-BFGS should converge for multi-atom systems."""
        n_atoms = 5
        x_eq = np.array([[i, 0.0, 0.0] for i in range(n_atoms)], dtype=float)
        positions = x_eq + 0.5 * np.random.RandomState(123).randn(n_atoms, 3)
        system = make_system(n_atoms=n_atoms, positions=positions)
        fc = MockHarmonicForceCalculator(k=2.0, x_eq=x_eq)
        minimizer = LBFGS(force_tol=1e-5, max_steps=5000)

        result = minimizer.minimize(system, fc)

        assert result.converged
        np.testing.assert_array_almost_equal(
            system.state.positions, x_eq, decimal=3
        )


# =============================================================================
# Interface / Cross-Algorithm Tests
# =============================================================================


class TestMinimizerInterface:
    """Test minimizer interface compliance across all algorithms."""

    @pytest.fixture(params=[SteepestDescent, ConjugateGradient, LBFGS])
    def minimizer(self, request) -> Minimizer:
        """Create each minimizer type."""
        return request.param(force_tol=1e-4)

    def test_is_minimizer(self, minimizer: Minimizer) -> None:
        """All minimizers should be instances of Minimizer."""
        assert isinstance(minimizer, Minimizer)

    def test_has_minimize_method(self, minimizer: Minimizer) -> None:
        """All minimizers should have a minimize method."""
        assert hasattr(minimizer, "minimize")
        assert callable(minimizer.minimize)

    def test_has_get_name_method(self, minimizer: Minimizer) -> None:
        """All minimizers should have a get_name method."""
        assert hasattr(minimizer, "get_name")
        assert callable(minimizer.get_name)

    def test_returns_minimization_result(self, minimizer: Minimizer) -> None:
        """All minimizers should return MinimizationResult."""
        system = make_system(n_atoms=1, positions=np.array([[1.0, 0.0, 0.0]]))
        fc = MockHarmonicForceCalculator(k=1.0)

        result = minimizer.minimize(system, fc)

        assert isinstance(result, MinimizationResult)

    def test_invalid_force_tol(self) -> None:
        """Non-positive force_tol should raise ValueError."""
        with pytest.raises(ValueError):
            SteepestDescent(force_tol=-1e-4)

    def test_invalid_energy_tol(self) -> None:
        """Non-positive energy_tol should raise ValueError."""
        with pytest.raises(ValueError):
            SteepestDescent(energy_tol=0.0)

    def test_invalid_max_steps(self) -> None:
        """Non-positive max_steps should raise ValueError."""
        with pytest.raises(ValueError):
            SteepestDescent(max_steps=0)
