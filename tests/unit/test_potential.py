"""
Unit tests for potential module.

Tests for LennardJones, Morse, and EAM potentials.
"""
import numpy as np
import pytest

from md_simulator.boundary import PeriodicBoundaryCondition
from md_simulator.potential import (
    CompositePotential,
    LennardJonesPotential,
    MorsePotential,
    PotentialEnergy,
    SuttonChenEAM,
)


class TestLennardJonesPotential:
    """Tests for LennardJonesPotential."""

    @pytest.fixture
    def lj(self) -> LennardJonesPotential:
        """Standard LJ potential."""
        return LennardJonesPotential(epsilon=1.0, sigma=1.0, cutoff=2.5)

    @pytest.fixture
    def box(self) -> np.ndarray:
        """Standard box."""
        return np.array([10.0, 10.0, 10.0])

    @pytest.fixture
    def bc(self) -> PeriodicBoundaryCondition:
        """Periodic BC."""
        return PeriodicBoundaryCondition()

    def test_pair_energy_at_sigma(self, lj: LennardJonesPotential) -> None:
        """At r=sigma, U(r) = 0."""
        energy = lj.compute_pair_energy(1.0)
        assert pytest.approx(energy, abs=1e-10) == 0.0

    def test_pair_energy_at_minimum(self, lj: LennardJonesPotential) -> None:
        """At r=2^(1/6)*sigma, U(r) = -epsilon."""
        r_min = 2 ** (1 / 6)  # ~1.122
        energy = lj.compute_pair_energy(r_min)
        assert pytest.approx(energy, abs=1e-10) == -1.0

    def test_pair_energy_beyond_cutoff(self, lj: LennardJonesPotential) -> None:
        """Beyond cutoff, energy is zero."""
        energy = lj.compute_pair_energy(3.0)
        assert energy == 0.0

    def test_compute_energy_two_atoms(
        self,
        lj: LennardJonesPotential,
        box: np.ndarray,
        bc: PeriodicBoundaryCondition,
    ) -> None:
        """Test energy for two atoms at sigma distance."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],  # Distance = sigma
        ])
        energy = lj.compute_energy(positions, box, bc)
        assert pytest.approx(energy, abs=1e-10) == 0.0

    def test_analytical_force(self, lj: LennardJonesPotential) -> None:
        """Test analytical force calculation."""
        # At r_min = 2^(1/6)*sigma ~= 1.122, force is zero (dU/dr = 0)
        r_min = 2 ** (1 / 6)
        force_at_min = lj.compute_pair_force(r_min)
        assert pytest.approx(force_at_min, abs=1e-6) == 0.0

        # At r < r_min (e.g., r=1.0), should be positive (repulsive)
        force = lj.compute_pair_force(1.0)
        assert force > 0

        # At r > r_min (e.g., r=1.5), should be negative (attractive)
        force = lj.compute_pair_force(1.5)
        assert force < 0

    def test_shifted_energy(self, bc: PeriodicBoundaryCondition) -> None:
        """Test energy shift at cutoff."""
        lj_shifted = LennardJonesPotential(
            epsilon=1.0, sigma=1.0, cutoff=2.5, shift_energy=True
        )
        # At cutoff, shifted energy should be zero
        energy_at_cutoff = lj_shifted.compute_pair_energy(2.49999)
        assert abs(energy_at_cutoff) < 0.01  # Close to zero


class TestMorsePotential:
    """Tests for MorsePotential."""

    @pytest.fixture
    def morse(self) -> MorsePotential:
        """Standard Morse potential."""
        return MorsePotential(D=0.5, a=1.5, r0=1.0, cutoff=5.0)

    def test_pair_energy_at_equilibrium(self, morse: MorsePotential) -> None:
        """At r=r0, U(r) = 0."""
        energy = morse.compute_pair_energy(1.0)
        assert pytest.approx(energy, abs=1e-10) == 0.0

    def test_pair_energy_asymptote(self, morse: MorsePotential) -> None:
        """As r->infinity, U(r) -> D."""
        # At large r, exp term -> 0, so U -> D
        energy = morse.compute_pair_energy(4.5)
        assert pytest.approx(energy, rel=0.1) == 0.5


class TestSuttonChenEAM:
    """Tests for SuttonChenEAM potential."""

    def test_copper_factory(self) -> None:
        """Test copper factory method."""
        cu_eam = SuttonChenEAM.copper(cutoff=5.5)
        assert cu_eam.epsilon == 0.0124
        assert cu_eam.a == 3.615

    def test_compute_energy_runs(self) -> None:
        """Test that EAM energy computation doesn't crash."""
        cu_eam = SuttonChenEAM.copper(cutoff=5.5)
        box = np.array([10.0, 10.0, 10.0])
        bc = PeriodicBoundaryCondition()

        # Two Cu atoms
        positions = np.array([
            [0.0, 0.0, 0.0],
            [2.5, 0.0, 0.0],
        ])
        atom_types = np.array([0, 0])

        energy = cu_eam.compute_energy(positions, box, bc, atom_types=atom_types)
        assert isinstance(energy, float)


class TestCompositePotential:
    """Tests for CompositePotential."""

    def test_composite_energy_is_sum(self) -> None:
        """Test that composite energy is sum of components."""
        lj1 = LennardJonesPotential(epsilon=1.0, sigma=1.0, cutoff=2.5)
        lj2 = LennardJonesPotential(epsilon=0.5, sigma=1.0, cutoff=2.5)
        composite = CompositePotential([lj1, lj2])

        box = np.array([10.0, 10.0, 10.0])
        bc = PeriodicBoundaryCondition()
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
        ])

        e1 = lj1.compute_energy(positions, box, bc)
        e2 = lj2.compute_energy(positions, box, bc)
        e_composite = composite.compute_energy(positions, box, bc)

        assert pytest.approx(e_composite) == e1 + e2

    def test_cutoff_is_max(self) -> None:
        """Test that composite cutoff is max of components."""
        lj1 = LennardJonesPotential(epsilon=1.0, sigma=1.0, cutoff=2.5)
        lj2 = LennardJonesPotential(epsilon=0.5, sigma=1.0, cutoff=3.5)
        composite = CompositePotential([lj1, lj2])

        assert composite.cutoff == 3.5


class TestPotentialInterface:
    """Test all potentials implement the interface."""

    @pytest.mark.parametrize("potential_class,args", [
        (LennardJonesPotential, (1.0, 1.0, 2.5)),
        (MorsePotential, (0.5, 1.5, 1.0, 5.0)),
    ])
    def test_is_potential_energy(
        self, potential_class: type, args: tuple
    ) -> None:
        """Test inheritance."""
        pot = potential_class(*args)
        assert isinstance(pot, PotentialEnergy)

    @pytest.mark.parametrize("potential_class,args", [
        (LennardJonesPotential, (1.0, 1.0, 2.5)),
        (MorsePotential, (0.5, 1.5, 1.0, 5.0)),
    ])
    def test_has_required_methods(
        self, potential_class: type, args: tuple
    ) -> None:
        """Test required methods exist."""
        pot = potential_class(*args)
        assert hasattr(pot, "compute_energy")
        assert hasattr(pot, "get_name")
        assert hasattr(pot, "cutoff")
