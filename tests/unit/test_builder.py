"""
Unit tests for builder module.
"""
import numpy as np
import pytest

from pyMD.builder import SystemBuilder
from pyMD.core import System, Units


class TestSystemBuilder:
    """Tests for SystemBuilder."""

    def test_fcc_lattice_atom_count(self) -> None:
        """FCC lattice should have 4 atoms per unit cell."""
        system = (
            SystemBuilder()
            .element("Ar", mass=1.0)
            .fcc_lattice(nx=2, ny=2, nz=2, a=1.0)
            .build()
        )
        # 2x2x2 unit cells * 4 atoms/cell = 32 atoms
        assert system.get_num_atoms() == 32

    def test_bcc_lattice_atom_count(self) -> None:
        """BCC lattice should have 2 atoms per unit cell."""
        system = (
            SystemBuilder()
            .element("Fe", mass=55.845)
            .bcc_lattice(nx=3, ny=3, nz=3, a=2.87)
            .build()
        )
        # 3x3x3 * 2 = 54 atoms
        assert system.get_num_atoms() == 54

    def test_sc_lattice_atom_count(self) -> None:
        """SC lattice should have 1 atom per unit cell."""
        system = (
            SystemBuilder()
            .element("X", mass=1.0)
            .sc_lattice(nx=4, ny=4, nz=4, a=1.0)
            .build()
        )
        # 4x4x4 * 1 = 64 atoms
        assert system.get_num_atoms() == 64

    def test_box_dimensions(self) -> None:
        """Box should be nx*a x ny*a x nz*a."""
        a = 2.5
        nx, ny, nz = 3, 4, 5
        system = (
            SystemBuilder()
            .element("Ar", mass=1.0)
            .fcc_lattice(nx=nx, ny=ny, nz=nz, a=a)
            .build()
        )
        expected_box = np.array([nx * a, ny * a, nz * a])
        np.testing.assert_array_almost_equal(system.get_box(), expected_box)

    def test_temperature_initializes_velocities(self) -> None:
        """Temperature should initialize velocities."""
        target_T = 1.0
        system = (
            SystemBuilder()
            .element("Ar", mass=1.0)
            .fcc_lattice(nx=3, ny=3, nz=3, a=1.5)
            .temperature(target_T)
            .units(Units.LJ())
            .build()
        )
        # Velocities should not be all zero
        assert np.any(system.state.velocities != 0)
        # Temperature should be close to target
        T = system.compute_temperature()
        assert pytest.approx(T, rel=0.1) == target_T

    def test_zero_temperature(self) -> None:
        """Zero temperature should give zero velocities."""
        system = (
            SystemBuilder()
            .element("Ar", mass=1.0)
            .fcc_lattice(nx=2, ny=2, nz=2, a=1.0)
            .temperature(0.0)
            .build()
        )
        np.testing.assert_array_equal(system.state.velocities, 0)

    def test_element_mass(self) -> None:
        """Element mass should be set correctly."""
        mass = 39.948
        system = (
            SystemBuilder()
            .element("Ar", mass=mass)
            .sc_lattice(nx=2, ny=2, nz=2, a=1.0)
            .build()
        )
        for atom in system.atoms:
            assert atom.mass == mass

    def test_random_positions_with_density(self) -> None:
        """Random positions should respect density."""
        n_atoms = 100
        density = 0.8
        system = (
            SystemBuilder()
            .element("Ar", mass=1.0)
            .random_positions(n_atoms=n_atoms, density=density)
            .build()
        )
        assert system.get_num_atoms() == n_atoms
        # V = N / rho, L = V^(1/3)
        expected_L = (n_atoms / density) ** (1.0 / 3.0)
        assert pytest.approx(system.get_box()[0], rel=0.01) == expected_L

    def test_fluent_api(self) -> None:
        """Builder should support fluent API."""
        builder = SystemBuilder()
        result = builder.element("Ar", 1.0)
        assert result is builder  # Returns self

    def test_build_without_positions_raises(self) -> None:
        """Build without positions should raise."""
        builder = SystemBuilder().element("Ar", 1.0)
        with pytest.raises(ValueError, match="Positions"):
            builder.build()

    def test_units_propagate(self) -> None:
        """Units should be set on system."""
        system = (
            SystemBuilder()
            .element("Ar", mass=1.0)
            .fcc_lattice(nx=2, ny=2, nz=2, a=1.0)
            .units(Units.METAL())
            .build()
        )
        assert system.units.name.value == "metal"


class TestLatticePrimitives:
    """Test lattice generation correctness."""

    def test_fcc_positions_in_box(self) -> None:
        """All FCC positions should be within box."""
        system = (
            SystemBuilder()
            .element("Ar", mass=1.0)
            .fcc_lattice(nx=3, ny=3, nz=3, a=2.0)
            .build()
        )
        box = system.get_box()
        positions = system.state.positions

        # All positions should be >= 0 and < box
        assert np.all(positions >= 0)
        assert np.all(positions < box)

    def test_fcc_nearest_neighbor_distance(self) -> None:
        """FCC nearest neighbor distance should be a/sqrt(2)."""
        a = 2.0
        system = (
            SystemBuilder()
            .element("Ar", mass=1.0)
            .fcc_lattice(nx=2, ny=2, nz=2, a=a)
            .build()
        )

        positions = system.state.positions
        # Find minimum distance
        min_dist = float("inf")
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < min_dist:
                    min_dist = dist

        expected_nn = a / np.sqrt(2)
        assert pytest.approx(min_dist, rel=0.01) == expected_nn
