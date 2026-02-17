"""
Unit tests for neighbor module.

Tests for NeighborList implementations: BruteForce, Verlet, and Cell lists.
"""
import numpy as np
import pytest

from pymd.boundary import PeriodicBoundaryCondition
from pymd.neighbor import (
    BruteForceNeighborList,
    CellList,
    NeighborList,
    VerletList,
)


class TestBruteForceNeighborList:
    """Tests for BruteForceNeighborList."""

    @pytest.fixture
    def box(self) -> np.ndarray:
        """Standard cubic box."""
        return np.array([10.0, 10.0, 10.0])

    @pytest.fixture
    def bc(self) -> PeriodicBoundaryCondition:
        """Periodic boundary condition."""
        return PeriodicBoundaryCondition()

    def test_build_finds_neighbors(
        self, box: np.ndarray, bc: PeriodicBoundaryCondition
    ) -> None:
        """Test that close atoms are found as neighbors."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],  # Within cutoff
            [5.0, 0.0, 0.0],  # Outside cutoff
        ])
        nl = BruteForceNeighborList(cutoff=2.0)
        nl.build(positions, box, bc)

        # Atom 0 should have atom 1 as neighbor
        neighbors_0 = nl.get_neighbors(0)
        assert 1 in neighbors_0
        assert 2 not in neighbors_0

    def test_periodic_boundary(
        self, box: np.ndarray, bc: PeriodicBoundaryCondition
    ) -> None:
        """Test neighbors across periodic boundary."""
        positions = np.array([
            [0.5, 0.0, 0.0],  # Near left edge
            [9.5, 0.0, 0.0],  # Near right edge, 1 unit away via PBC
        ])
        nl = BruteForceNeighborList(cutoff=2.0)
        nl.build(positions, box, bc)

        # Should be neighbors via periodic image
        neighbors_0 = nl.get_neighbors(0)
        assert 1 in neighbors_0

    def test_num_neighbors(
        self, box: np.ndarray, bc: PeriodicBoundaryCondition
    ) -> None:
        """Test neighbor count."""
        # Create positions where each atom has predictable neighbors
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ])
        nl = BruteForceNeighborList(cutoff=1.5)
        nl.build(positions, box, bc)

        # 0-1 and 1-2 are neighbors (2 pairs)
        assert nl.get_num_neighbors() == 2


class TestVerletList:
    """Tests for VerletList."""

    @pytest.fixture
    def box(self) -> np.ndarray:
        """Standard cubic box."""
        return np.array([10.0, 10.0, 10.0])

    @pytest.fixture
    def bc(self) -> PeriodicBoundaryCondition:
        """Periodic boundary condition."""
        return PeriodicBoundaryCondition()

    def test_skin_includes_extra_atoms(
        self, box: np.ndarray, bc: PeriodicBoundaryCondition
    ) -> None:
        """Test that skin distance includes extra atoms."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [2.4, 0.0, 0.0],  # Just outside cutoff=2.0, inside cutoff+skin=2.5
        ])
        nl = VerletList(cutoff=2.0, skin=0.5)
        nl.build(positions, box, bc)

        # Should be in neighbor list due to skin
        neighbors_0 = nl.get_neighbors(0)
        assert 1 in neighbors_0

    def test_build_count(
        self, box: np.ndarray, bc: PeriodicBoundaryCondition
    ) -> None:
        """Test that build_count increments."""
        positions = np.zeros((5, 3))
        nl = VerletList(cutoff=2.0)

        assert nl.build_count == 0
        nl.build(positions, box, bc)
        assert nl.build_count == 1
        nl.build(positions, box, bc)
        assert nl.build_count == 2

    def test_needs_rebuild_initially(self) -> None:
        """Test needs_rebuild returns True before any build."""
        nl = VerletList(cutoff=2.0, skin=0.3)
        positions = np.zeros((5, 3))
        assert nl.needs_rebuild(positions) is True


class TestCellList:
    """Tests for CellList."""

    @pytest.fixture
    def box(self) -> np.ndarray:
        """Standard cubic box."""
        return np.array([10.0, 10.0, 10.0])

    @pytest.fixture
    def bc(self) -> PeriodicBoundaryCondition:
        """Periodic boundary condition."""
        return PeriodicBoundaryCondition()

    def test_cell_grid_created(
        self, box: np.ndarray, bc: PeriodicBoundaryCondition
    ) -> None:
        """Test that cell grid is properly sized."""
        positions = np.zeros((10, 3))
        nl = CellList(cutoff=2.5, skin=0.0)
        nl.build(positions, box, bc)

        # 10/2.5 = 4 cells per dimension
        assert nl.n_cells[0] >= 1
        assert nl.n_cells[1] >= 1
        assert nl.n_cells[2] >= 1

    def test_same_results_as_brute_force(
        self, box: np.ndarray, bc: PeriodicBoundaryCondition
    ) -> None:
        """Test CellList gives same results as BruteForce."""
        np.random.seed(42)
        positions = np.random.rand(20, 3) * 10

        nl_brute = BruteForceNeighborList(cutoff=2.5)
        nl_brute.build(positions, box, bc)

        nl_cell = CellList(cutoff=2.5, skin=0.0)
        nl_cell.build(positions, box, bc)

        # Should find same number of neighbors
        assert nl_brute.get_num_neighbors() == nl_cell.get_num_neighbors()


class TestNeighborListInterface:
    """Test that all neighbor lists implement the interface."""

    @pytest.mark.parametrize("nl_class,args", [
        (BruteForceNeighborList, (2.5,)),
        (VerletList, (2.5, 0.3)),
        (CellList, (2.5, 0.3)),
    ])
    def test_is_neighbor_list(
        self, nl_class: type, args: tuple
    ) -> None:
        """Test all NLs inherit from NeighborList."""
        nl = nl_class(*args)
        assert isinstance(nl, NeighborList)

    @pytest.mark.parametrize("nl_class,args", [
        (BruteForceNeighborList, (2.5,)),
        (VerletList, (2.5, 0.3)),
        (CellList, (2.5, 0.3)),
    ])
    def test_has_required_methods(
        self, nl_class: type, args: tuple
    ) -> None:
        """Test all NLs have required methods."""
        nl = nl_class(*args)
        assert hasattr(nl, "build")
        assert hasattr(nl, "get_neighbors")
        assert hasattr(nl, "get_name")
        assert hasattr(nl, "needs_rebuild")
