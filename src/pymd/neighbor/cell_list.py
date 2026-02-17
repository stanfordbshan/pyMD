"""
Cell list (link-cell) neighbor list implementation.

O(N) scaling algorithm for large systems. Divides box into cells
and only checks neighboring cells.
"""
from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

from .neighbor_list import NeighborList

if TYPE_CHECKING:
    from pymd.boundary import BoundaryCondition


class CellList(NeighborList):
    """
    Cell list (link-cell) algorithm with O(N) scaling.

    Divides the simulation box into cells of size >= cutoff.
    For each atom, only checks atoms in the same cell and 26
    neighboring cells.

    Complexity:
        - Build: O(N) for uniform density
        - Lookup: O(1) per atom

    Good for:
        - Large systems (N > 10000)
        - Best asymptotic scaling

    Attributes:
        cells: Dictionary mapping cell indices to list of atom indices.
        cell_size: Actual cell size used.
        n_cells: Number of cells in each dimension.
        build_count: Number of times the list has been rebuilt.

    Example:
        >>> from pymd.neighbor import CellList
        >>> nl = CellList(cutoff=2.5, skin=0.3)
        >>> nl.build(positions, box, pbc)
        >>> print(f"Grid: {nl.n_cells}")
    """

    def __init__(self, cutoff: float, skin: float = 0.3) -> None:
        """
        Initialize cell list.

        Args:
            cutoff: Interaction cutoff distance.
            skin: Buffer distance (default 0.3).
        """
        super().__init__(cutoff, skin)
        self.cells: Dict[Tuple[int, int, int], List[int]] = {}
        self.cell_size: float = 0.0
        self.n_cells: NDArray[np.intp] = np.array([1, 1, 1], dtype=np.intp)
        self.build_count = 0

    def build(
        self,
        positions: NDArray[np.floating],
        box: NDArray[np.floating],
        boundary_condition: "BoundaryCondition",
    ) -> None:
        """
        Build cell list.

        Args:
            positions: (N, 3) atomic positions.
            box: (3,) box dimensions.
            boundary_condition: For minimum image convention.
        """
        n_atoms = len(positions)

        # Determine cell grid
        self.cell_size = self.build_cutoff
        self.n_cells = np.maximum(1, (box / self.cell_size).astype(np.intp))
        actual_cell_size = box / self.n_cells

        # Initialize cells
        self.cells = {}
        for i in range(self.n_cells[0]):
            for j in range(self.n_cells[1]):
                for k in range(self.n_cells[2]):
                    self.cells[(i, j, k)] = []

        # Assign atoms to cells
        for atom_idx in range(n_atoms):
            pos = positions[atom_idx]
            cell_idx = tuple(
                (pos / actual_cell_size).astype(np.intp) % self.n_cells
            )
            self.cells[cell_idx].append(atom_idx)

        # Build neighbor list by checking neighboring cells
        self.neighbors = {i: [] for i in range(n_atoms)}

        for cell_idx, atoms_in_cell in self.cells.items():
            neighbor_cells = self._get_neighbor_cells(cell_idx)

            for atom_i in atoms_in_cell:
                for neighbor_cell in neighbor_cells:
                    for atom_j in self.cells[neighbor_cell]:
                        if atom_j <= atom_i:
                            continue

                        dr = positions[atom_j] - positions[atom_i]
                        dr = boundary_condition.apply_minimum_image(
                            dr.reshape(1, 3), box
                        ).flatten()
                        distance = np.linalg.norm(dr)

                        if distance < self.build_cutoff:
                            self.neighbors[atom_i].append(atom_j)

        self.last_build_positions = positions.copy()
        self.build_count += 1

    def _get_neighbor_cells(
        self, cell_idx: Tuple[int, int, int]
    ) -> List[Tuple[int, int, int]]:
        """
        Get indices of neighboring cells (including periodic wrapping).

        Args:
            cell_idx: (i, j, k) cell index.

        Returns:
            List of neighboring cell indices.
        """
        i, j, k = cell_idx
        neighbors = []

        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                for dk in [-1, 0, 1]:
                    ni = (i + di) % self.n_cells[0]
                    nj = (j + dj) % self.n_cells[1]
                    nk = (k + dk) % self.n_cells[2]
                    neighbors.append((ni, nj, nk))

        return neighbors

    def get_neighbors(self, atom_index: int) -> NDArray[np.intp]:
        """Get neighbors for atom at given index."""
        if self.neighbors is None:
            return np.array([], dtype=np.intp)
        return np.array(self.neighbors.get(atom_index, []), dtype=np.intp)

    def get_name(self) -> str:
        """Return algorithm name with stats."""
        return f"CellList(cells={tuple(self.n_cells)}, builds={self.build_count})"

    def get_num_neighbors(self) -> int:
        """Return total number of neighbor pairs."""
        if self.neighbors is None:
            return 0
        return sum(len(v) for v in self.neighbors.values())
