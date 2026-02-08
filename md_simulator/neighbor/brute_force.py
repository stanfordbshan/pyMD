"""
Brute-force neighbor list implementation.

Simple O(N²) algorithm that checks all pairs. Good for small systems
and debugging/educational purposes.
"""
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .neighbor_list import NeighborList

if TYPE_CHECKING:
    from md_simulator.boundary import BoundaryCondition


class BruteForceNeighborList(NeighborList):
    """
    Brute-force neighbor list (O(N²) scaling).

    Checks all pairs of atoms at every build. No optimization,
    but simple and correct. Good for:
    - Small systems (N < 1000)
    - Debugging
    - Educational purposes
    - Reference implementation

    Example:
        >>> from md_simulator.neighbor import BruteForceNeighborList
        >>> nl = BruteForceNeighborList(cutoff=2.5)
        >>> nl.build(positions, box, pbc)
        >>> print(nl.get_num_neighbors())
    """

    def build(
        self,
        positions: NDArray[np.floating],
        box: NDArray[np.floating],
        boundary_condition: "BoundaryCondition",
    ) -> None:
        """
        Build neighbor list by checking all pairs.

        Args:
            positions: (N, 3) atomic positions.
            box: (3,) box dimensions.
            boundary_condition: For minimum image convention.
        """
        n_atoms = len(positions)
        self.neighbors = {i: [] for i in range(n_atoms)}

        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                dr = positions[j] - positions[i]
                dr = boundary_condition.apply_minimum_image(
                    dr.reshape(1, 3), box
                ).flatten()
                distance = np.linalg.norm(dr)

                if distance < self.build_cutoff:
                    self.neighbors[i].append(j)

        self.last_build_positions = positions.copy()

    def get_neighbors(self, atom_index: int) -> NDArray[np.intp]:
        """Get neighbors for atom at given index."""
        if self.neighbors is None:
            return np.array([], dtype=np.intp)
        return np.array(self.neighbors.get(atom_index, []), dtype=np.intp)

    def get_name(self) -> str:
        """Return algorithm name."""
        return "BruteForce"

    def get_num_neighbors(self) -> int:
        """Return total number of neighbor pairs."""
        if self.neighbors is None:
            return 0
        return sum(len(v) for v in self.neighbors.values())
