"""
Verlet neighbor list implementation.

Verlet list with skin distance for reduced rebuild frequency.
O(N²) build but infrequent rebuilds make it efficient.
"""
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .neighbor_list import NeighborList

if TYPE_CHECKING:
    from pyMD.boundary import BoundaryCondition


class VerletList(NeighborList):
    """
    Verlet neighbor list with skin distance.

    Classic Verlet list algorithm that includes a "skin" buffer around
    the cutoff. Atoms within (cutoff + skin) are stored, but only those
    within cutoff are used for forces. This reduces rebuild frequency.

    Complexity:
        - Build: O(N²)
        - Lookup: O(1) per atom

    Good for:
        - Medium systems (1000 < N < 10000)
        - Uniform density systems

    Attributes:
        build_count: Number of times the list has been rebuilt.

    Example:
        >>> from pyMD.neighbor import VerletList
        >>> nl = VerletList(cutoff=2.5, skin=0.3)
        >>> nl.build(positions, box, pbc)
        >>> print(f"Built {nl.build_count} times")
    """

    def __init__(self, cutoff: float, skin: float = 0.3) -> None:
        """
        Initialize Verlet list.

        Args:
            cutoff: Interaction cutoff distance.
            skin: Buffer distance (default 0.3). Larger values reduce
                  rebuild frequency but increase neighbor count.
        """
        super().__init__(cutoff, skin)
        self.build_count = 0

    def build(
        self,
        positions: NDArray[np.floating],
        box: NDArray[np.floating],
        boundary_condition: "BoundaryCondition",
    ) -> None:
        """
        Build neighbor list with skin distance.

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
        self.build_count += 1

    def get_neighbors(self, atom_index: int) -> NDArray[np.intp]:
        """Get neighbors for atom at given index."""
        if self.neighbors is None:
            return np.array([], dtype=np.intp)
        return np.array(self.neighbors.get(atom_index, []), dtype=np.intp)

    def get_name(self) -> str:
        """Return algorithm name with stats."""
        return f"VerletList(skin={self.skin:.2f}, builds={self.build_count})"

    def get_num_neighbors(self) -> int:
        """Return total number of neighbor pairs."""
        if self.neighbors is None:
            return 0
        return sum(len(v) for v in self.neighbors.values())
