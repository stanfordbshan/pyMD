"""
Abstract base class for neighbor list algorithms.

This module provides the NeighborList ABC that defines the interface
for all neighbor list strategies (Verlet, Cell, BruteForce).
"""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from pymd.boundary import BoundaryCondition


class NeighborList(ABC):
    """
    Abstract base for neighbor list algorithms (Strategy Pattern).

    Neighbor lists speed up force calculations by tracking which atoms
    are close enough to interact. Different algorithms have different
    scaling and are suited for different system sizes.

    Attributes:
        cutoff: Interaction cutoff distance.
        skin: Extra distance buffer to reduce rebuild frequency.
        build_cutoff: Total cutoff for building (cutoff + skin).
        neighbors: Dictionary mapping atom index to list of neighbor indices.
        last_build_positions: Positions at last build (for rebuild check).

    Example:
        >>> from pymd.neighbor import VerletList
        >>> nl = VerletList(cutoff=2.5, skin=0.3)
        >>> nl.build(positions, box, boundary_condition)
        >>> neighbors_of_atom_0 = nl.get_neighbors(0)
    """

    def __init__(self, cutoff: float, skin: float = 0.0) -> None:
        """
        Initialize neighbor list.

        Args:
            cutoff: Interaction cutoff distance.
            skin: Extra buffer distance (reduces rebuild frequency).
        """
        if cutoff <= 0:
            raise ValueError(f"Cutoff must be positive, got {cutoff}")
        if skin < 0:
            raise ValueError(f"Skin must be non-negative, got {skin}")

        self.cutoff = cutoff
        self.skin = skin
        self.build_cutoff = cutoff + skin
        self.neighbors: Optional[Dict[int, List[int]]] = None
        self.last_build_positions: Optional[NDArray[np.floating]] = None

    @abstractmethod
    def build(
        self,
        positions: NDArray[np.floating],
        box: NDArray[np.floating],
        boundary_condition: "BoundaryCondition",
    ) -> None:
        """
        Build or rebuild the neighbor list.

        Args:
            positions: (N, 3) atomic positions.
            box: (3,) box dimensions.
            boundary_condition: For handling periodicity.
        """
        pass

    @abstractmethod
    def get_neighbors(self, atom_index: int) -> NDArray[np.intp]:
        """
        Get neighbor indices for a given atom.

        Args:
            atom_index: Index of the atom.

        Returns:
            Array of neighbor atom indices.
        """
        pass

    def needs_rebuild(self, positions: NDArray[np.floating]) -> bool:
        """
        Check if the neighbor list needs rebuilding.

        Uses displacement criterion: if any atom moved more than skin/2
        since the last build, the list should be rebuilt.

        Args:
            positions: Current (N, 3) atomic positions.

        Returns:
            True if rebuild is needed, False otherwise.
        """
        if self.last_build_positions is None:
            return True

        if self.skin == 0:
            return True  # No skin means always rebuild

        max_displacement = np.max(
            np.linalg.norm(positions - self.last_build_positions, axis=1)
        )
        return max_displacement > self.skin / 2.0

    @abstractmethod
    def get_name(self) -> str:
        """Get human-readable name of this neighbor list algorithm."""
        pass

    @abstractmethod
    def get_num_neighbors(self) -> int:
        """Return total number of neighbor pairs."""
        pass

    def get_all_pairs(self) -> List[tuple]:
        """
        Get all neighbor pairs as a list of (i, j) tuples.

        Returns:
            List of tuples (i, j) where i < j.
        """
        if self.neighbors is None:
            return []

        pairs = []
        for i, js in self.neighbors.items():
            for j in js:
                pairs.append((i, j))
        return pairs
