"""
Open boundary condition implementation.

This module provides open (non-periodic) boundaries for simulations
of isolated systems like gas-phase molecules or clusters.
"""
import numpy as np
from numpy.typing import NDArray

from .boundary_condition import BoundaryCondition


class OpenBoundaryCondition(BoundaryCondition):
    """
    Open boundaries (no periodicity).

    Used for gas-phase simulations, isolated clusters, or any system
    that doesn't require periodic images. No wrapping is applied,
    and minimum image convention passes vectors unchanged.

    Example:
        >>> import numpy as np
        >>> from md_simulator.boundary import OpenBoundaryCondition
        >>> bc = OpenBoundaryCondition()
        >>> box = np.array([10.0, 10.0, 10.0])
        >>> # Vectors pass through unchanged
        >>> vector = np.array([[15.0, 0.0, 0.0]])
        >>> corrected = bc.apply_minimum_image(vector, box)
        >>> print(corrected)  # [[15.0, 0.0, 0.0]]
    """

    def apply_minimum_image(
        self,
        vector: NDArray[np.floating],
        box: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        No modification for open boundaries.

        Args:
            vector: (N, 3) or (3,) displacement vectors.
            box: (3,) box dimensions (ignored for open BC).

        Returns:
            Unchanged vectors.
        """
        return np.asarray(vector, dtype=np.float64)

    def wrap_positions(
        self,
        positions: NDArray[np.floating],
        box: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        No wrapping for open boundaries.

        Args:
            positions: (N, 3) atomic positions.
            box: (3,) box dimensions (ignored for open BC).

        Returns:
            Unchanged positions.
        """
        return np.asarray(positions, dtype=np.float64)

    def get_name(self) -> str:
        """Return 'Open' as the boundary condition name."""
        return "Open"
