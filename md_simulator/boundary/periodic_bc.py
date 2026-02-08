"""
Periodic boundary condition implementation.

This module provides fully periodic boundaries in all three dimensions,
typical for bulk simulations.
"""
import numpy as np
from numpy.typing import NDArray

from .boundary_condition import BoundaryCondition


class PeriodicBoundaryCondition(BoundaryCondition):
    """
    Fully periodic boundaries in all dimensions.

    This is the most common boundary condition for bulk material
    simulations. Atoms that exit one side of the box re-enter from
    the opposite side, and forces are computed using the minimum
    image convention.

    Design Note:
        This class is STATELESS - no box is stored here.
        The box is passed as a parameter from System.state.box.

    Example:
        >>> import numpy as np
        >>> from md_simulator.boundary import PeriodicBoundaryCondition
        >>> bc = PeriodicBoundaryCondition()
        >>> box = np.array([10.0, 10.0, 10.0])
        >>> # Vector of 8 units wraps to -2
        >>> vector = np.array([[8.0, 0.0, 0.0]])
        >>> corrected = bc.apply_minimum_image(vector, box)
        >>> print(corrected)  # [[-2.0, 0.0, 0.0]]
    """

    def apply_minimum_image(
        self,
        vector: NDArray[np.floating],
        box: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Apply minimum image convention for periodic boundaries.

        Maps all distance vectors to the range [-box/2, box/2).

        Args:
            vector: (N, 3) or (3,) displacement vectors.
            box: (3,) box dimensions.

        Returns:
            Corrected vectors in the range [-box/2, box/2).
        """
        # Ensure inputs are arrays
        vector = np.asarray(vector, dtype=np.float64)
        box = np.asarray(box, dtype=np.float64)

        # vector - box * round(vector / box) maps to [-box/2, box/2)
        return vector - box * np.round(vector / box)

    def wrap_positions(
        self,
        positions: NDArray[np.floating],
        box: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Wrap positions back into [0, box).

        Args:
            positions: (N, 3) atomic positions.
            box: (3,) box dimensions.

        Returns:
            Wrapped positions in [0, box).
        """
        # Ensure inputs are arrays
        positions = np.asarray(positions, dtype=np.float64)
        box = np.asarray(box, dtype=np.float64)

        # positions - box * floor(positions / box) maps to [0, box)
        return positions - box * np.floor(positions / box)

    def get_name(self) -> str:
        """Return 'Periodic' as the boundary condition name."""
        return "Periodic"
