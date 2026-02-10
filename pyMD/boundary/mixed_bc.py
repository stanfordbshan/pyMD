"""
Mixed boundary condition implementation.

This module provides mixed boundaries where each dimension can be
either periodic or open, useful for slab geometries.
"""
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from .boundary_condition import BoundaryCondition


class MixedBoundaryCondition(BoundaryCondition):
    """
    Mixed boundaries with different conditions per dimension.

    Allows each dimension (X, Y, Z) to be independently set as
    periodic or open. Common use cases:
    - Surface/slab: periodic in XY, open in Z
    - Nanowire: periodic in Z, open in XY
    - Nanotube: periodic in axial direction

    Attributes:
        periodic_dims: Boolean array indicating which dimensions are periodic.

    Example:
        >>> import numpy as np
        >>> from pyMD.boundary import MixedBoundaryCondition
        >>> # Periodic in X and Y, open in Z (slab geometry)
        >>> bc = MixedBoundaryCondition(periodic_dims=(True, True, False))
        >>> box = np.array([10.0, 10.0, 50.0])
        >>> # X will wrap, Z will not
        >>> vector = np.array([[8.0, 0.0, 30.0]])
        >>> corrected = bc.apply_minimum_image(vector, box)
        >>> print(corrected)  # [[-2.0, 0.0, 30.0]]
    """

    def __init__(self, periodic_dims: Tuple[bool, bool, bool]) -> None:
        """
        Initialize mixed boundary condition.

        Args:
            periodic_dims: Tuple of (x_periodic, y_periodic, z_periodic).
                True means that dimension is periodic.

        Example:
            >>> # Surface: periodic XY, open Z
            >>> bc = MixedBoundaryCondition((True, True, False))
            >>> # Nanowire: open XY, periodic Z
            >>> bc = MixedBoundaryCondition((False, False, True))
        """
        self.periodic_dims = np.array(periodic_dims, dtype=bool)
        if self.periodic_dims.shape != (3,):
            raise ValueError("periodic_dims must be a tuple of 3 booleans")

    def apply_minimum_image(
        self,
        vector: NDArray[np.floating],
        box: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Apply minimum image only in periodic dimensions.

        Args:
            vector: (N, 3) or (3,) displacement vectors.
            box: (3,) box dimensions.

        Returns:
            Corrected vectors with minimum image in periodic dimensions.
        """
        vector = np.asarray(vector, dtype=np.float64)
        box = np.asarray(box, dtype=np.float64)

        # Make a copy to avoid modifying input
        result = vector.copy()

        # Apply minimum image only to periodic dimensions
        if vector.ndim == 1:
            # Single vector (3,)
            for dim in range(3):
                if self.periodic_dims[dim]:
                    result[dim] -= box[dim] * np.round(result[dim] / box[dim])
        else:
            # Array of vectors (N, 3)
            for dim in range(3):
                if self.periodic_dims[dim]:
                    result[:, dim] -= box[dim] * np.round(result[:, dim] / box[dim])

        return result

    def wrap_positions(
        self,
        positions: NDArray[np.floating],
        box: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Wrap positions only in periodic dimensions.

        Args:
            positions: (N, 3) atomic positions.
            box: (3,) box dimensions.

        Returns:
            Positions wrapped in periodic dimensions only.
        """
        positions = np.asarray(positions, dtype=np.float64)
        box = np.asarray(box, dtype=np.float64)

        # Make a copy to avoid modifying input
        wrapped = positions.copy()

        # Wrap only periodic dimensions
        for dim in range(3):
            if self.periodic_dims[dim]:
                wrapped[:, dim] -= box[dim] * np.floor(wrapped[:, dim] / box[dim])

        return wrapped

    def get_name(self) -> str:
        """
        Return descriptive name showing which dimensions are periodic.

        Returns:
            String like "Mixed(XY periodic)" or "Mixed(Z periodic)".
        """
        dim_names = ['X', 'Y', 'Z']
        periodic_names = [dim_names[i] for i in range(3) if self.periodic_dims[i]]

        if not periodic_names:
            return "Mixed(none periodic)"
        elif len(periodic_names) == 3:
            return "Mixed(all periodic)"
        else:
            return f"Mixed({''.join(periodic_names)} periodic)"
