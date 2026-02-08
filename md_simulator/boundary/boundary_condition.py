"""
Abstract base class for boundary conditions.

This module provides the BoundaryCondition ABC that defines
the interface for all boundary condition strategies.
"""
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class BoundaryCondition(ABC):
    """
    Abstract base for boundary conditions (Strategy Pattern).

    BoundaryCondition defines the interface for handling periodic boundaries,
    minimum image convention, and position wrapping. Different implementations
    (Periodic, Open, Mixed) are interchangeable.

    Design Notes:
        - BoundaryCondition is STATELESS - it does NOT own the box.
        - The box is passed as a parameter from System.state.box.
        - This design keeps BC pure and reusable, and allows the box
          to change during NPT simulations without recreating the BC.

    Example:
        >>> # Using with System
        >>> from md_simulator.boundary import PeriodicBoundaryCondition
        >>> bc = PeriodicBoundaryCondition()
        >>> # BC receives box from System, doesn't store it
        >>> wrapped = bc.wrap_positions(positions, box)
    """

    @abstractmethod
    def apply_minimum_image(
        self,
        vector: NDArray[np.floating],
        box: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Apply minimum image convention to distance vectors.

        Critical for force calculations with periodic boundaries.
        Ensures that the shortest distance between atoms is used,
        accounting for periodic images.

        Args:
            vector: (N, 3) or (3,) displacement vectors r_j - r_i.
            box: (3,) current box dimensions FROM SYSTEM.

        Returns:
            Corrected vectors with minimum image applied.

        Example:
            >>> # If box = [10, 10, 10] and vector = [8, 0, 0]
            >>> # The minimum image is [-2, 0, 0] (across boundary)
            >>> corrected = bc.apply_minimum_image(vector, box)
        """
        pass

    @abstractmethod
    def wrap_positions(
        self,
        positions: NDArray[np.floating],
        box: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Wrap positions back into primary simulation box.

        Ensures all atoms remain within the box [0, box).

        Args:
            positions: (N, 3) atomic positions.
            box: (3,) current box dimensions FROM SYSTEM.

        Returns:
            Wrapped positions within the primary box.
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Get human-readable name of this boundary condition.

        Returns:
            Name string (e.g., "Periodic", "Open", "Mixed(XY periodic)").
        """
        pass

    def compute_distance_vectors(
        self,
        positions_i: NDArray[np.floating],
        positions_j: NDArray[np.floating],
        box: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Compute distance vectors with minimum image convention.

        Convenience method that computes r_j - r_i and applies
        minimum image convention.

        Args:
            positions_i: (N, 3) or (3,) positions of atoms i.
            positions_j: (N, 3) or (3,) positions of atoms j.
            box: (3,) box dimensions.

        Returns:
            Distance vectors from i to j with minimum image applied.
        """
        dr = positions_j - positions_i
        return self.apply_minimum_image(dr, box)

    def compute_distances(
        self,
        positions_i: NDArray[np.floating],
        positions_j: NDArray[np.floating],
        box: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Compute scalar distances with minimum image convention.

        Args:
            positions_i: (N, 3) or (3,) positions of atoms i.
            positions_j: (N, 3) or (3,) positions of atoms j.
            box: (3,) box dimensions.

        Returns:
            Scalar distances |r_j - r_i| with minimum image applied.
        """
        dr = self.compute_distance_vectors(positions_i, positions_j, box)
        return np.linalg.norm(dr, axis=-1)
