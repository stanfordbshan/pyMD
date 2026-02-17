"""
Abstract base class for potential energy functions.

This module provides the PotentialEnergy ABC that all potentials
must implement. Users only write compute_energy(); forces are
computed automatically via autodiff.
"""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from pyMD.boundary import BoundaryCondition
    from pyMD.neighbor import NeighborList


class PotentialEnergy(ABC):
    """
    Abstract base for ALL potential energy functions.

    ╔══════════════════════════════════════════════════════════╗
    ║  CRITICAL: Users ONLY implement compute_energy()         ║
    ║  Forces computed AUTOMATICALLY via autodiff              ║
    ║                                                          ║
    ║  This applies to:                                        ║
    ║  - Simple pairwise (LJ, Morse)                          ║
    ║  - Complex many-body (EAM, MEAM, ADP)                   ║
    ║  - Custom research potentials                           ║
    ╚══════════════════════════════════════════════════════════╝

    Example:
        >>> class MyPotential(PotentialEnergy):
        ...     def compute_energy(self, positions, box, bc, **kwargs):
        ...         # Just write the energy formula!
        ...         return np.sum(positions**2)  # Simple example
        ...
        ...     def get_name(self):
        ...         return "MyPotential"
    """

    @abstractmethod
    def compute_energy(
        self,
        positions: NDArray[np.floating],
        box: NDArray[np.floating],
        boundary_condition: "BoundaryCondition",
        atom_types: Optional[NDArray[np.intp]] = None,
        neighbor_list: Optional["NeighborList"] = None,
        **kwargs: Any,
    ) -> float:
        """
        Compute total potential energy from atomic positions.

        User writes ONLY this method.
        Forces = -∇E computed automatically via autodiff.

        Args:
            positions: (N, 3) atomic coordinates.
            box: (3,) box dimensions.
            boundary_condition: For handling periodicity.
            atom_types: (N,) atom type indices (for multi-species).
            neighbor_list: Optional neighbor list for efficiency.
            **kwargs: Additional potential-specific parameters.

        Returns:
            Total potential energy (scalar float).
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get human-readable name of this potential."""
        pass

    @property
    def cutoff(self) -> float:
        """
        Return the interaction cutoff distance.

        Subclasses should override this if they have a cutoff.
        """
        return float("inf")
