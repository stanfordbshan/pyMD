"""
Morse potential implementation.

Morse potential for molecular bonding interactions.
Forces are computed automatically via autodiff.
"""
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from numpy.typing import NDArray

from ..potential_energy import PotentialEnergy

if TYPE_CHECKING:
    from pyMD.boundary import BoundaryCondition
    from pyMD.neighbor import NeighborList


class MorsePotential(PotentialEnergy):
    """
    Morse potential for molecular bonds.

    U(r) = D * [1 - exp(-a*(r - r₀))]²

    where:
        - D: Well depth (dissociation energy)
        - a: Width parameter (controls steepness)
        - r₀: Equilibrium bond length

    Unlike harmonic bonds, Morse potential allows for bond breaking
    at large separations and has an asymmetric energy profile.

    Forces are computed AUTOMATICALLY via autodiff!

    Attributes:
        D: Well depth (dissociation energy).
        a: Width parameter.
        r0: Equilibrium distance.
        _cutoff: Interaction cutoff distance.

    Example:
        >>> from pyMD.potential import MorsePotential
        >>> morse = MorsePotential(D=0.5, a=1.5, r0=1.0, cutoff=5.0)
        >>> energy = morse.compute_energy(positions, box, bc)
    """

    def __init__(
        self,
        D: float,
        a: float,
        r0: float,
        cutoff: float,
    ) -> None:
        """
        Initialize Morse potential.

        Args:
            D: Well depth (energy units).
            a: Width parameter (1/length units).
            r0: Equilibrium distance (length units).
            cutoff: Interaction cutoff distance.
        """
        if D <= 0:
            raise ValueError(f"D must be positive, got {D}")
        if a <= 0:
            raise ValueError(f"a must be positive, got {a}")
        if r0 <= 0:
            raise ValueError(f"r0 must be positive, got {r0}")
        if cutoff <= 0:
            raise ValueError(f"cutoff must be positive, got {cutoff}")

        self.D = D
        self.a = a
        self.r0 = r0
        self._cutoff = cutoff

    @property
    def cutoff(self) -> float:
        """Return the cutoff distance."""
        return self._cutoff

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
        Compute Morse energy.

        Args:
            positions: (N, 3) atomic positions.
            box: (3,) box dimensions.
            boundary_condition: For minimum image convention.
            atom_types: Ignored - single species assumed.
            neighbor_list: Optional neighbor list for efficiency.

        Returns:
            Total Morse potential energy.
        """
        energy = 0.0
        n_atoms = len(positions)

        if neighbor_list is not None:
            for i in range(n_atoms):
                neighbors = neighbor_list.get_neighbors(i)
                for j in neighbors:
                    dr = positions[j] - positions[i]
                    dr = boundary_condition.apply_minimum_image(
                        dr.reshape(1, 3), box
                    ).flatten()
                    r = np.linalg.norm(dr)

                    if r < self._cutoff and r > 0:
                        exp_term = np.exp(-self.a * (r - self.r0))
                        energy += self.D * (1 - exp_term) ** 2
        else:
            for i in range(n_atoms):
                for j in range(i + 1, n_atoms):
                    dr = positions[j] - positions[i]
                    dr = boundary_condition.apply_minimum_image(
                        dr.reshape(1, 3), box
                    ).flatten()
                    r = np.linalg.norm(dr)

                    if r < self._cutoff and r > 0:
                        exp_term = np.exp(-self.a * (r - self.r0))
                        energy += self.D * (1 - exp_term) ** 2

        return energy

    def get_name(self) -> str:
        """Return potential name with parameters."""
        return f"Morse(D={self.D}, a={self.a}, r0={self.r0})"

    def compute_pair_energy(self, r: float) -> float:
        """
        Compute Morse energy for a single pair at distance r.

        Args:
            r: Interatomic distance.

        Returns:
            Pair energy.
        """
        if r >= self._cutoff or r <= 0:
            return 0.0

        exp_term = np.exp(-self.a * (r - self.r0))
        return self.D * (1 - exp_term) ** 2
