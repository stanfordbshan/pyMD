"""
Composite potential for combining multiple potentials.

Allows combining different potentials (e.g., EAM + Coulomb).
"""
from typing import TYPE_CHECKING, Any, List, Optional

import numpy as np
from numpy.typing import NDArray

from .potential_energy import PotentialEnergy

if TYPE_CHECKING:
    from md_simulator.boundary import BoundaryCondition
    from md_simulator.neighbor import NeighborList


class CompositePotential(PotentialEnergy):
    """
    Combine multiple potentials (Composite Pattern).

    Total energy is the sum of all component potentials.
    Forces are computed automatically via autodiff on the sum.

    Example:
        >>> from md_simulator.potential import CompositePotential
        >>> from md_simulator.potential import LennardJonesPotential
        >>> lj = LennardJonesPotential(epsilon=1.0, sigma=1.0, cutoff=2.5)
        >>> # Could add more potentials here
        >>> composite = CompositePotential([lj])
    """

    def __init__(self, potentials: List[PotentialEnergy]) -> None:
        """
        Initialize composite potential.

        Args:
            potentials: List of PotentialEnergy objects to combine.
        """
        if not potentials:
            raise ValueError("Must provide at least one potential")
        self.potentials = potentials

    @property
    def cutoff(self) -> float:
        """Return maximum cutoff of all potentials."""
        return max(p.cutoff for p in self.potentials)

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
        Compute total energy as sum of all potentials.

        Args:
            positions: (N, 3) atomic positions.
            box: (3,) box dimensions.
            boundary_condition: For minimum image convention.
            atom_types: (N,) atom type indices.
            neighbor_list: Optional neighbor list.

        Returns:
            Sum of all potential energies.
        """
        return sum(
            p.compute_energy(
                positions, box, boundary_condition, atom_types, neighbor_list, **kwargs
            )
            for p in self.potentials
        )

    def get_name(self) -> str:
        """Return composite name listing all potentials."""
        names = [p.get_name() for p in self.potentials]
        return f"Composite({', '.join(names)})"

    def add_potential(self, potential: PotentialEnergy) -> None:
        """Add a potential to the composite."""
        self.potentials.append(potential)
