"""
Force calculator for molecular dynamics simulations.

This module provides the ForceCalculator class that computes forces
from potential energy using autodiff backends.
"""
from typing import TYPE_CHECKING, Optional

import numpy as np
from numpy.typing import NDArray

from .autodiff_backend import AutoDiffBackend

if TYPE_CHECKING:
    from pymd.core import System
    from pymd.neighbor import NeighborList
    from pymd.potential import PotentialEnergy


class ForceCalculator:
    """
    Computes forces from potential energy using autodiff.

    This is where the magic happens:
    - User provides energy function E(positions)
    - ForceCalculator computes F = -∇E automatically

    The calculator manages:
    - Neighbor list rebuilding (if used)
    - Energy function wrapping
    - Autodiff backend invocation

    Attributes:
        potential: The potential energy function.
        backend: Autodiff backend for gradient computation.
        neighbor_list: Optional neighbor list for efficiency.

    Example:
        >>> from pymd.force import ForceCalculator, JAXBackend
        >>> from pymd.potential import LennardJonesPotential
        >>> potential = LennardJonesPotential(epsilon=1.0, sigma=1.0, cutoff=2.5)
        >>> backend = JAXBackend()
        >>> calculator = ForceCalculator(potential, backend)
        >>> forces = calculator.compute_forces(system)
    """

    def __init__(
        self,
        potential: "PotentialEnergy",
        backend: AutoDiffBackend,
        neighbor_list: Optional["NeighborList"] = None,
    ) -> None:
        """
        Initialize force calculator.

        Args:
            potential: Potential energy object with compute_energy method.
            backend: Autodiff backend for gradient computation.
            neighbor_list: Optional neighbor list for efficiency.
        """
        self.potential = potential
        self.backend = backend
        self.neighbor_list = neighbor_list

    def compute_forces(self, system: "System") -> NDArray[np.floating]:
        """
        Compute forces on all atoms using autodiff.

        Process:
        1. Check if neighbor list needs rebuilding
        2. Create energy function with fixed parameters
        3. Call autodiff backend to compute -∇E
        4. Return forces

        Args:
            system: System containing atoms and state.

        Returns:
            (N, 3) array of forces on each atom.
        """
        # Rebuild neighbor list if needed
        if self.neighbor_list is not None:
            if self.neighbor_list.needs_rebuild(system.state.positions):
                self.neighbor_list.build(
                    system.state.positions,
                    system.state.box,
                    system.boundary_condition,
                )

        # Create energy function closure
        def energy_fn(positions: NDArray) -> float:
            return self.potential.compute_energy(
                positions,
                system.state.box,
                system.boundary_condition,
                atom_types=system.get_atom_types(),
                neighbor_list=self.neighbor_list,
            )

        # Compute forces using autodiff: F = -∇E
        forces = self.backend.compute_forces(energy_fn, system.state.positions)

        return forces

    def compute_energy(self, system: "System") -> float:
        """
        Compute potential energy of the system.

        Args:
            system: System containing atoms and state.

        Returns:
            Total potential energy.
        """
        # Rebuild neighbor list if needed
        if self.neighbor_list is not None:
            if self.neighbor_list.needs_rebuild(system.state.positions):
                self.neighbor_list.build(
                    system.state.positions,
                    system.state.box,
                    system.boundary_condition,
                )

        return self.potential.compute_energy(
            system.state.positions,
            system.state.box,
            system.boundary_condition,
            atom_types=system.get_atom_types(),
            neighbor_list=self.neighbor_list,
        )

    def compute_forces_and_energy(
        self, system: "System"
    ) -> tuple[NDArray[np.floating], float]:
        """
        Compute both forces and energy (more efficient than separate calls).

        Args:
            system: System containing atoms and state.

        Returns:
            Tuple of (forces, potential_energy).
        """
        # Rebuild neighbor list if needed
        if self.neighbor_list is not None:
            if self.neighbor_list.needs_rebuild(system.state.positions):
                self.neighbor_list.build(
                    system.state.positions,
                    system.state.box,
                    system.boundary_condition,
                )

        # Compute energy first
        energy = self.potential.compute_energy(
            system.state.positions,
            system.state.box,
            system.boundary_condition,
            atom_types=system.get_atom_types(),
            neighbor_list=self.neighbor_list,
        )

        # Then compute forces
        def energy_fn(positions: NDArray) -> float:
            return self.potential.compute_energy(
                positions,
                system.state.box,
                system.boundary_condition,
                atom_types=system.get_atom_types(),
                neighbor_list=self.neighbor_list,
            )

        forces = self.backend.compute_forces(energy_fn, system.state.positions)

        return forces, energy
