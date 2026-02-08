"""
System class for molecular dynamics simulations.

This module provides the System class that orchestrates atoms, state,
boundary conditions, and units for an MD simulation.
"""
from typing import TYPE_CHECKING, List, Optional

import numpy as np
from numpy.typing import NDArray

from .atom import Atom
from .state import State
from .units import UnitSystem

if TYPE_CHECKING:
    from md_simulator.boundary import BoundaryCondition


class System:
    """
    Central object holding all atoms and state for MD simulation.

    The System class is the main container for an MD simulation. It manages:
    - A collection of Atom objects (static properties)
    - The current State (dynamic properties: positions, velocities, forces)
    - A reference to the boundary condition strategy
    - The unit system for the simulation

    Design Notes:
        - The simulation box is stored in System.state, not in BoundaryCondition.
          This allows the BC to be stateless and reusable, and enables NPT
          simulations where the box changes over time.
        - BoundaryCondition is a Strategy pattern - different BC algorithms
          are interchangeable.

    Attributes:
        atoms: List of Atom objects in the system.
        state: Current State containing positions, velocities, forces.
        boundary_condition: Strategy for handling boundaries.
        units: Unit system for the simulation.
        atom_types: Integer type indices for multi-species potentials.

    Example:
        >>> from md_simulator.core import System, Atom, State, Units
        >>> from md_simulator.boundary import PeriodicBoundaryCondition
        >>> atoms = [Atom('Ar', mass=39.948, index=i) for i in range(100)]
        >>> positions = np.random.rand(100, 3) * 10.0
        >>> state = State(
        ...     positions=positions,
        ...     velocities=np.zeros((100, 3)),
        ...     forces=np.zeros((100, 3)),
        ...     box=np.array([10.0, 10.0, 10.0])
        ... )
        >>> system = System(atoms, state, PeriodicBoundaryCondition(), Units.LJ())
    """

    def __init__(
        self,
        atoms: List[Atom],
        state: State,
        boundary_condition: "BoundaryCondition",
        units: UnitSystem,
    ) -> None:
        """
        Initialize a System.

        Args:
            atoms: List of Atom objects.
            state: Initial State with positions, velocities, forces, box.
            boundary_condition: BoundaryCondition strategy.
            units: UnitSystem for the simulation.

        Raises:
            ValueError: If number of atoms doesn't match state dimensions.
        """
        if len(atoms) != state.n_atoms:
            raise ValueError(
                f"Number of atoms ({len(atoms)}) must match "
                f"state dimensions ({state.n_atoms})"
            )

        self.atoms = atoms
        self.state = state
        self.boundary_condition = boundary_condition
        self.units = units
        self._atom_types: Optional[NDArray[np.intp]] = None
        self._species_map: Optional[dict] = None

    @property
    def atom_types(self) -> NDArray[np.intp]:
        """
        Get integer type indices for multi-species potentials.

        Lazily computed and cached. Maps unique atom_type strings
        to integer indices (0, 1, 2, ...).

        Returns:
            (N,) array of integer type indices.

        Example:
            For a Cu-Ni alloy system:
                Cu atoms → 0
                Ni atoms → 1
        """
        if self._atom_types is None:
            self._atom_types, self._species_map = self._assign_atom_types()
        return self._atom_types

    @property
    def species_map(self) -> dict:
        """
        Get mapping from species name to type index.

        Returns:
            Dictionary mapping atom_type strings to integer indices.
        """
        if self._species_map is None:
            self._atom_types, self._species_map = self._assign_atom_types()
        return self._species_map

    def _assign_atom_types(self) -> tuple:
        """
        Assign integer type indices to atoms based on species.

        Returns:
            Tuple of (atom_types array, species_to_index dict).
        """
        species_to_index: dict = {}
        atom_types = np.zeros(len(self.atoms), dtype=np.intp)

        for i, atom in enumerate(self.atoms):
            if atom.atom_type not in species_to_index:
                species_to_index[atom.atom_type] = len(species_to_index)
            atom_types[i] = species_to_index[atom.atom_type]

        return atom_types, species_to_index

    def get_masses(self) -> NDArray[np.floating]:
        """
        Return array of atomic masses.

        Returns:
            (N,) array of atomic masses in current unit system.
        """
        return np.array([atom.mass for atom in self.atoms], dtype=np.float64)

    def get_charges(self) -> NDArray[np.floating]:
        """
        Return array of atomic charges.

        Returns:
            (N,) array of atomic charges in elementary charge units.
        """
        return np.array([atom.charge for atom in self.atoms], dtype=np.float64)

    def get_atom_types(self) -> NDArray[np.intp]:
        """
        Return array of atom type indices.

        This is an alias for the atom_types property for compatibility
        with the design document interface.

        Returns:
            (N,) array of integer type indices.
        """
        return self.atom_types

    def wrap_positions(self) -> None:
        """
        Wrap positions using boundary condition strategy.

        Modifies state.positions in-place to bring atoms back into
        the primary simulation box.

        Note:
            The box is passed TO the BoundaryCondition, not owned BY it.
            This allows BC to be stateless and reusable.
        """
        self.state.positions = self.boundary_condition.wrap_positions(
            self.state.positions, self.state.box
        )

    def set_box(self, new_box: NDArray[np.floating]) -> None:
        """
        Update box dimensions (e.g., for NPT ensemble).

        Args:
            new_box: (3,) array of new box dimensions.
        """
        self.state.box = np.asarray(new_box, dtype=np.float64)

    def get_box(self) -> NDArray[np.floating]:
        """
        Get current box dimensions.

        Returns:
            (3,) array of box dimensions.
        """
        return self.state.box

    def get_volume(self) -> float:
        """
        Compute simulation box volume.

        Returns:
            Volume of the orthorhombic box.
        """
        return float(np.prod(self.state.box))

    def compute_kinetic_energy(self) -> float:
        """
        Compute total kinetic energy of the system.

        KE = (1/2) * Σ_i m_i * |v_i|²

        Returns:
            Total kinetic energy in current unit system.
        """
        masses = self.get_masses()
        velocities = self.state.velocities
        # (1/2) * m * v^2 summed over all atoms
        return 0.5 * np.sum(masses[:, np.newaxis] * velocities ** 2)

    def compute_temperature(self) -> float:
        """
        Compute instantaneous temperature from kinetic energy.

        Uses the equipartition theorem:
            T = 2 * KE / (n_dof * kB)

        where n_dof = 3*N - 3 (subtracting center of mass motion).

        Returns:
            Instantaneous temperature in Kelvin.

        Raises:
            ValueError: If system has fewer than 2 atoms.
        """
        if len(self.atoms) < 2:
            raise ValueError("Need at least 2 atoms to compute temperature")

        ke = self.compute_kinetic_energy()
        n_dof = 3 * len(self.atoms) - 3  # Subtract center of mass DOF
        return 2.0 * ke / (n_dof * self.units.boltzmann)

    def get_num_atoms(self) -> int:
        """
        Return the number of atoms in the system.

        Returns:
            Number of atoms.
        """
        return len(self.atoms)

    def get_center_of_mass(self) -> NDArray[np.floating]:
        """
        Compute center of mass position.

        Returns:
            (3,) array of center of mass coordinates.
        """
        masses = self.get_masses()
        total_mass = np.sum(masses)
        return np.sum(masses[:, np.newaxis] * self.state.positions, axis=0) / total_mass

    def get_momentum(self) -> NDArray[np.floating]:
        """
        Compute total momentum of the system.

        Returns:
            (3,) array of total momentum.
        """
        masses = self.get_masses()
        return np.sum(masses[:, np.newaxis] * self.state.velocities, axis=0)

    def zero_momentum(self) -> None:
        """
        Remove center of mass velocity.

        Adjusts all velocities so that total momentum is zero.
        """
        masses = self.get_masses()
        total_mass = np.sum(masses)
        com_velocity = self.get_momentum() / total_mass
        self.state.velocities -= com_velocity

    def __repr__(self) -> str:
        """Return string representation of the system."""
        species = list(self.species_map.keys())
        return (
            f"System(n_atoms={len(self.atoms)}, "
            f"species={species}, "
            f"box={self.state.box}, "
            f"units={self.units.name.value})"
        )
