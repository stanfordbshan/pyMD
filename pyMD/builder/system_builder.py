"""
SystemBuilder for constructing molecular dynamics systems.

Provides the Builder pattern for creating System objects with
various lattice types and initialization options.
"""
from typing import List, Optional, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from pyMD.core import Atom, State, System, UnitSystem, Units
from pyMD.boundary import BoundaryCondition, PeriodicBoundaryCondition

if TYPE_CHECKING:
    pass


class SystemBuilder:
    """
    Builder for constructing System objects.

    Fluent interface for creating MD systems with various options:
    - Lattice types (FCC, BCC, SC, random)
    - Element specification
    - Velocity initialization
    - Box dimensions

    Example:
        >>> builder = SystemBuilder()
        >>> system = (builder
        ...     .element("Ar", mass=39.948)
        ...     .fcc_lattice(nx=4, ny=4, nz=4, a=5.26)
        ...     .temperature(100.0)
        ...     .units(Units.REAL())
        ...     .periodic_boundary()
        ...     .build())
    """

    def __init__(self) -> None:
        """Initialize builder with default values."""
        self._atoms: List[Atom] = []
        self._positions: Optional[NDArray] = None
        self._velocities: Optional[NDArray] = None
        self._box: Optional[NDArray] = None
        self._units: UnitSystem = Units.LJ()
        self._bc: BoundaryCondition = PeriodicBoundaryCondition()
        self._element_type: str = "X"
        self._element_mass: float = 1.0
        self._target_temperature: Optional[float] = None

    def element(self, atom_type: str, mass: float) -> "SystemBuilder":
        """
        Set the element type and mass.

        Args:
            atom_type: Atom type string (e.g., "Ar", "Cu").
            mass: Atomic mass in simulation units.

        Returns:
            Self for chaining.
        """
        self._element_type = atom_type
        self._element_mass = mass
        return self

    def box(self, lx: float, ly: float, lz: float) -> "SystemBuilder":
        """
        Set box dimensions explicitly.

        Args:
            lx, ly, lz: Box dimensions in each direction.

        Returns:
            Self for chaining.
        """
        self._box = np.array([lx, ly, lz], dtype=np.float64)
        return self

    def units(self, unit_system: UnitSystem) -> "SystemBuilder":
        """
        Set the unit system.

        Args:
            unit_system: UnitSystem instance (e.g., Units.LJ()).

        Returns:
            Self for chaining.
        """
        self._units = unit_system
        return self

    def periodic_boundary(self) -> "SystemBuilder":
        """Use periodic boundary conditions."""
        self._bc = PeriodicBoundaryCondition()
        return self

    def boundary_condition(self, bc: BoundaryCondition) -> "SystemBuilder":
        """
        Set custom boundary condition.

        Args:
            bc: BoundaryCondition instance.

        Returns:
            Self for chaining.
        """
        self._bc = bc
        return self

    def temperature(self, temp: float) -> "SystemBuilder":
        """
        Set target temperature for velocity initialization.

        Args:
            temp: Target temperature in simulation units.

        Returns:
            Self for chaining.
        """
        self._target_temperature = temp
        return self

    def fcc_lattice(
        self,
        nx: int,
        ny: int,
        nz: int,
        a: float,
    ) -> "SystemBuilder":
        """
        Create FCC (face-centered cubic) lattice.

        FCC has 4 atoms per unit cell at:
            (0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)

        Args:
            nx, ny, nz: Number of unit cells in each direction.
            a: Lattice constant (unit cell side length).

        Returns:
            Self for chaining.
        """
        # FCC basis positions (in fractional coordinates)
        basis = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ])

        positions = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    for b in basis:
                        pos = (np.array([i, j, k]) + b) * a
                        positions.append(pos)

        self._positions = np.array(positions, dtype=np.float64)
        self._box = np.array([nx * a, ny * a, nz * a], dtype=np.float64)

        # Create atoms
        n_atoms = len(positions)
        self._atoms = [
            Atom(mass=self._element_mass, atom_type=self._element_type)
            for _ in range(n_atoms)
        ]

        return self

    def bcc_lattice(
        self,
        nx: int,
        ny: int,
        nz: int,
        a: float,
    ) -> "SystemBuilder":
        """
        Create BCC (body-centered cubic) lattice.

        BCC has 2 atoms per unit cell at:
            (0, 0, 0), (0.5, 0.5, 0.5)

        Args:
            nx, ny, nz: Number of unit cells in each direction.
            a: Lattice constant.

        Returns:
            Self for chaining.
        """
        basis = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
        ])

        positions = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    for b in basis:
                        pos = (np.array([i, j, k]) + b) * a
                        positions.append(pos)

        self._positions = np.array(positions, dtype=np.float64)
        self._box = np.array([nx * a, ny * a, nz * a], dtype=np.float64)

        n_atoms = len(positions)
        self._atoms = [
            Atom(mass=self._element_mass, atom_type=self._element_type)
            for _ in range(n_atoms)
        ]

        return self

    def sc_lattice(
        self,
        nx: int,
        ny: int,
        nz: int,
        a: float,
    ) -> "SystemBuilder":
        """
        Create SC (simple cubic) lattice.

        SC has 1 atom per unit cell at (0, 0, 0).

        Args:
            nx, ny, nz: Number of unit cells in each direction.
            a: Lattice constant.

        Returns:
            Self for chaining.
        """
        positions = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    pos = np.array([i, j, k], dtype=np.float64) * a
                    positions.append(pos)

        self._positions = np.array(positions, dtype=np.float64)
        self._box = np.array([nx * a, ny * a, nz * a], dtype=np.float64)

        n_atoms = len(positions)
        self._atoms = [
            Atom(mass=self._element_mass, atom_type=self._element_type)
            for _ in range(n_atoms)
        ]

        return self

    def random_positions(
        self,
        n_atoms: int,
        density: Optional[float] = None,
    ) -> "SystemBuilder":
        """
        Create random positions within the box.

        Args:
            n_atoms: Number of atoms.
            density: Number density (atoms per unit volume).
                     If provided, box is computed from density.

        Returns:
            Self for chaining.
        """
        if density is not None:
            # Compute box for given density: V = N / rho
            volume = n_atoms / density
            L = volume ** (1.0 / 3.0)
            self._box = np.array([L, L, L], dtype=np.float64)
        elif self._box is None:
            raise ValueError("Must set box or provide density for random positions")

        self._positions = np.random.rand(n_atoms, 3) * self._box
        self._atoms = [
            Atom(mass=self._element_mass, atom_type=self._element_type)
            for _ in range(n_atoms)
        ]

        return self

    def positions(self, pos: NDArray) -> "SystemBuilder":
        """
        Set positions directly.

        Args:
            pos: (N, 3) array of positions.

        Returns:
            Self for chaining.
        """
        self._positions = np.asarray(pos, dtype=np.float64)
        n_atoms = len(self._positions)
        self._atoms = [
            Atom(mass=self._element_mass, atom_type=self._element_type)
            for _ in range(n_atoms)
        ]
        return self

    def velocities(self, vel: NDArray) -> "SystemBuilder":
        """
        Set velocities directly.

        Args:
            vel: (N, 3) array of velocities.

        Returns:
            Self for chaining.
        """
        self._velocities = np.asarray(vel, dtype=np.float64)
        return self

    def _initialize_velocities(self) -> NDArray:
        """Initialize velocities from temperature."""
        n_atoms = len(self._atoms)

        if self._target_temperature is None or self._target_temperature == 0:
            return np.zeros((n_atoms, 3), dtype=np.float64)

        # Maxwell-Boltzmann distribution
        # sigma = sqrt(kB * T / m) for each component
        kB = self._units.boltzmann
        masses = np.array([a.mass for a in self._atoms])

        # Generate random velocities
        velocities = np.random.randn(n_atoms, 3)

        # Scale each atom's velocity by sqrt(kB*T/m)
        for i in range(n_atoms):
            sigma = np.sqrt(kB * self._target_temperature / masses[i])
            velocities[i] *= sigma

        # Remove center of mass velocity
        total_mass = np.sum(masses)
        com_velocity = np.sum(masses[:, np.newaxis] * velocities, axis=0) / total_mass
        velocities -= com_velocity

        # Rescale to exact target temperature
        # Current KE
        ke = 0.5 * np.sum(masses[:, np.newaxis] * velocities ** 2)
        n_dof = 3 * n_atoms - 3
        current_temp = 2.0 * ke / (n_dof * kB)

        if current_temp > 0:
            scale = np.sqrt(self._target_temperature / current_temp)
            velocities *= scale

        return velocities

    def build(self) -> System:
        """
        Build and return the System.

        Returns:
            Configured System instance.

        Raises:
            ValueError: If required components are missing.
        """
        if self._positions is None:
            raise ValueError("Positions not set. Use a lattice method or positions().")
        if self._box is None:
            raise ValueError("Box dimensions not set.")
        if not self._atoms:
            raise ValueError("No atoms defined.")

        # Initialize velocities
        if self._velocities is None:
            velocities = self._initialize_velocities()
        else:
            velocities = self._velocities

        n_atoms = len(self._atoms)
        forces = np.zeros((n_atoms, 3), dtype=np.float64)

        state = State(
            positions=self._positions,
            velocities=velocities,
            forces=forces,
            box=self._box,
        )

        return System(
            atoms=self._atoms,
            state=state,
            boundary_condition=self._bc,
            units=self._units,
        )
