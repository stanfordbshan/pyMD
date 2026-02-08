"""
State class for molecular dynamics simulations.

This module provides the State dataclass representing a snapshot
of the system at a single timestep.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class State:
    """
    Snapshot of the system state at one timestep.

    Contains all time-dependent quantities that define the system.
    These are the quantities that change during the simulation,
    as opposed to static properties stored in Atom objects.

    Attributes:
        positions: (N, 3) array of atomic positions in the current unit system.
        velocities: (N, 3) array of atomic velocities.
        forces: (N, 3) array of forces on each atom.
        box: (3,) array of simulation box dimensions (orthorhombic box).
        time: Current simulation time (default: 0.0).
        step: Current timestep number (default: 0).

    Note:
        The box is stored in the State (not BoundaryCondition) to allow
        for NPT simulations where the box can change over time.

    Example:
        >>> import numpy as np
        >>> from md_simulator.core import State
        >>> state = State(
        ...     positions=np.zeros((10, 3)),
        ...     velocities=np.zeros((10, 3)),
        ...     forces=np.zeros((10, 3)),
        ...     box=np.array([10.0, 10.0, 10.0])
        ... )
    """
    positions: NDArray[np.floating]
    velocities: NDArray[np.floating]
    forces: NDArray[np.floating]
    box: NDArray[np.floating]
    time: float = 0.0
    step: int = 0

    def __post_init__(self) -> None:
        """Validate state arrays after initialization."""
        # Ensure arrays are numpy arrays
        self.positions = np.asarray(self.positions, dtype=np.float64)
        self.velocities = np.asarray(self.velocities, dtype=np.float64)
        self.forces = np.asarray(self.forces, dtype=np.float64)
        self.box = np.asarray(self.box, dtype=np.float64)

        # Validate shapes
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError(
                f"Positions must be (N, 3) array, got shape {self.positions.shape}"
            )
        if self.velocities.shape != self.positions.shape:
            raise ValueError(
                f"Velocities shape {self.velocities.shape} must match "
                f"positions shape {self.positions.shape}"
            )
        if self.forces.shape != self.positions.shape:
            raise ValueError(
                f"Forces shape {self.forces.shape} must match "
                f"positions shape {self.positions.shape}"
            )
        if self.box.shape != (3,):
            raise ValueError(f"Box must be (3,) array, got shape {self.box.shape}")

    @property
    def n_atoms(self) -> int:
        """Return the number of atoms in the state."""
        return self.positions.shape[0]

    def copy(self) -> "State":
        """Create a deep copy of the state."""
        return State(
            positions=self.positions.copy(),
            velocities=self.velocities.copy(),
            forces=self.forces.copy(),
            box=self.box.copy(),
            time=self.time,
            step=self.step,
        )
