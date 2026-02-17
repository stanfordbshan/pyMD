"""
Atom class for molecular dynamics simulations.

This module provides the Atom dataclass representing a single atom
in the simulation.
"""
from dataclasses import dataclass


@dataclass
class Atom:
    """
    Represents a single atom in the simulation.

    The Atom class is a value object (dataclass) that stores the static
    properties of an atom. Dynamic properties like position and velocity
    are stored in the State class.

    Attributes:
        atom_type: Chemical symbol or type identifier (e.g., 'Cu', 'Ar', 'H').
            Used for element identification and multi-species potentials.
        mass: Atomic mass in the current unit system (e.g., g/mol for REAL/METAL).
        charge: Electric charge in elementary charge units (default: 0.0).
        index: Unique identifier/index for this atom in the system (default: -1).

    Example:
        >>> from pyMD.core import Atom
        >>> cu_atom = Atom(atom_type='Cu', mass=63.546, charge=0.0, index=0)
        >>> print(cu_atom.atom_type)
        Cu
    """
    atom_type: str
    mass: float
    charge: float = 0.0
    index: int = -1

    def __post_init__(self) -> None:
        """Validate atom properties after initialization."""
        if self.mass <= 0:
            raise ValueError(f"Atom mass must be positive, got {self.mass}")
        if not self.atom_type:
            raise ValueError("Atom type cannot be empty")
