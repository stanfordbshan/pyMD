"""
Core module for molecular dynamics simulations.

This module provides the fundamental classes for MD simulations:
- Atom: Represents a single atom with its static properties
- State: Snapshot of system state (positions, velocities, forces)
- System: Central container for atoms and simulation state
- Units: Factory for LAMMPS-style unit systems
- ElementRegistry: Database of chemical element properties
"""

from .atom import Atom
from .constants import (
    ANGSTROM_TO_METER,
    AMU,
    AVOGADRO,
    BOLTZMANN_EV,
    BOLTZMANN_KCAL,
    BOLTZMANN_SI,
    COULOMB_CONSTANT,
    ELEMENTARY_CHARGE,
    ELECTRON_MASS,
    EPSILON_0,
    EV_TO_JOULE,
    FS_TO_S,
    HBAR,
    KCAL_MOL_TO_JOULE,
    PLANCK,
    PS_TO_S,
    SPEED_OF_LIGHT,
)
from .element_registry import ElementData, ElementRegistry, elements
from .state import State
from .system import System
from .units import Units, UnitSystem, UnitSystemType

__all__ = [
    # Classes
    "Atom",
    "State",
    "System",
    "Units",
    "UnitSystem",
    "UnitSystemType",
    "ElementData",
    "ElementRegistry",
    # Singleton instance
    "elements",
    # Constants
    "AVOGADRO",
    "BOLTZMANN_SI",
    "BOLTZMANN_EV",
    "BOLTZMANN_KCAL",
    "ELEMENTARY_CHARGE",
    "ELECTRON_MASS",
    "AMU",
    "SPEED_OF_LIGHT",
    "PLANCK",
    "HBAR",
    "EPSILON_0",
    "COULOMB_CONSTANT",
    "ANGSTROM_TO_METER",
    "EV_TO_JOULE",
    "KCAL_MOL_TO_JOULE",
    "FS_TO_S",
    "PS_TO_S",
]
