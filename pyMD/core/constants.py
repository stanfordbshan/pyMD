"""
Physical constants for molecular dynamics simulations.

This module provides fundamental physical constants used throughout
the MD simulator.
"""
from typing import Final

# Avogadro's number (mol^-1)
AVOGADRO: Final[float] = 6.02214076e23

# Boltzmann constant in SI units (J/K)
BOLTZMANN_SI: Final[float] = 1.380649e-23

# Boltzmann constant in eV/K (for METAL units)
BOLTZMANN_EV: Final[float] = 8.617333262e-5

# Boltzmann constant in kcal/(mol·K) (for REAL units)
BOLTZMANN_KCAL: Final[float] = 0.001987204

# Elementary charge (C)
ELEMENTARY_CHARGE: Final[float] = 1.602176634e-19

# Electron mass (kg)
ELECTRON_MASS: Final[float] = 9.1093837015e-31

# Atomic mass unit (kg)
AMU: Final[float] = 1.66053906660e-27

# Speed of light (m/s)
SPEED_OF_LIGHT: Final[float] = 299792458.0

# Planck constant (J·s)
PLANCK: Final[float] = 6.62607015e-34

# Reduced Planck constant (J·s)
HBAR: Final[float] = 1.054571817e-34

# Vacuum permittivity (F/m)
EPSILON_0: Final[float] = 8.8541878128e-12

# Coulomb constant (N·m²/C²)
COULOMB_CONSTANT: Final[float] = 8.9875517923e9

# Angstrom to meter conversion
ANGSTROM_TO_METER: Final[float] = 1e-10

# eV to Joule conversion
EV_TO_JOULE: Final[float] = 1.602176634e-19

# kcal/mol to Joule conversion
KCAL_MOL_TO_JOULE: Final[float] = 4184.0 / AVOGADRO

# Femtosecond to second conversion
FS_TO_S: Final[float] = 1e-15

# Picosecond to second conversion
PS_TO_S: Final[float] = 1e-12
