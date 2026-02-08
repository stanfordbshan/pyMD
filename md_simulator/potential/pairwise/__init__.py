"""
Pairwise potentials module.

Simple two-body potentials like Lennard-Jones and Morse.
"""

from .lennard_jones import LennardJonesPotential
from .morse import MorsePotential

__all__ = [
    "LennardJonesPotential",
    "MorsePotential",
]
