"""
Potential energy module for molecular dynamics simulations.

This module provides potential energy functions where users ONLY
implement compute_energy(). Forces F = -∇E are computed automatically
via autodiff.

Available potentials:
- Pairwise: LennardJonesPotential, MorsePotential
- Many-body: EAMBasePotential, SuttonChenEAM
- Composite: CompositePotential (combine multiple potentials)
"""

from .composite_potential import CompositePotential
from .many_body import EAMBasePotential, SuttonChenEAM
from .pairwise import LennardJonesPotential, MorsePotential
from .potential_energy import PotentialEnergy

__all__ = [
    # Base class
    "PotentialEnergy",
    # Pairwise
    "LennardJonesPotential",
    "MorsePotential",
    # Many-body
    "EAMBasePotential",
    "SuttonChenEAM",
    # Composite
    "CompositePotential",
]
