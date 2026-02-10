"""
Boundary condition module for molecular dynamics simulations.

This module provides the Strategy pattern implementation for
boundary conditions:
- PeriodicBoundaryCondition: Fully periodic (bulk simulations)
- OpenBoundaryCondition: No boundaries (clusters, gas phase)
- MixedBoundaryCondition: Per-dimension settings (surfaces, slabs)
"""

from .boundary_condition import BoundaryCondition
from .mixed_bc import MixedBoundaryCondition
from .open_bc import OpenBoundaryCondition
from .periodic_bc import PeriodicBoundaryCondition

__all__ = [
    "BoundaryCondition",
    "PeriodicBoundaryCondition",
    "OpenBoundaryCondition",
    "MixedBoundaryCondition",
]
