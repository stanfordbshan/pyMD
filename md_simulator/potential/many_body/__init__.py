"""
Many-body potentials module.

Complex potentials like EAM, MEAM, and ADP for metallic systems.
"""

from .eam_base import EAMBasePotential, SuttonChenEAM

__all__ = [
    "EAMBasePotential",
    "SuttonChenEAM",
]
