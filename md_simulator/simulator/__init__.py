"""
Simulator module for molecular dynamics simulations.

Provides the main simulation driver:
- Simulator: Orchestrates the MD loop
- SimulatorBuilder: Builder pattern for construction
"""

from .simulator import Simulator, SimulatorBuilder

__all__ = [
    "Simulator",
    "SimulatorBuilder",
]
