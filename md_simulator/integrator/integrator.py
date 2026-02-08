"""
Abstract base class for time integrators.

This module provides the Integrator ABC that defines the interface
for all integration algorithms (Velocity Verlet, Leapfrog, etc.).
"""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from md_simulator.core import System
    from md_simulator.force import ForceCalculator


class Integrator(ABC):
    """
    Abstract base for time integration algorithms (Strategy Pattern).

    Integrators advance the simulation by one timestep, updating
    positions and velocities according to Newton's equations.

    Attributes:
        dt: Time step size.

    Example:
        >>> from md_simulator.integrator import VelocityVerlet
        >>> integrator = VelocityVerlet(dt=0.001)
        >>> integrator.step(system, force_calculator)
    """

    def __init__(self, dt: float) -> None:
        """
        Initialize integrator.

        Args:
            dt: Time step size in simulation time units.
        """
        if dt <= 0:
            raise ValueError(f"Time step must be positive, got {dt}")
        self.dt = dt

    @abstractmethod
    def step(
        self,
        system: "System",
        force_calculator: "ForceCalculator",
    ) -> float:
        """
        Advance the system by one time step.

        Updates system.state.positions, velocities, forces, time, and step.

        Args:
            system: The molecular system to integrate.
            force_calculator: For computing forces from potential.

        Returns:
            Potential energy after the step.
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get human-readable name of this integrator."""
        pass

    def initialize(
        self,
        system: "System",
        force_calculator: "ForceCalculator",
    ) -> None:
        """
        Initialize the integrator (compute initial forces, etc.).

        Called once before the simulation starts.

        Args:
            system: The molecular system.
            force_calculator: For computing initial forces.
        """
        # Default: compute initial forces
        forces = force_calculator.compute_forces(system)
        system.state.forces = forces
