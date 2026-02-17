"""
Abstract base class for energy minimizers.

This module provides the Minimizer ABC that defines the interface
for all energy minimization algorithms (Steepest Descent, Conjugate Gradient,
L-BFGS, etc.) and the MinimizationResult dataclass.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

import numpy as np

if TYPE_CHECKING:
    from pymd.core import System
    from pymd.force import ForceCalculator


@dataclass
class MinimizationResult:
    """
    Result of an energy minimization run.

    Attributes:
        converged: Whether the minimization converged within tolerances.
        n_steps: Number of minimization steps taken.
        initial_energy: Potential energy before minimization.
        final_energy: Potential energy after minimization.
        max_force: Maximum force component magnitude at final configuration.
        energy_history: List of potential energies at each step.
        message: Human-readable description of the outcome.
    """
    converged: bool
    n_steps: int
    initial_energy: float
    final_energy: float
    max_force: float
    energy_history: List[float] = field(default_factory=list)
    message: str = ""


class Minimizer(ABC):
    """
    Abstract base for energy minimization algorithms (Strategy + Template Method).

    Minimizers find a local energy minimum by iteratively adjusting atomic
    positions. The minimize() method implements the convergence loop (template),
    while subclasses provide the algorithm-specific _step() logic.

    This is standalone and does NOT use or extend Simulator/Integrator.

    Attributes:
        force_tol: Convergence threshold for maximum force component.
        energy_tol: Convergence threshold for energy change between steps.
        max_steps: Maximum number of minimization steps.

    Example:
        >>> from pymd.minimizer import ConjugateGradient
        >>> minimizer = ConjugateGradient(force_tol=1e-4)
        >>> result = minimizer.minimize(system, force_calculator)
        >>> print(result.converged, result.final_energy)
    """

    def __init__(
        self,
        force_tol: float = 1e-4,
        energy_tol: float = 1e-8,
        max_steps: int = 10000,
    ) -> None:
        """
        Initialize minimizer.

        Args:
            force_tol: Convergence threshold for max force component.
            energy_tol: Convergence threshold for energy change.
            max_steps: Maximum number of minimization steps.

        Raises:
            ValueError: If any parameter is non-positive.
        """
        if force_tol <= 0:
            raise ValueError(f"force_tol must be positive, got {force_tol}")
        if energy_tol <= 0:
            raise ValueError(f"energy_tol must be positive, got {energy_tol}")
        if max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {max_steps}")

        self.force_tol = force_tol
        self.energy_tol = energy_tol
        self.max_steps = max_steps

    def minimize(
        self,
        system: "System",
        force_calculator: "ForceCalculator",
    ) -> MinimizationResult:
        """
        Minimize the energy of the system (template method).

        Process:
        1. Compute initial forces and energy
        2. Check if already converged
        3. Call _initialize() for algorithm-specific setup
        4. Loop: _step() -> check convergence
        5. Zero velocities (minimized structure has no dynamics)
        6. Return MinimizationResult

        Args:
            system: The molecular system to minimize.
            force_calculator: For computing forces and energy.

        Returns:
            MinimizationResult with convergence info and energy history.
        """
        # Compute initial forces and energy
        forces, energy = force_calculator.compute_forces_and_energy(system)
        system.state.forces = forces
        energy_history = [energy]
        initial_energy = energy

        # Check if already converged
        max_force = float(np.max(np.abs(forces)))
        if max_force < self.force_tol:
            system.state.velocities = np.zeros_like(system.state.velocities)
            return MinimizationResult(
                converged=True,
                n_steps=0,
                initial_energy=initial_energy,
                final_energy=energy,
                max_force=max_force,
                energy_history=energy_history,
                message="Already converged (forces below tolerance)",
            )

        # Algorithm-specific initialization
        self._initialize(system, forces, energy)

        # Main minimization loop
        converged = False
        n_steps = 0
        for step in range(self.max_steps):
            n_steps += 1
            forces, energy = self._step(system, force_calculator, forces, energy)
            energy_history.append(energy)
            system.state.forces = forces

            # Check force convergence (primary)
            max_force = float(np.max(np.abs(forces)))
            if max_force < self.force_tol:
                converged = True
                break

            # Check energy convergence (secondary)
            if abs(energy_history[-1] - energy_history[-2]) < self.energy_tol:
                converged = True
                break

        # Zero velocities — minimized structure has no dynamics
        system.state.velocities = np.zeros_like(system.state.velocities)

        if converged:
            message = f"Converged after {n_steps} steps"
        else:
            message = f"Did not converge after {n_steps} steps (max_force={max_force:.2e})"

        return MinimizationResult(
            converged=converged,
            n_steps=n_steps,
            initial_energy=initial_energy,
            final_energy=energy,
            max_force=max_force,
            energy_history=energy_history,
            message=message,
        )

    @abstractmethod
    def _step(
        self,
        system: "System",
        force_calculator: "ForceCalculator",
        forces: np.ndarray,
        energy: float,
    ) -> tuple:
        """
        Perform one minimization step (algorithm-specific).

        Must update system.state.positions in place and call
        system.wrap_positions() after updating positions.

        Args:
            system: The molecular system.
            force_calculator: For computing forces and energy.
            forces: Current forces on all atoms.
            energy: Current potential energy.

        Returns:
            Tuple of (new_forces, new_energy).
        """
        pass

    def _initialize(
        self,
        system: "System",
        forces: np.ndarray,
        energy: float,
    ) -> None:
        """
        Optional algorithm-specific initialization.

        Called once before the minimization loop begins.
        Override in subclasses that need setup (e.g., CG search direction).

        Args:
            system: The molecular system.
            forces: Initial forces.
            energy: Initial potential energy.
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get human-readable name of this minimizer."""
        pass
