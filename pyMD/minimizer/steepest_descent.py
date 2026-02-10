"""
Steepest Descent energy minimizer.

The simplest minimization algorithm: moves along the force direction
(negative gradient) at each step with backtracking line search.
"""
from typing import TYPE_CHECKING

import numpy as np

from .minimizer import Minimizer

if TYPE_CHECKING:
    from pyMD.core import System
    from pyMD.force import ForceCalculator


class SteepestDescent(Minimizer):
    """
    Steepest Descent minimizer.

    At each step, moves atoms along the force direction (steepest descent
    of the energy surface). Uses backtracking line search to ensure
    energy decrease.

    Attributes:
        initial_step_size: Starting step size for line search.
        max_step_size: Maximum allowed step size.

    Example:
        >>> from pyMD.minimizer import SteepestDescent
        >>> minimizer = SteepestDescent(force_tol=1e-4, initial_step_size=0.01)
        >>> result = minimizer.minimize(system, force_calculator)
    """

    def __init__(
        self,
        force_tol: float = 1e-4,
        energy_tol: float = 1e-8,
        max_steps: int = 10000,
        initial_step_size: float = 0.01,
        max_step_size: float = 0.1,
    ) -> None:
        """
        Initialize Steepest Descent minimizer.

        Args:
            force_tol: Convergence threshold for max force component.
            energy_tol: Convergence threshold for energy change.
            max_steps: Maximum number of minimization steps.
            initial_step_size: Starting step size for line search.
            max_step_size: Maximum allowed step size.

        Raises:
            ValueError: If step size parameters are non-positive.
        """
        super().__init__(force_tol=force_tol, energy_tol=energy_tol, max_steps=max_steps)

        if initial_step_size <= 0:
            raise ValueError(
                f"initial_step_size must be positive, got {initial_step_size}"
            )
        if max_step_size <= 0:
            raise ValueError(f"max_step_size must be positive, got {max_step_size}")

        self._step_size = initial_step_size
        self._max_step_size = max_step_size

    def _step(
        self,
        system: "System",
        force_calculator: "ForceCalculator",
        forces: np.ndarray,
        energy: float,
    ) -> tuple:
        """
        Perform one steepest descent step.

        Search direction is the normalized force vector.
        Uses backtracking line search to ensure energy decrease.
        """
        # Search direction = forces (normalized)
        direction = forces.copy()
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction /= norm

        # Save original positions
        original_positions = system.state.positions.copy()

        # Backtracking line search
        step_size = self._step_size
        backtracked = False
        for _ in range(10):
            system.state.positions = original_positions + step_size * direction
            system.wrap_positions()
            new_forces, new_energy = force_calculator.compute_forces_and_energy(system)

            if new_energy <= energy:
                break
            # Energy increased — halve step size
            step_size *= 0.5
            backtracked = True
        else:
            # All line search attempts failed; accept last position
            pass

        # Adaptive step size: grow if no backtracking, already shrunk if backtracked
        if not backtracked:
            self._step_size = min(step_size * 1.2, self._max_step_size)
        else:
            self._step_size = step_size

        return new_forces, new_energy

    def get_name(self) -> str:
        """Get human-readable name."""
        return f"SteepestDescent(force_tol={self.force_tol}, max_steps={self.max_steps})"
