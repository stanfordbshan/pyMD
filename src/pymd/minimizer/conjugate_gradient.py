"""
Conjugate Gradient energy minimizer (Polak-Ribiere).

More efficient than steepest descent for most systems by using
conjugate search directions that avoid redundant exploration.
"""
from typing import TYPE_CHECKING

import numpy as np

from .minimizer import Minimizer

if TYPE_CHECKING:
    from pymd.core import System
    from pymd.force import ForceCalculator


class ConjugateGradient(Minimizer):
    """
    Conjugate Gradient minimizer using Polak-Ribiere formula.

    Builds on steepest descent by choosing search directions that are
    conjugate to previous directions, leading to faster convergence.
    Automatically restarts (resets to steepest descent) when beta <= 0
    or at periodic intervals.

    Attributes:
        initial_step_size: Starting step size for line search.
        max_step_size: Maximum allowed step size.
        restart_interval: Steps between forced restarts (0 = 3*N_atoms).

    Example:
        >>> from pymd.minimizer import ConjugateGradient
        >>> minimizer = ConjugateGradient(force_tol=1e-4)
        >>> result = minimizer.minimize(system, force_calculator)
    """

    def __init__(
        self,
        force_tol: float = 1e-4,
        energy_tol: float = 1e-8,
        max_steps: int = 10000,
        initial_step_size: float = 0.01,
        max_step_size: float = 0.1,
        restart_interval: int = 0,
    ) -> None:
        """
        Initialize Conjugate Gradient minimizer.

        Args:
            force_tol: Convergence threshold for max force component.
            energy_tol: Convergence threshold for energy change.
            max_steps: Maximum number of minimization steps.
            initial_step_size: Starting step size for line search.
            max_step_size: Maximum allowed step size.
            restart_interval: Steps between forced CG restarts.
                0 means automatic (3 * N_atoms).

        Raises:
            ValueError: If step size parameters are non-positive or
                restart_interval is negative.
        """
        super().__init__(force_tol=force_tol, energy_tol=energy_tol, max_steps=max_steps)

        if initial_step_size <= 0:
            raise ValueError(
                f"initial_step_size must be positive, got {initial_step_size}"
            )
        if max_step_size <= 0:
            raise ValueError(f"max_step_size must be positive, got {max_step_size}")
        if restart_interval < 0:
            raise ValueError(
                f"restart_interval must be non-negative, got {restart_interval}"
            )

        self._step_size = initial_step_size
        self._max_step_size = max_step_size
        self._restart_interval = restart_interval

        # Internal state set during _initialize
        self._search_direction: np.ndarray = np.array([])
        self._old_forces: np.ndarray = np.array([])
        self._step_count: int = 0
        self._actual_restart_interval: int = 0

    def _initialize(
        self,
        system: "System",
        forces: np.ndarray,
        energy: float,
    ) -> None:
        """Initialize CG with first search direction = forces."""
        self._search_direction = forces.copy()
        self._old_forces = forces.copy()
        self._step_count = 0
        n_atoms = system.get_num_atoms()
        self._actual_restart_interval = (
            self._restart_interval if self._restart_interval > 0 else 3 * n_atoms
        )

    def _step(
        self,
        system: "System",
        force_calculator: "ForceCalculator",
        forces: np.ndarray,
        energy: float,
    ) -> tuple:
        """
        Perform one conjugate gradient step.

        Uses Polak-Ribiere formula for beta with automatic restart.
        """
        self._step_count += 1

        # Normalize search direction
        direction = self._search_direction.copy()
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
            step_size *= 0.5
            backtracked = True
        else:
            pass

        # Adaptive step size
        if not backtracked:
            self._step_size = min(step_size * 1.2, self._max_step_size)
        else:
            self._step_size = step_size

        # Polak-Ribiere beta
        f_old_flat = self._old_forces.ravel()
        f_new_flat = new_forces.ravel()
        old_dot_old = np.dot(f_old_flat, f_old_flat)

        if old_dot_old > 0:
            beta = np.dot(f_new_flat, f_new_flat - f_old_flat) / old_dot_old
            beta = max(0.0, beta)  # Restart if beta < 0
        else:
            beta = 0.0

        # Periodic restart
        if self._step_count % self._actual_restart_interval == 0:
            beta = 0.0

        # Update search direction
        self._search_direction = new_forces + beta * self._search_direction
        self._old_forces = new_forces.copy()

        return new_forces, new_energy

    def get_name(self) -> str:
        """Get human-readable name."""
        return (
            f"ConjugateGradient(force_tol={self.force_tol}, "
            f"max_steps={self.max_steps})"
        )
