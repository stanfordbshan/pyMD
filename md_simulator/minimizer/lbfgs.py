"""
L-BFGS energy minimizer.

Limited-memory Broyden-Fletcher-Goldfarb-Shanno algorithm. Approximates
the inverse Hessian using a history of position and gradient changes,
providing quasi-Newton convergence with minimal memory overhead.
"""
from collections import deque
from typing import TYPE_CHECKING

import numpy as np

from .minimizer import Minimizer

if TYPE_CHECKING:
    from md_simulator.core import System
    from md_simulator.force import ForceCalculator


class LBFGS(Minimizer):
    """
    L-BFGS minimizer using two-loop recursion.

    Stores a limited history of position and gradient changes to
    approximate the inverse Hessian matrix. More efficient than CG
    for many problems, especially near the minimum.

    Attributes:
        memory: Number of correction pairs to store.
        initial_step_size: Starting step size for line search.
        max_step_size: Maximum allowed step size.

    Example:
        >>> from md_simulator.minimizer import LBFGS
        >>> minimizer = LBFGS(force_tol=1e-4, memory=10)
        >>> result = minimizer.minimize(system, force_calculator)
    """

    def __init__(
        self,
        force_tol: float = 1e-4,
        energy_tol: float = 1e-8,
        max_steps: int = 10000,
        memory: int = 10,
        initial_step_size: float = 0.01,
        max_step_size: float = 0.1,
    ) -> None:
        """
        Initialize L-BFGS minimizer.

        Args:
            force_tol: Convergence threshold for max force component.
            energy_tol: Convergence threshold for energy change.
            max_steps: Maximum number of minimization steps.
            memory: Number of correction pairs to store.
            initial_step_size: Starting step size for line search.
            max_step_size: Maximum allowed step size.

        Raises:
            ValueError: If memory or step size parameters are non-positive.
        """
        super().__init__(force_tol=force_tol, energy_tol=energy_tol, max_steps=max_steps)

        if memory <= 0:
            raise ValueError(f"memory must be positive, got {memory}")
        if initial_step_size <= 0:
            raise ValueError(
                f"initial_step_size must be positive, got {initial_step_size}"
            )
        if max_step_size <= 0:
            raise ValueError(f"max_step_size must be positive, got {max_step_size}")

        self._memory = memory
        self._step_size = initial_step_size
        self._max_step_size = max_step_size

        # History of correction pairs (s_k, y_k)
        self._s_history: deque = deque(maxlen=memory)
        self._y_history: deque = deque(maxlen=memory)
        self._old_positions: np.ndarray = np.array([])
        self._old_gradient: np.ndarray = np.array([])

    def _initialize(
        self,
        system: "System",
        forces: np.ndarray,
        energy: float,
    ) -> None:
        """Initialize L-BFGS history."""
        self._s_history.clear()
        self._y_history.clear()
        self._old_positions = system.state.positions.ravel().copy()
        self._old_gradient = -forces.ravel().copy()  # gradient = -forces

    def _two_loop_recursion(self, gradient: np.ndarray) -> np.ndarray:
        """
        L-BFGS two-loop recursion to compute search direction.

        Computes H_k * gradient using stored correction pairs,
        where H_k is the approximate inverse Hessian.

        Args:
            gradient: Current gradient (flattened).

        Returns:
            Search direction (flattened), approximating H_k * gradient.
        """
        q = gradient.copy()
        m = len(self._s_history)

        if m == 0:
            # No history yet — use steepest descent direction
            return q

        alphas = np.zeros(m)
        rhos = np.zeros(m)

        # Forward pass (most recent to oldest)
        for i in range(m - 1, -1, -1):
            s_i = self._s_history[i]
            y_i = self._y_history[i]
            rho_i = 1.0 / np.dot(y_i, s_i)
            rhos[i] = rho_i
            alphas[i] = rho_i * np.dot(s_i, q)
            q = q - alphas[i] * y_i

        # Initial Hessian scaling
        s_latest = self._s_history[-1]
        y_latest = self._y_history[-1]
        gamma = np.dot(s_latest, y_latest) / np.dot(y_latest, y_latest)
        r = gamma * q

        # Backward pass (oldest to most recent)
        for i in range(m):
            s_i = self._s_history[i]
            y_i = self._y_history[i]
            beta = rhos[i] * np.dot(y_i, r)
            r = r + (alphas[i] - beta) * s_i

        return r

    def _step(
        self,
        system: "System",
        force_calculator: "ForceCalculator",
        forces: np.ndarray,
        energy: float,
    ) -> tuple:
        """
        Perform one L-BFGS step.

        Uses two-loop recursion to compute quasi-Newton search direction,
        then applies backtracking line search.
        """
        gradient = -forces.ravel()

        # Compute search direction via two-loop recursion
        # direction = -H_k * gradient (descent direction)
        hg = self._two_loop_recursion(gradient)
        direction = -hg.reshape(forces.shape)

        # Normalize direction
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

        # Update history
        new_positions = system.state.positions.ravel()
        new_gradient = -new_forces.ravel()
        s = new_positions - self._old_positions
        y = new_gradient - self._old_gradient

        # Only add to history if curvature condition holds
        if np.dot(s, y) > 0:
            self._s_history.append(s.copy())
            self._y_history.append(y.copy())

        self._old_positions = new_positions.copy()
        self._old_gradient = new_gradient.copy()

        return new_forces, new_energy

    def get_name(self) -> str:
        """Get human-readable name."""
        return (
            f"LBFGS(force_tol={self.force_tol}, max_steps={self.max_steps}, "
            f"memory={self._memory})"
        )
