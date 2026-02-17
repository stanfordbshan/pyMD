"""
Velocity Verlet integrator implementation.

The most common integrator for molecular dynamics simulations.
Time-reversible and symplectic with good energy conservation.
"""
from typing import TYPE_CHECKING

import numpy as np

from .integrator import Integrator

if TYPE_CHECKING:
    from pyMD.core import System
    from pyMD.force import ForceCalculator


class VelocityVerlet(Integrator):
    """
    Velocity Verlet integrator.

    Algorithm (for each time step dt):
        1. v(t + dt/2) = v(t) + (1/2) * a(t) * dt
        2. r(t + dt) = r(t) + v(t + dt/2) * dt
        3. Compute a(t + dt) from new positions
        4. v(t + dt) = v(t + dt/2) + (1/2) * a(t + dt) * dt

    Properties:
        - Time-reversible
        - Symplectic (preserves phase space volume)
        - Second-order accurate in positions
        - Good long-term energy conservation

    Example:
        >>> from pyMD.integrator import VelocityVerlet
        >>> integrator = VelocityVerlet(dt=0.001)
        >>> for step in range(1000):
        ...     potential_energy = integrator.step(system, force_calc)
    """

    def step(
        self,
        system: "System",
        force_calculator: "ForceCalculator",
    ) -> float:
        """
        Advance system by one Velocity Verlet step.

        Args:
            system: The molecular system to integrate.
            force_calculator: For computing forces.

        Returns:
            Potential energy after the step.
        """
        dt = self.dt
        state = system.state
        masses = system.get_masses()

        # Get current accelerations: a = F / m
        # masses is (N,), forces is (N, 3), need broadcasting
        accelerations = state.forces / masses[:, np.newaxis]

        # Step 1: Half-step velocity update
        # v(t + dt/2) = v(t) + 0.5 * a(t) * dt
        state.velocities = state.velocities + 0.5 * accelerations * dt

        # Step 2: Full-step position update
        # r(t + dt) = r(t) + v(t + dt/2) * dt
        state.positions = state.positions + state.velocities * dt

        # Apply periodic boundary wrapping
        system.wrap_positions()

        # Step 3: Compute new forces at r(t + dt)
        forces, potential_energy = force_calculator.compute_forces_and_energy(system)
        state.forces = forces

        # Step 4: Second half-step velocity update
        # v(t + dt) = v(t + dt/2) + 0.5 * a(t + dt) * dt
        new_accelerations = forces / masses[:, np.newaxis]
        state.velocities = state.velocities + 0.5 * new_accelerations * dt

        # Update time and step count
        state.time += dt
        state.step += 1

        return potential_energy

    def get_name(self) -> str:
        """Return integrator name with timestep."""
        return f"VelocityVerlet(dt={self.dt})"
