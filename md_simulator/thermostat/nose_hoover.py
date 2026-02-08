"""
Nose-Hoover thermostat implementation.

Extended system thermostat that correctly samples the canonical (NVT)
ensemble. Uses an additional degree of freedom to couple the system
to a heat bath.
"""
from typing import TYPE_CHECKING

import numpy as np

from .thermostat import Thermostat

if TYPE_CHECKING:
    from md_simulator.core import System


class NoseHooverThermostat(Thermostat):
    """
    Nose-Hoover thermostat for canonical (NVT) ensemble.

    Uses an extended system with an additional heat bath variable xi
    that couples to the kinetic energy:
        dxi/dt = (2*KE - N_dof*kB*T) / Q

    Properties:
        + Correctly samples canonical ensemble
        + Proper energy fluctuations
        - Slower equilibration than Berendsen
        - Requires careful choice of Q

    The thermal mass Q controls coupling strength:
        Q ~ N_dof * kB * T * tau^2

    Attributes:
        tau: Characteristic time scale.
        xi: Heat bath variable (friction coefficient).
        Q: Thermal mass (computed from tau).

    Example:
        >>> thermostat = NoseHooverThermostat(target_temp=300.0, tau=0.5)
        >>> thermostat.apply(system, dt)
    """

    def __init__(self, target_temperature: float, tau: float) -> None:
        """
        Initialize Nose-Hoover thermostat.

        Args:
            target_temperature: Target temperature.
            tau: Characteristic oscillation period.
                 Typical: 20*dt to 200*dt.
        """
        super().__init__(target_temperature)
        if tau <= 0:
            raise ValueError(f"Tau must be positive, got {tau}")
        self.tau = tau
        self.xi = 0.0  # Heat bath variable (starts at equilibrium)
        self._Q_initialized = False
        self._Q = 1.0  # Will be set on first apply

    def _initialize_Q(self, system: "System") -> None:
        """Initialize thermal mass Q based on system properties."""
        n_dof = 3 * system.get_num_atoms() - 3  # Remove COM motion
        kB = system.units.boltzmann
        # Q = N_dof * kB * T * tau^2
        self._Q = n_dof * kB * self.target_temperature * self.tau ** 2
        self._Q_initialized = True

    def apply(self, system: "System", dt: float) -> None:
        """
        Apply Nose-Hoover thermostat (velocity rescaling + xi update).

        Uses a simple integration scheme:
        1. Update xi from kinetic energy mismatch
        2. Rescale velocities by exp(-xi * dt)

        Args:
            system: The molecular system.
            dt: Current time step.
        """
        if not self._Q_initialized:
            self._initialize_Q(system)

        n_dof = 3 * system.get_num_atoms() - 3
        kB = system.units.boltzmann

        # Current kinetic energy
        kinetic_energy = system.compute_kinetic_energy()

        # Target kinetic energy for NVT: KE_target = (1/2) * N_dof * kB * T
        target_ke = 0.5 * n_dof * kB * self.target_temperature

        # Update heat bath variable: dxi/dt = (2*KE - N_dof*kB*T) / Q
        # Using simple Euler: xi(t+dt) = xi(t) + dxi/dt * dt
        dxi_dt = (2.0 * kinetic_energy - n_dof * kB * self.target_temperature) / self._Q
        self.xi += dxi_dt * dt

        # Rescale velocities: v_new = v * exp(-xi * dt)
        # For small xi*dt, this is approximately v * (1 - xi * dt)
        scale_factor = np.exp(-self.xi * dt)
        system.state.velocities *= scale_factor

    def get_name(self) -> str:
        """Return thermostat name with parameters."""
        return f"NoseHoover(T={self.target_temperature}, tau={self.tau})"

    def reset(self) -> None:
        """Reset heat bath variable to zero."""
        self.xi = 0.0
