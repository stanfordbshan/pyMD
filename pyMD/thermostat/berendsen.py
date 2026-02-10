"""
Berendsen thermostat implementation.

Simple velocity rescaling with exponential relaxation toward
target temperature. Fast equilibration but does not sample
the canonical ensemble correctly.
"""
from typing import TYPE_CHECKING

import numpy as np

from .thermostat import Thermostat

if TYPE_CHECKING:
    from pyMD.core import System


class BerendsenThermostat(Thermostat):
    """
    Berendsen (weak coupling) thermostat.

    Rescales velocities each step to drive temperature toward target:
        v_new = v * lambda
        lambda = sqrt(1 + (dt/tau) * (T_target/T_current - 1))

    Properties:
        + Fast equilibration
        + Simple and stable
        - Does NOT sample canonical ensemble
        - Suppresses energy fluctuations

    Use for: Equilibration, then switch to Nose-Hoover for production.

    Attributes:
        tau: Coupling time constant (larger = weaker coupling).

    Example:
        >>> thermostat = BerendsenThermostat(target_temp=300.0, tau=100*dt)
        >>> thermostat.apply(system, dt)
    """

    def __init__(self, target_temperature: float, tau: float) -> None:
        """
        Initialize Berendsen thermostat.

        Args:
            target_temperature: Target temperature.
            tau: Coupling time constant. Typical: 100*dt to 1000*dt.
                 Smaller tau = stronger coupling = faster equilibration.
        """
        super().__init__(target_temperature)
        if tau <= 0:
            raise ValueError(f"Coupling time tau must be positive, got {tau}")
        self.tau = tau

    def apply(self, system: "System", dt: float) -> None:
        """
        Apply Berendsen velocity rescaling.

        Args:
            system: The molecular system.
            dt: Current time step.
        """
        current_temp = system.compute_temperature()

        # Avoid division by zero for frozen systems
        if current_temp < 1e-10:
            return

        # Compute scaling factor
        # lambda^2 = 1 + (dt/tau) * (T_target/T_current - 1)
        ratio = self.target_temperature / current_temp
        lambda_sq = 1.0 + (dt / self.tau) * (ratio - 1.0)

        # Ensure lambda_sq is positive (can go negative if T >> T_target)
        if lambda_sq < 0:
            lambda_sq = 0.0

        lambda_factor = np.sqrt(lambda_sq)

        # Rescale velocities
        system.state.velocities *= lambda_factor

    def get_name(self) -> str:
        """Return thermostat name with parameters."""
        return f"Berendsen(T={self.target_temperature}, tau={self.tau})"
