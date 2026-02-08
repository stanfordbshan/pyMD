"""
Abstract base class for thermostats.

This module provides the Thermostat ABC that defines the interface
for temperature control algorithms (NVE, Berendsen, Nose-Hoover, etc.).
"""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from md_simulator.core import System


class Thermostat(ABC):
    """
    Abstract base for thermostats (Strategy Pattern).

    Thermostats control the temperature of the simulation by modifying
    velocities. Different thermostats sample different ensembles:
    - NoThermostat: NVE (constant energy)
    - Berendsen: Weak coupling, not proper NVT
    - NoseHoover: Proper canonical (NVT) ensemble

    Attributes:
        target_temperature: Target temperature in simulation units.

    Example:
        >>> from md_simulator.thermostat import BerendsenThermostat
        >>> thermostat = BerendsenThermostat(target_temp=300.0, tau=0.1)
        >>> thermostat.apply(system, dt)
    """

    def __init__(self, target_temperature: float) -> None:
        """
        Initialize thermostat.

        Args:
            target_temperature: Target temperature in simulation units.
        """
        if target_temperature < 0:
            raise ValueError(
                f"Target temperature must be non-negative, got {target_temperature}"
            )
        self.target_temperature = target_temperature

    @abstractmethod
    def apply(self, system: "System", dt: float) -> None:
        """
        Apply thermostat to the system.

        Modifies system.state.velocities to control temperature.

        Args:
            system: The molecular system.
            dt: Current time step size.
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get human-readable name of this thermostat."""
        pass

    def get_kinetic_temperature(self, system: "System") -> float:
        """
        Get current kinetic temperature of the system.

        Args:
            system: The molecular system.

        Returns:
            Current temperature.
        """
        return system.compute_temperature()
