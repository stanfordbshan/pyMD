"""
No thermostat (NVE ensemble).

Simply passes through without modifying velocities.
Conserves total energy in an isolated system.
"""
from typing import TYPE_CHECKING

from .thermostat import Thermostat

if TYPE_CHECKING:
    from pyMD.core import System


class NoThermostat(Thermostat):
    """
    No thermostat - NVE (microcanonical) ensemble.

    Does not modify velocities. The system conserves total energy,
    and temperature fluctuates naturally.

    Use this for:
        - Energy conservation tests
        - True microcanonical simulations
        - Equilibration checks

    Example:
        >>> from pyMD.thermostat import NoThermostat
        >>> thermostat = NoThermostat()  # No target temperature needed
        >>> thermostat.apply(system, dt)  # Does nothing
    """

    def __init__(self) -> None:
        """Initialize NVE thermostat (no target temperature)."""
        # Use 0.0 as placeholder - not actually used
        super().__init__(target_temperature=0.0)

    def apply(self, system: "System", dt: float) -> None:
        """
        Do nothing - NVE conserves energy naturally.

        Args:
            system: The molecular system (unchanged).
            dt: Time step (ignored).
        """
        pass  # No velocity rescaling

    def get_name(self) -> str:
        """Return thermostat name."""
        return "NVE (No Thermostat)"
