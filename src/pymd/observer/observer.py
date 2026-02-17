"""
Observer module for monitoring simulation progress.

Provides the Observer pattern for logging, trajectory output,
and property calculations during simulation.
"""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from pymd.core import System


class Observer(ABC):
    """
    Abstract base for simulation observers (Observer Pattern).

    Observers are notified at each step to record properties,
    write trajectories, or perform analysis.

    Attributes:
        interval: How often to call observe() (in steps).

    Example:
        >>> observer = EnergyObserver(interval=100)
        >>> if step % observer.interval == 0:
        ...     observer.observe(system, step, potential_energy)
    """

    def __init__(self, interval: int = 1) -> None:
        """
        Initialize observer.

        Args:
            interval: Observation interval in steps. Default=1 (every step).
        """
        if interval < 1:
            raise ValueError(f"Interval must be >= 1, got {interval}")
        self.interval = interval

    @abstractmethod
    def observe(
        self,
        system: "System",
        step: int,
        potential_energy: float,
    ) -> None:
        """
        Record observation.

        Args:
            system: Current system state.
            step: Current step number.
            potential_energy: Current potential energy.
        """
        pass

    def finalize(self) -> None:
        """Called at end of simulation for cleanup."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get observer name."""
        pass


class CompositeObserver(Observer):
    """
    Composite observer that wraps multiple observers.

    Delegates to child observers based on their individual intervals.
    """

    def __init__(self, observers: List[Observer]) -> None:
        """
        Initialize with list of observers.

        Args:
            observers: List of Observer instances.
        """
        super().__init__(interval=1)  # Check every step
        self.observers = observers

    def observe(
        self,
        system: "System",
        step: int,
        potential_energy: float,
    ) -> None:
        """Delegate to child observers based on their intervals."""
        for obs in self.observers:
            if step % obs.interval == 0:
                obs.observe(system, step, potential_energy)

    def finalize(self) -> None:
        """Finalize all child observers."""
        for obs in self.observers:
            obs.finalize()

    def get_name(self) -> str:
        """Return composite name."""
        names = [o.get_name() for o in self.observers]
        return f"Composite[{', '.join(names)}]"


class EnergyObserver(Observer):
    """
    Records energy components during simulation.

    Tracks potential, kinetic, and total energy over time.
    """

    def __init__(self, interval: int = 1) -> None:
        """Initialize energy observer."""
        super().__init__(interval)
        self.steps: List[int] = []
        self.potential_energies: List[float] = []
        self.kinetic_energies: List[float] = []
        self.total_energies: List[float] = []
        self.temperatures: List[float] = []

    def observe(
        self,
        system: "System",
        step: int,
        potential_energy: float,
    ) -> None:
        """Record energy values."""
        kinetic = system.compute_kinetic_energy()
        temperature = system.compute_temperature()

        self.steps.append(step)
        self.potential_energies.append(potential_energy)
        self.kinetic_energies.append(kinetic)
        self.total_energies.append(potential_energy + kinetic)
        self.temperatures.append(temperature)

    def get_name(self) -> str:
        """Return observer name."""
        return f"EnergyObserver(interval={self.interval})"

    def get_energy_drift(self) -> float:
        """
        Compute relative energy drift.

        Returns:
            (E_final - E_initial) / |E_initial|
        """
        if len(self.total_energies) < 2:
            return 0.0
        E0 = self.total_energies[0]
        E_final = self.total_energies[-1]
        if abs(E0) < 1e-10:
            return 0.0
        return (E_final - E0) / abs(E0)


class TrajectoryObserver(Observer):
    """
    Records atomic positions over time for trajectory output.
    """

    def __init__(self, interval: int = 100) -> None:
        """Initialize trajectory observer."""
        super().__init__(interval)
        self.frames: List[dict] = []

    def observe(
        self,
        system: "System",
        step: int,
        potential_energy: float,
    ) -> None:
        """Record frame."""
        import numpy as np

        self.frames.append({
            "step": step,
            "time": system.state.time,
            "positions": system.state.positions.copy(),
            "velocities": system.state.velocities.copy(),
            "box": system.boundary_conditions.get_box_dimensions(
                np.array([10.0, 10.0, 10.0])  # Fallback
            ),
            "potential_energy": potential_energy,
        })

    def get_name(self) -> str:
        """Return observer name."""
        return f"TrajectoryObserver(interval={self.interval})"


class PrintObserver(Observer):
    """
    Prints simulation progress to console.
    """

    def __init__(self, interval: int = 100) -> None:
        """Initialize print observer."""
        super().__init__(interval)

    def observe(
        self,
        system: "System",
        step: int,
        potential_energy: float,
    ) -> None:
        """Print step info."""
        kinetic = system.compute_kinetic_energy()
        total = potential_energy + kinetic
        temperature = system.compute_temperature()

        print(
            f"Step {step:6d} | "
            f"T={temperature:8.2f} | "
            f"PE={potential_energy:12.4f} | "
            f"KE={kinetic:12.4f} | "
            f"E_total={total:12.4f}"
        )

    def get_name(self) -> str:
        """Return observer name."""
        return f"PrintObserver(interval={self.interval})"
