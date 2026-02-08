"""
Main Simulator class that orchestrates the MD simulation.

Brings together System, Integrator, Thermostat, ForceCalculator,
and Observers into a cohesive simulation loop.
"""
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from md_simulator.core import System
    from md_simulator.force import ForceCalculator
    from md_simulator.integrator import Integrator
    from md_simulator.observer import Observer
    from md_simulator.thermostat import Thermostat


class Simulator:
    """
    Main MD simulation driver.

    Orchestrates the simulation loop:
    1. Initialize forces
    2. For each step:
       a. Integrate equations of motion
       b. Apply thermostat
       c. Notify observers
    3. Finalize observers

    Example:
        >>> sim = Simulator(
        ...     system=system,
        ...     integrator=VelocityVerlet(dt=0.001),
        ...     force_calculator=ForceCalculator(potential, backend),
        ...     thermostat=BerendsenThermostat(300.0, tau=0.1),
        ...     observers=[EnergyObserver(), PrintObserver(interval=100)]
        ... )
        >>> sim.run(num_steps=10000)
    """

    def __init__(
        self,
        system: "System",
        integrator: "Integrator",
        force_calculator: "ForceCalculator",
        thermostat: Optional["Thermostat"] = None,
        observers: Optional[List["Observer"]] = None,
    ) -> None:
        """
        Initialize simulator.

        Args:
            system: The molecular system to simulate.
            integrator: Time integration algorithm.
            force_calculator: For computing forces from potential.
            thermostat: Temperature control (optional, defaults to NVE).
            observers: List of observers (optional).
        """
        self.system = system
        self.integrator = integrator
        self.force_calculator = force_calculator
        self.thermostat = thermostat
        self.observers = observers or []

        # Track simulation statistics
        self._initialized = False
        self._total_steps_run = 0

    def initialize(self) -> None:
        """
        Initialize the simulation (compute initial forces).

        Called automatically by run() if not already done.
        """
        if not self._initialized:
            self.integrator.initialize(self.system, self.force_calculator)
            self._initialized = True

    def run(self, num_steps: int) -> None:
        """
        Run the simulation for a given number of steps.

        Args:
            num_steps: Number of time steps to run.
        """
        # Ensure initialized
        self.initialize()

        # Main simulation loop
        for _ in range(num_steps):
            current_step = self.system.state.step

            # 1. Integrate equations of motion (updates positions, velocities)
            potential_energy = self.integrator.step(
                self.system, self.force_calculator
            )

            # 2. Apply thermostat (modifies velocities)
            if self.thermostat is not None:
                self.thermostat.apply(self.system, self.integrator.dt)

            # 3. Notify observers
            for observer in self.observers:
                if current_step % observer.interval == 0:
                    observer.observe(
                        self.system,
                        current_step,
                        potential_energy,
                    )

            self._total_steps_run += 1

        # Finalize observers
        for observer in self.observers:
            observer.finalize()

    def run_until(self, target_time: float) -> None:
        """
        Run simulation until a target simulation time.

        Args:
            target_time: Target time in simulation units.
        """
        dt = self.integrator.dt
        current_time = self.system.state.time
        remaining_time = target_time - current_time
        if remaining_time <= 0:
            return

        num_steps = int(remaining_time / dt)
        self.run(num_steps)

    def get_total_steps(self) -> int:
        """Return total steps run so far."""
        return self._total_steps_run

    def reset(self) -> None:
        """Reset simulator state (not system state)."""
        self._initialized = False
        self._total_steps_run = 0


class SimulatorBuilder:
    """
    Builder pattern for constructing Simulator instances.

    Example:
        >>> sim = (SimulatorBuilder()
        ...     .with_system(system)
        ...     .with_integrator(VelocityVerlet(dt=0.001))
        ...     .with_force_calculator(force_calc)
        ...     .with_thermostat(BerendsenThermostat(300.0, 0.1))
        ...     .add_observer(EnergyObserver())
        ...     .build())
    """

    def __init__(self) -> None:
        """Initialize builder."""
        self._system: Optional["System"] = None
        self._integrator: Optional["Integrator"] = None
        self._force_calculator: Optional["ForceCalculator"] = None
        self._thermostat: Optional["Thermostat"] = None
        self._observers: List["Observer"] = []

    def with_system(self, system: "System") -> "SimulatorBuilder":
        """Set the system."""
        self._system = system
        return self

    def with_integrator(self, integrator: "Integrator") -> "SimulatorBuilder":
        """Set the integrator."""
        self._integrator = integrator
        return self

    def with_force_calculator(
        self, force_calculator: "ForceCalculator"
    ) -> "SimulatorBuilder":
        """Set the force calculator."""
        self._force_calculator = force_calculator
        return self

    def with_thermostat(self, thermostat: "Thermostat") -> "SimulatorBuilder":
        """Set the thermostat."""
        self._thermostat = thermostat
        return self

    def add_observer(self, observer: "Observer") -> "SimulatorBuilder":
        """Add an observer."""
        self._observers.append(observer)
        return self

    def build(self) -> Simulator:
        """
        Build the simulator.

        Returns:
            Configured Simulator instance.

        Raises:
            ValueError: If required components are missing.
        """
        if self._system is None:
            raise ValueError("System is required")
        if self._integrator is None:
            raise ValueError("Integrator is required")
        if self._force_calculator is None:
            raise ValueError("ForceCalculator is required")

        return Simulator(
            system=self._system,
            integrator=self._integrator,
            force_calculator=self._force_calculator,
            thermostat=self._thermostat,
            observers=self._observers,
        )
