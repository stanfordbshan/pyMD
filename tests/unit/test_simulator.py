"""
Unit tests for observer and simulator modules.
"""
import numpy as np
import pytest

from md_simulator.core import Atom, State, System, Units
from md_simulator.boundary import PeriodicBoundaryCondition
from md_simulator.integrator import VelocityVerlet
from md_simulator.observer import (
    CompositeObserver,
    EnergyObserver,
    Observer,
    PrintObserver,
)
from md_simulator.simulator import Simulator, SimulatorBuilder
from md_simulator.thermostat import NoThermostat


class MockForceCalculator:
    """Mock force calculator for testing."""
    
    def compute_forces(self, system: System) -> np.ndarray:
        """Return zero forces."""
        n_atoms = system.get_num_atoms()
        return np.zeros((n_atoms, 3))
    
    def compute_forces_and_energy(self, system: System) -> tuple:
        """Return forces and energy."""
        return self.compute_forces(system), -1.0  # Constant PE


@pytest.fixture
def simple_system() -> System:
    """Create a simple system for testing."""
    atoms = [Atom(mass=1.0, atom_type="Ar") for _ in range(5)]
    np.random.seed(123)
    positions = np.random.rand(5, 3) * 10.0
    velocities = np.random.randn(5, 3) * 0.1
    box = np.array([10.0, 10.0, 10.0])
    state = State(
        positions=positions,
        velocities=velocities,
        forces=np.zeros((5, 3)),
        box=box,
    )
    bc = PeriodicBoundaryCondition()
    units = Units.LJ()
    return System(atoms, state, bc, units)


class TestEnergyObserver:
    """Tests for EnergyObserver."""

    def test_records_energy(self, simple_system: System) -> None:
        """Test that energy is recorded."""
        observer = EnergyObserver(interval=1)
        
        observer.observe(simple_system, step=0, potential_energy=-1.0)
        observer.observe(simple_system, step=1, potential_energy=-0.9)
        
        assert len(observer.steps) == 2
        assert len(observer.potential_energies) == 2
        assert observer.potential_energies[0] == -1.0

    def test_calculates_kinetic_energy(self, simple_system: System) -> None:
        """Test that kinetic energy is calculated."""
        observer = EnergyObserver()
        observer.observe(simple_system, step=0, potential_energy=0.0)
        
        assert len(observer.kinetic_energies) == 1
        assert observer.kinetic_energies[0] >= 0

    def test_energy_drift(self, simple_system: System) -> None:
        """Test energy drift calculation."""
        observer = EnergyObserver()
        
        # Mock some total energies
        observer.total_energies = [-10.0, -10.01]
        
        drift = observer.get_energy_drift()
        # (-10.01 - (-10.0)) / |-10.0| = -0.001
        assert pytest.approx(drift, rel=0.01) == -0.001


class TestCompositeObserver:
    """Tests for CompositeObserver."""

    def test_delegates_to_children(self, simple_system: System) -> None:
        """Test delegation to child observers."""
        obs1 = EnergyObserver(interval=1)
        obs2 = EnergyObserver(interval=2)
        composite = CompositeObserver([obs1, obs2])
        
        composite.observe(simple_system, step=0, potential_energy=0.0)
        composite.observe(simple_system, step=1, potential_energy=0.0)
        composite.observe(simple_system, step=2, potential_energy=0.0)
        
        # obs1 observes all, obs2 only even steps
        assert len(obs1.steps) == 3
        assert len(obs2.steps) == 2  # steps 0 and 2


class TestSimulator:
    """Tests for Simulator."""

    def test_run_advances_steps(self, simple_system: System) -> None:
        """Test that run advances step count."""
        integrator = VelocityVerlet(dt=0.001)
        force_calc = MockForceCalculator()
        
        sim = Simulator(
            system=simple_system,
            integrator=integrator,
            force_calculator=force_calc,
        )
        
        initial_step = simple_system.state.step
        sim.run(num_steps=10)
        
        assert simple_system.state.step == initial_step + 10

    def test_observers_called(self, simple_system: System) -> None:
        """Test that observers are called during simulation."""
        integrator = VelocityVerlet(dt=0.001)
        force_calc = MockForceCalculator()
        observer = EnergyObserver(interval=1)
        
        sim = Simulator(
            system=simple_system,
            integrator=integrator,
            force_calculator=force_calc,
            observers=[observer],
        )
        
        sim.run(num_steps=5)
        
        # Should have observations for steps 0-4
        assert len(observer.steps) == 5

    def test_thermostat_applied(self, simple_system: System) -> None:
        """Test that thermostat is applied."""
        integrator = VelocityVerlet(dt=0.001)
        force_calc = MockForceCalculator()
        thermostat = NoThermostat()  # Won't change anything but tests the path
        
        sim = Simulator(
            system=simple_system,
            integrator=integrator,
            force_calculator=force_calc,
            thermostat=thermostat,
        )
        
        # Should not raise
        sim.run(num_steps=5)


class TestSimulatorBuilder:
    """Tests for SimulatorBuilder."""

    def test_build_requires_system(self) -> None:
        """Test that build requires system."""
        builder = SimulatorBuilder()
        builder.with_integrator(VelocityVerlet(dt=0.001))
        builder.with_force_calculator(MockForceCalculator())
        
        with pytest.raises(ValueError, match="System"):
            builder.build()

    def test_build_requires_integrator(self, simple_system: System) -> None:
        """Test that build requires integrator."""
        builder = SimulatorBuilder()
        builder.with_system(simple_system)
        builder.with_force_calculator(MockForceCalculator())
        
        with pytest.raises(ValueError, match="Integrator"):
            builder.build()

    def test_fluent_api(self, simple_system: System) -> None:
        """Test fluent API works."""
        observer = EnergyObserver()
        
        sim = (
            SimulatorBuilder()
            .with_system(simple_system)
            .with_integrator(VelocityVerlet(dt=0.001))
            .with_force_calculator(MockForceCalculator())
            .with_thermostat(NoThermostat())
            .add_observer(observer)
            .build()
        )
        
        assert isinstance(sim, Simulator)


class TestObserverInterface:
    """Test observer interface compliance."""

    @pytest.mark.parametrize("observer_class,args", [
        (EnergyObserver, ()),
        (PrintObserver, (100,)),
    ])
    def test_is_observer(
        self, observer_class: type, args: tuple
    ) -> None:
        """Test inheritance."""
        obs = observer_class(*args)
        assert isinstance(obs, Observer)
