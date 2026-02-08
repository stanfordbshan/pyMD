"""
Unit tests for integrator module.
"""
import numpy as np
import pytest

from md_simulator.core import Atom, State, System, Units
from md_simulator.boundary import PeriodicBoundaryCondition
from md_simulator.integrator import Integrator, VelocityVerlet


class MockForceCalculator:
    """Mock force calculator for testing."""
    
    def __init__(self, constant_force: float = 0.0):
        self.constant_force = constant_force
        
    def compute_forces(self, system: System) -> np.ndarray:
        """Return constant forces."""
        n_atoms = system.get_num_atoms()
        return np.full((n_atoms, 3), self.constant_force)
    
    def compute_forces_and_energy(self, system: System) -> tuple:
        """Return forces and energy."""
        forces = self.compute_forces(system)
        return forces, 0.0


class TestVelocityVerlet:
    """Tests for VelocityVerlet integrator."""

    @pytest.fixture
    def simple_system(self) -> System:
        """Create a simple single-atom system."""
        atoms = [Atom(mass=1.0, atom_type="Ar")]
        positions = np.array([[0.0, 0.0, 0.0]])
        velocities = np.array([[1.0, 0.0, 0.0]])
        box = np.array([10.0, 10.0, 10.0])
        state = State(
            positions=positions,
            velocities=velocities,
            forces=np.zeros((1, 3)),
            box=box,
        )
        bc = PeriodicBoundaryCondition()
        units = Units.LJ()
        return System(atoms, state, bc, units)

    def test_invalid_timestep(self) -> None:
        """Test that negative timestep raises error."""
        with pytest.raises(ValueError):
            VelocityVerlet(dt=-0.001)

    def test_position_update_no_force(self, simple_system: System) -> None:
        """Test position update with zero force."""
        dt = 0.1
        integrator = VelocityVerlet(dt=dt)
        force_calc = MockForceCalculator(constant_force=0.0)
        
        initial_pos = simple_system.state.positions.copy()
        initial_vel = simple_system.state.velocities.copy()
        
        integrator.step(simple_system, force_calc)
        
        # With v=1.0 in x and no force: x_new = x + v*dt = 0 + 1*0.1 = 0.1
        expected_pos = initial_pos + initial_vel * dt
        np.testing.assert_array_almost_equal(
            simple_system.state.positions, expected_pos, decimal=10
        )

    def test_velocity_update_with_force(self, simple_system: System) -> None:
        """Test velocity update with constant force."""
        dt = 0.1
        integrator = VelocityVerlet(dt=dt)
        force = 2.0  # Constant force
        force_calc = MockForceCalculator(constant_force=force)
        
        # Set initial forces so acceleration is known
        simple_system.state.forces = np.full((1, 3), force)
        
        initial_vel = simple_system.state.velocities.copy()
        
        integrator.step(simple_system, force_calc)
        
        # v_new = v + a*dt where a = F/m = 2.0/1.0 = 2.0
        # But VV does half step before and after, so:
        # v_half = v + 0.5*a*dt = 1.0 + 0.5*2.0*0.1 = 1.1
        # After step, a_new = 2.0 again
        # v_new = v_half + 0.5*a_new*dt = 1.1 + 0.5*2.0*0.1 = 1.2
        expected_vel_x = 1.0 + 2.0 * dt  # = 1.2
        assert pytest.approx(simple_system.state.velocities[0, 0]) == expected_vel_x

    def test_step_count_increments(self, simple_system: System) -> None:
        """Test that step count increments."""
        integrator = VelocityVerlet(dt=0.1)
        force_calc = MockForceCalculator()
        
        assert simple_system.state.step == 0
        integrator.step(simple_system, force_calc)
        assert simple_system.state.step == 1
        integrator.step(simple_system, force_calc)
        assert simple_system.state.step == 2

    def test_time_advances(self, simple_system: System) -> None:
        """Test that time advances by dt."""
        dt = 0.1
        integrator = VelocityVerlet(dt=dt)
        force_calc = MockForceCalculator()
        
        assert simple_system.state.time == 0.0
        integrator.step(simple_system, force_calc)
        assert pytest.approx(simple_system.state.time) == dt

    def test_get_name(self) -> None:
        """Test integrator name."""
        integrator = VelocityVerlet(dt=0.001)
        assert "VelocityVerlet" in integrator.get_name()
        assert "0.001" in integrator.get_name()


class TestIntegratorInterface:
    """Test integrator interface compliance."""

    def test_is_integrator(self) -> None:
        """Test inheritance."""
        integrator = VelocityVerlet(dt=0.001)
        assert isinstance(integrator, Integrator)

    def test_has_required_methods(self) -> None:
        """Test required methods exist."""
        integrator = VelocityVerlet(dt=0.001)
        assert hasattr(integrator, "step")
        assert hasattr(integrator, "get_name")
        assert hasattr(integrator, "initialize")
