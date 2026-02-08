"""
Unit tests for thermostat module.
"""
import numpy as np
import pytest

from md_simulator.core import Atom, State, System, Units
from md_simulator.boundary import PeriodicBoundaryCondition
from md_simulator.thermostat import (
    BerendsenThermostat,
    NoThermostat,
    NoseHooverThermostat,
    Thermostat,
)


@pytest.fixture
def multi_atom_system() -> System:
    """Create a system with multiple atoms for testing."""
    atoms = [Atom(mass=1.0, atom_type="Ar") for _ in range(10)]
    np.random.seed(42)
    positions = np.random.rand(10, 3) * 10.0
    velocities = np.random.randn(10, 3)  # Random velocities
    box = np.array([10.0, 10.0, 10.0])
    state = State(
        positions=positions,
        velocities=velocities,
        forces=np.zeros((10, 3)),
        box=box,
    )
    bc = PeriodicBoundaryCondition()
    units = Units.LJ()
    return System(atoms, state, bc, units)


class TestNoThermostat:
    """Tests for NoThermostat (NVE)."""

    def test_does_nothing(self, multi_atom_system: System) -> None:
        """Test that NVE thermostat does not modify velocities."""
        thermostat = NoThermostat()
        original_velocities = multi_atom_system.state.velocities.copy()
        
        thermostat.apply(multi_atom_system, dt=0.001)
        
        np.testing.assert_array_equal(
            multi_atom_system.state.velocities, original_velocities
        )

    def test_get_name(self) -> None:
        """Test thermostat name."""
        thermostat = NoThermostat()
        assert "NVE" in thermostat.get_name()


class TestBerendsenThermostat:
    """Tests for BerendsenThermostat."""

    def test_invalid_tau(self) -> None:
        """Test that negative tau raises error."""
        with pytest.raises(ValueError):
            BerendsenThermostat(target_temperature=300.0, tau=-0.1)

    def test_temperature_moves_toward_target(
        self, multi_atom_system: System
    ) -> None:
        """Test that temperature moves toward target."""
        target_temp = 2.0  # In LJ units
        thermostat = BerendsenThermostat(target_temperature=target_temp, tau=0.1)
        
        initial_temp = multi_atom_system.compute_temperature()
        
        # Apply thermostat multiple times
        for _ in range(10):
            thermostat.apply(multi_atom_system, dt=0.01)
        
        final_temp = multi_atom_system.compute_temperature()
        
        # Temperature should move toward target
        if initial_temp > target_temp:
            assert final_temp < initial_temp
        else:
            assert final_temp > initial_temp

    def test_get_name(self) -> None:
        """Test thermostat name with parameters."""
        thermostat = BerendsenThermostat(target_temperature=300.0, tau=0.5)
        name = thermostat.get_name()
        assert "Berendsen" in name
        assert "300" in name


class TestNoseHooverThermostat:
    """Tests for NoseHooverThermostat."""

    def test_invalid_tau(self) -> None:
        """Test that negative tau raises error."""
        with pytest.raises(ValueError):
            NoseHooverThermostat(target_temperature=300.0, tau=-1.0)

    def test_xi_starts_at_zero(self) -> None:
        """Test initial xi value."""
        thermostat = NoseHooverThermostat(target_temperature=1.0, tau=0.5)
        assert thermostat.xi == 0.0

    def test_reset_clears_xi(self, multi_atom_system: System) -> None:
        """Test that reset clears heat bath variable."""
        thermostat = NoseHooverThermostat(target_temperature=1.0, tau=0.5)
        
        # Apply to change xi
        thermostat.apply(multi_atom_system, dt=0.01)
        
        # Reset
        thermostat.reset()
        assert thermostat.xi == 0.0

    def test_get_name(self) -> None:
        """Test thermostat name."""
        thermostat = NoseHooverThermostat(target_temperature=1.0, tau=0.5)
        name = thermostat.get_name()
        assert "NoseHoover" in name


class TestThermostatInterface:
    """Test thermostat interface compliance."""

    @pytest.mark.parametrize("thermostat_class,args", [
        (NoThermostat, ()),
        (BerendsenThermostat, (1.0, 0.1)),
        (NoseHooverThermostat, (1.0, 0.5)),
    ])
    def test_is_thermostat(
        self, thermostat_class: type, args: tuple
    ) -> None:
        """Test inheritance."""
        thermostat = thermostat_class(*args)
        assert isinstance(thermostat, Thermostat)

    @pytest.mark.parametrize("thermostat_class,args", [
        (NoThermostat, ()),
        (BerendsenThermostat, (1.0, 0.1)),
        (NoseHooverThermostat, (1.0, 0.5)),
    ])
    def test_has_required_methods(
        self, thermostat_class: type, args: tuple
    ) -> None:
        """Test required methods exist."""
        thermostat = thermostat_class(*args)
        assert hasattr(thermostat, "apply")
        assert hasattr(thermostat, "get_name")
