"""
Unit tests for core module.

Tests for Atom, State, System, Units, and ElementRegistry classes.
"""
import numpy as np
import pytest

from pyMD.core import (
    Atom,
    ElementData,
    ElementRegistry,
    State,
    System,
    Units,
    UnitSystemType,
    elements,
)
from pyMD.boundary import PeriodicBoundaryCondition


class TestAtom:
    """Tests for Atom dataclass."""

    def test_atom_creation(self) -> None:
        """Test basic atom creation."""
        atom = Atom(atom_type="Cu", mass=63.546, charge=0.0, index=0)
        assert atom.atom_type == "Cu"
        assert atom.mass == 63.546
        assert atom.charge == 0.0
        assert atom.index == 0

    def test_atom_defaults(self) -> None:
        """Test default values for charge and index."""
        atom = Atom(atom_type="Ar", mass=39.948)
        assert atom.charge == 0.0
        assert atom.index == -1

    def test_atom_invalid_mass(self) -> None:
        """Test that negative mass raises ValueError."""
        with pytest.raises(ValueError, match="mass must be positive"):
            Atom(atom_type="X", mass=-1.0)

    def test_atom_zero_mass(self) -> None:
        """Test that zero mass raises ValueError."""
        with pytest.raises(ValueError, match="mass must be positive"):
            Atom(atom_type="X", mass=0.0)

    def test_atom_empty_type(self) -> None:
        """Test that empty atom type raises ValueError."""
        with pytest.raises(ValueError, match="type cannot be empty"):
            Atom(atom_type="", mass=1.0)


class TestState:
    """Tests for State dataclass."""

    def test_state_creation(self) -> None:
        """Test basic state creation."""
        n_atoms = 10
        state = State(
            positions=np.zeros((n_atoms, 3)),
            velocities=np.zeros((n_atoms, 3)),
            forces=np.zeros((n_atoms, 3)),
            box=np.array([10.0, 10.0, 10.0]),
        )
        assert state.n_atoms == n_atoms
        assert state.time == 0.0
        assert state.step == 0

    def test_state_arrays_converted(self) -> None:
        """Test that lists are converted to numpy arrays."""
        state = State(
            positions=[[0, 0, 0], [1, 1, 1]],
            velocities=[[0, 0, 0], [0, 0, 0]],
            forces=[[0, 0, 0], [0, 0, 0]],
            box=[10, 10, 10],
        )
        assert isinstance(state.positions, np.ndarray)
        assert isinstance(state.box, np.ndarray)
        assert state.positions.dtype == np.float64

    def test_state_invalid_position_shape(self) -> None:
        """Test that invalid position shape raises ValueError."""
        with pytest.raises(ValueError, match="Positions must be"):
            State(
                positions=np.zeros((10,)),  # Wrong shape
                velocities=np.zeros((10, 3)),
                forces=np.zeros((10, 3)),
                box=np.array([10.0, 10.0, 10.0]),
            )

    def test_state_mismatched_velocities(self) -> None:
        """Test that mismatched velocities shape raises ValueError."""
        with pytest.raises(ValueError, match="Velocities shape"):
            State(
                positions=np.zeros((10, 3)),
                velocities=np.zeros((5, 3)),  # Wrong size
                forces=np.zeros((10, 3)),
                box=np.array([10.0, 10.0, 10.0]),
            )

    def test_state_copy(self) -> None:
        """Test that copy creates independent arrays."""
        state = State(
            positions=np.ones((5, 3)),
            velocities=np.ones((5, 3)),
            forces=np.ones((5, 3)),
            box=np.array([10.0, 10.0, 10.0]),
        )
        state_copy = state.copy()

        # Modify original
        state.positions[0, 0] = 999.0

        # Copy should be unchanged
        assert state_copy.positions[0, 0] == 1.0


class TestUnits:
    """Tests for UnitSystem and Units factory."""

    def test_real_units(self) -> None:
        """Test REAL units creation."""
        units = Units.REAL()
        assert units.name == UnitSystemType.REAL
        assert units.energy_unit == "kcal/mol"
        assert units.time_unit == "femtosecond"
        assert pytest.approx(units.boltzmann, rel=1e-4) == 0.001987204

    def test_metal_units(self) -> None:
        """Test METAL units creation."""
        units = Units.METAL()
        assert units.name == UnitSystemType.METAL
        assert units.energy_unit == "eV"
        assert units.time_unit == "picosecond"
        assert pytest.approx(units.boltzmann, rel=1e-4) == 8.617333e-5

    def test_lj_units(self) -> None:
        """Test LJ (reduced) units creation."""
        units = Units.LJ()
        assert units.name == UnitSystemType.LJ
        assert units.boltzmann == 1.0

    def test_si_units(self) -> None:
        """Test SI units creation."""
        units = Units.SI()
        assert units.name == UnitSystemType.SI
        assert pytest.approx(units.boltzmann, rel=1e-6) == 1.380649e-23


class TestElementRegistry:
    """Tests for ElementRegistry singleton."""

    def test_singleton_pattern(self) -> None:
        """Test that ElementRegistry is a singleton."""
        reg1 = ElementRegistry()
        reg2 = ElementRegistry()
        assert reg1 is reg2

    def test_get_mass(self) -> None:
        """Test getting atomic mass by symbol."""
        assert pytest.approx(elements.get_mass("Cu"), rel=1e-3) == 63.546
        assert pytest.approx(elements.get_mass("Ar"), rel=1e-3) == 39.948
        assert pytest.approx(elements.get_mass("H"), rel=1e-3) == 1.008

    def test_get_element(self) -> None:
        """Test getting full element data."""
        cu = elements.get_element("Cu")
        assert cu is not None
        assert cu.symbol == "Cu"
        assert cu.name == "Copper"
        assert cu.atomic_number == 29

    def test_get_element_by_number(self) -> None:
        """Test getting element by atomic number."""
        fe = elements.get_element(26)
        assert fe is not None
        assert fe.symbol == "Fe"

    def test_unknown_element(self) -> None:
        """Test that unknown element raises KeyError."""
        with pytest.raises(KeyError, match="not found"):
            elements.get_mass("Xx")

    def test_contains(self) -> None:
        """Test 'in' operator."""
        assert "Cu" in elements
        assert "Fe" in elements
        assert "Xx" not in elements

    def test_getitem(self) -> None:
        """Test indexing access."""
        au = elements["Au"]
        assert au.symbol == "Au"
        assert au.atomic_number == 79

    def test_list_elements(self) -> None:
        """Test listing all elements."""
        elem_list = elements.list_elements()
        assert "Cu" in elem_list
        assert "Ar" in elem_list
        assert len(elem_list) > 20  # Should have many elements

    def test_add_custom_element(self) -> None:
        """Test adding custom element."""
        # Use a unique name to avoid conflicts
        custom = ElementData("TEST1", "TestElement", 0, 100.0)
        elements.add_custom_element(custom)
        assert "TEST1" in elements
        assert elements.get_mass("TEST1") == 100.0


class TestSystem:
    """Tests for System class."""

    @pytest.fixture
    def simple_system(self) -> System:
        """Create a simple Argon system for testing."""
        n_atoms = 10
        atoms = [Atom("Ar", mass=39.948, index=i) for i in range(n_atoms)]
        positions = np.random.rand(n_atoms, 3) * 10.0
        velocities = np.random.randn(n_atoms, 3)

        state = State(
            positions=positions,
            velocities=velocities,
            forces=np.zeros((n_atoms, 3)),
            box=np.array([10.0, 10.0, 10.0]),
        )

        return System(
            atoms=atoms,
            state=state,
            boundary_condition=PeriodicBoundaryCondition(),
            units=Units.LJ(),
        )

    def test_system_creation(self, simple_system: System) -> None:
        """Test basic system creation."""
        assert simple_system.get_num_atoms() == 10
        assert simple_system.get_box().shape == (3,)

    def test_get_masses(self, simple_system: System) -> None:
        """Test getting masses array."""
        masses = simple_system.get_masses()
        assert masses.shape == (10,)
        assert np.allclose(masses, 39.948)

    def test_atom_types_single_species(self, simple_system: System) -> None:
        """Test atom type assignment for single species."""
        types = simple_system.get_atom_types()
        assert np.all(types == 0)

    def test_atom_types_multi_species(self) -> None:
        """Test atom type assignment for multi-species system."""
        atoms = [
            Atom("Cu", mass=63.546, index=0),
            Atom("Ni", mass=58.693, index=1),
            Atom("Cu", mass=63.546, index=2),
            Atom("Ni", mass=58.693, index=3),
        ]
        state = State(
            positions=np.zeros((4, 3)),
            velocities=np.zeros((4, 3)),
            forces=np.zeros((4, 3)),
            box=np.array([10.0, 10.0, 10.0]),
        )
        system = System(atoms, state, PeriodicBoundaryCondition(), Units.METAL())

        types = system.get_atom_types()
        assert types[0] == types[2]  # Cu atoms same type
        assert types[1] == types[3]  # Ni atoms same type
        assert types[0] != types[1]  # Cu != Ni

    def test_compute_kinetic_energy(self) -> None:
        """Test kinetic energy computation."""
        atoms = [Atom("Ar", mass=1.0, index=i) for i in range(2)]
        state = State(
            positions=np.zeros((2, 3)),
            velocities=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            forces=np.zeros((2, 3)),
            box=np.array([10.0, 10.0, 10.0]),
        )
        system = System(atoms, state, PeriodicBoundaryCondition(), Units.LJ())

        # KE = 0.5 * m * v^2 = 0.5 * 1 * 1^2 * 2 = 1.0
        ke = system.compute_kinetic_energy()
        assert pytest.approx(ke) == 1.0

    def test_compute_temperature(self, simple_system: System) -> None:
        """Test temperature computation returns positive value."""
        temp = simple_system.compute_temperature()
        assert temp > 0

    def test_zero_momentum(self) -> None:
        """Test removing center of mass velocity."""
        atoms = [Atom("Ar", mass=1.0, index=i) for i in range(3)]
        state = State(
            positions=np.zeros((3, 3)),
            velocities=np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
            forces=np.zeros((3, 3)),
            box=np.array([10.0, 10.0, 10.0]),
        )
        system = System(atoms, state, PeriodicBoundaryCondition(), Units.LJ())

        system.zero_momentum()

        # Total momentum should now be zero
        momentum = system.get_momentum()
        assert np.allclose(momentum, 0.0, atol=1e-10)

    def test_wrap_positions(self, simple_system: System) -> None:
        """Test position wrapping."""
        # Set a position outside the box
        simple_system.state.positions[0] = [15.0, -5.0, 25.0]
        simple_system.wrap_positions()

        # Position should now be wrapped into [0, box)
        pos = simple_system.state.positions[0]
        box = simple_system.get_box()
        assert np.all(pos >= 0) and np.all(pos < box)

    def test_mismatched_atoms_state(self) -> None:
        """Test that mismatched atoms and state raises ValueError."""
        atoms = [Atom("Ar", mass=39.948, index=i) for i in range(5)]
        state = State(
            positions=np.zeros((10, 3)),  # 10 != 5 atoms
            velocities=np.zeros((10, 3)),
            forces=np.zeros((10, 3)),
            box=np.array([10.0, 10.0, 10.0]),
        )

        with pytest.raises(ValueError, match="must match"):
            System(atoms, state, PeriodicBoundaryCondition(), Units.LJ())
