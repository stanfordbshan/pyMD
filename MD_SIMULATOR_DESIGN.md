# Molecular Dynamics Simulation Framework - Complete Design Document

**Project**: Educational Object-Oriented MD Simulator with Autodiff Forces  
**Date**: February 2026  
**Purpose**: Comprehensive design plan for implementation by Claude Opus 4.6

---

## Table of Contents

1. [Core Design Philosophy](#core-design-philosophy)
2. [Project Structure](#project-structure)
3. [Detailed Component Design](#detailed-component-design)
4. [Design Patterns Summary](#design-patterns-summary)
5. [Implementation Roadmap](#implementation-roadmap)
6. [Testing Strategy](#testing-strategy)
7. [Key Design Decisions](#key-design-decisions)
8. [Usage Examples](#usage-examples)

---

## Core Design Philosophy

```
╔════════════════════════════════════════════════════════════════╗
║  FUNDAMENTAL PRINCIPLE:                                        ║
║  Users write ONLY energy functions E(positions)                ║
║  Forces F = -∇E computed AUTOMATICALLY via autodiff            ║
║                                                                ║
║  This applies to ALL potentials:                               ║
║  - Simple pairwise (LJ, Morse)                                 ║
║  - Complex many-body (EAM, MEAM, ADP)                          ║
║  - Custom research potentials                                  ║
║                                                                ║
║  → Educational: Students learn physics, not calculus           ║
║  → Flexible: Researchers test new potentials instantly         ║
║  → Correct: Autodiff is mathematically exact                   ║
╚════════════════════════════════════════════════════════════════╝
```

### Key Requirements

1. **User writes ONLY energy functions** - never forces
2. **Autodiff backend switchable**: JAX ↔ PyTorch ↔ Autograd
3. **Unit systems**: LAMMPS-style (REAL, METAL, LJ, SI)
4. **EAM/MEAM support**: Full many-body potentials with automatic forces
5. **Neighbor lists**: Independent class with multiple algorithms
6. **YAML configuration**: Complete simulation setup from files
7. **OVITO output**: XYZ, LAMMPS dump, Extended XYZ formats
8. **Design patterns**: Strategy, Observer, Builder, Factory, Template Method
9. **Comprehensive testing**: Unit + integration tests
10. **Element registry**: Automatic mass lookup by chemical symbol

---

## Project Structure

```
pyMD/
├── core/
│   ├── atom.py              # Atom class (type, mass, charge)
│   ├── state.py             # State class (positions, velocities, forces)
│   ├── system.py            # System class (orchestrates atoms + state)
│   ├── constants.py         # Physical constants
│   ├── units.py             # Unit system management (LAMMPS-style)
│   └── element_registry.py  # Chemical element database (NEW!)
│
├── boundary/                 # Strategy Pattern for BC
│   ├── boundary_condition.py      # ABC for boundary conditions
│   ├── periodic_bc.py             # Periodic boundaries
│   ├── open_bc.py                 # No boundaries (free space)
│   └── mixed_bc.py                # Mixed BC (e.g., periodic XY, open Z)
│
├── neighbor/                 # Strategy Pattern for neighbor lists
│   ├── neighbor_list.py           # ABC for neighbor lists
│   ├── verlet_list.py             # Verlet list algorithm
│   ├── cell_list.py               # Cell/link-cell list (O(N))
│   └── brute_force.py             # No neighbor list (for small systems)
│
├── potential/                # Energy functions (autodiff computes forces!)
│   ├── potential_energy.py        # ABC for all potentials
│   │
│   ├── pairwise/                  # Simple pairwise potentials
│   │   ├── lennard_jones.py       # LJ 12-6 potential
│   │   ├── morse.py               # Morse potential
│   │   ├── buckingham.py          # Buckingham potential
│   │   └── custom_pairwise.py     # User-defined pairwise
│   │
│   ├── bonded/                    # Bonded interactions
│   │   ├── harmonic_bond.py       # Harmonic bonds
│   │   ├── harmonic_angle.py      # Angle potentials
│   │   └── dihedral.py            # Dihedral/torsion
│   │
│   ├── many_body/                 # Complex many-body potentials
│   │   ├── eam_base.py            # Base class for EAM-like potentials
│   │   ├── eam.py                 # Standard EAM (single species)
│   │   ├── eam_alloy.py           # EAM/Alloy (multi-species)
│   │   ├── eam_fs.py              # Finnis-Sinclair EAM
│   │   ├── meam.py                # Modified EAM (angular terms)
│   │   ├── adp.py                 # Angular Dependent Potential
│   │   └── custom_many_body.py    # User-defined many-body
│   │
│   ├── electrostatics/
│   │   ├── coulomb.py             # Coulombic interactions
│   │   └── ewald.py               # Ewald summation
│   │
│   └── composite_potential.py     # Combines multiple potentials
│
├── force/                    # Autodiff force computation
│   ├── force_calculator.py        # Main force calculator
│   └── autodiff_backend.py        # ABC + JAX/PyTorch/Autograd backends
│
├── integrator/               # Strategy Pattern for time integration
│   ├── integrator.py              # ABC for integrators
│   ├── velocity_verlet.py         # Velocity Verlet (symplectic)
│   ├── leapfrog.py                # Leapfrog integrator
│   └── langevin.py                # Langevin dynamics (stochastic)
│
├── thermostat/               # Strategy Pattern for temperature control
│   ├── thermostat.py              # ABC for thermostats
│   ├── no_thermostat.py           # NVE ensemble (no thermostat)
│   ├── berendsen.py               # Berendsen weak coupling
│   ├── nose_hoover.py             # Nosé-Hoover (canonical NVT)
│   └── andersen.py                # Andersen stochastic
│
├── observer/                 # Observer Pattern for monitoring
│   ├── observer.py                # ABC for observers
│   ├── observer_manager.py        # Manages collection of observers
│   ├── energy_observer.py         # Tracks KE, PE, total energy
│   ├── temperature_observer.py    # Tracks temperature
│   ├── trajectory_writer.py       # Writes OVITO-compatible formats
│   ├── pressure_observer.py       # Tracks pressure (virial)
│   └── rdf_observer.py            # Radial distribution function
│
├── simulator/                # Template Method + Facade
│   ├── pyMD.py            # Main simulation engine
│   └── simulation_factory.py      # Factory for complete simulations
│
├── builder/                  # Builder + Factory Patterns
│   ├── system_builder.py          # Builder for System objects
│   ├── potential_factory.py       # Factory for potentials
│   ├── neighbor_list_factory.py   # Factory for neighbor lists
│   ├── backend_factory.py         # Factory for autodiff backends
│   └── config_loader.py           # YAML configuration loader
│
├── utils/
│   ├── geometry.py                # Geometric calculations
│   └── io.py                      # File I/O helpers
│
├── tests/                    # Comprehensive testing
│   ├── unit/                      # Unit tests for each module
│   │   ├── test_boundary.py
│   │   ├── test_neighbor_list.py
│   │   ├── test_potential.py
│   │   ├── test_force_calculator.py
│   │   ├── test_integrator.py
│   │   ├── test_thermostat.py
│   │   ├── test_units.py
│   │   ├── test_element_registry.py
│   │   └── ...
│   │
│   ├── integration/               # Integration tests
│   │   ├── test_nve_conservation.py
│   │   ├── test_nvt_temperature.py
│   │   ├── test_backend_consistency.py
│   │   ├── test_neighbor_list_correctness.py
│   │   ├── test_eam_simulation.py
│   │   └── test_meam_simulation.py
│   │
│   └── fixtures/
│       └── test_data/
│
└── examples/                 # Example simulations
    ├── configs/                   # YAML configuration files
    │   ├── lennard_jones_gas.yaml
    │   ├── copper_fcc_eam.yaml
    │   ├── water_molecules.yaml
    │   └── tio2_meam.yaml
    │
    ├── 01_lennard_jones_gas.py
    ├── 02_copper_eam.py
    ├── 03_water_simulation.py
    ├── 04_tio2_meam.py
    ├── 05_custom_potential.py
    │
    ├── notebooks/
    │   ├── tutorial_01_basics.ipynb
    │   ├── tutorial_02_eam.ipynb
    │   └── tutorial_03_custom_potentials.ipynb
    │
    └── benchmarks/
        ├── backend_comparison.py
        └── neighbor_list_comparison.py
```

---

## Detailed Component Design

### 1. Core Module

#### **Atom** (Value Object)
```python
from dataclasses import dataclass

@dataclass
class Atom:
    """
    Represents a single atom
    
    Attributes:
        atom_type: Chemical symbol or type identifier (e.g., 'Cu', 'Ar', 'H')
        mass: Atomic mass in current unit system
        charge: Electric charge in elementary charge units
        index: Unique identifier for this atom
    """
    atom_type: str
    mass: float
    charge: float = 0.0
    index: int = -1
```

#### **State** (Value Object)
```python
@dataclass
class State:
    """
    Snapshot of system state at one timestep
    
    Contains all time-dependent quantities that define the system
    """
    positions: np.ndarray   # (N, 3) atomic positions
    velocities: np.ndarray  # (N, 3) atomic velocities
    forces: np.ndarray      # (N, 3) forces on atoms
    box: np.ndarray         # (3,) or (3,3) box dimensions
    time: float = 0.0       # Current simulation time
    step: int = 0           # Current timestep number
```

#### **System** (Entity)
```python
class System:
    """
    Central object holding all atoms and state
    
    DESIGN NOTE: Box lives in System.state, not in BoundaryCondition!
    
    Responsibilities:
    - Manage collection of atoms
    - Maintain current state (positions, velocities, forces, BOX)
    - Reference boundary condition strategy (but doesn't own box)
    - Track unit system
    - Compute system properties (temperature, KE, etc.)
    """
    
    def __init__(self, atoms: List[Atom], 
                 initial_positions: np.ndarray,
                 box: np.ndarray,  # ← Box owned by System
                 boundary_condition: BoundaryCondition,
                 units: UnitSystem):
        self.atoms = atoms
        self.state = State(
            positions=initial_positions,
            velocities=np.zeros_like(initial_positions),
            forces=np.zeros_like(initial_positions),
            box=box  # ← Stored in State
        )
        self.boundary_condition = boundary_condition  # ← Strategy reference
        self.units = units
        self.atom_types = self._assign_atom_types()
    
    def _assign_atom_types(self) -> np.ndarray:
        """
        Assign integer type indices to atoms based on species
        
        For multi-species potentials (EAM/Alloy, MEAM)
        Example: For Cu-Ni alloy: Cu → 0, Ni → 1
        """
        species_to_index = {}
        atom_types = np.zeros(len(self.atoms), dtype=int)
        
        for i, atom in enumerate(self.atoms):
            if atom.atom_type not in species_to_index:
                species_to_index[atom.atom_type] = len(species_to_index)
            atom_types[i] = species_to_index[atom.atom_type]
        
        return atom_types
    
    def get_masses(self) -> np.ndarray:
        """Return (N,) array of atomic masses"""
        return np.array([atom.mass for atom in self.atoms])
    
    def get_charges(self) -> np.ndarray:
        """Return (N,) array of atomic charges"""
        return np.array([atom.charge for atom in self.atoms])
    
    def get_atom_types(self) -> np.ndarray:
        """Return (N,) array of atom type indices"""
        return self.atom_types
    
    def wrap_positions(self) -> None:
        """
        Wrap positions using boundary condition strategy
        
        CRITICAL: Box is passed TO the BC, not owned BY the BC
        This allows BC to be stateless/reusable
        """
        self.state.positions = self.boundary_condition.wrap_positions(
            self.state.positions,
            self.state.box  # ← Pass box from System to BC
        )
    
    def set_box(self, new_box: np.ndarray) -> None:
        """
        Update box dimensions (e.g., for NPT ensemble)
        
        BC doesn't need to know - it receives box as parameter
        """
        self.state.box = new_box
    
    def get_box(self) -> np.ndarray:
        """Get current box dimensions"""
        return self.state.box
    
    def get_volume(self) -> float:
        """Compute simulation box volume"""
        return np.prod(self.state.box)
    
    def compute_kinetic_energy(self) -> float:
        """Compute total kinetic energy"""
        masses = self.get_masses()
        velocities = self.state.velocities
        return 0.5 * np.sum(masses[:, np.newaxis] * velocities**2)
    
    def compute_temperature(self) -> float:
        """Compute instantaneous temperature from kinetic energy"""
        ke = self.compute_kinetic_energy()
        n_dof = 3 * len(self.atoms) - 3  # subtract center of mass
        # T = 2*KE / (n_dof * kB)
        return 2.0 * ke / (n_dof * self.units.boltzmann)
    
    def get_num_atoms(self) -> int:
        return len(self.atoms)
```

#### **Units** (LAMMPS-style Unit Systems)
```python
from enum import Enum
from dataclasses import dataclass

class UnitSystemType(Enum):
    """Supported unit systems"""
    REAL = "real"       # Å, kcal/mol, fs, K (biomolecular)
    METAL = "metal"     # Å, eV, ps, K (materials)
    SI = "si"           # m, J, s, K (base units)
    LJ = "lj"           # Reduced/dimensionless units

@dataclass
class UnitSystem:
    """
    Complete unit system specification
    
    Defines all units and conversion factors.
    User selects one at simulation start.
    """
    name: UnitSystemType
    length_unit: str        # "angstrom", "meter", "sigma"
    energy_unit: str        # "kcal/mol", "eV", "epsilon"
    time_unit: str          # "femtosecond", "picosecond", "tau"
    mass_unit: str          # "g/mol", "amu"
    temperature_unit: str   # "kelvin"
    charge_unit: str        # "e" (elementary charge)
    boltzmann: float        # kB in this unit system
    pressure_unit: str      # Derived
    force_unit: str         # Derived
    # Conversion factors to SI
    length_to_si: float
    energy_to_si: float
    time_to_si: float
    mass_to_si: float

class Units:
    """Factory for standard unit systems"""
    
    @staticmethod
    def REAL() -> UnitSystem:
        """
        REAL units (common in biomolecular simulations)
        - distance: Ångströms
        - energy: kcal/mol
        - time: femtoseconds
        - mass: g/mol
        - temperature: Kelvin
        """
        return UnitSystem(
            name=UnitSystemType.REAL,
            length_unit="angstrom",
            energy_unit="kcal/mol",
            time_unit="femtosecond",
            mass_unit="g/mol",
            temperature_unit="kelvin",
            charge_unit="e",
            boltzmann=0.001987204,  # kcal/(mol·K)
            pressure_unit="atm",
            force_unit="kcal/mol/angstrom",
            length_to_si=1e-10,
            energy_to_si=4184/6.022e23,
            time_to_si=1e-15,
            mass_to_si=1.66054e-27
        )
    
    @staticmethod
    def METAL() -> UnitSystem:
        """
        METAL units (common in materials science)
        - distance: Ångströms
        - energy: eV
        - time: picoseconds
        - mass: g/mol
        - temperature: Kelvin
        """
        return UnitSystem(
            name=UnitSystemType.METAL,
            length_unit="angstrom",
            energy_unit="eV",
            time_unit="picosecond",
            mass_unit="g/mol",
            temperature_unit="kelvin",
            charge_unit="e",
            boltzmann=8.617333e-5,  # eV/K
            pressure_unit="bar",
            force_unit="eV/angstrom",
            length_to_si=1e-10,
            energy_to_si=1.60218e-19,
            time_to_si=1e-12,
            mass_to_si=1.66054e-27
        )
    
    @staticmethod
    def LJ() -> UnitSystem:
        """
        LJ (Lennard-Jones) reduced units (dimensionless)
        - distance: σ (LJ length parameter)
        - energy: ε (LJ energy parameter)
        - time: σ√(m/ε)
        - mass: m (particle mass)
        - temperature: ε/kB
        """
        return UnitSystem(
            name=UnitSystemType.LJ,
            length_unit="sigma",
            energy_unit="epsilon",
            time_unit="tau",
            mass_unit="m",
            temperature_unit="epsilon/kB",
            charge_unit="q",
            boltzmann=1.0,
            pressure_unit="epsilon/sigma^3",
            force_unit="epsilon/sigma",
            length_to_si=1.0,
            energy_to_si=1.0,
            time_to_si=1.0,
            mass_to_si=1.0
        )
    
    @staticmethod
    def SI() -> UnitSystem:
        """SI base units"""
        return UnitSystem(
            name=UnitSystemType.SI,
            length_unit="meter",
            energy_unit="joule",
            time_unit="second",
            mass_unit="kilogram",
            temperature_unit="kelvin",
            charge_unit="coulomb",
            boltzmann=1.380649e-23,  # J/K
            pressure_unit="pascal",
            force_unit="newton",
            length_to_si=1.0,
            energy_to_si=1.0,
            time_to_si=1.0,
            mass_to_si=1.0
        )
```

#### **ElementRegistry** (NEW! - Singleton Pattern)
```python
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass(frozen=True)
class ElementData:
    """
    Immutable data for a chemical element
    
    Contains standard atomic properties that don't change.
    """
    symbol: str              # Chemical symbol (e.g., "Cu", "Ar", "H")
    name: str                # Full name (e.g., "Copper", "Argon")
    atomic_number: int       # Z (1 for H, 6 for C, 29 for Cu, etc.)
    atomic_mass: float       # Standard atomic mass in amu/g/mol
    
    # Optional: Additional properties
    covalent_radius: Optional[float] = None      # Å
    vdw_radius: Optional[float] = None           # van der Waals radius (Å)
    electronegativity: Optional[float] = None    # Pauling scale
    melting_point: Optional[float] = None        # K
    color: Optional[str] = None                  # CPK color for visualization


class ElementRegistry:
    """
    Registry for chemical element properties (Singleton pattern)
    
    Provides convenient lookup of atomic masses and other properties.
    Users can access by symbol or atomic number.
    
    Usage:
        registry = ElementRegistry()
        cu_mass = registry.get_mass('Cu')
        ar_data = registry.get_element('Ar')
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """Singleton: Only one registry instance exists"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize periodic table data (only once)"""
        if not ElementRegistry._initialized:
            self._elements_by_symbol: Dict[str, ElementData] = {}
            self._elements_by_number: Dict[int, ElementData] = {}
            self._initialize_periodic_table()
            ElementRegistry._initialized = True
    
    def _initialize_periodic_table(self):
        """
        Populate registry with standard element data
        
        Data sources:
        - Atomic masses: IUPAC standard atomic weights
        - Radii: Cordero et al. (2008)
        - Colors: CPK coloring
        """
        elements = [
            # Noble gases (often used for simple potentials)
            ElementData("He", "Helium", 2, 4.0026, 0.28, 1.40, None, "#D9FFFF"),
            ElementData("Ne", "Neon", 10, 20.180, 0.58, 1.54, None, "#B3E3F5"),
            ElementData("Ar", "Argon", 18, 39.948, 0.71, 1.88, None, "#80D1E3"),
            ElementData("Kr", "Krypton", 36, 83.798, 0.88, 2.02, 3.00, "#8DD9CC"),
            ElementData("Xe", "Xenon", 54, 131.29, 1.08, 2.16, 2.60, "#429EB0"),
            
            # Common metals (EAM/MEAM simulations)
            ElementData("Cu", "Copper", 29, 63.546, 1.32, 1.40, 1.90, "#C88033"),
            ElementData("Ag", "Silver", 47, 107.87, 1.45, 1.72, 1.93, "#C0C0C0"),
            ElementData("Au", "Gold", 79, 196.97, 1.36, 1.66, 2.54, "#FFD123"),
            ElementData("Ni", "Nickel", 28, 58.693, 1.24, 1.63, 1.91, "#50D050"),
            ElementData("Pd", "Palladium", 46, 106.42, 1.39, 1.63, 2.20, "#006985"),
            ElementData("Pt", "Platinum", 78, 195.08, 1.36, 1.75, 2.28, "#D0D0E0"),
            ElementData("Al", "Aluminum", 13, 26.982, 1.21, 1.84, 1.61, "#BFA6A6"),
            ElementData("Fe", "Iron", 26, 55.845, 1.32, 2.00, 1.83, "#E06633"),
            ElementData("Ti", "Titanium", 22, 47.867, 1.60, 2.15, 1.54, "#BFC2C7"),
            ElementData("Cr", "Chromium", 24, 51.996, 1.39, 2.05, 1.66, "#8A99C7"),
            ElementData("Mo", "Molybdenum", 42, 95.95, 1.54, 2.10, 2.16, "#54B5B5"),
            ElementData("W", "Tungsten", 74, 183.84, 1.62, 2.10, 2.36, "#2194D6"),
            ElementData("Ta", "Tantalum", 73, 180.95, 1.70, 2.20, 1.50, "#4DC2FF"),
            
            # Light elements (biomolecular simulations)
            ElementData("H", "Hydrogen", 1, 1.0080, 0.31, 1.10, 2.20, "#FFFFFF"),
            ElementData("C", "Carbon", 6, 12.011, 0.76, 1.70, 2.55, "#909090"),
            ElementData("N", "Nitrogen", 7, 14.007, 0.71, 1.55, 3.04, "#3050F8"),
            ElementData("O", "Oxygen", 8, 15.999, 0.66, 1.52, 3.44, "#FF0D0D"),
            ElementData("S", "Sulfur", 16, 32.06, 1.05, 1.80, 2.58, "#FFFF30"),
            ElementData("P", "Phosphorus", 15, 30.974, 1.07, 1.80, 2.19, "#FF8000"),
            
            # Add more elements as needed...
        ]
        
        for element in elements:
            self._elements_by_symbol[element.symbol] = element
            self._elements_by_number[element.atomic_number] = element
    
    def get_element(self, identifier) -> Optional[ElementData]:
        """
        Get element data by symbol or atomic number
        
        Args:
            identifier: Element symbol (str) or atomic number (int)
        """
        if isinstance(identifier, str):
            return self._elements_by_symbol.get(identifier)
        elif isinstance(identifier, int):
            return self._elements_by_number.get(identifier)
        else:
            raise TypeError(f"Identifier must be str or int, got {type(identifier)}")
    
    def get_mass(self, symbol: str) -> float:
        """Get atomic mass by element symbol"""
        element = self.get_element(symbol)
        if element is None:
            raise KeyError(f"Element '{symbol}' not found in registry")
        return element.atomic_mass
    
    def has_element(self, identifier) -> bool:
        """Check if element exists in registry"""
        return self.get_element(identifier) is not None
    
    def list_elements(self) -> list[str]:
        """Return list of all available element symbols"""
        return sorted(self._elements_by_symbol.keys())
    
    def add_custom_element(self, element: ElementData) -> None:
        """
        Add custom element/pseudoatom
        
        Useful for coarse-grained simulations, united atoms, dummy atoms
        """
        if element.symbol in self._elements_by_symbol:
            raise ValueError(f"Element '{element.symbol}' already exists")
        
        self._elements_by_symbol[element.symbol] = element
        if element.atomic_number > 0:
            self._elements_by_number[element.atomic_number] = element
    
    def __contains__(self, identifier) -> bool:
        """Support 'in' operator"""
        return self.has_element(identifier)
    
    def __getitem__(self, identifier) -> ElementData:
        """Support indexing: registry['Cu']"""
        element = self.get_element(identifier)
        if element is None:
            raise KeyError(f"Element '{identifier}' not found")
        return element

# Module-level convenience instance
elements = ElementRegistry()
```

---

### 2. Boundary Module (Strategy Pattern)

#### **BoundaryCondition** (ABC)
```python
from abc import ABC, abstractmethod
import numpy as np

class BoundaryCondition(ABC):
    """
    Abstract base for boundary conditions
    
    DESIGN NOTE: BoundaryCondition is a STATELESS strategy.
    It does NOT own the box - the box is passed as a parameter.
    
    This design:
    - Keeps BC pure and reusable
    - Allows box to change without recreating BC
    - Follows Strategy pattern correctly
    
    Strategy Pattern: Different BC algorithms are interchangeable
    """
    
    @abstractmethod
    def apply_minimum_image(self, 
                          vector: np.ndarray, 
                          box: np.ndarray) -> np.ndarray:
        """
        Apply minimum image convention to distance vectors
        
        Critical for force calculations with periodic boundaries
        
        Args:
            vector: (N, 3) displacement vectors
            box: (3,) current box dimensions FROM SYSTEM
            
        Returns:
            Corrected vectors
        """
        pass
    
    @abstractmethod
    def wrap_positions(self, 
                      positions: np.ndarray, 
                      box: np.ndarray) -> np.ndarray:
        """
        Wrap positions back into primary simulation box
        
        Args:
            positions: (N, 3) atomic positions
            box: (3,) current box dimensions FROM SYSTEM
            
        Returns:
            Wrapped positions
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        pass
```

#### **Concrete Implementations**
```python
class PeriodicBoundaryCondition(BoundaryCondition):
    """
    Fully periodic boundaries (typical for bulk simulations)
    
    NO box stored here! Box comes from System.state.box
    """
    
    def apply_minimum_image(self, vector: np.ndarray, box: np.ndarray) -> np.ndarray:
        """Minimum image convention for periodic boundaries"""
        return vector - box * np.round(vector / box)
    
    def wrap_positions(self, positions: np.ndarray, box: np.ndarray) -> np.ndarray:
        """Wrap positions back into [0, box)"""
        return positions - box * np.floor(positions / box)
    
    def get_name(self) -> str:
        return "Periodic"


class OpenBoundaryCondition(BoundaryCondition):
    """No boundaries (gas phase, clusters)"""
    
    def apply_minimum_image(self, vector: np.ndarray, box: np.ndarray) -> np.ndarray:
        """No modification for open boundaries"""
        return vector
    
    def wrap_positions(self, positions: np.ndarray, box: np.ndarray) -> np.ndarray:
        """No wrapping for open boundaries"""
        return positions
    
    def get_name(self) -> str:
        return "Open"


class MixedBoundaryCondition(BoundaryCondition):
    """
    Different BC per dimension
    Example: periodic in XY, open in Z (slab geometry)
    """
    
    def __init__(self, periodic_dims: Tuple[bool, bool, bool]):
        """
        Args:
            periodic_dims: (x_periodic, y_periodic, z_periodic)
        """
        self.periodic_dims = np.array(periodic_dims)
    
    def apply_minimum_image(self, vector: np.ndarray, box: np.ndarray) -> np.ndarray:
        """Apply minimum image only in periodic dimensions"""
        correction = np.zeros_like(vector)
        correction[:, self.periodic_dims] = (
            -box[self.periodic_dims] * 
            np.round(vector[:, self.periodic_dims] / box[self.periodic_dims])
        )
        return vector + correction
    
    def wrap_positions(self, positions: np.ndarray, box: np.ndarray) -> np.ndarray:
        """Wrap only in periodic dimensions"""
        wrapped = positions.copy()
        wrapped[:, self.periodic_dims] = (
            positions[:, self.periodic_dims] - 
            box[self.periodic_dims] * 
            np.floor(positions[:, self.periodic_dims] / box[self.periodic_dims])
        )
        return wrapped
    
    def get_name(self) -> str:
        dims = ['X' if self.periodic_dims[i] else '' for i in range(3)]
        return f"Mixed({''.join(dims)} periodic)"
```

---

### 3. Neighbor List Module (Strategy Pattern)

#### **NeighborList** (ABC)
```python
class NeighborList(ABC):
    """
    Abstract base for neighbor list algorithms
    
    Neighbor lists speed up force calculations by tracking
    which atoms are close enough to interact.
    
    Strategy Pattern: Different algorithms (Verlet, Cell, BruteForce)
    """
    
    def __init__(self, cutoff: float, skin: float = 0.0):
        """
        Args:
            cutoff: Interaction cutoff distance
            skin: Extra distance for buffer (reduces rebuild frequency)
        """
        self.cutoff = cutoff
        self.skin = skin
        self.build_cutoff = cutoff + skin
        self.neighbors = None
        self.last_build_positions = None
    
    @abstractmethod
    def build(self, positions: np.ndarray, box: np.ndarray, 
             boundary_condition: BoundaryCondition) -> None:
        """Build/rebuild the neighbor list"""
        pass
    
    @abstractmethod
    def get_neighbors(self, atom_index: int) -> np.ndarray:
        """Get neighbor indices for given atom"""
        pass
    
    def needs_rebuild(self, positions: np.ndarray) -> bool:
        """
        Check if list needs rebuilding based on displacement
        
        Uses displacement criterion: if any atom moved more than skin/2,
        rebuild the list.
        """
        if self.last_build_positions is None:
            return True
        
        max_displacement = np.max(np.linalg.norm(
            positions - self.last_build_positions, axis=1
        ))
        
        return max_displacement > self.skin / 2.0
    
    @abstractmethod
    def get_name(self) -> str:
        pass
    
    @abstractmethod
    def get_num_neighbors(self) -> int:
        """Return total number of neighbor pairs"""
        pass
```

#### **Concrete Implementations**
```python
class BruteForceNeighborList(NeighborList):
    """
    No optimization - check all pairs
    O(N²) scaling
    Good for: N < 1000, debugging, educational
    """
    
    def build(self, positions, box, boundary_condition):
        """Always returns all pairs within cutoff"""
        n_atoms = len(positions)
        self.neighbors = {i: [] for i in range(n_atoms)}
        
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                dr = positions[j] - positions[i]
                dr = boundary_condition.apply_minimum_image(dr, box)
                distance = np.linalg.norm(dr)
                
                if distance < self.build_cutoff:
                    self.neighbors[i].append(j)
        
        self.last_build_positions = positions.copy()
    
    def get_neighbors(self, atom_index):
        return np.array(self.neighbors[atom_index])
    
    def get_name(self):
        return "BruteForce"
    
    def get_num_neighbors(self):
        return sum(len(v) for v in self.neighbors.values())


class VerletList(NeighborList):
    """
    Verlet neighbor list with skin
    
    O(N²) build but infrequent
    Good for: 1000 < N < 10000, uniform density
    """
    
    def __init__(self, cutoff: float, skin: float = 0.3):
        super().__init__(cutoff, skin)
        self.build_count = 0
    
    def build(self, positions, box, boundary_condition):
        """Build neighbor list with skin distance"""
        n_atoms = len(positions)
        self.neighbors = {i: [] for i in range(n_atoms)}
        
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                dr = positions[j] - positions[i]
                dr = boundary_condition.apply_minimum_image(dr, box)
                distance = np.linalg.norm(dr)
                
                if distance < self.build_cutoff:
                    self.neighbors[i].append(j)
        
        self.last_build_positions = positions.copy()
        self.build_count += 1
    
    def get_neighbors(self, atom_index):
        return np.array(self.neighbors[atom_index])
    
    def get_name(self):
        return f"VerletList(skin={self.skin:.2f}, builds={self.build_count})"
    
    def get_num_neighbors(self):
        return sum(len(v) for v in self.neighbors.values())


class CellList(NeighborList):
    """
    Cell list (link-cell) algorithm
    
    O(N) scaling for build and lookup
    Good for: N > 10000, best asymptotic scaling
    
    Algorithm:
    1. Divide box into cells of size ≥ cutoff
    2. Assign each atom to a cell
    3. For atom in cell i, only check atoms in cell i and 26 neighbors
    """
    
    def __init__(self, cutoff: float, skin: float = 0.3):
        super().__init__(cutoff, skin)
        self.cells = None
        self.cell_size = None
        self.n_cells = None
        self.build_count = 0
    
    def build(self, positions, box, boundary_condition):
        """Build cell list"""
        # Determine cell grid
        self.cell_size = self.build_cutoff
        self.n_cells = np.maximum(1, (box / self.cell_size).astype(int))
        actual_cell_size = box / self.n_cells
        
        # Initialize cells
        self.cells = {}
        for i in range(self.n_cells[0]):
            for j in range(self.n_cells[1]):
                for k in range(self.n_cells[2]):
                    self.cells[(i, j, k)] = []
        
        # Assign atoms to cells
        for atom_idx, pos in enumerate(positions):
            cell_idx = tuple((pos / actual_cell_size).astype(int) % self.n_cells)
            self.cells[cell_idx].append(atom_idx)
        
        # Build neighbor list by checking neighboring cells
        n_atoms = len(positions)
        self.neighbors = {i: [] for i in range(n_atoms)}
        
        for cell_idx, atoms_in_cell in self.cells.items():
            neighbor_cells = self._get_neighbor_cells(cell_idx)
            
            for atom_i in atoms_in_cell:
                for neighbor_cell in neighbor_cells:
                    for atom_j in self.cells[neighbor_cell]:
                        if atom_j <= atom_i:
                            continue
                        
                        dr = positions[atom_j] - positions[atom_i]
                        dr = boundary_condition.apply_minimum_image(dr, box)
                        distance = np.linalg.norm(dr)
                        
                        if distance < self.build_cutoff:
                            self.neighbors[atom_i].append(atom_j)
        
        self.last_build_positions = positions.copy()
        self.build_count += 1
    
    def _get_neighbor_cells(self, cell_idx):
        """Get indices of neighboring cells (including periodic wrapping)"""
        i, j, k = cell_idx
        neighbors = []
        
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                for dk in [-1, 0, 1]:
                    ni = (i + di) % self.n_cells[0]
                    nj = (j + dj) % self.n_cells[1]
                    nk = (k + dk) % self.n_cells[2]
                    neighbors.append((ni, nj, nk))
        
        return neighbors
    
    def get_neighbors(self, atom_index):
        return np.array(self.neighbors[atom_index])
    
    def get_name(self):
        return f"CellList(cells={self.n_cells}, builds={self.build_count})"
    
    def get_num_neighbors(self):
        return sum(len(v) for v in self.neighbors.values())
```

---

### 4. Potential Module (Energy-Only Interface)

#### **PotentialEnergy** (ABC)
```python
class PotentialEnergy(ABC):
    """
    Abstract base for ALL potential energy functions
    
    ╔══════════════════════════════════════════════════════════╗
    ║  CRITICAL: Users ONLY implement compute_energy()        ║
    ║  Forces computed AUTOMATICALLY via autodiff             ║
    ║                                                          ║
    ║  This applies to:                                        ║
    ║  - Simple pairwise (LJ, Morse)                          ║
    ║  - Complex many-body (EAM, MEAM, ADP)                   ║
    ║  - Custom research potentials                           ║
    ╚══════════════════════════════════════════════════════════╝
    """
    
    @abstractmethod
    def compute_energy(self, 
                      positions: np.ndarray, 
                      box: np.ndarray,
                      boundary_condition: BoundaryCondition,
                      atom_types: Optional[np.ndarray] = None,
                      neighbor_list: Optional[NeighborList] = None,
                      **kwargs) -> float:
        """
        Compute total potential energy from atomic positions
        
        User writes ONLY this method.
        Forces = -∇E computed automatically via autodiff.
        
        Args:
            positions: (N, 3) atomic coordinates
            box: (3,) box dimensions
            boundary_condition: For handling periodicity
            atom_types: (N,) atom type indices (for multi-species)
            neighbor_list: Optional for efficiency
            
        Returns:
            Total potential energy (scalar)
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        pass
```

#### **Pairwise Potentials**
```python
class LennardJonesPotential(PotentialEnergy):
    """
    Lennard-Jones 12-6 potential
    U(r) = 4ε[(σ/r)¹² - (σ/r)⁶]
    
    User writes energy, forces computed automatically!
    """
    
    def __init__(self, epsilon: float, sigma: float, cutoff: float):
        self.epsilon = epsilon
        self.sigma = sigma
        self.cutoff = cutoff
    
    def compute_energy(self, positions, box, boundary_condition,
                      atom_types=None, neighbor_list=None, **kwargs):
        """Compute LJ energy - autodiff will compute forces!"""
        energy = 0.0
        n_atoms = len(positions)
        
        if neighbor_list is not None:
            # Use neighbor list
            for i in range(n_atoms):
                neighbors = neighbor_list.get_neighbors(i)
                for j in neighbors:
                    dr = positions[j] - positions[i]
                    dr = boundary_condition.apply_minimum_image(dr, box)
                    r = np.linalg.norm(dr)
                    
                    if r < self.cutoff:
                        sr = self.sigma / r
                        energy += 4.0 * self.epsilon * (sr**12 - sr**6)
        else:
            # Brute force all pairs
            for i in range(n_atoms):
                for j in range(i + 1, n_atoms):
                    dr = positions[j] - positions[i]
                    dr = boundary_condition.apply_minimum_image(dr, box)
                    r = np.linalg.norm(dr)
                    
                    if r < self.cutoff:
                        sr = self.sigma / r
                        energy += 4.0 * self.epsilon * (sr**12 - sr**6)
        
        return energy
    
    def get_name(self):
        return f"LJ(ε={self.epsilon}, σ={self.sigma})"


class MorsePotential(PotentialEnergy):
    """
    Morse potential: U = D[1 - exp(-a(r-r0))]²
    
    User writes energy, forces via autodiff!
    """
    
    def __init__(self, D: float, a: float, r0: float, cutoff: float):
        self.D = D
        self.a = a
        self.r0 = r0
        self.cutoff = cutoff
    
    def compute_energy(self, positions, box, boundary_condition,
                      atom_types=None, neighbor_list=None, **kwargs):
        """Compute Morse energy"""
        # Similar structure to LJ
        pass
    
    def get_name(self):
        return f"Morse(D={self.D}, a={self.a}, r0={self.r0})"
```

#### **Bonded Potentials**
```python
class HarmonicBondPotential(PotentialEnergy):
    """
    Harmonic bonds: U = (1/2)*k*(r - r0)²
    
    User specifies bond list and parameters.
    Forces computed automatically!
    """
    
    def __init__(self, bond_list: List[Tuple[int, int]], k: float, r0: float):
        """
        Args:
            bond_list: List of bonded atom pairs [(i, j), ...]
            k: Spring constant
            r0: Equilibrium bond length
        """
        self.bonds = bond_list
        self.k = k
        self.r0 = r0
    
    def compute_energy(self, positions, box, boundary_condition,
                      atom_types=None, neighbor_list=None, **kwargs):
        """Compute harmonic bond energy"""
        energy = 0.0
        
        for i, j in self.bonds:
            dr = positions[j] - positions[i]
            dr = boundary_condition.apply_minimum_image(dr, box)
            r = np.linalg.norm(dr)
            
            energy += 0.5 * self.k * (r - self.r0)**2
        
        return energy
    
    def get_name(self):
        return f"HarmonicBond(k={self.k}, r0={self.r0}, {len(self.bonds)} bonds)"
```

#### **Many-Body Potentials: EAM Family**

```python
class EAMBasePotential(PotentialEnergy):
    """
    Base class for Embedded Atom Method (EAM)
    
    Total energy:
    E = Σ_i F_i(ρ_i) + (1/2) Σ_i Σ_j≠i φ_ij(r_ij)
    
    where:
    - F(ρ): Embedding energy
    - ρ_i = Σ_j f_j(r_ij): Electron density at atom i
    - φ(r): Pair interaction
    
    KEY: Even with nested functions (density → embedding),
    autodiff handles ALL derivatives automatically!
    User just writes the energy formula.
    """
    
    def __init__(self, 
                 cutoff: float,
                 embedding_functions: Dict[int, Callable],
                 pair_potentials: Dict[Tuple[int, int], Callable],
                 density_functions: Dict[int, Callable]):
        """
        Args:
            cutoff: Interaction cutoff distance
            embedding_functions: Dict[atom_type → F(ρ)]
            pair_potentials: Dict[(type_i, type_j) → φ(r)]
            density_functions: Dict[atom_type → f(r)]
        """
        self.cutoff = cutoff
        self.embedding_functions = embedding_functions
        self.pair_potentials = pair_potentials
        self.density_functions = density_functions
    
    def compute_energy(self, positions, box, boundary_condition,
                      atom_types, neighbor_list=None, **kwargs):
        """
        Compute EAM energy - autodiff will handle force calculation!
        
        User only needs to understand this energy form.
        Forces will be computed automatically.
        """
        n_atoms = len(positions)
        
        # Step 1: Compute electron density at each atom
        rho = self._compute_densities(positions, box, boundary_condition, 
                                     atom_types, neighbor_list)
        
        # Step 2: Compute embedding energy
        embedding_energy = 0.0
        for i in range(n_atoms):
            atom_type = atom_types[i]
            F = self.embedding_functions[atom_type]
            embedding_energy += F(rho[i])
        
        # Step 3: Compute pair interactions
        pair_energy = self._compute_pair_energy(positions, box, boundary_condition,
                                               atom_types, neighbor_list)
        
        total_energy = embedding_energy + pair_energy
        return total_energy
    
    def _compute_densities(self, positions, box, boundary_condition, 
                          atom_types, neighbor_list):
        """
        Compute electron density at each atom: ρ_i = Σ_j≠i f_j(r_ij)
        
        This is an intermediate quantity, but autodiff tracks all operations!
        """
        n_atoms = len(positions)
        rho = np.zeros(n_atoms)
        
        if neighbor_list is not None:
            for i in range(n_atoms):
                neighbors = neighbor_list.get_neighbors(i)
                for j in neighbors:
                    r_ij = self._compute_distance(i, j, positions, box, boundary_condition)
                    
                    if r_ij < self.cutoff:
                        atom_type_j = atom_types[j]
                        f_j = self.density_functions[atom_type_j]
                        rho[i] += f_j(r_ij)
                        rho[j] += f_j(r_ij)
        else:
            for i in range(n_atoms):
                for j in range(i + 1, n_atoms):
                    r_ij = self._compute_distance(i, j, positions, box, boundary_condition)
                    
                    if r_ij < self.cutoff:
                        atom_type_i = atom_types[i]
                        atom_type_j = atom_types[j]
                        f_i = self.density_functions[atom_type_i]
                        f_j = self.density_functions[atom_type_j]
                        
                        rho[i] += f_j(r_ij)
                        rho[j] += f_i(r_ij)
        
        return rho
    
    def _compute_pair_energy(self, positions, box, boundary_condition,
                            atom_types, neighbor_list):
        """Compute pairwise energy: (1/2) Σ_i Σ_j≠i φ_ij(r_ij)"""
        pair_energy = 0.0
        n_atoms = len(positions)
        
        if neighbor_list is not None:
            for i in range(n_atoms):
                neighbors = neighbor_list.get_neighbors(i)
                for j in neighbors:
                    r_ij = self._compute_distance(i, j, positions, box, boundary_condition)
                    
                    if r_ij < self.cutoff:
                        type_i = atom_types[i]
                        type_j = atom_types[j]
                        pair_key = tuple(sorted([type_i, type_j]))
                        phi = self.pair_potentials[pair_key]
                        pair_energy += phi(r_ij)
        else:
            for i in range(n_atoms):
                for j in range(i + 1, n_atoms):
                    r_ij = self._compute_distance(i, j, positions, box, boundary_condition)
                    
                    if r_ij < self.cutoff:
                        type_i = atom_types[i]
                        type_j = atom_types[j]
                        pair_key = tuple(sorted([type_i, type_j]))
                        phi = self.pair_potentials[pair_key]
                        pair_energy += phi(r_ij)
        
        return pair_energy
    
    def _compute_distance(self, i, j, positions, box, boundary_condition):
        """Compute distance between atoms i and j with PBC"""
        dr = positions[j] - positions[i]
        dr = boundary_condition.apply_minimum_image(dr, box)
        return np.linalg.norm(dr)
    
    def get_name(self):
        return "EAMBase"


class EAMPotential(EAMBasePotential):
    """
    Standard EAM for single-species metals
    
    Common parameterizations:
    - Sutton-Chen
    - Johnson
    - Voter-Chen
    
    User only provides functional forms - forces computed automatically!
    """
    
    @classmethod
    def sutton_chen(cls, epsilon: float, a: float, c: float, 
                    m: int, n: int, cutoff: float):
        """
        Sutton-Chen EAM (FCC metals: Cu, Ag, Au, Ni, Pd, Pt)
        
        F(ρ) = -c * ε * √ρ
        φ(r) = ε * (a/r)^n
        f(r) = (a/r)^m
        
        Example (Copper):
            eam_cu = EAMPotential.sutton_chen(
                epsilon=0.0124,  # eV
                a=3.615,         # Å
                c=39.432, m=6, n=12, cutoff=5.5
            )
        
        Forces computed automatically via autodiff!
        """
        embedding_fn = lambda rho: -c * epsilon * np.sqrt(rho)
        pair_fn = lambda r: epsilon * (a / r) ** n
        density_fn = lambda r: (a / r) ** m
        
        return cls(
            cutoff=cutoff,
            embedding_functions={0: embedding_fn},
            pair_potentials={(0, 0): pair_fn},
            density_functions={0: density_fn}
        )
    
    def get_name(self):
        return "EAM(Sutton-Chen)"


class MEAMPotential(PotentialEnergy):
    """
    Modified EAM with angular-dependent terms
    
    Total energy includes:
    - ρ⁽⁰⁾: Spherical (like EAM)
    - ρ⁽¹⁾: Dipole (angular l=1)
    - ρ⁽²⁾: Quadrupole (angular l=2)
    - ρ⁽³⁾: Octupole (angular l=3)
    
    COMPLEXITY: High - many angular terms, screening functions
    
    BUT: User STILL only writes energy function!
    Autodiff handles ALL complex derivatives automatically.
    """
    
    def __init__(self, parameters: Dict):
        """
        Args:
            parameters: MEAM parameter dictionary
        """
        self.params = parameters
        self.cutoff = parameters['cutoff']
    
    def compute_energy(self, positions, box, boundary_condition,
                      atom_types, neighbor_list=None, **kwargs):
        """
        Compute MEAM energy with angular moments
        
        Even with:
        - Vector/tensor sums for angular moments
        - 3-body screening functions
        - Complex nested functions
        
        Autodiff computes forces automatically!
        """
        n_atoms = len(positions)
        
        # Compute partial densities (spherical + angular)
        rho_0, rho_1, rho_2, rho_3 = self._compute_partial_densities(
            positions, box, boundary_condition, atom_types, neighbor_list
        )
        
        # Background density with angular corrections
        rho_bar = np.zeros(n_atoms)
        for i in range(n_atoms):
            t1, t2, t3 = self.params['t1'], self.params['t2'], self.params['t3']
            
            rho_bar[i] = rho_0[i]
            if rho_0[i] > 1e-10:
                rho_bar[i] += (t1 * np.sum(rho_1[i]**2) + 
                              t2 * np.sum(rho_2[i]**2) + 
                              t3 * np.sum(rho_3[i]**2)) / rho_0[i]
        
        # Embedding energy
        embedding_energy = 0.0
        for i in range(n_atoms):
            F = self._get_embedding_function(atom_types[i])
            embedding_energy += F(rho_bar[i])
        
        # Screened pair energy
        pair_energy = self._compute_screened_pair_energy(
            positions, box, boundary_condition, atom_types, neighbor_list
        )
        
        return embedding_energy + pair_energy
    
    def _compute_partial_densities(self, positions, box, boundary_condition,
                                   atom_types, neighbor_list):
        """Compute partial electron densities including angular terms"""
        # Implementation details...
        pass
    
    def get_name(self):
        return "MEAM"


class CustomManyBodyPotential(PotentialEnergy):
    """
    Framework for user-defined research potentials
    
    Users can implement ANY differentiable energy function.
    Autodiff computes forces automatically!
    """
    
    def __init__(self, energy_function: Callable, cutoff: float, **params):
        """
        Args:
            energy_function: User's function(positions, box, bc, ...) → energy
            cutoff: Interaction cutoff
            **params: Any additional parameters
        """
        self.energy_function = energy_function
        self.cutoff = cutoff
        self.params = params
    
    def compute_energy(self, positions, box, boundary_condition,
                      atom_types=None, neighbor_list=None, **kwargs):
        """Call user's custom energy - autodiff handles forces!"""
        return self.energy_function(
            positions, box, boundary_condition,
            cutoff=self.cutoff, **self.params, **kwargs
        )
    
    def get_name(self):
        return f"CustomManyBody({self.energy_function.__name__})"
```

#### **Composite Potential**
```python
class CompositePotential(PotentialEnergy):
    """
    Combine multiple potentials (Composite Pattern)
    
    Example: EAM + Coulomb + Harmonic bonds
    """
    
    def __init__(self, potentials: List[PotentialEnergy]):
        self.potentials = potentials
    
    def compute_energy(self, positions, box, boundary_condition, **kwargs):
        return sum(p.compute_energy(positions, box, boundary_condition, **kwargs)
                   for p in self.potentials)
    
    def get_name(self):
        names = [p.get_name() for p in self.potentials]
        return f"Composite({', '.join(names)})"
```

---

### 5. Force Module (Autodiff)

#### **AutoDiffBackend** (ABC - Strategy)
```python
class AutoDiffBackend(ABC):
    """
    Abstract backend for automatic differentiation
    
    Strategy Pattern: JAX, PyTorch, or Autograd
    User can easily switch between backends
    """
    
    @abstractmethod
    def compute_forces(self, energy_fn: Callable, positions: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute forces as F = -∇U
        
        Args:
            energy_fn: Function that computes energy from positions
            positions: (N, 3) atomic positions
            
        Returns:
            forces: (N, 3) force array
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        pass
```

#### **Concrete Backends**
```python
class JAXBackend(AutoDiffBackend):
    """
    JAX autodiff backend
    
    Advantages: Fastest, GPU/TPU, jit compilation
    """
    
    def __init__(self, use_jit: bool = True):
        import jax
        import jax.numpy as jnp
        self.jax = jax
        self.jnp = jnp
        self.use_jit = use_jit
    
    def compute_forces(self, energy_fn, positions, **kwargs):
        """Compute forces using JAX autodiff"""
        positions_jax = self.jnp.array(positions)
        
        grad_fn = self.jax.grad(lambda pos: energy_fn(pos, **kwargs))
        
        if self.use_jit:
            grad_fn = self.jax.jit(grad_fn)
        
        gradient = grad_fn(positions_jax)
        forces = -np.array(gradient)
        
        return forces
    
    def get_name(self):
        return f"JAX(jit={self.use_jit})"


class PyTorchBackend(AutoDiffBackend):
    """
    PyTorch autodiff backend
    
    Advantages: Familiar, GPU support, extensive ecosystem
    """
    
    def __init__(self, device: str = 'cpu'):
        import torch
        self.torch = torch
        self.device = device
    
    def compute_forces(self, energy_fn, positions, **kwargs):
        """Compute forces using PyTorch autodiff"""
        positions_torch = self.torch.tensor(
            positions, 
            requires_grad=True, 
            dtype=self.torch.float64,
            device=self.device
        )
        
        energy = energy_fn(positions_torch.cpu().numpy(), **kwargs)
        energy_torch = self.torch.tensor(energy, device=self.device)
        
        energy_torch.backward()
        forces = -positions_torch.grad.cpu().numpy()
        
        return forces
    
    def get_name(self):
        return f"PyTorch(device={self.device})"


class AutogradBackend(AutoDiffBackend):
    """
    Autograd backend (NumPy-based)
    
    Advantages: Lightweight, pure NumPy
    """
    
    def __init__(self):
        import autograd.numpy as anp
        from autograd import grad
        self.anp = anp
        self.grad = grad
    
    def compute_forces(self, energy_fn, positions, **kwargs):
        """Compute forces using autograd"""
        def energy_fn_wrapped(pos):
            return energy_fn(pos, **kwargs)
        
        grad_fn = self.grad(energy_fn_wrapped)
        gradient = grad_fn(positions)
        forces = -gradient
        
        return forces
    
    def get_name(self):
        return "Autograd(NumPy)"


class BackendFactory:
    """Factory for creating autodiff backends"""
    
    @staticmethod
    def create(backend_name: str, **kwargs) -> AutoDiffBackend:
        """
        Create backend by name
        
        Args:
            backend_name: 'jax', 'pytorch', or 'autograd'
            **kwargs: Backend-specific options
            
        Example:
            backend = BackendFactory.create('jax', use_jit=True)
            backend = BackendFactory.create('pytorch', device='cuda')
        """
        backend_name = backend_name.lower()
        
        if backend_name == 'jax':
            return JAXBackend(**kwargs)
        elif backend_name in ['pytorch', 'torch']:
            return PyTorchBackend(**kwargs)
        elif backend_name == 'autograd':
            return AutogradBackend(**kwargs)
        else:
            raise ValueError(f"Unknown backend: {backend_name}")
```

#### **ForceCalculator**
```python
class ForceCalculator:
    """
    Computes forces from potential energy using autodiff
    
    This is where the magic happens:
    User provides energy function E(positions)
    This class computes F = -∇E automatically
    """
    
    def __init__(self, potential: PotentialEnergy,
                 backend: AutoDiffBackend,
                 neighbor_list: Optional[NeighborList] = None):
        self.potential = potential
        self.backend = backend
        self.neighbor_list = neighbor_list
    
    def compute_forces(self, system: System) -> np.ndarray:
        """
        Compute forces on all atoms using autodiff
        
        Process:
        1. Check if neighbor list needs rebuilding
        2. Create energy function with fixed parameters
        3. Call autodiff backend to compute -∇E
        4. Return forces
        """
        # Rebuild neighbor list if needed
        if self.neighbor_list is not None:
            if self.neighbor_list.needs_rebuild(system.state.positions):
                self.neighbor_list.build(
                    system.state.positions,
                    system.state.box,
                    system.boundary_condition
                )
        
        # Create energy function
        def energy_fn(positions):
            return self.potential.compute_energy(
                positions,
                system.state.box,
                system.boundary_condition,
                atom_types=system.get_atom_types(),
                neighbor_list=self.neighbor_list
            )
        
        # Autodiff computes forces = -∇E
        forces = self.backend.compute_forces(energy_fn, system.state.positions)
        return forces
```

---

### 6. Integrator Module (Strategy Pattern)

```python
class Integrator(ABC):
    """Abstract base for time integration schemes"""
    
    def __init__(self, timestep: float):
        self.dt = timestep
    
    @abstractmethod
    def step(self, system: System, force_calculator: ForceCalculator) -> None:
        """Advance system by one timestep"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        pass


class VelocityVerlet(Integrator):
    """
    Velocity Verlet algorithm
    
    Symplectic, time-reversible, second-order accurate
    
    Algorithm:
        v(t+dt/2) = v(t) + (F(t)/m) * dt/2
        r(t+dt) = r(t) + v(t+dt/2) * dt
        F(t+dt) = compute forces at new positions
        v(t+dt) = v(t+dt/2) + (F(t+dt)/m) * dt/2
    """
    
    def step(self, system, force_calculator):
        """Advance by one timestep using Velocity Verlet"""
        dt = self.dt
        masses = system.get_masses()[:, np.newaxis]  # (N, 1)
        
        # Half-step velocity update
        system.state.velocities += 0.5 * system.state.forces / masses * dt
        
        # Full-step position update
        system.state.positions += system.state.velocities * dt
        
        # Compute new forces
        system.state.forces = force_calculator.compute_forces(system)
        
        # Half-step velocity update
        system.state.velocities += 0.5 * system.state.forces / masses * dt
    
    def get_name(self):
        return f"VelocityVerlet(dt={self.dt})"


class LeapfrogIntegrator(Integrator):
    """Leapfrog (positions and velocities staggered)"""
    
    def step(self, system, force_calculator):
        # Implementation...
        pass
    
    def get_name(self):
        return f"Leapfrog(dt={self.dt})"


class LangevinIntegrator(Integrator):
    """Langevin dynamics (stochastic, includes friction)"""
    
    def __init__(self, timestep: float, temperature: float, friction: float):
        super().__init__(timestep)
        self.temperature = temperature
        self.friction = friction
    
    def step(self, system, force_calculator):
        # Implementation with friction and random forces...
        pass
    
    def get_name(self):
        return f"Langevin(dt={self.dt}, T={self.temperature})"
```

---

### 7. Thermostat Module (Strategy Pattern)

```python
class Thermostat(ABC):
    """Abstract base for temperature control"""
    
    @abstractmethod
    def apply(self, system: System, target_temperature: float) -> None:
        """Apply temperature control to velocities"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        pass


class NoThermostat(Thermostat):
    """No temperature control (NVE ensemble)"""
    
    def apply(self, system, target_temperature):
        """Do nothing - NVE ensemble"""
        pass
    
    def get_name(self):
        return "NoThermostat(NVE)"


class BerendsenThermostat(Thermostat):
    """
    Berendsen weak coupling
    
    Fast equilibration but not canonical ensemble
    Good for equilibration phase
    """
    
    def __init__(self, coupling_time: float):
        self.tau = coupling_time
    
    def apply(self, system, target_temperature):
        """Scale velocities toward target temperature"""
        current_T = system.compute_temperature()
        
        # Berendsen scaling factor
        lambda_factor = np.sqrt(1.0 + (1.0 / self.tau) * 
                               (target_temperature / current_T - 1.0))
        
        system.state.velocities *= lambda_factor
    
    def get_name(self):
        return f"Berendsen(τ={self.tau})"


class NoseHooverThermostat(Thermostat):
    """
    Nosé-Hoover chains
    
    Proper canonical NVT ensemble
    """
    
    def __init__(self, coupling_time: float):
        self.tau = coupling_time
        self.xi = 0.0  # Extended variable
    
    def apply(self, system, target_temperature):
        """Apply Nosé-Hoover thermostat"""
        # Implementation with extended variable dynamics...
        pass
    
    def get_name(self):
        return f"NoseHoover(τ={self.tau})"


class AndersenThermostat(Thermostat):
    """Stochastic collisions with heat bath"""
    
    def __init__(self, collision_frequency: float):
        self.nu = collision_frequency
    
    def apply(self, system, target_temperature):
        """Randomly reassign velocities"""
        # Implementation with stochastic collisions...
        pass
    
    def get_name(self):
        return f"Andersen(ν={self.nu})"
```

---

### 8. Observer Module (Observer Pattern)

```python
class Observer(ABC):
    """Abstract base for observing simulation"""
    
    @abstractmethod
    def update(self, system: System, step: int) -> None:
        """Called at observation intervals"""
        pass
    
    @abstractmethod
    def finalize(self) -> None:
        """Called when simulation ends"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        pass


class ObserverManager:
    """Manages collection of observers (Mediator pattern)"""
    
    def __init__(self):
        self.observers: List[Observer] = []
    
    def add_observer(self, observer: Observer) -> None:
        self.observers.append(observer)
    
    def notify_all(self, system: System, step: int) -> None:
        for observer in self.observers:
            observer.update(system, step)
    
    def finalize_all(self) -> None:
        for observer in self.observers:
            observer.finalize()


class EnergyObserver(Observer):
    """Track KE, PE, total energy"""
    
    def __init__(self, interval: int = 10):
        self.interval = interval
        self.energies = {
            'kinetic': [],
            'potential': [],
            'total': [],
            'time': []
        }
    
    def update(self, system, step):
        if step % self.interval == 0:
            ke = system.compute_kinetic_energy()
            # PE would be computed by force calculator
            # For now, store what we can
            self.energies['time'].append(system.state.time)
            self.energies['kinetic'].append(ke)
    
    def finalize(self):
        """Could save to file or plot"""
        print(f"Tracked {len(self.energies['time'])} energy measurements")
    
    def get_name(self):
        return f"EnergyObserver(interval={self.interval})"


class TrajectoryWriter(Observer):
    """
    Write trajectories in OVITO-compatible formats
    
    Formats: XYZ, LAMMPS dump, Extended XYZ
    """
    
    def __init__(self, filename: str, interval: int = 100,
                 format: str = 'xyz', properties: List[str] = None):
        self.filename = filename
        self.interval = interval
        self.format = format.lower()
        self.properties = properties or []
        self.file = None
        self.frame_count = 0
    
    def update(self, system, step):
        """Write current frame if at interval"""
        if step % self.interval != 0:
            return
        
        if self.file is None:
            self.file = open(self.filename, 'w')
        
        if self.format == 'xyz':
            self._write_xyz(system, step)
        elif self.format == 'lammps':
            self._write_lammps_dump(system, step)
        elif self.format == 'exyz':
            self._write_extended_xyz(system, step)
        
        self.frame_count += 1
    
    def _write_xyz(self, system, step):
        """Write standard XYZ format (OVITO-compatible)"""
        n_atoms = system.get_num_atoms()
        
        # Header
        self.file.write(f"{n_atoms}\n")
        
        # Comment line with metadata
        comment = f"Step={step} Time={system.state.time:.3f}"
        self.file.write(comment + "\n")
        
        # Atom data
        for i, atom in enumerate(system.atoms):
            pos = system.state.positions[i]
            self.file.write(f"{atom.atom_type} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")
        
        self.file.flush()
    
    def _write_lammps_dump(self, system, step):
        """Write LAMMPS dump format"""
        # Implementation...
        pass
    
    def _write_extended_xyz(self, system, step):
        """Write extended XYZ format"""
        # Implementation...
        pass
    
    def finalize(self):
        """Close file"""
        if self.file is not None:
            self.file.close()
            print(f"Wrote {self.frame_count} frames to {self.filename}")
    
    def get_name(self):
        return f"TrajectoryWriter({self.format}, {self.filename})"
```

---

### 9. Simulator Module (Template Method)

```python
class MDSimulator:
    """
    Main simulation engine
    
    Template Method Pattern: Defines simulation algorithm
    Facade Pattern: Simple interface to complex subsystem
    """
    
    def __init__(self, system: System,
                 force_calculator: ForceCalculator,
                 integrator: Integrator,
                 thermostat: Thermostat,
                 target_temperature: Optional[float] = None):
        self.system = system
        self.force_calculator = force_calculator
        self.integrator = integrator
        self.thermostat = thermostat
        self.target_temperature = target_temperature
        self.observer_manager = ObserverManager()
    
    def add_observer(self, observer: Observer) -> None:
        """Add observer to monitor simulation"""
        self.observer_manager.add_observer(observer)
    
    def setup(self) -> None:
        """Initialize simulation"""
        self.system.state.forces = self.force_calculator.compute_forces(self.system)
        print(f"Simulation initialized: {self.system.get_num_atoms()} atoms")
    
    def step(self) -> None:
        """
        Single MD step (Template Method)
        
        1. Integrate equations of motion
        2. Apply boundary conditions
        3. Apply thermostat
        4. Update state
        """
        # Integration
        self.integrator.step(self.system, self.force_calculator)
        
        # Boundary conditions
        self.system.wrap_positions()
        
        # Thermostat
        if self.target_temperature is not None:
            self.thermostat.apply(self.system, self.target_temperature)
        
        # Update counters
        self.system.state.step += 1
        self.system.state.time += self.integrator.dt
    
    def run(self, n_steps: int, observe_interval: int = 1) -> None:
        """Run simulation for n_steps"""
        self.setup()
        
        print(f"Running {n_steps} steps...")
        for i in range(n_steps):
            self.step()
            
            if i % observe_interval == 0:
                self.observer_manager.notify_all(self.system, i)
            
            if (i + 1) % (n_steps // 10) == 0:
                print(f"  Step {i+1}/{n_steps} ({100*(i+1)//n_steps}%)")
        
        self.finalize()
    
    def finalize(self) -> None:
        """Clean up"""
        self.observer_manager.finalize_all()
        print("Simulation complete!")
```

---

### 10. Builder Module

```python
class SystemBuilder:
    """
    Builder for System objects with automatic element lookup
    
    Users can specify atoms by symbol - masses looked up automatically!
    """
    
    def __init__(self, units: UnitSystem):
        self.units = units
        self.atoms = []
        self.positions = []
        self.box = None
        self.boundary_condition = None
        self.element_registry = ElementRegistry()
    
    def add_atom(self, 
                 atom_type: str, 
                 position: np.ndarray,
                 mass: Optional[float] = None,
                 charge: float = 0.0) -> 'SystemBuilder':
        """
        Add atom to system
        
        Args:
            atom_type: Element symbol (e.g., 'Cu', 'Ar', 'H')
            position: (3,) array of coordinates
            mass: Optional mass override. If None, looked up from registry.
            charge: Electric charge
            
        Example:
            builder.add_atom('Cu', position=[0, 0, 0])  # Mass automatic!
        """
        # Lookup mass if not provided
        if mass is None:
            mass = self.element_registry.get_mass(atom_type)
        
        atom = Atom(
            atom_type=atom_type,
            mass=mass,
            charge=charge,
            index=len(self.atoms)
        )
        self.atoms.append(atom)
        self.positions.append(position)
        
        return self
    
    def set_box(self, box: np.ndarray) -> 'SystemBuilder':
        self.box = box
        return self
    
    def set_boundary_condition(self, bc: BoundaryCondition) -> 'SystemBuilder':
        self.boundary_condition = bc
        return self
    
    def initialize_velocities(self, temperature: float) -> 'SystemBuilder':
        """Initialize velocities from Maxwell-Boltzmann distribution"""
        n_atoms = len(self.atoms)
        velocities = np.zeros((n_atoms, 3))
        
        for i, atom in enumerate(self.atoms):
            sigma = np.sqrt(self.units.boltzmann * temperature / atom.mass)
            velocities[i] = np.random.normal(0, sigma, size=3)
        
        # Remove center-of-mass velocity
        total_mass = sum(atom.mass for atom in self.atoms)
        com_velocity = sum(atom.mass * velocities[i] 
                          for i, atom in enumerate(self.atoms)) / total_mass
        velocities -= com_velocity
        
        self.velocities = velocities
        return self
    
    def build(self) -> System:
        """Construct final System object"""
        if self.boundary_condition is None:
            self.boundary_condition = PeriodicBoundaryCondition()
        
        if not hasattr(self, 'velocities'):
            self.velocities = np.zeros((len(self.atoms), 3))
        
        positions = np.array(self.positions)
        
        system = System(
            atoms=self.atoms,
            initial_positions=positions,
            box=self.box,
            boundary_condition=self.boundary_condition,
            units=self.units
        )
        system.state.velocities = self.velocities
        
        return system
```

---

## Design Patterns Summary

| Pattern | Location | Purpose |
|---------|----------|---------|
| **Strategy** | BoundaryCondition, NeighborList, PotentialEnergy, Integrator, Thermostat, AutoDiffBackend | Interchangeable algorithms |
| **Composite** | CompositePotential | Combine multiple potentials |
| **Observer** | Observer + ObserverManager | Decouple monitoring from simulation |
| **Template Method** | MDSimulator.step() | Define algorithm skeleton |
| **Builder** | SystemBuilder | Construct complex System objects |
| **Factory** | BackendFactory, PotentialFactory, etc. | Create objects from specifications |
| **Facade** | MDSimulator | Simple interface to complex subsystem |
| **Singleton** | ElementRegistry | Single global element database |

---

## Implementation Roadmap

### **Phase 1: Core Foundation** (Week 1)
- [ ] Units module (REAL, METAL, LJ, SI)
- [ ] Atom, State, System classes
- [ ] ElementRegistry with common elements
- [ ] BoundaryCondition (Periodic, Open, Mixed)
- [ ] Unit tests for core

### **Phase 2: Neighbor Lists** (Week 1)
- [ ] NeighborList ABC
- [ ] BruteForce, Verlet, Cell implementations
- [ ] Unit tests for neighbor lists

### **Phase 3: Autodiff Infrastructure** (Week 2)
- [ ] AutoDiffBackend ABC
- [ ] JAXBackend
- [ ] PyTorchBackend
- [ ] BackendFactory
- [ ] ForceCalculator
- [ ] Unit tests

### **Phase 4: Simple Potentials** (Week 2)
- [ ] PotentialEnergy ABC
- [ ] LennardJones
- [ ] Harmonic bond
- [ ] Test forces vs analytical
- [ ] Integration test: backend consistency

### **Phase 5: Integration & Simulation** (Week 2-3)
- [ ] Integrator ABC + VelocityVerlet
- [ ] Basic MDSimulator
- [ ] Run first LJ simulation!
- [ ] Integration test: NVE conservation

### **Phase 6: Thermostats** (Week 3)
- [ ] Thermostat ABC
- [ ] NoThermostat, Berendsen, NoseHoover
- [ ] Integration test: NVT temperature

### **Phase 7: EAM Potentials** (Week 3-4)
- [ ] EAMBasePotential
- [ ] EAMPotential (Sutton-Chen)
- [ ] EAMAlloyPotential
- [ ] FinnisSinclair
- [ ] Test EAM forces via autodiff
- [ ] Integration test: Cu FCC simulation

### **Phase 8: MEAM & Advanced** (Week 4)
- [ ] MEAMPotential with angular terms
- [ ] ADPPotential
- [ ] CustomManyBodyPotential
- [ ] Integration test: MEAM simulation

### **Phase 9: Observers** (Week 4-5)
- [ ] Observer pattern infrastructure
- [ ] EnergyObserver, TemperatureObserver
- [ ] TrajectoryWriter (XYZ, LAMMPS, Extended XYZ)
- [ ] PressureObserver, RDFObserver

### **Phase 10: Config & Factories** (Week 5)
- [ ] YAML configuration support
- [ ] SystemBuilder (with ElementRegistry)
- [ ] PotentialFactory
- [ ] ConfigLoader
- [ ] Example YAML configs

### **Phase 11: Polish & Examples** (Week 5-6)
- [ ] Complete docstrings
- [ ] AutogradBackend
- [ ] More integrators
- [ ] Coulomb potential
- [ ] Example notebooks

### **Phase 12: Documentation** (Week 6)
- [ ] README with quickstart
- [ ] Tutorials
- [ ] OVITO visualization guide

---

## Testing Strategy

### **Unit Tests**
```python
tests/unit/
├── test_boundary.py         # BC algorithms
├── test_neighbor_list.py    # Neighbor list algorithms
├── test_potential.py        # Energy calculations
├── test_force_calculator.py # Force accuracy vs analytical
├── test_integrator.py       # Single step correctness
├── test_thermostat.py       # Temperature control
├── test_units.py            # Unit conversions
└── test_element_registry.py # Element lookup
```

### **Integration Tests**
```python
tests/integration/
├── test_nve_conservation.py       # Energy conservation
├── test_nvt_temperature.py        # Temperature control
├── test_backend_consistency.py    # JAX vs PyTorch
├── test_neighbor_list_correctness.py  # Verlet vs Brute
├── test_eam_simulation.py         # EAM works correctly
└── test_meam_simulation.py        # MEAM works correctly
```

---

## Key Design Decisions

### **1. Box Location: System.state**
```
✓ System owns box (data/state)
✓ BoundaryCondition receives box as parameter (strategy)

WHY:
- Box is physical property of system
- BC is pure strategy (stateless)
- Allows box to change (NPT, deformation)
- BC is reusable across systems
```

### **2. Energy-Only Interface**
```
╔═══════════════════════════════════════════════════╗
║  USER WRITES:  E(positions)                      ║
║  AUTODIFF GIVES: F = -∇E                         ║
║                                                  ║
║  Works for ALL potentials including EAM/MEAM!    ║
╚═══════════════════════════════════════════════════╝
```

### **3. Neighbor Lists: Independent Strategy**
```
✓ Independent class (not in System or ForceCalculator)
✓ Multiple algorithms (Brute, Verlet, Cell)
✓ Passed to ForceCalculator
```

### **4. Element Registry: Singleton**
```
✓ Single global registry
✓ Automatic mass lookup
✓ User convenience
✓ Supports custom elements
```

---

## Usage Examples

### **Example 1: Simple LJ Gas**
```python
from pyMD import *

# Setup
units = Units.LJ()
builder = SystemBuilder(units)

# Add atoms (mass automatic!)
for i in range(100):
    builder.add_atom('Ar', position=np.random.rand(3) * 10)

system = (builder
          .set_box(np.array([10, 10, 10]))
          .set_boundary_condition(PeriodicBoundaryCondition())
          .initialize_velocities(temperature=1.0)
          .build())

# Potential (user writes energy only!)
potential = LennardJonesPotential(epsilon=1.0, sigma=1.0, cutoff=2.5)

# Neighbor list
neighbor_list = CellList(cutoff=2.5, skin=0.3)

# Backend (easy switching!)
backend = BackendFactory.create('jax', use_jit=True)

# Forces automatic!
force_calc = ForceCalculator(potential, backend, neighbor_list)

# Simulation
integrator = VelocityVerlet(timestep=0.005)
thermostat = BerendsenThermostat(coupling_time=0.5)

sim = MDSimulator(system, force_calc, integrator, thermostat, temperature=1.0)
sim.add_observer(TrajectoryWriter('output.xyz', interval=100))
sim.add_observer(EnergyObserver(interval=10))

sim.run(n_steps=10000)
```

### **Example 2: Copper EAM**
```python
# Copper FCC with EAM
units = Units.METAL()
builder = SystemBuilder(units)

# Build FCC lattice (mass automatic!)
builder.add_atoms_from_lattice('Cu', 'fcc', 3.615, 5, 5, 5)
system = builder.build()

# EAM potential (user writes energy only!)
eam_cu = EAMPotential.sutton_chen(
    epsilon=0.0124, a=3.615, c=39.432, m=6, n=12, cutoff=5.5
)

# Forces computed automatically via autodiff!
backend = BackendFactory.create('jax')
force_calc = ForceCalculator(eam_cu, backend, neighbor_list)

sim = MDSimulator(...)
sim.run(n_steps=10000)
```

### **Example 3: Custom Research Potential**
```python
def my_novel_potential(positions, box, bc, cutoff, alpha):
    """User's research potential - only write energy!"""
    energy = 0.0
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            r_vec = positions[j] - positions[i]
            r_vec = bc.apply_minimum_image(r_vec, box)
            r = np.linalg.norm(r_vec)
            if r < cutoff:
                energy += alpha * np.exp(-r) * np.sin(r)
    return energy

# Use in simulation - forces automatic!
custom_pot = CustomManyBodyPotential(my_novel_potential, cutoff=8.0, alpha=2.5)
force_calc = ForceCalculator(custom_pot, backend)
```

---

## Summary

This design provides a complete, educational, well-architected MD framework where:

✅ Users write ONLY energy functions
✅ Forces computed automatically via autodiff
✅ Full EAM/MEAM support
✅ Easy backend switching (JAX/PyTorch)
✅ LAMMPS-style units
✅ Independent neighbor lists
✅ Element registry for convenience
✅ Clean OOP with design patterns
✅ Comprehensive testing
✅ OVITO-compatible output

**Ready for implementation by Claude Opus 4.6!**
