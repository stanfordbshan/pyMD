"""
Element registry for chemical element properties.

This module provides a registry of chemical elements with their
properties (mass, atomic number, radii, etc.) using a Singleton pattern.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Union


@dataclass(frozen=True)
class ElementData:
    """
    Immutable data for a chemical element.

    Contains standard atomic properties that don't change during simulation.

    Attributes:
        symbol: Chemical symbol (e.g., "Cu", "Ar", "H").
        name: Full element name (e.g., "Copper", "Argon").
        atomic_number: Atomic number Z (1 for H, 29 for Cu, etc.).
        atomic_mass: Standard atomic mass in amu (g/mol).
        covalent_radius: Covalent radius in Ångströms (optional).
        vdw_radius: Van der Waals radius in Ångströms (optional).
        electronegativity: Pauling electronegativity scale (optional).
        color: CPK color for visualization as hex string (optional).

    Example:
        >>> from pyMD.core.element_registry import ElementData
        >>> cu = ElementData("Cu", "Copper", 29, 63.546, 1.32, 1.40, 1.90, "#C88033")
    """
    symbol: str
    name: str
    atomic_number: int
    atomic_mass: float
    covalent_radius: Optional[float] = None
    vdw_radius: Optional[float] = None
    electronegativity: Optional[float] = None
    color: Optional[str] = None


class ElementRegistry:
    """
    Registry for chemical element properties (Singleton pattern).

    Provides convenient lookup of atomic masses and other properties.
    Users can access elements by symbol or atomic number.

    The registry is a singleton - only one instance exists throughout
    the application lifetime.

    Attributes:
        _instance: The singleton instance.
        _initialized: Whether the periodic table has been initialized.

    Example:
        >>> from pyMD.core import elements
        >>> # Get mass by symbol
        >>> cu_mass = elements.get_mass('Cu')
        >>> print(cu_mass)
        63.546
        >>> # Get full element data
        >>> ar_data = elements.get_element('Ar')
        >>> print(ar_data.atomic_number)
        18
        >>> # Check if element exists
        >>> print('Fe' in elements)
        True
    """

    _instance: Optional["ElementRegistry"] = None
    _initialized: bool = False

    def __new__(cls) -> "ElementRegistry":
        """Singleton: Only one registry instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize periodic table data (only once)."""
        if not ElementRegistry._initialized:
            self._elements_by_symbol: Dict[str, ElementData] = {}
            self._elements_by_number: Dict[int, ElementData] = {}
            self._initialize_periodic_table()
            ElementRegistry._initialized = True

    def _initialize_periodic_table(self) -> None:
        """
        Populate registry with standard element data.

        Data sources:
            - Atomic masses: IUPAC standard atomic weights
            - Radii: Cordero et al. (2008)
            - Colors: CPK coloring scheme
        """
        elements_data = [
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
            ElementData("Mn", "Manganese", 25, 54.938, 1.39, 2.05, 1.55, "#9C7AC7"),
            ElementData("Co", "Cobalt", 27, 58.933, 1.26, 2.00, 1.88, "#F090A0"),
            ElementData("Zn", "Zinc", 30, 65.38, 1.22, 1.39, 1.65, "#7D80B0"),
            ElementData("Mo", "Molybdenum", 42, 95.95, 1.54, 2.10, 2.16, "#54B5B5"),
            ElementData("W", "Tungsten", 74, 183.84, 1.62, 2.10, 2.36, "#2194D6"),
            ElementData("Ta", "Tantalum", 73, 180.95, 1.70, 2.20, 1.50, "#4DC2FF"),
            ElementData("Nb", "Niobium", 41, 92.906, 1.64, 2.10, 1.60, "#73C2C9"),
            ElementData("V", "Vanadium", 23, 50.942, 1.53, 2.05, 1.63, "#A6A6AB"),
            ElementData("Zr", "Zirconium", 40, 91.224, 1.75, 2.15, 1.33, "#94E0E0"),
            ElementData("Hf", "Hafnium", 72, 178.49, 1.75, 2.20, 1.30, "#4DC2FF"),

            # Light elements (biomolecular simulations)
            ElementData("H", "Hydrogen", 1, 1.0080, 0.31, 1.10, 2.20, "#FFFFFF"),
            ElementData("C", "Carbon", 6, 12.011, 0.76, 1.70, 2.55, "#909090"),
            ElementData("N", "Nitrogen", 7, 14.007, 0.71, 1.55, 3.04, "#3050F8"),
            ElementData("O", "Oxygen", 8, 15.999, 0.66, 1.52, 3.44, "#FF0D0D"),
            ElementData("F", "Fluorine", 9, 18.998, 0.57, 1.47, 3.98, "#90E050"),
            ElementData("S", "Sulfur", 16, 32.06, 1.05, 1.80, 2.58, "#FFFF30"),
            ElementData("P", "Phosphorus", 15, 30.974, 1.07, 1.80, 2.19, "#FF8000"),
            ElementData("Cl", "Chlorine", 17, 35.45, 1.02, 1.75, 3.16, "#1FF01F"),
            ElementData("Br", "Bromine", 35, 79.904, 1.20, 1.85, 2.96, "#A62929"),
            ElementData("I", "Iodine", 53, 126.90, 1.39, 1.98, 2.66, "#940094"),

            # Alkali metals
            ElementData("Li", "Lithium", 3, 6.941, 1.28, 1.82, 0.98, "#CC80FF"),
            ElementData("Na", "Sodium", 11, 22.990, 1.66, 2.27, 0.93, "#AB5CF2"),
            ElementData("K", "Potassium", 19, 39.098, 2.03, 2.75, 0.82, "#8F40D4"),
            ElementData("Rb", "Rubidium", 37, 85.468, 2.20, 3.03, 0.82, "#702EB0"),
            ElementData("Cs", "Cesium", 55, 132.91, 2.44, 3.43, 0.79, "#57178F"),

            # Alkaline earth metals
            ElementData("Be", "Beryllium", 4, 9.0122, 0.96, 1.53, 1.57, "#C2FF00"),
            ElementData("Mg", "Magnesium", 12, 24.305, 1.41, 1.73, 1.31, "#8AFF00"),
            ElementData("Ca", "Calcium", 20, 40.078, 1.76, 2.31, 1.00, "#3DFF00"),
            ElementData("Sr", "Strontium", 38, 87.62, 1.95, 2.49, 0.95, "#00FF00"),
            ElementData("Ba", "Barium", 56, 137.33, 2.15, 2.68, 0.89, "#00C900"),

            # Semiconductors and other common elements
            ElementData("Si", "Silicon", 14, 28.085, 1.11, 2.10, 1.90, "#F0C8A0"),
            ElementData("Ge", "Germanium", 32, 72.63, 1.20, 2.11, 2.01, "#668F8F"),
            ElementData("Ga", "Gallium", 31, 69.723, 1.22, 1.87, 1.81, "#C28F8F"),
            ElementData("As", "Arsenic", 33, 74.922, 1.19, 1.85, 2.18, "#BD80E3"),
            ElementData("Se", "Selenium", 34, 78.971, 1.20, 1.90, 2.55, "#FFA100"),
            ElementData("Sn", "Tin", 50, 118.71, 1.39, 2.17, 1.96, "#668080"),
            ElementData("Pb", "Lead", 82, 207.2, 1.46, 2.02, 2.33, "#575961"),
            ElementData("Bi", "Bismuth", 83, 208.98, 1.48, 2.07, 2.02, "#9E4FB5"),
        ]

        for element in elements_data:
            self._elements_by_symbol[element.symbol] = element
            self._elements_by_number[element.atomic_number] = element

    def get_element(self, identifier: Union[str, int]) -> Optional[ElementData]:
        """
        Get element data by symbol or atomic number.

        Args:
            identifier: Element symbol (str, e.g., 'Cu') or atomic number (int, e.g., 29).

        Returns:
            ElementData if found, None otherwise.

        Raises:
            TypeError: If identifier is neither str nor int.

        Example:
            >>> registry = ElementRegistry()
            >>> cu = registry.get_element('Cu')
            >>> fe = registry.get_element(26)
        """
        if isinstance(identifier, str):
            return self._elements_by_symbol.get(identifier)
        elif isinstance(identifier, int):
            return self._elements_by_number.get(identifier)
        else:
            raise TypeError(
                f"Identifier must be str or int, got {type(identifier).__name__}"
            )

    def get_mass(self, symbol: str) -> float:
        """
        Get atomic mass by element symbol.

        Args:
            symbol: Chemical element symbol (e.g., 'Cu', 'Ar').

        Returns:
            Atomic mass in amu (g/mol).

        Raises:
            KeyError: If element not found in registry.

        Example:
            >>> from pyMD.core import elements
            >>> mass = elements.get_mass('Cu')
            >>> print(mass)
            63.546
        """
        element = self.get_element(symbol)
        if element is None:
            raise KeyError(f"Element '{symbol}' not found in registry")
        return element.atomic_mass

    def has_element(self, identifier: Union[str, int]) -> bool:
        """
        Check if element exists in registry.

        Args:
            identifier: Element symbol (str) or atomic number (int).

        Returns:
            True if element exists, False otherwise.
        """
        return self.get_element(identifier) is not None

    def list_elements(self) -> List[str]:
        """
        Return list of all available element symbols.

        Returns:
            Sorted list of element symbols.
        """
        return sorted(self._elements_by_symbol.keys())

    def add_custom_element(self, element: ElementData) -> None:
        """
        Add custom element or pseudoatom to registry.

        Useful for coarse-grained simulations, united atoms, or dummy atoms.

        Args:
            element: ElementData for the custom element.

        Raises:
            ValueError: If element symbol already exists.

        Example:
            >>> from pyMD.core import elements, ElementData
            >>> ch3 = ElementData("CH3", "Methyl", 0, 15.035)
            >>> elements.add_custom_element(ch3)
        """
        if element.symbol in self._elements_by_symbol:
            raise ValueError(f"Element '{element.symbol}' already exists")

        self._elements_by_symbol[element.symbol] = element
        if element.atomic_number > 0:
            self._elements_by_number[element.atomic_number] = element

    def __contains__(self, identifier: Union[str, int]) -> bool:
        """Support 'in' operator."""
        return self.has_element(identifier)

    def __getitem__(self, identifier: Union[str, int]) -> ElementData:
        """
        Support indexing: registry['Cu'] or registry[29].

        Raises:
            KeyError: If element not found.
        """
        element = self.get_element(identifier)
        if element is None:
            raise KeyError(f"Element '{identifier}' not found")
        return element

    def __len__(self) -> int:
        """Return number of elements in registry."""
        return len(self._elements_by_symbol)


# Module-level convenience instance (Singleton)
elements = ElementRegistry()
