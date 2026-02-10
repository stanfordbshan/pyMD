"""
Unit system management for molecular dynamics simulations.

This module provides LAMMPS-style unit systems (REAL, METAL, LJ, SI)
with complete unit definitions and conversion factors.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Final


class UnitSystemType(Enum):
    """Supported unit systems for MD simulations."""
    REAL = "real"    # Å, kcal/mol, fs, K (biomolecular)
    METAL = "metal"  # Å, eV, ps, K (materials science)
    SI = "si"        # m, J, s, K (base SI units)
    LJ = "lj"        # Reduced/dimensionless units


@dataclass(frozen=True)
class UnitSystem:
    """
    Complete unit system specification.

    Defines all units and conversion factors for a specific unit system.
    Users select one unit system at simulation start, and all quantities
    are interpreted in that system.

    Attributes:
        name: The type of unit system (REAL, METAL, SI, LJ).
        length_unit: String name of length unit (e.g., "angstrom", "meter").
        energy_unit: String name of energy unit (e.g., "kcal/mol", "eV").
        time_unit: String name of time unit (e.g., "femtosecond", "picosecond").
        mass_unit: String name of mass unit (e.g., "g/mol", "kilogram").
        temperature_unit: String name of temperature unit (typically "kelvin").
        charge_unit: String name of charge unit (e.g., "e" for elementary charge).
        boltzmann: Boltzmann constant in this unit system.
        pressure_unit: String name of pressure unit (derived).
        force_unit: String name of force unit (derived).
        length_to_si: Conversion factor from this length unit to meters.
        energy_to_si: Conversion factor from this energy unit to Joules.
        time_to_si: Conversion factor from this time unit to seconds.
        mass_to_si: Conversion factor from this mass unit to kilograms.

    Example:
        >>> from pyMD.core import Units
        >>> metal = Units.METAL()
        >>> print(f"Energy unit: {metal.energy_unit}")
        Energy unit: eV
        >>> print(f"kB = {metal.boltzmann} {metal.energy_unit}/K")
        kB = 8.617333e-05 eV/K
    """
    name: UnitSystemType
    length_unit: str
    energy_unit: str
    time_unit: str
    mass_unit: str
    temperature_unit: str
    charge_unit: str
    boltzmann: float
    pressure_unit: str
    force_unit: str
    length_to_si: float
    energy_to_si: float
    time_to_si: float
    mass_to_si: float


class Units:
    """
    Factory for standard unit systems.

    Provides static methods to create the four standard unit systems
    used in MD simulations. Each method returns a fully configured
    UnitSystem object.

    Example:
        >>> units = Units.METAL()
        >>> print(units.name)
        UnitSystemType.METAL
    """

    @staticmethod
    def REAL() -> UnitSystem:
        """
        Create REAL units (common in biomolecular simulations).

        Unit definitions:
            - Distance: Ångströms (Å)
            - Energy: kcal/mol
            - Time: femtoseconds (fs)
            - Mass: g/mol (grams per mole)
            - Temperature: Kelvin (K)
            - Charge: elementary charge (e)

        Returns:
            UnitSystem configured for REAL units.
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
            energy_to_si=4184.0 / 6.02214076e23,
            time_to_si=1e-15,
            mass_to_si=1.66054e-27,
        )

    @staticmethod
    def METAL() -> UnitSystem:
        """
        Create METAL units (common in materials science).

        Unit definitions:
            - Distance: Ångströms (Å)
            - Energy: electron volts (eV)
            - Time: picoseconds (ps)
            - Mass: g/mol (grams per mole)
            - Temperature: Kelvin (K)
            - Charge: elementary charge (e)

        Returns:
            UnitSystem configured for METAL units.
        """
        return UnitSystem(
            name=UnitSystemType.METAL,
            length_unit="angstrom",
            energy_unit="eV",
            time_unit="picosecond",
            mass_unit="g/mol",
            temperature_unit="kelvin",
            charge_unit="e",
            boltzmann=8.617333262e-5,  # eV/K
            pressure_unit="bar",
            force_unit="eV/angstrom",
            length_to_si=1e-10,
            energy_to_si=1.602176634e-19,
            time_to_si=1e-12,
            mass_to_si=1.66054e-27,
        )

    @staticmethod
    def LJ() -> UnitSystem:
        """
        Create LJ (Lennard-Jones) reduced units (dimensionless).

        In reduced units, all quantities are expressed in terms of
        the LJ parameters σ (length), ε (energy), and m (mass).

        Unit definitions:
            - Distance: σ (LJ length parameter)
            - Energy: ε (LJ energy parameter)
            - Time: τ = σ√(m/ε)
            - Mass: m (particle mass)
            - Temperature: ε/kB

        Returns:
            UnitSystem configured for reduced LJ units.
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
            length_to_si=1.0,  # User must define
            energy_to_si=1.0,  # User must define
            time_to_si=1.0,    # User must define
            mass_to_si=1.0,    # User must define
        )

    @staticmethod
    def SI() -> UnitSystem:
        """
        Create SI base units.

        Standard International System of Units.

        Unit definitions:
            - Distance: meters (m)
            - Energy: Joules (J)
            - Time: seconds (s)
            - Mass: kilograms (kg)
            - Temperature: Kelvin (K)
            - Charge: Coulombs (C)

        Returns:
            UnitSystem configured for SI units.
        """
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
            mass_to_si=1.0,
        )
