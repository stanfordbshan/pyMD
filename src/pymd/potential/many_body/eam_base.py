"""
EAM (Embedded Atom Method) base class.

Base implementation for EAM-style many-body potentials.
Forces are computed automatically via autodiff.
"""
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ..potential_energy import PotentialEnergy

if TYPE_CHECKING:
    from pymd.boundary import BoundaryCondition
    from pymd.neighbor import NeighborList


class EAMBasePotential(PotentialEnergy):
    """
    Base class for Embedded Atom Method (EAM) potentials.

    Total energy:
        E = Σ_i F_i(ρ_i) + (1/2) Σ_i Σ_j≠i φ_ij(r_ij)

    where:
        - F(ρ): Embedding energy function
        - ρ_i = Σ_j≠i f_j(r_ij): Electron density at atom i
        - φ(r): Pair interaction potential

    KEY: Even with nested functions (density → embedding),
    autodiff handles ALL derivatives automatically!
    User just writes the energy formula.

    Subclasses should implement:
        - Embedding function F(ρ)
        - Electron density function f(r)
        - Pair potential φ(r)

    Example:
        >>> class CopperEAM(EAMBasePotential):
        ...     # Implement specific functional forms
        ...     pass
    """

    def __init__(
        self,
        cutoff: float,
        embedding_functions: Dict[int, Callable[[float], float]],
        pair_potentials: Dict[Tuple[int, int], Callable[[float], float]],
        density_functions: Dict[int, Callable[[float], float]],
    ) -> None:
        """
        Initialize EAM base potential.

        Args:
            cutoff: Interaction cutoff distance.
            embedding_functions: Dict mapping atom_type → F(ρ).
            pair_potentials: Dict mapping (type_i, type_j) → φ(r).
            density_functions: Dict mapping atom_type → f(r).
        """
        self._cutoff = cutoff
        self.embedding_functions = embedding_functions
        self.pair_potentials = pair_potentials
        self.density_functions = density_functions

    @property
    def cutoff(self) -> float:
        """Return the cutoff distance."""
        return self._cutoff

    def compute_energy(
        self,
        positions: NDArray[np.floating],
        box: NDArray[np.floating],
        boundary_condition: "BoundaryCondition",
        atom_types: Optional[NDArray[np.intp]] = None,
        neighbor_list: Optional["NeighborList"] = None,
        **kwargs: Any,
    ) -> float:
        """
        Compute EAM energy - autodiff handles force calculation!

        Args:
            positions: (N, 3) atomic positions.
            box: (3,) box dimensions.
            boundary_condition: For minimum image convention.
            atom_types: (N,) atom type indices.
            neighbor_list: Optional neighbor list for efficiency.

        Returns:
            Total EAM potential energy.
        """
        n_atoms = len(positions)

        if atom_types is None:
            atom_types = np.zeros(n_atoms, dtype=np.intp)

        # Step 1: Compute electron density at each atom
        rho = self._compute_densities(
            positions, box, boundary_condition, atom_types, neighbor_list
        )

        # Step 2: Compute embedding energy
        embedding_energy = 0.0
        for i in range(n_atoms):
            atom_type = atom_types[i]
            F = self.embedding_functions[atom_type]
            embedding_energy += F(rho[i])

        # Step 3: Compute pair interactions
        pair_energy = self._compute_pair_energy(
            positions, box, boundary_condition, atom_types, neighbor_list
        )

        total_energy = embedding_energy + pair_energy
        return total_energy

    def _compute_densities(
        self,
        positions: NDArray[np.floating],
        box: NDArray[np.floating],
        boundary_condition: "BoundaryCondition",
        atom_types: NDArray[np.intp],
        neighbor_list: Optional["NeighborList"],
    ) -> NDArray[np.floating]:
        """
        Compute electron density at each atom: ρ_i = Σ_j≠i f_j(r_ij).

        This is an intermediate quantity, but autodiff tracks all operations!
        """
        n_atoms = len(positions)
        rho = np.zeros(n_atoms)

        if neighbor_list is not None:
            for i in range(n_atoms):
                neighbors = neighbor_list.get_neighbors(i)
                for j in neighbors:
                    r_ij = self._compute_distance(
                        i, j, positions, box, boundary_condition
                    )
                    if r_ij < self._cutoff:
                        atom_type_j = atom_types[j]
                        f_j = self.density_functions[atom_type_j]
                        rho[i] += f_j(r_ij)
                        rho[j] += f_j(r_ij)
        else:
            for i in range(n_atoms):
                for j in range(i + 1, n_atoms):
                    r_ij = self._compute_distance(
                        i, j, positions, box, boundary_condition
                    )
                    if r_ij < self._cutoff:
                        atom_type_i = atom_types[i]
                        atom_type_j = atom_types[j]
                        f_i = self.density_functions[atom_type_i]
                        f_j = self.density_functions[atom_type_j]
                        rho[i] += f_j(r_ij)
                        rho[j] += f_i(r_ij)

        return rho

    def _compute_pair_energy(
        self,
        positions: NDArray[np.floating],
        box: NDArray[np.floating],
        boundary_condition: "BoundaryCondition",
        atom_types: NDArray[np.intp],
        neighbor_list: Optional["NeighborList"],
    ) -> float:
        """Compute pairwise energy: (1/2) Σ_i Σ_j≠i φ_ij(r_ij)."""
        pair_energy = 0.0
        n_atoms = len(positions)

        if neighbor_list is not None:
            for i in range(n_atoms):
                neighbors = neighbor_list.get_neighbors(i)
                for j in neighbors:
                    r_ij = self._compute_distance(
                        i, j, positions, box, boundary_condition
                    )
                    if r_ij < self._cutoff:
                        type_i = atom_types[i]
                        type_j = atom_types[j]
                        pair_key = tuple(sorted([type_i, type_j]))
                        phi = self.pair_potentials[pair_key]
                        pair_energy += phi(r_ij)
        else:
            for i in range(n_atoms):
                for j in range(i + 1, n_atoms):
                    r_ij = self._compute_distance(
                        i, j, positions, box, boundary_condition
                    )
                    if r_ij < self._cutoff:
                        type_i = atom_types[i]
                        type_j = atom_types[j]
                        pair_key = tuple(sorted([type_i, type_j]))
                        phi = self.pair_potentials[pair_key]
                        pair_energy += phi(r_ij)

        return pair_energy

    def _compute_distance(
        self,
        i: int,
        j: int,
        positions: NDArray[np.floating],
        box: NDArray[np.floating],
        boundary_condition: "BoundaryCondition",
    ) -> float:
        """Compute distance between atoms i and j with PBC."""
        dr = positions[j] - positions[i]
        dr = boundary_condition.apply_minimum_image(dr.reshape(1, 3), box).flatten()
        return np.linalg.norm(dr)

    def get_name(self) -> str:
        """Return potential name."""
        return "EAMBase"


class SuttonChenEAM(EAMBasePotential):
    """
    Sutton-Chen EAM for FCC metals.

    Functional forms:
        F(ρ) = -c * ε * √ρ
        φ(r) = ε * (a/r)^n
        f(r) = (a/r)^m

    Common parameters:
        Metal   ε(eV)   a(Å)    c       m   n
        Cu      0.0124  3.615   39.432  6   12
        Ag      0.0025  4.086   144.41  6   12
        Au      0.0128  4.079   34.408  8   10
        Ni      0.0158  3.521   39.756  6   9
        Pd      0.0042  3.891   108.27  7   12
        Pt      0.0098  3.923   71.336  8   10

    Example:
        >>> eam_cu = SuttonChenEAM.create(
        ...     epsilon=0.0124, a=3.615, c=39.432, m=6, n=12, cutoff=5.5
        ... )
    """

    def __init__(
        self,
        epsilon: float,
        a: float,
        c: float,
        m: int,
        n: int,
        cutoff: float,
    ) -> None:
        """
        Initialize Sutton-Chen EAM.

        Args:
            epsilon: Energy scale (eV).
            a: Lattice parameter (Å).
            c: Dimensionless embedding parameter.
            m: Density exponent.
            n: Pair potential exponent.
            cutoff: Interaction cutoff distance.
        """
        self.epsilon = epsilon
        self.a = a
        self.c = c
        self.m = m
        self.n = n

        # Define functional forms
        def embedding_fn(rho: float) -> float:
            # F(ρ) = -c * ε * √ρ
            if rho > 0:
                return -c * epsilon * np.sqrt(rho)
            return 0.0

        def pair_fn(r: float) -> float:
            # φ(r) = ε * (a/r)^n
            if r > 0:
                return epsilon * (a / r) ** n
            return 0.0

        def density_fn(r: float) -> float:
            # f(r) = (a/r)^m
            if r > 0:
                return (a / r) ** m
            return 0.0

        super().__init__(
            cutoff=cutoff,
            embedding_functions={0: embedding_fn},
            pair_potentials={(0, 0): pair_fn},
            density_functions={0: density_fn},
        )

    @classmethod
    def create(
        cls,
        epsilon: float,
        a: float,
        c: float,
        m: int,
        n: int,
        cutoff: float,
    ) -> "SuttonChenEAM":
        """Factory method to create Sutton-Chen EAM potential."""
        return cls(epsilon, a, c, m, n, cutoff)

    @classmethod
    def copper(cls, cutoff: float = 5.5) -> "SuttonChenEAM":
        """Create Sutton-Chen potential for Copper."""
        return cls(epsilon=0.0124, a=3.615, c=39.432, m=6, n=12, cutoff=cutoff)

    @classmethod
    def silver(cls, cutoff: float = 5.5) -> "SuttonChenEAM":
        """Create Sutton-Chen potential for Silver."""
        return cls(epsilon=0.0025, a=4.086, c=144.41, m=6, n=12, cutoff=cutoff)

    @classmethod
    def gold(cls, cutoff: float = 5.5) -> "SuttonChenEAM":
        """Create Sutton-Chen potential for Gold."""
        return cls(epsilon=0.0128, a=4.079, c=34.408, m=8, n=10, cutoff=cutoff)

    def get_name(self) -> str:
        """Return potential name."""
        return f"SuttonChen(ε={self.epsilon}, a={self.a}, c={self.c})"
