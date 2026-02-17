"""
Lennard-Jones potential implementation.

The classic 12-6 Lennard-Jones potential for noble gases and simple fluids.
Forces are computed automatically via autodiff.
"""
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from numpy.typing import NDArray

from ..potential_energy import PotentialEnergy

if TYPE_CHECKING:
    from pyMD.boundary import BoundaryCondition
    from pyMD.neighbor import NeighborList


class LennardJonesPotential(PotentialEnergy):
    """
    Lennard-Jones 12-6 potential.

    U(r) = 4ε[(σ/r)¹² - (σ/r)⁶]

    where:
        - ε (epsilon): Depth of the potential well
        - σ (sigma): Distance at which U(r) = 0
        - r: Interatomic distance

    The minimum of U(r) is at r_min = 2^(1/6) * σ ≈ 1.122 * σ
    with value U_min = -ε.

    Forces are computed AUTOMATICALLY via autodiff!

    Attributes:
        epsilon: Well depth ε.
        sigma: Zero-crossing distance σ.
        _cutoff: Interaction cutoff distance.

    Example:
        >>> from pyMD.potential import LennardJonesPotential
        >>> # Argon LJ parameters in reduced units
        >>> lj = LennardJonesPotential(epsilon=1.0, sigma=1.0, cutoff=2.5)
        >>> energy = lj.compute_energy(positions, box, bc)
    """

    def __init__(
        self,
        epsilon: float,
        sigma: float,
        cutoff: float,
        shift_energy: bool = False,
    ) -> None:
        """
        Initialize Lennard-Jones potential.

        Args:
            epsilon: Well depth ε (energy units).
            sigma: Zero-crossing distance σ (length units).
            cutoff: Interaction cutoff distance (length units).
            shift_energy: If True, shift potential so U(cutoff) = 0.
        """
        if epsilon <= 0:
            raise ValueError(f"Epsilon must be positive, got {epsilon}")
        if sigma <= 0:
            raise ValueError(f"Sigma must be positive, got {sigma}")
        if cutoff <= 0:
            raise ValueError(f"Cutoff must be positive, got {cutoff}")

        self.epsilon = epsilon
        self.sigma = sigma
        self._cutoff = cutoff
        self.shift_energy = shift_energy

        # Compute energy shift if requested
        if shift_energy:
            sr_cut = sigma / cutoff
            self._energy_shift = 4.0 * epsilon * (sr_cut**12 - sr_cut**6)
        else:
            self._energy_shift = 0.0

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
        Compute LJ energy - autodiff will compute forces!

        Args:
            positions: (N, 3) atomic positions.
            box: (3,) box dimensions.
            boundary_condition: For minimum image convention.
            atom_types: Ignored - single species assumed.
            neighbor_list: Optional neighbor list for efficiency.

        Returns:
            Total LJ potential energy.
        """
        energy = 0.0
        n_atoms = len(positions)

        if neighbor_list is not None:
            # Use neighbor list
            for i in range(n_atoms):
                neighbors = neighbor_list.get_neighbors(i)
                for j in neighbors:
                    dr = positions[j] - positions[i]
                    dr = boundary_condition.apply_minimum_image(
                        dr.reshape(1, 3), box
                    ).flatten()
                    r = np.linalg.norm(dr)

                    if r < self._cutoff and r > 0:
                        sr = self.sigma / r
                        sr6 = sr**6
                        sr12 = sr6**2
                        energy += 4.0 * self.epsilon * (sr12 - sr6) - self._energy_shift
        else:
            # Brute force all pairs
            for i in range(n_atoms):
                for j in range(i + 1, n_atoms):
                    dr = positions[j] - positions[i]
                    dr = boundary_condition.apply_minimum_image(
                        dr.reshape(1, 3), box
                    ).flatten()
                    r = np.linalg.norm(dr)

                    if r < self._cutoff and r > 0:
                        sr = self.sigma / r
                        sr6 = sr**6
                        sr12 = sr6**2
                        energy += 4.0 * self.epsilon * (sr12 - sr6) - self._energy_shift

        return energy

    def get_name(self) -> str:
        """Return potential name with parameters."""
        return f"LJ(ε={self.epsilon}, σ={self.sigma}, rc={self._cutoff})"

    def compute_pair_energy(self, r: float) -> float:
        """
        Compute LJ energy for a single pair at distance r.

        Useful for debugging and visualization.

        Args:
            r: Interatomic distance.

        Returns:
            Pair energy (not counted twice).
        """
        if r >= self._cutoff or r <= 0:
            return 0.0

        sr = self.sigma / r
        sr6 = sr**6
        sr12 = sr6**2
        return 4.0 * self.epsilon * (sr12 - sr6) - self._energy_shift

    def compute_pair_force(self, r: float) -> float:
        """
        Compute ANALYTICAL LJ force magnitude for a single pair.

        F(r) = -dU/dr = 24ε/r * [2(σ/r)¹² - (σ/r)⁶]

        This is provided for validation against autodiff results.

        Args:
            r: Interatomic distance.

        Returns:
            Force magnitude (positive = repulsive).
        """
        if r >= self._cutoff or r <= 0:
            return 0.0

        sr = self.sigma / r
        sr6 = sr**6
        sr12 = sr6**2
        # F = -dU/dr = 24*epsilon/r * [2*sr^12 - sr^6]
        return 24.0 * self.epsilon / r * (2.0 * sr12 - sr6)
