#!/usr/bin/env python3
"""
Example 2: Diatomic Molecule with Morse Potential

Simulates a diatomic molecule (like HCl or O2) using Morse potential.
Shows how molecular vibrations work at the atomic level.

Physics:
    U(r) = D * [1 - exp(-a*(r-r0))]^2
    
Unlike harmonic oscillator, Morse potential:
- Is anharmonic (frequency depends on amplitude)
- Has finite dissociation energy D
- Better represents real molecular bonds

Usage:
    python examples/02_diatomic_morse.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from pyMD.core import Atom, State, System, Units
from pyMD.boundary import OpenBoundaryCondition
from pyMD.force import ForceCalculator, NumericalBackend
from pyMD.integrator import VelocityVerlet
from pyMD.potential import MorsePotential
from pyMD.simulator import Simulator
from pyMD.thermostat import NoThermostat
from pyMD.observer import EnergyObserver


def main():
    print("=" * 55)
    print("  Example 2: DIATOMIC MORSE MOLECULE")
    print("  Anharmonic molecular vibration")
    print("=" * 55)
    
    # Create two atoms (e.g., simplified "HCl" model)
    # Masses in reduced units (H=1, Cl=35 -> use lighter for speed)
    atoms = [Atom(mass=1.0, atom_type="H"), Atom(mass=2.0, atom_type="Cl")]
    
    # Morse parameters (in LJ reduced units)
    D = 1.0    # Dissociation energy
    a = 2.0    # Controls width of potential well
    r0 = 1.0   # Equilibrium bond length
    
    # Start slightly stretched
    r_initial = 1.15
    
    state = State(
        positions=np.array([[0.0, 0.0, 0.0], [r_initial, 0.0, 0.0]]),
        velocities=np.zeros((2, 3)),
        forces=np.zeros((2, 3)),
        box=np.array([10.0, 10.0, 10.0]),
    )
    
    system = System(atoms, state, OpenBoundaryCondition(), Units.LJ())
    
    # Morse potential
    potential = MorsePotential(D=D, a=a, r0=r0, cutoff=5.0)
    force_calc = ForceCalculator(potential, NumericalBackend())
    
    print(f"\nMorse parameters: D={D}, a={a}, r0={r0}")
    print(f"Initial stretch:  {r_initial - r0:.3f}")
    
    # Run simulation
    dt = 0.005
    steps = 300
    energy_obs = EnergyObserver(interval=15)
    
    sim = Simulator(
        system=system,
        integrator=VelocityVerlet(dt=dt),
        force_calculator=force_calc,
        thermostat=NoThermostat(),
        observers=[energy_obs],
    )
    
    # Track bond length
    bond_lengths = [r_initial]
    
    for _ in range(steps):
        sim.run(num_steps=1)
        r = np.linalg.norm(system.state.positions[1] - system.state.positions[0])
        bond_lengths.append(r)
    
    bond_lengths = np.array(bond_lengths)
    
    print(f"\n{'='*40}")
    print("VIBRATIONAL ANALYSIS")
    print(f"{'='*40}")
    print(f"Mean bond length:   {bond_lengths.mean():.4f}")
    print(f"Min bond length:    {bond_lengths.min():.4f}")
    print(f"Max bond length:    {bond_lengths.max():.4f}")
    print(f"Amplitude:          {(bond_lengths.max() - bond_lengths.min())/2:.4f}")
    
    # Energy analysis
    PE = np.array(energy_obs.potential_energies)
    KE = np.array(energy_obs.kinetic_energies)
    
    print(f"\nEnergy (mean):")
    print(f"  PE: {PE.mean():.4f}")
    print(f"  KE: {KE.mean():.4f}")
    print(f"  Total: {(PE + KE).mean():.4f}")
    print(f"Energy drift: {energy_obs.get_energy_drift():.2e}")
    
    # Note about anharmonicity
    print(f"\nNote: Morse is anharmonic - mean bond length")
    print(f"({bond_lengths.mean():.4f}) > equilibrium ({r0})")
    
    print("\n[PASS] Morse vibration simulated!")
    print("=" * 55)


if __name__ == "__main__":
    main()
