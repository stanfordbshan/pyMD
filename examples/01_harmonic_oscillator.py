#!/usr/bin/env python3
"""
Example 1: Harmonic Oscillator

A simple 2-atom system connected by a harmonic spring.
Demonstrates the most basic MD simulation - periodic oscillation.

Physics:
    U(r) = 0.5 * k * (r - r0)^2
    
The two atoms oscillate around equilibrium distance r0.
Total energy is conserved (kinetic <-> potential).

Usage:
    python examples/01_harmonic_oscillator.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from md_simulator.core import Atom, State, System, Units
from md_simulator.boundary import OpenBoundaryCondition
from md_simulator.force import ForceCalculator, NumericalBackend
from md_simulator.integrator import VelocityVerlet
from md_simulator.potential import PotentialEnergy
from md_simulator.simulator import Simulator
from md_simulator.thermostat import NoThermostat
from md_simulator.observer import EnergyObserver


class HarmonicPotential(PotentialEnergy):
    """Simple harmonic spring between two atoms."""
    
    def __init__(self, k: float = 1.0, r0: float = 1.0):
        self.k = k    # Spring constant
        self.r0 = r0  # Equilibrium distance
        self._cutoff = 100.0  # Large cutoff
    
    def compute_energy(self, positions, box, boundary_condition, 
                       atom_types=None, neighbor_list=None, **kwargs):
        r = np.linalg.norm(positions[1] - positions[0])
        return 0.5 * self.k * (r - self.r0) ** 2
    
    @property
    def cutoff(self):
        return self._cutoff
    
    def get_name(self):
        return f"Harmonic(k={self.k}, r0={self.r0})"


def main():
    print("=" * 55)
    print("  Example 1: HARMONIC OSCILLATOR")
    print("  Two atoms connected by a spring")
    print("=" * 55)
    
    # Two unit-mass atoms
    atoms = [Atom(mass=1.0, atom_type="A"), Atom(mass=1.0, atom_type="A")]
    
    # Start stretched from equilibrium (r0=1.0, start at r=1.3)
    r0 = 1.0
    r_initial = 1.3
    
    state = State(
        positions=np.array([[0.0, 0.0, 0.0], [r_initial, 0.0, 0.0]]),
        velocities=np.zeros((2, 3)),  # Start at rest
        forces=np.zeros((2, 3)),
        box=np.array([10.0, 10.0, 10.0]),
    )
    
    system = System(atoms, state, OpenBoundaryCondition(), Units.LJ())
    
    # Harmonic potential
    k = 10.0  # Spring constant
    potential = HarmonicPotential(k=k, r0=r0)
    force_calc = ForceCalculator(potential, NumericalBackend())
    
    # Calculate expected period: T = 2*pi*sqrt(m/k)
    # For reduced mass of 2 equal masses: mu = m/2
    mu = 0.5  # Reduced mass
    expected_period = 2 * np.pi * np.sqrt(mu / k)
    print(f"\nExpected oscillation period: {expected_period:.3f}")
    
    # Run simulation
    dt = 0.01
    steps = 200
    energy_obs = EnergyObserver(interval=10)
    
    sim = Simulator(
        system=system,
        integrator=VelocityVerlet(dt=dt),
        force_calculator=force_calc,
        thermostat=NoThermostat(),
        observers=[energy_obs],
    )
    
    # Track distance over time
    distances = [r_initial]
    times = [0.0]
    
    for step in range(steps):
        sim.run(num_steps=1)
        r = np.linalg.norm(system.state.positions[1] - system.state.positions[0])
        distances.append(r)
        times.append(system.state.time)
    
    # Analyze
    distances = np.array(distances)
    times = np.array(times)
    
    print(f"\n{'='*40}")
    print("RESULTS")
    print(f"{'='*40}")
    print(f"Simulation time:    {times[-1]:.2f}")
    print(f"Distance range:     [{distances.min():.3f}, {distances.max():.3f}]")
    print(f"Equilibrium:        {r0}")
    
    # Energy conservation
    E_total = np.array(energy_obs.total_energies)
    print(f"Energy drift:       {energy_obs.get_energy_drift():.2e}")
    
    # Check if system oscillates (max - min should be ~2*(r_initial - r0))
    amplitude = (distances.max() - distances.min()) / 2
    expected_amplitude = r_initial - r0
    print(f"Amplitude:          {amplitude:.3f} (expected: {expected_amplitude:.3f})")
    
    print("\n[PASS] Harmonic oscillation observed!")
    print("=" * 55)


if __name__ == "__main__":
    main()
