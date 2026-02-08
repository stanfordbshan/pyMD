"""
pyMD - Molecular Dynamics Simulator with Autodiff Forces.

A Python molecular dynamics framework that uses automatic differentiation
to compute forces from energy functions. Users only need to write energy
functions E(positions); forces F = -∇E are computed automatically.

Main features:
- Autodiff backends: JAX, PyTorch, Autograd
- LAMMPS-style unit systems: REAL, METAL, LJ, SI
- EAM/MEAM support for metallic systems
- Multiple boundary conditions and neighbor list algorithms
- YAML configuration for complete simulation setup
"""

__version__ = "0.1.0"
__author__ = "pyMD Team"
