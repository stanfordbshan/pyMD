"""
Force computation module for molecular dynamics simulations.

This module provides:
- AutoDiffBackend ABC and implementations (JAX, PyTorch, Autograd, Numerical)
- ForceCalculator for computing F = -∇E from potential energy functions
- BackendFactory for easy backend selection
"""

from .autodiff_backend import (
    AutoDiffBackend,
    AutogradBackend,
    BackendFactory,
    JAXBackend,
    NumericalBackend,
    PyTorchBackend,
)
from .force_calculator import ForceCalculator

__all__ = [
    # Backend ABC
    "AutoDiffBackend",
    # Concrete backends
    "JAXBackend",
    "PyTorchBackend",
    "AutogradBackend",
    "NumericalBackend",
    # Factory
    "BackendFactory",
    # Calculator
    "ForceCalculator",
]
