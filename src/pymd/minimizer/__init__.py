"""
Minimizer module for energy minimization.

Provides energy minimization algorithms:
- SteepestDescent: Simple gradient descent with line search
- ConjugateGradient: Polak-Ribiere CG with automatic restarts
- LBFGS: Limited-memory BFGS quasi-Newton method
"""

from .minimizer import Minimizer, MinimizationResult
from .steepest_descent import SteepestDescent
from .conjugate_gradient import ConjugateGradient
from .lbfgs import LBFGS

__all__ = [
    "Minimizer",
    "MinimizationResult",
    "SteepestDescent",
    "ConjugateGradient",
    "LBFGS",
]
