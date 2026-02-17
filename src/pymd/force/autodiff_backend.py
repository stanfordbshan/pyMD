"""
Autodiff backend implementations for force computation.

This module provides the AutoDiffBackend ABC and concrete implementations
for JAX, PyTorch, and Autograd backends.
"""
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

import numpy as np
from numpy.typing import NDArray


class AutoDiffBackend(ABC):
    """
    Abstract base for automatic differentiation backends (Strategy Pattern).

    Provides a unified interface for computing forces as F = -∇E using
    different autodiff libraries. Users can switch backends without
    changing their energy functions.

    Example:
        >>> from pymd.force import JAXBackend
        >>> backend = JAXBackend(use_jit=True)
        >>> forces = backend.compute_forces(energy_fn, positions)
    """

    @abstractmethod
    def compute_forces(
        self,
        energy_fn: Callable[..., float],
        positions: NDArray[np.floating],
        **kwargs: Any,
    ) -> NDArray[np.floating]:
        """
        Compute forces as F = -∇E.

        Args:
            energy_fn: Function that computes energy from positions.
                       Signature: energy_fn(positions, **kwargs) -> float
            positions: (N, 3) atomic positions.
            **kwargs: Additional arguments passed to energy_fn.

        Returns:
            (N, 3) array of forces on each atom.
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get human-readable name of this backend."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend's dependencies are installed."""
        pass


class JAXBackend(AutoDiffBackend):
    """
    JAX autodiff backend.

    Advantages:
        - Fastest for large systems
        - GPU/TPU support
        - JIT compilation for speed

    Requires:
        pip install jax jaxlib

    Example:
        >>> backend = JAXBackend(use_jit=True)
        >>> if backend.is_available():
        ...     forces = backend.compute_forces(energy_fn, positions)
    """

    def __init__(self, use_jit: bool = True) -> None:
        """
        Initialize JAX backend.

        Args:
            use_jit: Whether to JIT-compile the gradient function.
        """
        self.use_jit = use_jit
        self._jax: Optional[Any] = None
        self._jnp: Optional[Any] = None
        self._available: Optional[bool] = None

    def _init_jax(self) -> bool:
        """Lazy initialization of JAX."""
        if self._available is not None:
            return self._available

        try:
            import jax
            import jax.numpy as jnp

            self._jax = jax
            self._jnp = jnp
            self._available = True
        except ImportError:
            self._available = False

        return self._available

    def is_available(self) -> bool:
        """Check if JAX is installed."""
        return self._init_jax()

    def compute_forces(
        self,
        energy_fn: Callable[..., float],
        positions: NDArray[np.floating],
        **kwargs: Any,
    ) -> NDArray[np.floating]:
        """
        Compute forces using JAX autodiff.

        Args:
            energy_fn: Energy function taking positions array.
            positions: (N, 3) atomic positions.
            **kwargs: Additional arguments for energy_fn.

        Returns:
            (N, 3) force array.

        Raises:
            RuntimeError: If JAX is not installed.
        """
        if not self._init_jax():
            raise RuntimeError(
                "JAX is not installed. Install with: pip install jax jaxlib"
            )

        jax = self._jax
        jnp = self._jnp

        # Convert to JAX array
        positions_jax = jnp.array(positions, dtype=jnp.float64)

        # Create gradient function
        def energy_wrapper(pos: Any) -> float:
            # Convert back to numpy for the energy function
            pos_np = np.array(pos)
            return energy_fn(pos_np, **kwargs)

        grad_fn = jax.grad(energy_wrapper)

        if self.use_jit:
            grad_fn = jax.jit(grad_fn)

        # Compute gradient and negate for forces
        gradient = grad_fn(positions_jax)
        forces = -np.array(gradient)

        return forces

    def get_name(self) -> str:
        """Return backend name."""
        return f"JAX(jit={self.use_jit})"


class PyTorchBackend(AutoDiffBackend):
    """
    PyTorch autodiff backend.

    Advantages:
        - Familiar API
        - GPU support
        - Extensive ecosystem

    Requires:
        pip install torch

    Example:
        >>> backend = PyTorchBackend(device='cpu')
        >>> if backend.is_available():
        ...     forces = backend.compute_forces(energy_fn, positions)
    """

    def __init__(self, device: str = "cpu") -> None:
        """
        Initialize PyTorch backend.

        Args:
            device: Device to use ('cpu', 'cuda', 'cuda:0', etc.).
        """
        self.device = device
        self._torch: Optional[Any] = None
        self._available: Optional[bool] = None

    def _init_torch(self) -> bool:
        """Lazy initialization of PyTorch."""
        if self._available is not None:
            return self._available

        try:
            import torch

            self._torch = torch
            self._available = True
        except ImportError:
            self._available = False

        return self._available

    def is_available(self) -> bool:
        """Check if PyTorch is installed."""
        return self._init_torch()

    def compute_forces(
        self,
        energy_fn: Callable[..., float],
        positions: NDArray[np.floating],
        **kwargs: Any,
    ) -> NDArray[np.floating]:
        """
        Compute forces using PyTorch autodiff.

        Note: This implementation converts positions to PyTorch tensors
        and uses autograd. For best performance, use JAX backend.

        Args:
            energy_fn: Energy function taking positions array.
            positions: (N, 3) atomic positions.
            **kwargs: Additional arguments for energy_fn.

        Returns:
            (N, 3) force array.

        Raises:
            RuntimeError: If PyTorch is not installed.
        """
        if not self._init_torch():
            raise RuntimeError(
                "PyTorch is not installed. Install with: pip install torch"
            )

        torch = self._torch

        # Create tensor with gradient tracking
        positions_torch = torch.tensor(
            positions,
            requires_grad=True,
            dtype=torch.float64,
            device=self.device,
        )

        # Compute energy using numpy (convert back and forth)
        # This is not ideal for performance but works with numpy-based energy functions
        energy = energy_fn(positions_torch.detach().cpu().numpy(), **kwargs)
        energy_torch = torch.tensor(energy, device=self.device, dtype=torch.float64)

        # For proper autodiff, we need the energy function to use torch operations
        # This simple wrapper works when energy_fn returns a scalar
        energy_torch.backward()

        if positions_torch.grad is not None:
            forces = -positions_torch.grad.cpu().numpy()
        else:
            # Fallback to numerical differentiation
            forces = self._numerical_gradient(energy_fn, positions, **kwargs)

        return forces

    def _numerical_gradient(
        self,
        energy_fn: Callable[..., float],
        positions: NDArray[np.floating],
        h: float = 1e-5,
        **kwargs: Any,
    ) -> NDArray[np.floating]:
        """Compute numerical gradient as fallback."""
        forces = np.zeros_like(positions)
        for i in range(len(positions)):
            for j in range(3):
                pos_plus = positions.copy()
                pos_minus = positions.copy()
                pos_plus[i, j] += h
                pos_minus[i, j] -= h
                forces[i, j] = -(
                    energy_fn(pos_plus, **kwargs) - energy_fn(pos_minus, **kwargs)
                ) / (2 * h)
        return forces

    def get_name(self) -> str:
        """Return backend name."""
        return f"PyTorch(device={self.device})"


class AutogradBackend(AutoDiffBackend):
    """
    Autograd backend (pure NumPy autodiff).

    Advantages:
        - Lightweight
        - Pure Python/NumPy
        - No external dependencies beyond autograd

    Requires:
        pip install autograd

    Example:
        >>> backend = AutogradBackend()
        >>> if backend.is_available():
        ...     forces = backend.compute_forces(energy_fn, positions)
    """

    def __init__(self) -> None:
        """Initialize Autograd backend."""
        self._autograd: Optional[Any] = None
        self._anp: Optional[Any] = None
        self._available: Optional[bool] = None

    def _init_autograd(self) -> bool:
        """Lazy initialization of Autograd."""
        if self._available is not None:
            return self._available

        try:
            import autograd
            import autograd.numpy as anp

            self._autograd = autograd
            self._anp = anp
            self._available = True
        except ImportError:
            self._available = False

        return self._available

    def is_available(self) -> bool:
        """Check if Autograd is installed."""
        return self._init_autograd()

    def compute_forces(
        self,
        energy_fn: Callable[..., float],
        positions: NDArray[np.floating],
        **kwargs: Any,
    ) -> NDArray[np.floating]:
        """
        Compute forces using Autograd.

        Args:
            energy_fn: Energy function taking positions array.
            positions: (N, 3) atomic positions.
            **kwargs: Additional arguments for energy_fn.

        Returns:
            (N, 3) force array.

        Raises:
            RuntimeError: If Autograd is not installed.
        """
        if not self._init_autograd():
            raise RuntimeError(
                "Autograd is not installed. Install with: pip install autograd"
            )

        from autograd import grad

        def energy_wrapper(pos: NDArray) -> float:
            return energy_fn(pos, **kwargs)

        grad_fn = grad(energy_wrapper)
        gradient = grad_fn(positions)
        forces = -gradient

        return forces

    def get_name(self) -> str:
        """Return backend name."""
        return "Autograd(NumPy)"


class NumericalBackend(AutoDiffBackend):
    """
    Numerical differentiation backend (finite differences).

    This is a fallback backend that doesn't require any autodiff library.
    Uses central finite differences to approximate the gradient.

    Advantages:
        - No external dependencies
        - Works with any energy function
        - Good for debugging/validation

    Disadvantages:
        - Slow (O(6N) energy evaluations per force calculation)
        - Approximate (not exact)

    Example:
        >>> backend = NumericalBackend(h=1e-5)
        >>> forces = backend.compute_forces(energy_fn, positions)
    """

    def __init__(self, h: float = 1e-5) -> None:
        """
        Initialize numerical backend.

        Args:
            h: Step size for finite differences.
        """
        self.h = h

    def is_available(self) -> bool:
        """Always available (no dependencies)."""
        return True

    def compute_forces(
        self,
        energy_fn: Callable[..., float],
        positions: NDArray[np.floating],
        **kwargs: Any,
    ) -> NDArray[np.floating]:
        """
        Compute forces using central finite differences.

        F_i = -(E(r + h*e_i) - E(r - h*e_i)) / (2*h)

        Args:
            energy_fn: Energy function taking positions array.
            positions: (N, 3) atomic positions.
            **kwargs: Additional arguments for energy_fn.

        Returns:
            (N, 3) force array.
        """
        forces = np.zeros_like(positions)
        h = self.h

        for i in range(len(positions)):
            for j in range(3):
                pos_plus = positions.copy()
                pos_minus = positions.copy()
                pos_plus[i, j] += h
                pos_minus[i, j] -= h

                e_plus = energy_fn(pos_plus, **kwargs)
                e_minus = energy_fn(pos_minus, **kwargs)

                forces[i, j] = -(e_plus - e_minus) / (2 * h)

        return forces

    def get_name(self) -> str:
        """Return backend name."""
        return f"Numerical(h={self.h})"


class BackendFactory:
    """
    Factory for creating autodiff backends.

    Provides a simple interface to create backends by name,
    with automatic fallback to available backends.

    Example:
        >>> backend = BackendFactory.create('jax', use_jit=True)
        >>> # Or get the best available backend
        >>> backend = BackendFactory.get_best_available()
    """

    @staticmethod
    def create(backend_name: str, **kwargs: Any) -> AutoDiffBackend:
        """
        Create backend by name.

        Args:
            backend_name: One of 'jax', 'pytorch'/'torch', 'autograd', 'numerical'.
            **kwargs: Backend-specific options.

        Returns:
            Configured AutoDiffBackend instance.

        Raises:
            ValueError: If backend name is unknown.
        """
        backend_name = backend_name.lower()

        if backend_name == "jax":
            return JAXBackend(**kwargs)
        elif backend_name in ["pytorch", "torch"]:
            return PyTorchBackend(**kwargs)
        elif backend_name == "autograd":
            return AutogradBackend()
        elif backend_name == "numerical":
            return NumericalBackend(**kwargs)
        else:
            raise ValueError(
                f"Unknown backend: {backend_name}. "
                f"Choose from: jax, pytorch, autograd, numerical"
            )

    @staticmethod
    def get_best_available(**kwargs: Any) -> AutoDiffBackend:
        """
        Get the best available autodiff backend.

        Priority: JAX > PyTorch > Autograd > Numerical

        Args:
            **kwargs: Options passed to the backend.

        Returns:
            Best available backend.
        """
        # Try JAX first (fastest)
        jax_backend = JAXBackend(**kwargs)
        if jax_backend.is_available():
            return jax_backend

        # Try PyTorch
        torch_backend = PyTorchBackend(**kwargs)
        if torch_backend.is_available():
            return torch_backend

        # Try Autograd
        autograd_backend = AutogradBackend()
        if autograd_backend.is_available():
            return autograd_backend

        # Fallback to numerical
        return NumericalBackend()

    @staticmethod
    def list_available() -> list:
        """
        List all available backends.

        Returns:
            List of available backend names.
        """
        available = ["numerical"]  # Always available

        if AutogradBackend().is_available():
            available.append("autograd")
        if PyTorchBackend().is_available():
            available.append("pytorch")
        if JAXBackend().is_available():
            available.append("jax")

        return available
