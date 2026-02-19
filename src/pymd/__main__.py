"""Allow running with: python -m pymd

Prints version info and available commands.
"""
import pymd


def main():
    print(f"pymd {pymd.__version__} - Molecular Dynamics Simulator with Autodiff Forces")
    print()
    print("Usage:")
    print("  python -m pymd.gui        Launch the desktop GUI")
    print("  python -m pytest tests/   Run tests")
    print()
    print("Quick start:")
    print("  from pymd.builder import SystemBuilder")
    print("  from pymd.core import Units")
    print("  from pymd.potential import LennardJonesPotential")
    print("  from pymd.force import ForceCalculator, NumericalBackend")


main()
