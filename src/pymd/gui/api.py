"""Backward-compatibility shim — use :mod:`pymd.gui.bridge` instead."""
from pymd.gui.bridge import DirectBridge as SimulatorAPI

__all__ = ["SimulatorAPI"]
