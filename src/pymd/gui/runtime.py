"""
Compute-mode dispatcher for the GUI.

Supports three modes:
  - ``direct``  — in-process bridge call (no HTTP)
  - ``api``     — HTTP API only
  - ``auto``    — try direct first, fall back to API on import/setup failure
"""
from __future__ import annotations

from typing import Literal

ComputeMode = Literal["direct", "api", "auto"]


def create_js_api(
    mode: ComputeMode = "auto",
    api_url: str = "http://127.0.0.1:8000",
):
    """Return a js_api object appropriate for *mode*.

    All returned objects expose the same public method signatures so JS
    code does not need to know which path it is talking to.
    """
    if mode == "direct":
        from pymd.gui.bridge import DirectBridge
        return DirectBridge()

    if mode == "api":
        from pymd.gui.api_client import APIClient
        return APIClient(base_url=api_url)

    # auto: try direct, fall back to api
    try:
        from pymd.gui.bridge import DirectBridge
        return DirectBridge()
    except Exception:
        from pymd.gui.api_client import APIClient
        return APIClient(base_url=api_url)
