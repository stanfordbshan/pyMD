"""
pymd Desktop Application — consolidated launcher with embedded API management.

Supports three compute modes:
  - ``direct``  — in-process bridge calls (no HTTP)
  - ``api``     — HTTP API only (requires running API server or embedded)
  - ``auto``    — direct calls, with embedded API subprocess for JS fallback

Usage::

    python -m pymd.gui                          # auto mode (default)
    python -m pymd.gui --compute-mode direct    # direct in-process only
    python -m pymd.gui --compute-mode api       # HTTP API only
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

_GUI_DIR = os.path.dirname(os.path.abspath(__file__))
_ASSETS_DIR = os.path.join(_GUI_DIR, "assets")


# ------------------------------------------------------------------ #
#  Embedded API subprocess management
# ------------------------------------------------------------------ #


@dataclass
class EmbeddedApiProcess:
    """Tracks a child API subprocess."""

    process: subprocess.Popen
    base_url: str


def _wait_for_api_health(base_url: str, timeout: float = 10.0) -> bool:
    """Poll ``GET /health`` until the API is reachable or *timeout* expires."""
    deadline = time.monotonic() + timeout
    url = base_url.rstrip("/") + "/health"
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.25)
    return False


def _start_embedded_api(host: str = "127.0.0.1", port: int = 8000) -> EmbeddedApiProcess:
    """Launch ``python -m pymd.api`` as a child process."""
    cmd = [
        sys.executable, "-m", "pymd.api",
        "--host", host,
        "--port", str(port),
        "--log-level", "warning",
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base_url = f"http://{host}:{port}"
    if not _wait_for_api_health(base_url):
        proc.terminate()
        proc.wait(timeout=5)
        raise RuntimeError(f"Embedded API did not become healthy at {base_url}")
    return EmbeddedApiProcess(process=proc, base_url=base_url)


def _stop_embedded_api(embedded: Optional[EmbeddedApiProcess]) -> None:
    """Gracefully shut down the embedded API subprocess."""
    if embedded is None:
        return
    try:
        embedded.process.terminate()
        embedded.process.wait(timeout=5)
    except Exception:
        try:
            embedded.process.kill()
        except Exception:
            pass


def _resolve_api_backend(
    compute_mode: str,
    api_url: Optional[str],
    host: str,
    port: int,
) -> tuple[Optional[EmbeddedApiProcess], Optional[str]]:
    """Determine whether an embedded API is needed and return ``(embedded, resolved_url)``.

    - ``direct`` mode: no API needed.
    - ``api`` mode with explicit URL: use the external URL, no embedded subprocess.
    - ``api``/``auto`` without explicit URL: start embedded subprocess.
    """
    if compute_mode == "direct":
        return None, None

    if api_url:
        return None, api_url

    embedded = _start_embedded_api(host, port)
    return embedded, embedded.base_url


# ------------------------------------------------------------------ #
#  CLI argument parser
# ------------------------------------------------------------------ #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pymd.gui",
        description="Launch the pymd desktop GUI.",
    )
    parser.add_argument(
        "--compute-mode",
        choices=["direct", "api", "auto"],
        default="auto",
        help="Compute mode: direct (in-process), api (HTTP), auto (direct + API fallback)",
    )
    parser.add_argument(
        "--api-url",
        default=None,
        help="Base URL for an external API server (skips embedded API)",
    )
    parser.add_argument(
        "--api-host",
        default="127.0.0.1",
        help="Host for embedded API server (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=8000,
        help="Port for embedded API server (default: 8000)",
    )
    parser.add_argument("--width", type=int, default=1200, help="Window width")
    parser.add_argument("--height", type=int, default=800, help="Window height")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    return parser


# ------------------------------------------------------------------ #
#  Launch
# ------------------------------------------------------------------ #


def launch_gui(
    compute_mode: str = "auto",
    api_url: Optional[str] = None,
    api_host: str = "127.0.0.1",
    api_port: int = 8000,
    width: int = 1200,
    height: int = 800,
    debug: bool = False,
) -> None:
    """Create the pywebview window and run the main loop."""
    import webview

    from pymd.gui.bridge import DirectBridge

    embedded, resolved_url = _resolve_api_backend(
        compute_mode, api_url, api_host, api_port,
    )

    bridge = DirectBridge()
    index_path = os.path.join(_ASSETS_DIR, "index.html")

    window = webview.create_window(
        "pymd - Molecular Dynamics Simulator",
        url=index_path,
        js_api=bridge,
        width=width,
        height=height,
        min_size=(900, 600),
    )
    bridge.set_window(window)

    def _on_loaded():
        """Inject compute-mode config into the frontend after page load."""
        api_url_js = f"'{resolved_url}'" if resolved_url else "''"
        window.evaluate_js(
            f"window.__PYMD_COMPUTE_MODE = '{compute_mode}';"
            f"window.__PYMD_API_BASE_URL = {api_url_js};"
        )

    window.events.loaded += _on_loaded

    try:
        webview.start(debug=debug)
    finally:
        _stop_embedded_api(embedded)


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        launch_gui(
            compute_mode=args.compute_mode,
            api_url=args.api_url,
            api_host=args.api_host,
            api_port=args.api_port,
            width=args.width,
            height=args.height,
            debug=args.debug,
        )
        return 0
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
