"""
pymd Desktop Application entry point.

Launches a pywebview window with the pymd GUI.

Usage:
    python -m pymd.gui                     # auto mode (default)
    python -m pymd.gui --mode direct       # direct in-process calls
    python -m pymd.gui --mode api          # HTTP API only
"""
import argparse
import os

import webview

from pymd.gui.runtime import create_js_api

_GUI_DIR = os.path.dirname(os.path.abspath(__file__))
_ASSETS_DIR = os.path.join(_GUI_DIR, "assets")


def main():
    parser = argparse.ArgumentParser(description="pymd GUI")
    parser.add_argument(
        "--mode",
        choices=["direct", "api", "auto"],
        default="auto",
        help="Compute mode: direct (in-process), api (HTTP), auto (direct with API fallback)",
    )
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8000",
        help="Base URL for the API server (used in api/auto modes)",
    )
    args = parser.parse_args()

    api = create_js_api(mode=args.mode, api_url=args.api_url)
    index_path = os.path.join(_ASSETS_DIR, "index.html")

    window = webview.create_window(
        "pymd - Molecular Dynamics Simulator",
        url=index_path,
        js_api=api,
        width=1200,
        height=800,
        min_size=(900, 600),
    )
    api.set_window(window)
    webview.start(debug=False)


if __name__ == "__main__":
    main()
