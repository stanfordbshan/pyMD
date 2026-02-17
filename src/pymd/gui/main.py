"""
pymd Desktop Application entry point.

Launches a pywebview window with the pymd GUI.
Usage: python -m pymd.gui.main
"""
import os
import sys

import webview
from pymd.gui.api import SimulatorAPI

_GUI_DIR = os.path.dirname(os.path.abspath(__file__))
_ASSETS_DIR = os.path.join(_GUI_DIR, "assets")


def main():
    api = SimulatorAPI()
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
