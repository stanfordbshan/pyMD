"""
pyMD Desktop Application entry point.

Launches a pywebview window with the pyMD GUI.
Usage: python gui/main.py
"""
import os
import sys

# Add project root to path so pyMD package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import webview
from api import SimulatorAPI


def main():
    api = SimulatorAPI()
    ui_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui")
    index_path = os.path.join(ui_dir, "index.html")

    window = webview.create_window(
        "pyMD - Molecular Dynamics Simulator",
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
