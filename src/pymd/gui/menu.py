"""Native menu bar for the pymd desktop GUI.

All menu callbacks delegate to existing HTML buttons via ``window.evaluate_js()``,
so form-gathering, validation, status updates, and button state management
all funnel through the same JS code paths.
"""
from __future__ import annotations

import webview
from webview.menu import Menu, MenuAction, MenuSeparator

from pymd import __version__


def build_menu_bar(window: webview.Window) -> list[Menu]:
    """Build the application menu bar.

    Parameters
    ----------
    window:
        The pywebview window instance (used for ``evaluate_js`` and ``destroy``).
    """

    # -- File --
    file_menu = Menu(
        "File",
        [
            MenuAction("Load YAML\tCtrl+O", lambda: window.evaluate_js(
                "document.getElementById('btn-load-yaml').click()"
            )),
            MenuAction("Build System\tCtrl+B", lambda: window.evaluate_js(
                "document.getElementById('btn-build-system').click()"
            )),
            MenuSeparator(),
            MenuAction("Exit\tAlt+F4", lambda: window.destroy()),
        ],
    )

    # -- Simulation --
    simulation_menu = Menu(
        "Simulation",
        [
            MenuAction("Start Simulation\tCtrl+R", lambda: window.evaluate_js(
                "document.getElementById('btn-start-sim').click()"
            )),
            MenuAction("Stop Simulation\tCtrl+.", lambda: window.evaluate_js(
                "document.getElementById('btn-stop-sim').click()"
            )),
            MenuSeparator(),
            MenuAction("Run Minimization\tCtrl+M", lambda: window.evaluate_js(
                "document.getElementById('btn-minimize').click()"
            )),
        ],
    )

    # -- View --
    def _switch_tab(tab_name: str) -> None:
        window.evaluate_js(
            f"document.querySelector('.nav-item[data-tab=\"{tab_name}\"]').click()"
        )

    view_menu = Menu(
        "View",
        [
            MenuAction("Setup", lambda: _switch_tab("setup")),
            MenuAction("Visualization", lambda: _switch_tab("visualization")),
            MenuAction("Simulation", lambda: _switch_tab("simulation")),
            MenuAction("Minimization", lambda: _switch_tab("minimization")),
            MenuAction("Results", lambda: _switch_tab("results")),
            MenuSeparator(),
            MenuAction("Reset 3D View", lambda: window.evaluate_js(
                "document.getElementById('btn-reset-view').click()"
            )),
        ],
    )

    # -- Help --
    about_js = (
        "alert("
        "'pyMD v" + __version__ + "\\n\\n"
        "Molecular Dynamics Simulator with Autodiff Forces.\\n"
        "Users write energy functions E(positions);\\n"
        "forces F = -∇E are computed automatically.')"
    )

    help_menu = Menu(
        "Help",
        [
            MenuAction("About pyMD", lambda: window.evaluate_js(about_js)),
        ],
    )

    return [file_menu, simulation_menu, view_menu, help_menu]
