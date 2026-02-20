"""
Direct-call bridge: pywebview js_api that delegates to MDService in-process.

This is the ``compute_mode="direct"`` path — no HTTP involved.
"""
from __future__ import annotations

import json
import threading
import traceback
from typing import Optional

from pymd.builder.config_loader import load_yaml as _load_yaml
from pymd.core.schemas import BuildConfig, MinimizationParams, SimulationParams
from pymd.core.service import MDService


class DirectBridge:
    """Exposed to JS via pywebview js_api.  All methods return JSON strings."""

    def __init__(self, service: Optional[MDService] = None):
        self._svc = service or MDService()
        self._window = None
        self._sim_thread: Optional[threading.Thread] = None

    def set_window(self, window):
        self._window = window

    # ------------------------------------------------------------------ #
    #  Setup
    # ------------------------------------------------------------------ #

    def load_yaml(self):
        try:
            import webview

            result = self._window.create_file_dialog(
                dialog_type=webview.FileDialog.OPEN,
                file_types=("YAML Files (*.yaml;*.yml)", "All files (*.*)"),
            )
            if not result:
                return json.dumps({"error": "No file selected"})
            path = result[0] if isinstance(result, (list, tuple)) else result
            config = _load_yaml(path)
            return json.dumps({"ok": True, "config": config, "path": str(path)})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def build_system(self, config_json):
        try:
            raw = json.loads(config_json) if isinstance(config_json, str) else config_json
            cfg = BuildConfig.from_dict(raw)
            summary = self._svc.build_system(cfg)
            return json.dumps({"ok": True, "summary": summary.to_dict(), "xyz": summary.xyz})
        except Exception as e:
            traceback.print_exc()
            return json.dumps({"error": str(e)})

    def get_xyz(self):
        try:
            xyz = self._svc.get_xyz()
            return json.dumps({"ok": True, "xyz": xyz})
        except Exception as e:
            return json.dumps({"error": str(e)})

    # ------------------------------------------------------------------ #
    #  Simulation
    # ------------------------------------------------------------------ #

    def start_simulation(self, num_steps, viz_interval=50, init_temp=1.0):
        try:
            params = SimulationParams(
                num_steps=int(num_steps),
                viz_interval=int(viz_interval),
                init_temp=float(init_temp),
            )
            self._svc.prepare_simulation(params)
        except RuntimeError as e:
            return json.dumps({"error": str(e)})

        def run_loop():
            try:
                self._svc.run_steps(
                    num_steps=params.num_steps,
                    viz_interval=params.viz_interval,
                    on_update=lambda u: self._eval_js(
                        f"onSimulationUpdate({json.dumps(u.to_dict())}, {json.dumps(u.xyz)})"
                    ),
                )
            except Exception as e:
                traceback.print_exc()
                self._eval_js(f"onSimulationError({json.dumps(str(e))})")
            finally:
                self._eval_js("onSimulationDone()")

        self._sim_thread = threading.Thread(target=run_loop, daemon=True)
        self._sim_thread.start()
        return json.dumps({"ok": True, "message": f"Started {params.num_steps} steps"})

    def stop_simulation(self):
        self._svc.stop()
        return json.dumps({"ok": True})

    def get_energy_data(self):
        try:
            data = self._svc.get_energy_data()
            return json.dumps({"ok": True, "data": data.to_dict()})
        except Exception as e:
            return json.dumps({"error": str(e)})

    # ------------------------------------------------------------------ #
    #  Minimization
    # ------------------------------------------------------------------ #

    def run_minimization(self, algorithm, params_json):
        try:
            raw = json.loads(params_json) if isinstance(params_json, str) else params_json
            params = MinimizationParams.from_dict(algorithm, raw)
        except Exception as e:
            return json.dumps({"error": str(e)})

        def minimize_thread():
            try:
                result = self._svc.run_minimization(params)
                self._eval_js(
                    f"onMinimizationDone({json.dumps(result.to_dict())}, {json.dumps(result.xyz)})"
                )
            except Exception as e:
                traceback.print_exc()
                self._eval_js(f"onMinimizationError({json.dumps(str(e))})")

        self._sim_thread = threading.Thread(target=minimize_thread, daemon=True)
        self._sim_thread.start()
        return json.dumps({"ok": True, "message": "Minimization started"})

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _eval_js(self, js_code: str):
        try:
            if self._window:
                self._window.evaluate_js(js_code)
        except Exception:
            pass
