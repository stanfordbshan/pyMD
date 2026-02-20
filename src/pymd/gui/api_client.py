"""
HTTP API client: pywebview js_api that proxies calls to the FastAPI server.

This is the ``compute_mode="api"`` path.
"""
from __future__ import annotations

import json
import traceback
from typing import Optional

import urllib.request
import urllib.error


class APIClient:
    """Exposed to JS via pywebview js_api.  Mirrors DirectBridge's interface."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self._base_url = base_url.rstrip("/")
        self._window = None

    def set_window(self, window):
        self._window = window

    # ------------------------------------------------------------------ #
    #  Setup
    # ------------------------------------------------------------------ #

    def load_yaml(self):
        """load_yaml is GUI-only (file dialog) — always handled locally."""
        try:
            import webview
            import yaml

            result = self._window.create_file_dialog(
                dialog_type=webview.FileDialog.OPEN,
                file_types=("YAML Files (*.yaml;*.yml)", "All files (*.*)"),
            )
            if not result:
                return json.dumps({"error": "No file selected"})
            path = result[0] if isinstance(result, (list, tuple)) else result
            with open(path, "r") as f:
                config = yaml.safe_load(f)
            return json.dumps({"ok": True, "config": config, "path": str(path)})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def build_system(self, config_json):
        try:
            raw = json.loads(config_json) if isinstance(config_json, str) else config_json
            resp = self._post("/api/build", raw)
            summary = resp["summary"]
            xyz = summary.get("xyz", "")
            return json.dumps({"ok": True, "summary": summary, "xyz": xyz})
        except Exception as e:
            traceback.print_exc()
            return json.dumps({"error": str(e)})

    def get_xyz(self):
        try:
            resp = self._get("/api/xyz")
            return json.dumps({"ok": True, "xyz": resp["xyz"]})
        except Exception as e:
            return json.dumps({"error": str(e)})

    # ------------------------------------------------------------------ #
    #  Simulation
    # ------------------------------------------------------------------ #

    def start_simulation(self, num_steps, viz_interval=50, init_temp=1.0):
        try:
            resp = self._post("/api/simulation/start", {
                "num_steps": int(num_steps),
                "viz_interval": int(viz_interval),
                "init_temp": float(init_temp),
            })
            # Push final updates to JS
            for u in resp.get("updates", []):
                xyz = u.get("xyz", "")
                self._eval_js(f"onSimulationUpdate({json.dumps(u)}, {json.dumps(xyz)})")
            self._eval_js("onSimulationDone()")
            return json.dumps({"ok": True, "message": f"Started {num_steps} steps"})
        except Exception as e:
            traceback.print_exc()
            self._eval_js(f"onSimulationError({json.dumps(str(e))})")
            return json.dumps({"error": str(e)})

    def stop_simulation(self):
        try:
            self._post("/api/simulation/stop", {})
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def get_energy_data(self):
        try:
            resp = self._get("/api/energy")
            return json.dumps({"ok": True, "data": resp["data"]})
        except Exception as e:
            return json.dumps({"error": str(e)})

    # ------------------------------------------------------------------ #
    #  Minimization
    # ------------------------------------------------------------------ #

    def run_minimization(self, algorithm, params_json):
        try:
            raw = json.loads(params_json) if isinstance(params_json, str) else params_json
            body = {
                "algorithm": algorithm,
                "force_tol": raw.get("force_tol", 1e-4),
                "energy_tol": raw.get("energy_tol", 1e-8),
                "max_steps": raw.get("max_steps", 10000),
            }
            resp = self._post("/api/minimize", body)
            result = resp["result"]
            xyz = result.get("xyz", "")
            self._eval_js(
                f"onMinimizationDone({json.dumps(result)}, {json.dumps(xyz)})"
            )
            return json.dumps({"ok": True, "message": "Minimization started"})
        except Exception as e:
            traceback.print_exc()
            self._eval_js(f"onMinimizationError({json.dumps(str(e))})")
            return json.dumps({"error": str(e)})

    # ------------------------------------------------------------------ #
    #  HTTP helpers
    # ------------------------------------------------------------------ #

    def _post(self, path: str, body: dict) -> dict:
        url = self._base_url + path
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _get(self, path: str) -> dict:
        url = self._base_url + path
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _eval_js(self, js_code: str):
        try:
            if self._window:
                self._window.evaluate_js(js_code)
        except Exception:
            pass
