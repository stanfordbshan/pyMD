/* =============================================================
   pyMD Desktop App - Frontend Logic
   ============================================================= */

// ---- Compute Mode Dispatch ----
// Config is injected by Python via evaluate_js (window.__PYMD_*).
// Falls back to query params for standalone browser testing, then defaults.
function _getConfig(key, fallback) {
  const injected = window['__PYMD_' + key];
  if (injected !== undefined && injected !== '') return injected;
  const params = new URLSearchParams(window.location.search);
  return params.get(key.toLowerCase()) || fallback;
}
const COMPUTE_MODE = _getConfig('COMPUTE_MODE', 'direct');
const API_BASE_URL = _getConfig('API_BASE_URL', '');

async function callDirect(method, ...args) {
  const raw = await window.pywebview.api[method](...args);
  return JSON.parse(raw);
}

async function callApi(endpoint, httpMethod, body) {
  const opts = { method: httpMethod, headers: { 'Content-Type': 'application/json' } };
  if (body !== undefined) opts.body = JSON.stringify(body);
  const resp = await fetch(API_BASE_URL + endpoint, opts);
  if (!resp.ok) {
    const e = await resp.json().catch(() => ({ detail: 'API error' }));
    throw new Error(e.detail || 'API error');
  }
  return resp.json();
}

/**
 * Dispatch a call based on the current compute mode.
 *
 * @param {Function} directFn  — async function that calls pywebview.api.*
 * @param {Function} apiFn     — async function that calls fetch()
 * @returns {Promise<*>}
 */
async function dispatch(directFn, apiFn) {
  if (COMPUTE_MODE === 'direct') return directFn();
  if (COMPUTE_MODE === 'api') return apiFn();
  // auto: try direct first, fall back to API
  try {
    return await directFn();
  } catch (_directErr) {
    return apiFn();
  }
}

// ---- Global State ----
let viewer = null;
let systemBuilt = false;
let simEnergyData = { steps: [], pe: [], ke: [], total: [] };
let minEnergyHistory = [];
let lastSimResult = null;
let lastMinResult = null;
let loadedCoordinates = null; // Stores custom positions from YAML
let pendingXyz = null; // XYZ string waiting for viewer to be visible
let defaultView = null; // Saved default camera view for reset
let lastYamlPotential = null; // Snapshot of potential params from YAML load

// ---- Tab Switching ----
document.querySelectorAll(".nav-item").forEach((item) => {
  item.addEventListener("click", () => {
    document.querySelectorAll(".nav-item").forEach((i) => i.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    item.classList.add("active");
    const tab = item.dataset.tab;
    document.getElementById("tab-" + tab).classList.add("active");

    // Sync and redraw Potential Explorer when opening its tab
    if (tab === "potential") {
      peSyncFromSetup();
      peYRange = null; // Recompute Y range from fresh setup values
      peRedraw();
      peUpdateEducational();
    }

    // Initialize or refresh 3Dmol viewer when switching to visualization tab.
    // A short timeout ensures the container has its final dimensions before
    // 3Dmol measures the canvas — avoids garbled first render on Windows.
    if (tab === "visualization") {
      setTimeout(() => {
        if (!viewer) {
          initViewer();
        }
        if (viewer) {
          viewer.resize();
          if (pendingXyz) {
            loadXyzIntoViewer(pendingXyz);
          } else {
            viewer.render();
          }
        }
      }, 100);
    }
  });
});

// ---- 3Dmol.js Viewer ----
function initViewer() {
  const container = document.getElementById("mol-viewer");
  if (!container || viewer) return;

  viewer = $3Dmol.createViewer(container, {
    backgroundColor: 0x1a1a2e,
    antialias: true,
    disableFog: true,
    cartoonQuality: 6,
  });
  viewer.setHoverable({}, false);
  viewer.render();
}

// Handle window resize — keep viewer sized correctly
window.addEventListener("resize", () => {
  if (viewer) {
    viewer.resize();
    viewer.render();
  }
});

function getViewerStyle() {
  const sel = document.getElementById("cfg-viewer-style");
  const mode = sel ? sel.value : "ball-stick";
  const cs = { colorscheme: "Jmol" };
  switch (mode) {
    case "sphere":
      return { sphere: { ...cs } };
    case "ball-stick":
      return { sphere: { scale: 0.3, ...cs }, stick: { radius: 0.15, ...cs } };
    case "stick":
      return { stick: { radius: 0.15, ...cs } };
    case "line":
      return { line: { ...cs } };
    default:
      return { sphere: { scale: 0.3, ...cs } };
  }
}

function loadXyzIntoViewer(xyzString, preserveView) {
  if (!viewer) return;
  const savedView = preserveView ? viewer.getView() : null;
  viewer.removeAllModels();
  viewer.addModel(xyzString, "xyz");
  viewer.setStyle({}, getViewerStyle());
  if (savedView) {
    viewer.setView(savedView);
  } else {
    viewer.zoomTo();
  }
  viewer.render();
  if (!preserveView) {
    defaultView = viewer.getView();
  }
}

function updateViewer(xyzString, preserveView) {
  pendingXyz = xyzString;
  if (!xyzString) return;
  const visTab = document.getElementById("tab-visualization");
  if (!visTab.classList.contains("active")) return;
  if (!viewer) initViewer();
  if (!viewer) return;
  loadXyzIntoViewer(xyzString, preserveView);
}

// ---- Setup Tab: Dynamic Form ----
document.getElementById("cfg-lattice-type").addEventListener("change", (e) => {
  const val = e.target.value;
  document.getElementById("lattice-params-grid").style.display =
    (val !== "random" && val !== "positions") ? "grid" : "none";
  document.getElementById("random-params-grid").style.display = val === "random" ? "grid" : "none";
  document.getElementById("positions-params-grid").style.display = val === "positions" ? "grid" : "none";
});

document.getElementById("cfg-boundary").addEventListener("change", (e) => {
  document.getElementById("mixed-dims").style.display =
    e.target.value === "mixed" ? "block" : "none";
});

document.getElementById("cfg-potential-type").addEventListener("change", (e) => {
  document.getElementById("lj-params").style.display = e.target.value === "lj" ? "block" : "none";
  document.getElementById("morse-params").style.display =
    e.target.value === "morse" ? "block" : "none";
});

document.getElementById("cfg-thermostat-type").addEventListener("change", (e) => {
  document.getElementById("thermostat-tau-param").style.display =
    e.target.value === "nve" ? "none" : "block";
});

// ---- Load YAML ----
// load_yaml always uses pywebview (native file dialog) — direct-only.
document.getElementById("btn-load-yaml").addEventListener("click", async () => {
  setStatus("Loading YAML...", "running");
  try {
    const raw = await pywebview.api.load_yaml();
    const result = JSON.parse(raw);
    if (result.error) {
      setStatus("Error: " + result.error, "error");
      return;
    }
    populateForm(result.config);
    document.getElementById("yaml-path").textContent = result.path;
    setStatus("YAML loaded", "idle");
  } catch (err) {
    setStatus("Error: " + err.message, "error");
  }
});

function populateForm(config) {
  // Units
  if (config.units) {
    document.getElementById("cfg-units").value = config.units.toUpperCase();
  }

  // System
  const sys = config.system || {};
  if (sys.element) document.getElementById("cfg-element").value = sys.element;
  if (sys.mass) document.getElementById("cfg-mass").value = sys.mass;
  if (sys.temperature != null) {
    document.getElementById("cfg-thermo-temp").value = sys.temperature;
  }

  const lat = sys.lattice || {};
  if (lat.type) {
    document.getElementById("cfg-lattice-type").value = lat.type.toLowerCase();
    document.getElementById("cfg-lattice-type").dispatchEvent(new Event("change"));
  }
  if (lat.nx) document.getElementById("cfg-nx").value = lat.nx;
  if (lat.ny) document.getElementById("cfg-ny").value = lat.ny;
  if (lat.nz) document.getElementById("cfg-nz").value = lat.nz;
  if (lat.a) document.getElementById("cfg-a").value = lat.a;
  if (lat.n_atoms) document.getElementById("cfg-n-atoms").value = lat.n_atoms;
  if (lat.lx) document.getElementById("cfg-lx").value = lat.lx;
  if (lat.ly) document.getElementById("cfg-ly").value = lat.ly;
  if (lat.lz) document.getElementById("cfg-lz").value = lat.lz;

  // Custom positions
  if (lat.coordinates) {
    loadedCoordinates = lat.coordinates;
    document.getElementById("positions-info").textContent =
      loadedCoordinates.length + " custom positions loaded";
  }
  if (lat.type && lat.type.toLowerCase() === "positions") {
    if (lat.lx) document.getElementById("cfg-pos-lx").value = lat.lx;
    if (lat.ly) document.getElementById("cfg-pos-ly").value = lat.ly;
    if (lat.lz) document.getElementById("cfg-pos-lz").value = lat.lz;
  }

  // Boundary
  const bc = config.boundary || {};
  if (bc.type) {
    document.getElementById("cfg-boundary").value = bc.type.toLowerCase();
    document.getElementById("cfg-boundary").dispatchEvent(new Event("change"));
  }
  if (bc.periodic_dims) {
    document.getElementById("cfg-px").checked = bc.periodic_dims[0];
    document.getElementById("cfg-py").checked = bc.periodic_dims[1];
    document.getElementById("cfg-pz").checked = bc.periodic_dims[2];
  }

  // Potential
  const pot = config.potential || {};
  if (pot.type) {
    const ptype = pot.type.toLowerCase();
    document.getElementById("cfg-potential-type").value = ptype === "lennard_jones" ? "lj" : ptype;
    document.getElementById("cfg-potential-type").dispatchEvent(new Event("change"));
  }
  if (pot.epsilon != null) document.getElementById("cfg-epsilon").value = pot.epsilon;
  if (pot.sigma != null) document.getElementById("cfg-sigma").value = pot.sigma;
  if (pot.cutoff != null) document.getElementById("cfg-cutoff").value = pot.cutoff;
  if (pot.D != null) document.getElementById("cfg-morse-D").value = pot.D;
  if (pot.a != null) document.getElementById("cfg-morse-a").value = pot.a;
  if (pot.r0 != null) document.getElementById("cfg-morse-r0").value = pot.r0;
  // Morse cutoff
  if (pot.type && pot.type.toLowerCase() === "morse" && pot.cutoff != null) {
    document.getElementById("cfg-morse-cutoff").value = pot.cutoff;
  }

  // Snapshot potential state for "Reset to YAML" button
  lastYamlPotential = {
    type: document.getElementById("cfg-potential-type").value,
    epsilon: document.getElementById("cfg-epsilon").value,
    sigma: document.getElementById("cfg-sigma").value,
    cutoff: document.getElementById("cfg-cutoff").value,
    morseD: document.getElementById("cfg-morse-D").value,
    morseA: document.getElementById("cfg-morse-a").value,
    morseR0: document.getElementById("cfg-morse-r0").value,
    morseCutoff: document.getElementById("cfg-morse-cutoff").value,
  };
  const resetBtn = document.getElementById("pe-reset-yaml");
  if (resetBtn) resetBtn.disabled = false;

  // Integrator
  const intgr = config.integrator || {};
  if (intgr.dt) document.getElementById("cfg-dt").value = intgr.dt;

  // Thermostat
  const thermo = config.thermostat || {};
  if (thermo.type) {
    document.getElementById("cfg-thermostat-type").value = thermo.type.toLowerCase();
    document.getElementById("cfg-thermostat-type").dispatchEvent(new Event("change"));
  }
  if (thermo.temperature != null)
    document.getElementById("cfg-thermo-temp").value = thermo.temperature;
  if (thermo.tau != null) document.getElementById("cfg-thermo-tau").value = thermo.tau;

  // Observers
  const obs = config.observers || {};
  if (obs.energy_interval) document.getElementById("cfg-energy-interval").value = obs.energy_interval;
}

// ---- Build System ----
document.getElementById("btn-build-system").addEventListener("click", async () => {
  setStatus("Building system...", "running");
  const config = gatherConfig();
  try {
    const result = await dispatch(
      // direct path
      async () => {
        const raw = await pywebview.api.build_system(JSON.stringify(config));
        return JSON.parse(raw);
      },
      // API path
      async () => {
        const resp = await callApi('/api/build', 'POST', config);
        return { ok: true, summary: resp.summary, xyz: resp.summary.xyz };
      },
    );
    if (result.error) {
      setStatus("Build failed: " + result.error, "error");
      return;
    }
    systemBuilt = true;
    showSummary(result.summary);
    updateViewer(result.xyz);
    document.getElementById("viewer-info").textContent =
      result.summary.n_atoms + " atoms (" + result.summary.element + ")";
    setStatus("System built: " + result.summary.n_atoms + " atoms", "idle");
  } catch (err) {
    setStatus("Build error: " + err.message, "error");
  }
});

function gatherConfig() {
  const latticeType = document.getElementById("cfg-lattice-type").value;
  const potType = document.getElementById("cfg-potential-type").value;
  const bcType = document.getElementById("cfg-boundary").value;
  const thermoType = document.getElementById("cfg-thermostat-type").value;

  const temperature = parseFloat(document.getElementById("cfg-thermo-temp").value);

  const config = {
    units: document.getElementById("cfg-units").value,
    system: {
      element: document.getElementById("cfg-element").value,
      mass: parseFloat(document.getElementById("cfg-mass").value),
      lattice: { type: latticeType },
      temperature: temperature,
    },
    boundary: { type: bcType },
    integrator: { dt: parseFloat(document.getElementById("cfg-dt").value) },
    thermostat: { type: thermoType },
    observers: {
      energy: true,
      energy_interval: parseInt(document.getElementById("cfg-energy-interval").value),
    },
  };

  // Lattice params
  if (latticeType === "random") {
    config.system.lattice.n_atoms = parseInt(document.getElementById("cfg-n-atoms").value);
    config.system.lattice.lx = parseFloat(document.getElementById("cfg-lx").value);
    config.system.lattice.ly = parseFloat(document.getElementById("cfg-ly").value);
    config.system.lattice.lz = parseFloat(document.getElementById("cfg-lz").value);
  } else if (latticeType === "positions") {
    config.system.lattice.coordinates = loadedCoordinates || [];
    config.system.lattice.lx = parseFloat(document.getElementById("cfg-pos-lx").value);
    config.system.lattice.ly = parseFloat(document.getElementById("cfg-pos-ly").value);
    config.system.lattice.lz = parseFloat(document.getElementById("cfg-pos-lz").value);
  } else {
    config.system.lattice.nx = parseInt(document.getElementById("cfg-nx").value);
    config.system.lattice.ny = parseInt(document.getElementById("cfg-ny").value);
    config.system.lattice.nz = parseInt(document.getElementById("cfg-nz").value);
    config.system.lattice.a = parseFloat(document.getElementById("cfg-a").value);
  }

  // Boundary
  if (bcType === "mixed") {
    config.boundary.periodic_dims = [
      document.getElementById("cfg-px").checked,
      document.getElementById("cfg-py").checked,
      document.getElementById("cfg-pz").checked,
    ];
  }

  // Potential
  if (potType === "lj") {
    config.potential = {
      type: "lj",
      epsilon: parseFloat(document.getElementById("cfg-epsilon").value),
      sigma: parseFloat(document.getElementById("cfg-sigma").value),
      cutoff: parseFloat(document.getElementById("cfg-cutoff").value),
    };
  } else {
    config.potential = {
      type: "morse",
      D: parseFloat(document.getElementById("cfg-morse-D").value),
      a: parseFloat(document.getElementById("cfg-morse-a").value),
      r0: parseFloat(document.getElementById("cfg-morse-r0").value),
      cutoff: parseFloat(document.getElementById("cfg-morse-cutoff").value),
    };
  }

  // Thermostat
  if (thermoType !== "nve") {
    config.thermostat.temperature = temperature;
    config.thermostat.tau = parseFloat(document.getElementById("cfg-thermo-tau").value);
  }

  return config;
}

function showSummary(summary) {
  const box = document.getElementById("build-summary");
  box.style.display = "block";
  document.getElementById("summary-content").innerHTML = `
    <table>
      <tr><td>Atoms</td><td>${summary.n_atoms}</td></tr>
      <tr><td>Element</td><td>${summary.element}</td></tr>
      <tr><td>Box</td><td>[${summary.box.map((v) => v.toFixed(3)).join(", ")}]</td></tr>
      <tr><td>Units</td><td>${summary.units}</td></tr>
      <tr><td>Potential</td><td>${summary.potential}</td></tr>
      <tr><td>Boundary</td><td>${summary.boundary}</td></tr>
      <tr><td>Timestep</td><td>${summary.dt}</td></tr>
      <tr><td>Thermostat</td><td>${summary.thermostat}</td></tr>
    </table>`;
}

// ---- Simulation Tab ----
document.getElementById("btn-start-sim").addEventListener("click", async () => {
  if (!systemBuilt) {
    setStatus("Build a system first (Setup tab)", "error");
    return;
  }
  const steps = parseInt(document.getElementById("sim-steps").value);
  const initTemp = parseFloat(document.getElementById("cfg-thermo-temp").value);
  const vizInterval = parseInt(document.getElementById("sim-viz-interval").value);

  // Reset chart data
  simEnergyData = { steps: [], pe: [], ke: [], total: [] };
  clearCanvas("sim-chart");

  document.getElementById("btn-start-sim").disabled = true;
  document.getElementById("btn-stop-sim").disabled = false;
  setStatus("Simulation running...", "running");

  try {
    await dispatch(
      // direct path: bridge pushes updates via evaluate_js callbacks
      async () => {
        const raw = await pywebview.api.start_simulation(steps, vizInterval, initTemp);
        const result = JSON.parse(raw);
        if (result.error) throw new Error(result.error);
        return result;
      },
      // API path: receives all updates at once, replays them
      async () => {
        const resp = await callApi('/api/simulation/start', 'POST', {
          num_steps: steps,
          viz_interval: vizInterval,
          init_temp: initTemp,
        });
        // Replay updates sequentially into the UI
        for (const u of (resp.updates || [])) {
          onSimulationUpdate(u, u.xyz);
        }
        onSimulationDone();
        return resp;
      },
    );
  } catch (err) {
    setStatus("Error: " + err.message, "error");
    document.getElementById("btn-start-sim").disabled = false;
    document.getElementById("btn-stop-sim").disabled = true;
  }
});

document.getElementById("btn-stop-sim").addEventListener("click", async () => {
  try {
    await dispatch(
      async () => {
        await pywebview.api.stop_simulation();
      },
      async () => {
        await callApi('/api/simulation/stop', 'POST', {});
      },
    );
    setStatus("Stopping...", "running");
  } catch (err) {
    console.error(err);
  }
});

// Called from Python via evaluate_js (direct mode)
function onSimulationUpdate(data, xyzString) {
  // Update readout
  document.getElementById("ro-step").textContent = data.step;
  document.getElementById("ro-temp").textContent = data.temperature.toFixed(4);
  document.getElementById("ro-pe").textContent = data.pe.toFixed(4);
  document.getElementById("ro-ke").textContent = data.ke.toFixed(4);
  document.getElementById("ro-total").textContent = data.total_e.toFixed(4);

  // Update progress
  const pct = ((data.step / data.total) * 100).toFixed(1);
  document.getElementById("sim-progress").style.width = pct + "%";
  document.getElementById("sim-progress-text").textContent = data.step + " / " + data.total;

  // Accumulate chart data
  simEnergyData.steps.push(data.step);
  simEnergyData.pe.push(data.pe);
  simEnergyData.ke.push(data.ke);
  simEnergyData.total.push(data.total_e);

  // Draw chart
  drawEnergyChart("sim-chart", simEnergyData.steps, [
    { data: simEnergyData.pe, color: "#ff6b6b", label: "PE" },
    { data: simEnergyData.ke, color: "#51cf66", label: "KE" },
    { data: simEnergyData.total, color: "#6c8cff", label: "Total" },
  ]);

  // Update 3D viewer (preserve camera angle during simulation)
  updateViewer(xyzString, true);
}

function onSimulationDone() {
  document.getElementById("btn-start-sim").disabled = false;
  document.getElementById("btn-stop-sim").disabled = true;
  setStatus("Simulation complete", "idle");

  // Store for Results tab
  lastSimResult = {
    steps: simEnergyData.steps,
    pe: simEnergyData.pe,
    ke: simEnergyData.ke,
    total: simEnergyData.total,
  };
  updateResultsTab();
}

function onSimulationError(msg) {
  document.getElementById("btn-start-sim").disabled = false;
  document.getElementById("btn-stop-sim").disabled = true;
  setStatus("Simulation error: " + msg, "error");
}

// ---- Minimization Tab ----
document.getElementById("btn-minimize").addEventListener("click", async () => {
  if (!systemBuilt) {
    setStatus("Build a system first (Setup tab)", "error");
    return;
  }

  const algo = document.getElementById("min-algorithm").value;
  const params = {
    force_tol: parseFloat(document.getElementById("min-force-tol").value),
    energy_tol: parseFloat(document.getElementById("min-energy-tol").value),
    max_steps: parseInt(document.getElementById("min-max-steps").value),
  };

  document.getElementById("btn-minimize").disabled = true;
  document.getElementById("min-results").style.display = "none";
  minEnergyHistory = [];
  clearCanvas("min-chart");
  setStatus("Minimizing...", "running");

  try {
    await dispatch(
      // direct path: bridge pushes result via evaluate_js callback
      async () => {
        const raw = await pywebview.api.run_minimization(algo, JSON.stringify(params));
        const result = JSON.parse(raw);
        if (result.error) throw new Error(result.error);
        return result;
      },
      // API path: receives result directly
      async () => {
        const resp = await callApi('/api/minimize', 'POST', {
          algorithm: algo,
          ...params,
        });
        onMinimizationDone(resp.result, resp.result.xyz);
        return resp;
      },
    );
  } catch (err) {
    setStatus("Error: " + err.message, "error");
    document.getElementById("btn-minimize").disabled = false;
  }
});

// Called from Python via evaluate_js (direct mode)
function onMinimizationDone(result, xyzString) {
  document.getElementById("btn-minimize").disabled = false;
  setStatus("Minimization complete", "idle");

  // Show results
  const box = document.getElementById("min-results");
  box.style.display = "block";
  const convClass = result.converged ? "converged-yes" : "converged-no";
  const convText = result.converged ? "Yes" : "No";
  document.getElementById("min-results-content").innerHTML = `
    <table>
      <tr><td>Converged</td><td class="${convClass}">${convText}</td></tr>
      <tr><td>Steps Taken</td><td>${result.n_steps}</td></tr>
      <tr><td>Initial Energy</td><td>${result.initial_energy.toFixed(6)}</td></tr>
      <tr><td>Final Energy</td><td>${result.final_energy.toFixed(6)}</td></tr>
      <tr><td>Max Force</td><td>${result.max_force.toExponential(4)}</td></tr>
      <tr><td>Message</td><td>${result.message}</td></tr>
    </table>`;

  // Draw energy history chart
  if (result.energy_history && result.energy_history.length > 0) {
    minEnergyHistory = result.energy_history;
    const steps = result.energy_history.map((_, i) => i);
    drawEnergyChart("min-chart", steps, [
      { data: result.energy_history, color: "#6c8cff", label: "Energy" },
    ]);
  }

  // Update viewer
  updateViewer(xyzString);

  // Store for Results tab
  lastMinResult = result;
  updateResultsTab();
}

function onMinimizationError(msg) {
  document.getElementById("btn-minimize").disabled = false;
  setStatus("Minimization error: " + msg, "error");
}

// ---- Results Tab ----
function updateResultsTab() {
  const container = document.getElementById("results-content");
  let html = "<table>";

  if (lastSimResult && lastSimResult.steps.length > 0) {
    const n = lastSimResult.steps.length;
    const finalPE = lastSimResult.pe[n - 1];
    const finalKE = lastSimResult.ke[n - 1];
    const finalTotal = lastSimResult.total[n - 1];
    const initTotal = lastSimResult.total[0];
    const drift = initTotal !== 0 ? ((finalTotal - initTotal) / Math.abs(initTotal)) : 0;

    html += `
      <tr><td colspan="2"><strong>Simulation Results</strong></td></tr>
      <tr><td>Total Steps Recorded</td><td>${lastSimResult.steps[n - 1]}</td></tr>
      <tr><td>Final PE</td><td>${finalPE.toFixed(6)}</td></tr>
      <tr><td>Final KE</td><td>${finalKE.toFixed(6)}</td></tr>
      <tr><td>Final Total Energy</td><td>${finalTotal.toFixed(6)}</td></tr>
      <tr><td>Initial Total Energy</td><td>${initTotal.toFixed(6)}</td></tr>
      <tr><td>Energy Drift</td><td>${drift.toExponential(4)}</td></tr>`;
  }

  if (lastMinResult) {
    html += `
      <tr><td colspan="2" style="padding-top:16px"><strong>Minimization Results</strong></td></tr>
      <tr><td>Algorithm</td><td>${document.getElementById("min-algorithm").value}</td></tr>
      <tr><td>Converged</td><td>${lastMinResult.converged ? "Yes" : "No"}</td></tr>
      <tr><td>Steps</td><td>${lastMinResult.n_steps}</td></tr>
      <tr><td>Initial Energy</td><td>${lastMinResult.initial_energy.toFixed(6)}</td></tr>
      <tr><td>Final Energy</td><td>${lastMinResult.final_energy.toFixed(6)}</td></tr>
      <tr><td>Max Force</td><td>${lastMinResult.max_force.toExponential(4)}</td></tr>`;
  }

  if (!lastSimResult && !lastMinResult) {
    html += '<tr><td colspan="2" class="info-text">Run a simulation or minimization to see results here.</td></tr>';
  }

  html += "</table>";
  container.innerHTML = html;
}

// Copy energy data to clipboard
document.getElementById("btn-copy-data").addEventListener("click", () => {
  let text = "";
  if (lastSimResult && lastSimResult.steps.length > 0) {
    text += "# Simulation Energy Data\n";
    text += "# Step\tPE\tKE\tTotal\n";
    for (let i = 0; i < lastSimResult.steps.length; i++) {
      text += `${lastSimResult.steps[i]}\t${lastSimResult.pe[i]}\t${lastSimResult.ke[i]}\t${lastSimResult.total[i]}\n`;
    }
  }
  if (lastMinResult && lastMinResult.energy_history) {
    text += "\n# Minimization Energy History\n";
    text += "# Step\tEnergy\n";
    lastMinResult.energy_history.forEach((e, i) => {
      text += `${i}\t${e}\n`;
    });
  }
  if (!text) {
    text = "No data available.";
  }
  navigator.clipboard.writeText(text).then(() => {
    setStatus("Data copied to clipboard", "idle");
  });
});

// ---- Reset View ----
document.getElementById("btn-reset-view").addEventListener("click", () => {
  if (viewer) {
    viewer.resize();
    if (defaultView) {
      viewer.setView(defaultView);
      viewer.render();
    } else {
      viewer.zoomTo();
      viewer.render();
    }
  }
});

// ---- Viewer Style Change ----
document.getElementById("cfg-viewer-style").addEventListener("change", () => {
  if (viewer && pendingXyz) {
    loadXyzIntoViewer(pendingXyz, true);
  }
});

// ---- Canvas Chart Drawing ----
function clearCanvas(canvasId) {
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function drawEnergyChart(canvasId, xData, series) {
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;
  const pad = { top: 20, right: 100, bottom: 40, left: 70 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  ctx.clearRect(0, 0, W, H);

  if (xData.length < 2) return;

  // Compute global Y range
  let yMin = Infinity,
    yMax = -Infinity;
  series.forEach((s) => {
    s.data.forEach((v) => {
      if (v < yMin) yMin = v;
      if (v > yMax) yMax = v;
    });
  });
  if (yMin === yMax) {
    yMin -= 1;
    yMax += 1;
  }
  const yPad = (yMax - yMin) * 0.05;
  yMin -= yPad;
  yMax += yPad;

  const xMin = xData[0];
  const xMax = xData[xData.length - 1];

  // Helper: data to canvas
  function toX(v) {
    return pad.left + ((v - xMin) / (xMax - xMin || 1)) * plotW;
  }
  function toY(v) {
    return pad.top + plotH - ((v - yMin) / (yMax - yMin || 1)) * plotH;
  }

  // Grid
  ctx.strokeStyle = "#404060";
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= 5; i++) {
    const y = pad.top + (plotH * i) / 5;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(pad.left + plotW, y);
    ctx.stroke();
  }

  // Axes
  ctx.strokeStyle = "#606080";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, pad.top + plotH);
  ctx.lineTo(pad.left + plotW, pad.top + plotH);
  ctx.stroke();

  // Axis labels
  ctx.fillStyle = "#a0a0c0";
  ctx.font = "11px sans-serif";
  ctx.textAlign = "center";
  // X axis ticks
  for (let i = 0; i <= 5; i++) {
    const val = xMin + ((xMax - xMin) * i) / 5;
    const x = toX(val);
    ctx.fillText(Math.round(val).toString(), x, pad.top + plotH + 16);
  }
  // Y axis ticks
  ctx.textAlign = "right";
  for (let i = 0; i <= 5; i++) {
    const val = yMin + ((yMax - yMin) * i) / 5;
    const y = toY(val);
    ctx.fillText(formatNum(val), pad.left - 6, y + 4);
  }

  // X label
  ctx.textAlign = "center";
  ctx.fillText("Step", pad.left + plotW / 2, H - 4);

  // Draw series
  series.forEach((s) => {
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < s.data.length; i++) {
      const x = toX(xData[i]);
      const y = toY(s.data[i]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  });

  // Legend
  let ly = pad.top + 10;
  series.forEach((s) => {
    ctx.fillStyle = s.color;
    ctx.fillRect(pad.left + plotW + 10, ly - 4, 14, 3);
    ctx.fillStyle = "#e0e0f0";
    ctx.textAlign = "left";
    ctx.font = "11px sans-serif";
    ctx.fillText(s.label, pad.left + plotW + 28, ly);
    ly += 18;
  });
}

function formatNum(v) {
  if (Math.abs(v) >= 1000 || (Math.abs(v) < 0.01 && v !== 0)) {
    return v.toExponential(1);
  }
  return v.toFixed(2);
}

// ---- Status Bar ----
function setStatus(text, type) {
  const bar = document.getElementById("status-bar");
  bar.textContent = text;
  bar.className = "status-" + (type || "idle");
}

// ---- Keyboard Shortcuts ----
document.addEventListener("keydown", (e) => {
  if (!(e.ctrlKey || e.metaKey)) return;

  const shortcuts = {
    o: "btn-load-yaml",
    b: "btn-build-system",
    r: "btn-start-sim",
    ".": "btn-stop-sim",
    m: "btn-minimize",
  };

  const key = e.key.toLowerCase();
  const targetId = shortcuts[key];
  if (!targetId) return;

  const btn = document.getElementById(targetId);
  if (btn) {
    e.preventDefault();
    btn.click();
  }
});

// =====================================================================
//  Potential Explorer
// =====================================================================

let peYRange = null; // Fixed Y-axis range; reset on type change

// -- Sync map: PE slider id → Setup tab input id --
const PE_SYNC_MAP = {
  "pe-lj-epsilon": "cfg-epsilon",
  "pe-lj-sigma": "cfg-sigma",
  "pe-lj-cutoff": "cfg-cutoff",
  "pe-morse-D": "cfg-morse-D",
  "pe-morse-a": "cfg-morse-a",
  "pe-morse-r0": "cfg-morse-r0",
  "pe-morse-cutoff": "cfg-morse-cutoff",
};

// -- Math functions (matching Python source) --
function ljEnergy(r, eps, sig) {
  const s = sig / r;
  const s6 = s * s * s * s * s * s;
  return 4 * eps * (s6 * s6 - s6);
}

function ljForce(r, eps, sig) {
  const s = sig / r;
  const s6 = s * s * s * s * s * s;
  return 24 * eps / r * (2 * s6 * s6 - s6);
}

function morseEnergy(r, D, a, r0) {
  const ex = Math.exp(-a * (r - r0));
  return D * (1 - ex) * (1 - ex);
}

function morseForce(r, D, a, r0) {
  const ex = Math.exp(-a * (r - r0));
  return 2 * D * a * ex * (1 - ex);
}

// -- Type selector: toggle fieldsets --
document.getElementById("pe-type").addEventListener("change", (e) => {
  const isLJ = e.target.value === "lj";
  document.getElementById("pe-lj-fieldset").style.display = isLJ ? "block" : "none";
  document.getElementById("pe-morse-fieldset").style.display = isLJ ? "none" : "block";
  peYRange = null; // Reset Y-axis range on type change
  peRedraw();
  peUpdateEducational();
});

// -- Slider input handlers (no auto-sync to Setup; use Apply button) --
Object.keys(PE_SYNC_MAP).forEach((sliderId) => {
  const slider = document.getElementById(sliderId);
  if (!slider) return;
  slider.addEventListener("input", () => {
    document.getElementById(sliderId + "-val").textContent =
      parseFloat(slider.value).toFixed(2);
    peRedraw();
  });
});

// -- Checkbox handlers --
document.getElementById("pe-show-force").addEventListener("change", peRedraw);
document.getElementById("pe-annotations").addEventListener("change", peRedraw);

// -- Sync from Setup tab → sliders --
function peSyncFromSetup() {
  Object.keys(PE_SYNC_MAP).forEach((sliderId) => {
    const cfgInput = document.getElementById(PE_SYNC_MAP[sliderId]);
    const slider = document.getElementById(sliderId);
    if (!cfgInput || !slider) return;
    const v = parseFloat(cfgInput.value);
    if (!isNaN(v)) {
      // Clamp to slider range
      const clamped = Math.min(Math.max(v, parseFloat(slider.min)), parseFloat(slider.max));
      slider.value = clamped;
      document.getElementById(sliderId + "-val").textContent = clamped.toFixed(2);
    }
  });
}

// -- Reset to YAML --
document.getElementById("pe-reset-yaml").addEventListener("click", () => {
  if (!lastYamlPotential) return;
  // Restore Setup tab inputs
  document.getElementById("cfg-potential-type").value = lastYamlPotential.type;
  document.getElementById("cfg-potential-type").dispatchEvent(new Event("change"));
  document.getElementById("cfg-epsilon").value = lastYamlPotential.epsilon;
  document.getElementById("cfg-sigma").value = lastYamlPotential.sigma;
  document.getElementById("cfg-cutoff").value = lastYamlPotential.cutoff;
  document.getElementById("cfg-morse-D").value = lastYamlPotential.morseD;
  document.getElementById("cfg-morse-a").value = lastYamlPotential.morseA;
  document.getElementById("cfg-morse-r0").value = lastYamlPotential.morseR0;
  document.getElementById("cfg-morse-cutoff").value = lastYamlPotential.morseCutoff;
  // Update PE type selector to match
  document.getElementById("pe-type").value = lastYamlPotential.type;
  document.getElementById("pe-type").dispatchEvent(new Event("change"));
  // Sync sliders from restored Setup values
  peSyncFromSetup();
  peYRange = null; // Recompute Y range from YAML values
  peRedraw();
});

// -- Redraw chart --
function peRedraw() {
  const type = document.getElementById("pe-type").value;
  const showForce = document.getElementById("pe-show-force").checked;
  const showAnnotations = document.getElementById("pe-annotations").checked;

  let cutoff, rData, uData, fData, eqR, wellDepth, zeroCross;

  if (type === "lj") {
    const eps = parseFloat(document.getElementById("pe-lj-epsilon").value);
    const sig = parseFloat(document.getElementById("pe-lj-sigma").value);
    cutoff = parseFloat(document.getElementById("pe-lj-cutoff").value);
    const rMin = sig * 0.85;
    const rMax = cutoff;
    rData = []; uData = []; fData = [];
    for (let i = 0; i < 500; i++) {
      const r = rMin + (rMax - rMin) * i / 499;
      rData.push(r);
      uData.push(ljEnergy(r, eps, sig));
      fData.push(ljForce(r, eps, sig));
    }
    eqR = sig * Math.pow(2, 1 / 6);
    wellDepth = -eps;
    // Zero crossing: U(r)=0 at r = sigma for LJ
    zeroCross = sig;
  } else {
    const D = parseFloat(document.getElementById("pe-morse-D").value);
    const a = parseFloat(document.getElementById("pe-morse-a").value);
    const r0 = parseFloat(document.getElementById("pe-morse-r0").value);
    cutoff = parseFloat(document.getElementById("pe-morse-cutoff").value);
    const rMin = Math.max(0.01, r0 - 3 / a);
    const rMax = cutoff;
    rData = []; uData = []; fData = [];
    for (let i = 0; i < 500; i++) {
      const r = rMin + (rMax - rMin) * i / 499;
      rData.push(r);
      uData.push(morseEnergy(r, D, a, r0));
      fData.push(morseForce(r, D, a, r0));
    }
    eqR = r0;
    wellDepth = 0; // Morse minimum is 0 at r=r0
    zeroCross = null; // Morse doesn't cross zero (U >= 0)
  }

  // Update info box
  const infoEl = document.getElementById("pe-info-content");
  let infoHtml = `Equilibrium r: <strong>${eqR.toFixed(4)}</strong><br>`;
  infoHtml += `Well depth: <strong>${wellDepth.toFixed(4)}</strong><br>`;
  if (zeroCross != null) {
    infoHtml += `Zero crossing: <strong>${zeroCross.toFixed(4)}</strong><br>`;
  }
  infoHtml += `Cutoff: <strong>${cutoff.toFixed(2)}</strong>`;
  infoEl.innerHTML = infoHtml;

  // Draw chart
  drawPotentialChart("pe-chart", rData, uData, showForce ? fData : null, {
    showAnnotations,
    eqR,
    wellDepth,
    zeroCross,
    cutoff,
    type,
  });
}

function drawPotentialChart(canvasId, rData, uData, fData, opts) {
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;
  const pad = { top: 24, right: 120, bottom: 44, left: 70 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  ctx.clearRect(0, 0, W, H);

  if (rData.length < 2) return;

  // Y range: use fixed range if available, otherwise compute and store
  let yMin, yMax;
  if (peYRange) {
    yMin = peYRange[0];
    yMax = peYRange[1];
  } else {
    yMin = Infinity; yMax = -Infinity;
    uData.forEach((v) => { if (v < yMin) yMin = v; if (v > yMax) yMax = v; });
    if (fData) {
      fData.forEach((v) => { if (v < yMin) yMin = v; if (v > yMax) yMax = v; });
    }
    if (yMin === yMax) { yMin -= 1; yMax += 1; }
    const yPad = (yMax - yMin) * 0.08;
    yMin -= yPad; yMax += yPad;
    // Ensure Y floor accommodates full slider range
    if (opts.type === "lj") {
      const epsMax = parseFloat(document.getElementById("pe-lj-epsilon").max) || 5;
      yMin = Math.min(yMin, -epsMax * 1.1);
    }
    peYRange = [yMin, yMax];
  }

  const xMin = rData[0];
  const xMax = rData[rData.length - 1];

  function toX(v) { return pad.left + ((v - xMin) / (xMax - xMin || 1)) * plotW; }
  function toY(v) { return pad.top + plotH - ((v - yMin) / (yMax - yMin || 1)) * plotH; }

  // Vertical grid
  ctx.strokeStyle = "#404060";
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= 6; i++) {
    const x = pad.left + (plotW * i) / 6;
    ctx.beginPath(); ctx.moveTo(x, pad.top); ctx.lineTo(x, pad.top + plotH); ctx.stroke();
  }
  // Horizontal grid
  for (let i = 0; i <= 5; i++) {
    const y = pad.top + (plotH * i) / 5;
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + plotW, y); ctx.stroke();
  }

  // Axes
  ctx.strokeStyle = "#606080";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, pad.top + plotH);
  ctx.lineTo(pad.left + plotW, pad.top + plotH);
  ctx.stroke();

  // Dashed zero line (U=0)
  if (yMin < 0 && yMax > 0) {
    const y0 = toY(0);
    ctx.save();
    ctx.setLineDash([6, 4]);
    ctx.strokeStyle = "#808090";
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad.left, y0); ctx.lineTo(pad.left + plotW, y0); ctx.stroke();
    ctx.restore();
  }

  // Axis labels
  ctx.fillStyle = "#a0a0c0";
  ctx.font = "11px sans-serif";
  // X ticks
  ctx.textAlign = "center";
  for (let i = 0; i <= 6; i++) {
    const val = xMin + ((xMax - xMin) * i) / 6;
    ctx.fillText(val.toFixed(2), toX(val), pad.top + plotH + 16);
  }
  // Y ticks
  ctx.textAlign = "right";
  for (let i = 0; i <= 5; i++) {
    const val = yMin + ((yMax - yMin) * i) / 5;
    ctx.fillText(formatNum(val), pad.left - 6, toY(val) + 4);
  }

  // Axis titles
  ctx.fillStyle = "#a0a0c0";
  ctx.font = "12px sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("r (distance)", pad.left + plotW / 2, H - 4);
  ctx.save();
  ctx.translate(14, pad.top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("Energy / Force", 0, 0);
  ctx.restore();

  // Draw U(r) curve
  ctx.strokeStyle = "#6c8cff";
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < rData.length; i++) {
    const x = toX(rData[i]), y = toY(uData[i]);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Draw F(r) curve (dashed red)
  if (fData) {
    ctx.save();
    ctx.setLineDash([6, 4]);
    ctx.strokeStyle = "#ff6b6b";
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < rData.length; i++) {
      const x = toX(rData[i]), y = toY(fData[i]);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.restore();
  }

  // Annotations
  if (opts.showAnnotations) {
    // Equilibrium point (green dot + vertical guide)
    if (opts.eqR >= xMin && opts.eqR <= xMax) {
      const ex = toX(opts.eqR);
      const ey = toY(opts.wellDepth);
      ctx.save();
      ctx.setLineDash([3, 3]);
      ctx.strokeStyle = "#51cf66";
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(ex, pad.top); ctx.lineTo(ex, pad.top + plotH); ctx.stroke();
      ctx.restore();
      ctx.fillStyle = "#51cf66";
      ctx.beginPath(); ctx.arc(ex, ey, 5, 0, Math.PI * 2); ctx.fill();
    }

    // Well depth (yellow horizontal line)
    if (opts.wellDepth >= yMin && opts.wellDepth <= yMax) {
      const wy = toY(opts.wellDepth);
      ctx.save();
      ctx.setLineDash([4, 3]);
      ctx.strokeStyle = "#ffd43b";
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(pad.left, wy); ctx.lineTo(pad.left + plotW, wy); ctx.stroke();
      ctx.restore();
    }

    // Zero crossing (orange dot, LJ only)
    if (opts.zeroCross != null && opts.zeroCross >= xMin && opts.zeroCross <= xMax) {
      const zx = toX(opts.zeroCross);
      const zy = toY(0);
      ctx.fillStyle = "#ff922b";
      ctx.beginPath(); ctx.arc(zx, zy, 5, 0, Math.PI * 2); ctx.fill();
    }

    // Cutoff (red dashed vertical)
    if (opts.cutoff >= xMin && opts.cutoff <= xMax) {
      const cx = toX(opts.cutoff);
      ctx.save();
      ctx.setLineDash([5, 3]);
      ctx.strokeStyle = "#ff6b6b";
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(cx, pad.top); ctx.lineTo(cx, pad.top + plotH); ctx.stroke();
      ctx.restore();
    }
  }

  // Legend
  let ly = pad.top + 12;
  const lx = pad.left + plotW + 12;
  // U(r)
  ctx.fillStyle = "#6c8cff";
  ctx.fillRect(lx, ly - 2, 16, 3);
  ctx.fillStyle = "#e0e0f0";
  ctx.textAlign = "left";
  ctx.font = "11px sans-serif";
  ctx.fillText("U(r)", lx + 20, ly + 3);
  ly += 18;
  // F(r)
  if (fData) {
    ctx.save();
    ctx.setLineDash([4, 3]);
    ctx.strokeStyle = "#ff6b6b";
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(lx, ly); ctx.lineTo(lx + 16, ly); ctx.stroke();
    ctx.restore();
    ctx.fillStyle = "#e0e0f0";
    ctx.fillText("F(r)", lx + 20, ly + 3);
    ly += 18;
  }
  if (opts.showAnnotations) {
    // Equilibrium
    ctx.fillStyle = "#51cf66";
    ctx.beginPath(); ctx.arc(lx + 8, ly, 4, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = "#e0e0f0";
    ctx.fillText("Equilibrium", lx + 20, ly + 3);
    ly += 18;
    // Well depth
    ctx.fillStyle = "#ffd43b";
    ctx.fillRect(lx, ly - 2, 16, 3);
    ctx.fillStyle = "#e0e0f0";
    ctx.fillText("Well depth", lx + 20, ly + 3);
    ly += 18;
    if (opts.zeroCross != null) {
      ctx.fillStyle = "#ff922b";
      ctx.beginPath(); ctx.arc(lx + 8, ly, 4, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = "#e0e0f0";
      ctx.fillText("Zero cross", lx + 20, ly + 3);
      ly += 18;
    }
    // Cutoff
    ctx.save();
    ctx.setLineDash([4, 3]);
    ctx.strokeStyle = "#ff6b6b";
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(lx, ly); ctx.lineTo(lx + 16, ly); ctx.stroke();
    ctx.restore();
    ctx.fillStyle = "#e0e0f0";
    ctx.fillText("Cutoff", lx + 20, ly + 3);
  }
}

// =====================================================================
//  Potential Explorer — Sub-tabs
// =====================================================================

document.querySelectorAll(".pe-subtab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".pe-subtab").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".pe-subtab-content").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    const target = btn.dataset.subtab;
    document.getElementById("pe-subtab-" + target).classList.add("active");
  });
});

// =====================================================================
//  Potential Explorer — Apply to Setup
// =====================================================================

document.getElementById("pe-apply-setup").addEventListener("click", () => {
  const type = document.getElementById("pe-type").value;
  // Sync potential type
  document.getElementById("cfg-potential-type").value = type;
  document.getElementById("cfg-potential-type").dispatchEvent(new Event("change"));
  // Sync slider values → Setup tab inputs
  Object.keys(PE_SYNC_MAP).forEach((sliderId) => {
    const slider = document.getElementById(sliderId);
    const cfgInput = document.getElementById(PE_SYNC_MAP[sliderId]);
    if (slider && cfgInput) cfgInput.value = slider.value;
  });
  // Visual feedback
  const btn = document.getElementById("pe-apply-setup");
  const origText = btn.textContent;
  btn.textContent = "Applied!";
  btn.classList.add("applied");
  setTimeout(() => {
    btn.textContent = origText;
    btn.classList.remove("applied");
  }, 1500);
});

// =====================================================================
//  Potential Explorer — Educational Content
// =====================================================================

const PE_INTRO = {
  lj: `
<h3>History</h3>
<p>The Lennard-Jones potential was proposed by <strong>John Lennard-Jones</strong> in 1924 to model
interactions between noble gas atoms. It became one of the most widely used pair potentials
in molecular simulation due to its simplicity and reasonable accuracy for non-bonded interactions.</p>

<h3>Formula</h3>
<div class="pe-formula">U(r) = 4\u03B5 [(\u03C3/r)\u00B9\u00B2 \u2212 (\u03C3/r)\u2076]</div>
<p>The potential consists of two terms:</p>
<ul>
  <li>The <strong>r\u207B\u00B9\u00B2 term</strong> (repulsive): dominates at short range, modeling Pauli repulsion
  when electron clouds overlap.</li>
  <li>The <strong>r\u207B\u2076 term</strong> (attractive): captures London dispersion forces (van der Waals),
  arising from fluctuating induced dipoles.</li>
</ul>

<h3>Parameter Meanings</h3>
<ul>
  <li><strong>\u03B5 (epsilon)</strong> — Well depth: the strength of the interaction. Equals the
  magnitude of the energy minimum. Larger \u03B5 means stronger attraction.</li>
  <li><strong>\u03C3 (sigma)</strong> — Effective atomic diameter: the distance at which U(r) = 0.
  Atoms closer than \u03C3 repel; atoms farther attract.</li>
  <li><strong>r_min = 2\u00B9\u02B8\u2076 \u03C3 \u2248 1.122\u03C3</strong> — Equilibrium distance where
  the energy is minimized at U = \u2212\u03B5.</li>
</ul>

<h3>Physical Basis</h3>
<p>The attractive r\u207B\u2076 dependence has a rigorous quantum-mechanical origin in second-order
perturbation theory (London dispersion). The repulsive r\u207B\u00B9\u00B2 exponent, however, is
<em>empirical</em> — chosen for computational convenience (being the square of r\u207B\u2076) rather
than derived from first principles. Exponential repulsion (Born-Mayer) is more physically accurate
but more expensive.</p>

<h3>Applicability</h3>
<ul>
  <li>Noble gases: Ar, Ne, Kr, Xe (excellent agreement with experimental data)</li>
  <li>Simple fluids and Lennard-Jones fluids as model systems</li>
  <li>Coarse-grained models of hydrophobic interactions</li>
  <li>Non-bonded interactions in biomolecular force fields (combined with Coulomb)</li>
</ul>

<h3>Limitations</h3>
<ul>
  <li>Not suitable for covalent/metallic/ionic bonding</li>
  <li>The r\u207B\u00B9\u00B2 repulsive wall is too steep compared to quantum-mechanical calculations</li>
  <li>Single-species only (no cross-interactions without mixing rules)</li>
  <li>Cannot model bond formation or breaking</li>
  <li>Poor for anisotropic interactions (e.g., water, liquid crystals)</li>
</ul>`,

  morse: `
<h3>History</h3>
<p>The Morse potential was proposed by <strong>Philip M. Morse</strong> in 1929 as an analytically
solvable model for the vibrations of diatomic molecules. Unlike the harmonic oscillator, it
correctly captures bond dissociation and the anharmonicity of real molecular bonds.</p>

<h3>Formula</h3>
<div class="pe-formula">U(r) = D [1 \u2212 e\u207B\u1D43\u207D\u02B3\u207B\u02B3\u2080\u207E]\u00B2</div>
<p>The potential describes an asymmetric well:</p>
<ul>
  <li>At <strong>r = r\u2080</strong>: U = 0 (equilibrium, minimum energy)</li>
  <li>As <strong>r \u2192 \u221E</strong>: U \u2192 D (dissociation limit — bond breaks)</li>
  <li>As <strong>r \u2192 0</strong>: U grows steeply (strong repulsion)</li>
</ul>

<h3>Parameter Meanings</h3>
<ul>
  <li><strong>D</strong> — Dissociation energy (well depth): the energy required to completely
  separate two bonded atoms. Larger D means a stronger bond.</li>
  <li><strong>a</strong> — Width parameter: controls the steepness of the potential well.
  Larger a means a narrower, stiffer well (higher vibrational frequency).</li>
  <li><strong>r\u2080</strong> — Equilibrium bond length: the distance at which the energy is
  minimized (U = 0).</li>
</ul>

<h3>Physical Basis</h3>
<p>The Morse potential is an <em>exact</em> solution of the Schr\u00F6dinger equation for a
diatomic molecule — it yields analytical expressions for energy levels:
E_n = \u0127\u03C9(n + \u00BD) \u2212 [\u0127\u03C9(n + \u00BD)]\u00B2 / 4D.
The anharmonic correction (second term) naturally arises, predicting the convergence
of vibrational levels near the dissociation limit.</p>

<h3>Applicability</h3>
<ul>
  <li>Diatomic molecules (H\u2082, O\u2082, N\u2082, HCl, etc.)</li>
  <li>Covalent bonds in polyatomic molecules</li>
  <li>Metallic bonding in some embedded-atom model variants</li>
  <li>Near-equilibrium vibrations of any bond (better than harmonic)</li>
</ul>

<h3>Limitations</h3>
<ul>
  <li>Does not capture long-range van der Waals (r\u207B\u2076) attraction</li>
  <li>Not appropriate for non-bonded interactions between noble gas atoms</li>
  <li>Overestimates repulsion at very short distances (r \u226A r\u2080)</li>
  <li>Single bond only — does not describe three-body or angular effects</li>
  <li>Parameters (D, a, r\u2080) must be fitted for each specific bond type</li>
</ul>`,
};

function _highlightPython(code) {
  // Escape HTML
  let s = code.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  // Order matters: comments first (greedy), then strings, then keywords
  // Multi-line strings (triple-quoted)
  s = s.replace(/("""[\s\S]*?"""|'''[\s\S]*?''')/g, '<span class="st">$1</span>');
  // Single-line comments
  s = s.replace(/(#[^\n]*)/g, '<span class="cm">$1</span>');
  // Strings (double and single quoted, not already inside a span)
  s = s.replace(/(?<!<span class="cm">.*?)("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')/g, function(m) {
    if (m.includes('class=')) return m;
    return '<span class="st">' + m + '</span>';
  });
  // Decorators
  s = s.replace(/(@\w+)/g, '<span class="dec">$1</span>');
  // Keywords
  const kws = '\\b(class|def|return|if|else|elif|for|in|range|import|from|as|not|and|or|is|None|True|False|raise|with|try|except|finally|yield|lambda|pass|break|continue|while|assert|del|global|nonlocal)\\b';
  s = s.replace(new RegExp(kws, 'g'), '<span class="kw">$1</span>');
  // self
  s = s.replace(/\bself\b/g, '<span class="self">self</span>');
  // Built-in functions/types
  s = s.replace(/\b(len|float|int|str|print|isinstance|type|super|property|staticmethod|classmethod|abs|min|max|sum|map|filter|zip|enumerate|sorted|reversed|list|dict|set|tuple|ValueError|TypeError|Optional)\b/g, '<span class="nb">$1</span>');
  // Numbers (integers and floats, not inside words)
  s = s.replace(/(?<![.\w])(\d+\.?\d*(?:[eE][+-]?\d+)?)\b/g, '<span class="num">$1</span>');
  // Operators
  s = s.replace(/(\*\*|->|!=|==|&lt;=|&gt;=|&lt;|&gt;)/g, '<span class="op">$1</span>');
  return s;
}

const PE_CODE = {
  lj: `# Lennard-Jones 12-6 potential
# U(r) = 4\u03B5[(\u03C3/r)\u00B9\u00B2 - (\u03C3/r)\u2076]
# r_min = 2^(1/6) * \u03C3,  U_min = -\u03B5

class LennardJonesPotential(PotentialEnergy):

    def __init__(self, epsilon, sigma, cutoff):
        self.epsilon = epsilon
        self.sigma = sigma
        self._cutoff = cutoff

    def compute_pair_energy(self, r):
        """U(r) for a single pair."""
        sr = self.sigma / r
        sr6 = sr ** 6
        sr12 = sr6 ** 2
        return 4.0 * self.epsilon * (sr12 - sr6)

    def compute_pair_force(self, r):
        """F(r) = -dU/dr = 24\u03B5/r [2(\u03C3/r)\u00B9\u00B2 - (\u03C3/r)\u2076]"""
        sr = self.sigma / r
        sr6 = sr ** 6
        sr12 = sr6 ** 2
        return 24.0 * self.epsilon / r * (2.0 * sr12 - sr6)

    def compute_energy(self, positions, box, boundary_condition):
        """Total energy — forces computed via autodiff!"""
        energy = 0.0
        n = len(positions)
        for i in range(n):
            for j in range(i + 1, n):
                dr = positions[j] - positions[i]
                dr = boundary_condition.apply_minimum_image(dr, box)
                r = np.linalg.norm(dr)
                if 0 < r < self._cutoff:
                    energy += self.compute_pair_energy(r)
        return energy`,

  morse: `# Morse potential for molecular bonds
# U(r) = D [1 - exp(-a(r - r\u2080))]\u00B2
# U(r\u2080) = 0,  U(\u221E) = D

class MorsePotential(PotentialEnergy):

    def __init__(self, D, a, r0, cutoff):
        self.D = D
        self.a = a
        self.r0 = r0
        self._cutoff = cutoff

    def compute_pair_energy(self, r):
        """U(r) for a single pair."""
        exp_term = np.exp(-self.a * (r - self.r0))
        return self.D * (1 - exp_term) ** 2

    def compute_energy(self, positions, box, boundary_condition):
        """Total energy — forces computed via autodiff!"""
        energy = 0.0
        n = len(positions)
        for i in range(n):
            for j in range(i + 1, n):
                dr = positions[j] - positions[i]
                dr = boundary_condition.apply_minimum_image(dr, box)
                r = np.linalg.norm(dr)
                if 0 < r < self._cutoff:
                    energy += self.compute_pair_energy(r)
        return energy`,
};

function peUpdateEducational() {
  const type = document.getElementById("pe-type").value;
  document.getElementById("pe-intro-content").innerHTML = PE_INTRO[type] || "";
  document.getElementById("pe-code-content").innerHTML = _highlightPython(PE_CODE[type] || "");
}

// Initialize educational content on first load
peUpdateEducational();
