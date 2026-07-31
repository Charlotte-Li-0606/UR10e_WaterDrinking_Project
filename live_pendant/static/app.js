"use strict";

const byId = (id) => document.getElementById(id);
const POLL_MS = 500;
let lastSuccessfulPoll = 0;
let cameraSequence = 0;
let cameraWasAvailable = false;

function text(id, value) {
  byId(id).textContent = value ?? "—";
}

function fixed(value, digits = 1) {
  return Number.isFinite(value) ? Number(value).toFixed(digits) : "—";
}

function setDot(id, tone) {
  byId(id).className = `tile-dot ${tone || ""}`.trim();
}

function renderConnection(data) {
  const pill = byId("connection-pill");
  const state = data.connection_state || "offline";
  pill.className = `connection-pill ${state}`;
  text("connection-label", state === "live" ? "Live" : state === "partial" ? "Partial" : "Offline");
  const stamp = new Date(data.generated_at);
  text("last-update", Number.isNaN(stamp.valueOf()) ? "No telemetry" : `Updated ${stamp.toLocaleTimeString([], { hour12: false })}`);
  text("data-source", `${data.source === "demo" ? "Demo" : "ROS 2"} telemetry · ${state}`);
}

function renderRobot(robot = {}) {
  text("robot-mode", robot.mode);
  text("safety-mode", robot.safety_mode);
  const programState = robot.program_state || (robot.program_running === true ? "PLAYING" : robot.program_running === false ? "STOPPED" : null);
  text("program-state", programState);
  text("program-name", robot.program_name || "No program name");
  text("remote-control", robot.remote_control === true ? "REMOTE" : robot.remote_control === false ? "LOCAL" : null);

  setDot("robot-mode-dot", robot.mode === "RUNNING" ? "good" : robot.mode ? "warn" : "");
  setDot("safety-mode-dot", robot.safety_mode === "NORMAL" ? "good" : robot.safety_mode ? (robot.safety_mode === "REDUCED" ? "warn" : "bad") : "");
  setDot("program-dot", programState === "PLAYING" ? "good" : programState === "PAUSED" ? "warn" : programState ? "bad" : "");
  setDot("remote-dot", robot.remote_control === true ? "good" : robot.remote_control === false ? "warn" : "");

  const scaling = robot.speed_scaling == null ? Number.NaN : Number(robot.speed_scaling);
  const percent = Number.isFinite(scaling) ? Math.max(0, Math.min(100, scaling * 100)) : 0;
  text("speed-value", Number.isFinite(scaling) ? `${percent.toFixed(percent < 10 ? 1 : 0)}%` : "—%");
  byId("speed-bar").style.width = `${percent}%`;
  byId("speed-marker").style.left = `calc(${percent}% - 1px)`;
}

function renderTcp(tcp) {
  const position = tcp?.position_mm || [];
  const rotation = tcp?.rotation_vector_rad || [];
  ["x", "y", "z"].forEach((axis, index) => text(`tcp-${axis}`, fixed(position[index], 1)));
  ["rx", "ry", "rz"].forEach((axis, index) => text(`tcp-${axis}`, fixed(rotation[index], 3)));
  text("tcp-frame", (tcp?.frame || "base").toUpperCase());
}

function prettyJointName(name) {
  return name.replace(/_joint$/, "").replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function renderJoints(joints = []) {
  const target = byId("joint-list");
  if (!joints.length) {
    target.innerHTML = '<div class="empty-row">Waiting for joint states…</div>';
    return;
  }
  target.replaceChildren(...joints.map((joint, index) => {
    const row = document.createElement("div");
    row.className = "joint-row";
    const velocity = Number.isFinite(joint.velocity_deg_s) ? `${Math.abs(joint.velocity_deg_s).toFixed(2)}°/s` : "—";
    row.innerHTML = `<span class="joint-index">J${index + 1}</span><span class="joint-name"></span><strong class="joint-value">${fixed(joint.position_deg, 2)}°</strong><span class="joint-velocity">${velocity}</span>`;
    row.querySelector(".joint-name").textContent = prettyJointName(joint.name);
    return row;
  }));
}

function renderBits(id, countId, bits = []) {
  const target = byId(id);
  if (!bits.length) {
    target.innerHTML = '<span class="muted">No data</span>';
    text(countId, "0 active");
    return;
  }
  const sorted = [...bits].sort((a, b) => a.pin - b.pin);
  target.replaceChildren(...sorted.map((bit) => {
    const item = document.createElement("span");
    item.className = `bit ${bit.state ? "on" : ""}`;
    item.textContent = bit.pin;
    item.title = `Pin ${bit.pin}: ${bit.state ? "high" : "low"}`;
    return item;
  }));
  text(countId, `${sorted.filter((bit) => bit.state).length} active`);
}

function renderIo(io, tool) {
  renderBits("digital-inputs", "input-count", io?.digital_inputs);
  renderBits("digital-outputs", "output-count", io?.digital_outputs);
  text("tool-voltage", Number.isFinite(tool?.voltage_v) ? `${fixed(tool.voltage_v, 1)} V` : null);
  text("tool-current", Number.isFinite(tool?.current_a) ? `${fixed(tool.current_a, 3)} A` : null);
  text("tool-temp", Number.isFinite(tool?.temperature_c) ? `${fixed(tool.temperature_c, 1)} °C` : null);
}

function renderControllers(controllers = []) {
  const target = byId("controller-list");
  const active = controllers.filter((controller) => controller.state === "active").length;
  text("controller-count", `${active} active`);
  if (!controllers.length) {
    target.innerHTML = '<div class="empty-row">Waiting for controller manager…</div>';
    return;
  }
  const priority = (controller) => controller.state === "active" ? 0 : 1;
  const sorted = [...controllers].sort((a, b) => priority(a) - priority(b) || a.name.localeCompare(b.name));
  target.replaceChildren(...sorted.map((controller) => {
    const row = document.createElement("div");
    row.className = "controller-row";
    const dot = document.createElement("span");
    dot.className = `controller-state ${controller.state}`;
    const name = document.createElement("span");
    name.className = "controller-name";
    name.textContent = controller.name;
    const state = document.createElement("small");
    state.textContent = controller.state;
    row.append(dot, name, state);
    return row;
  }));
}

function renderCamera(camera = {}, streams = {}) {
  const available = Boolean(camera.available && streams.camera?.fresh);
  byId("camera-dot").className = `mini-dot ${available ? "live" : ""}`;
  text("camera-meta", available ? `${streams.camera.age_seconds.toFixed(1)}s ago` : "Waiting for frames");
  text("camera-resolution", camera.width && camera.height ? `${camera.width} × ${camera.height}` : "—");
  if (available && (!cameraWasAvailable || cameraSequence % 2 === 0)) {
    const image = byId("camera-feed");
    image.onload = () => byId("camera-stage").classList.add("has-image");
    image.onerror = () => byId("camera-stage").classList.remove("has-image");
    image.src = `/api/camera.jpg?t=${Date.now()}`;
  }
  if (!available) byId("camera-stage").classList.remove("has-image");
  cameraWasAvailable = available;
  cameraSequence += 1;
}

function render(data) {
  renderConnection(data);
  renderRobot(data.robot);
  renderTcp(data.tcp);
  renderJoints(data.joints);
  renderIo(data.io, data.tool);
  renderControllers(data.controllers);
  renderCamera(data.camera, data.streams);
}

async function poll() {
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    lastSuccessfulPoll = Date.now();
    render(data);
  } catch (_error) {
    if (Date.now() - lastSuccessfulPoll > POLL_MS * 3) {
      renderConnection({ connection_state: "offline", generated_at: null, source: "ros2" });
      text("last-update", "Pendant server unavailable");
    }
  }
}

function updateClock() {
  const now = new Date();
  text("clock-time", now.toLocaleTimeString([], { hour12: false }));
  text("clock-date", now.toLocaleDateString([], { weekday: "short", month: "short", day: "numeric" }));
}

updateClock();
poll();
setInterval(updateClock, 1000);
setInterval(poll, POLL_MS);
