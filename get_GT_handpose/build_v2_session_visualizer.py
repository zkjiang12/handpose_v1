#!/usr/bin/env python3
"""Build a static synchronized v2 video/audio/IMU visualizer."""

from __future__ import annotations

import json
import math
import wave
from pathlib import Path

import numpy as np


OUT_ROOT = Path("/Users/zikangjiang/dev/ego-exo/visualizations/v2")
VIS_ROOT = OUT_ROOT / "session_visualizer"
EGO_EXO_ROOT = Path("/Users/zikangjiang/dev/ego-exo")

SEGMENTS = [
    ("segment1", "take1"),
    ("segment2", "take2"),
]

CAMERA_LABELS = {
    "cam1": "Camera 1",
    "cam2": "Camera 2",
    "cam3": "Camera 3",
    "cam4": "Camera 4",
}


def load_report(report_folder: str) -> dict:
    path = OUT_ROOT / report_folder / "v2_calibration_and_audio_sync_report.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing report: {path}")
    return json.loads(path.read_text())


def read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, sample_rate


def downsample_audio(path: Path, bin_s: float = 0.01) -> dict:
    audio, sample_rate = read_wav_mono(path)
    bin_n = max(1, int(round(bin_s * sample_rate)))
    usable = (len(audio) // bin_n) * bin_n
    if usable == 0:
        return {"dt": bin_s, "rms": [], "onset": []}
    chunks = audio[:usable].reshape(-1, bin_n)
    rms = np.sqrt(np.mean(chunks * chunks, axis=1) + 1e-12)
    peak = np.max(np.abs(chunks), axis=1)
    energy = np.maximum(rms, peak * 0.35)
    log_energy = np.log1p(80.0 * energy)
    onset = np.maximum(np.diff(log_energy, prepend=log_energy[0]), 0.0)

    rms_scale = np.percentile(energy, 99.7) or 1.0
    onset_scale = np.percentile(onset, 99.7) or 1.0
    energy_n = np.clip(energy / rms_scale, 0.0, 1.5)
    onset_n = np.clip(onset / onset_scale, 0.0, 1.5)

    return {
        "dt": bin_s,
        "rms": [round(float(x), 4) for x in energy_n],
        "onset": [round(float(x), 4) for x in onset_n],
    }


def load_imu(path: Path, first_t_us: int) -> list[list[float]]:
    rows: list[list[float]] = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            t = (int(item["t_us"]) - first_t_us) / 1_000_000.0
            acc = item.get("acc", [0.0, 0.0, 0.0])
            gyro = item.get("gyro", [0.0, 0.0, 0.0])
            rows.append(
                [
                    round(t, 4),
                    round(float(acc[0]), 4),
                    round(float(acc[1]), 4),
                    round(float(acc[2]), 4),
                    round(float(gyro[0]), 5),
                    round(float(gyro[1]), 5),
                    round(float(gyro[2]), 5),
                ]
            )
    return rows


def build_session_data() -> dict:
    reports = {segment: load_report(folder) for segment, folder in SEGMENTS}
    first_ref_us = reports["segment1"]["extrinsics"]["cameras"]["cam1"]["imu_timestamps"]["first_t_us"]

    tracks = {}
    for cam in CAMERA_LABELS:
        tracks[cam] = {
            "id": cam,
            "label": CAMERA_LABELS[cam],
            "segments": [],
        }

    for segment_name, report_folder in SEGMENTS:
        report = reports[segment_name]
        ref_start_us = report["extrinsics"]["cameras"]["cam1"]["imu_timestamps"]["first_t_us"]
        ref_session_start_s = (ref_start_us - first_ref_us) / 1_000_000.0
        for cam, cam_report in report["extrinsics"]["cameras"].items():
            audio_report = report["audio_sync"]["cameras"][cam]
            imu_first = cam_report["imu_timestamps"]["first_t_us"]
            imu_path = Path(cam_report["imu"])
            wav_path = Path(audio_report["wav"])
            segment = {
                "id": segment_name,
                "sourceReportFolder": report_folder,
                "videoPath": cam_report["video"],
                "videoUrl": "/" + Path(cam_report["video"]).relative_to(EGO_EXO_ROOT).as_posix(),
                "durationS": round(float(cam_report["video_duration_s"]), 4),
                "deviceStartS": round((imu_first - first_ref_us) / 1_000_000.0, 4),
                "referenceStartS": round(ref_session_start_s, 4),
                "audioOffsetS": round(float(audio_report["sync_to_reference"]["offset_s"]), 4),
                "audioSyncConfidence": audio_report["sync_to_reference"]["confidence"],
                "audioCorrelationScore": round(float(audio_report["sync_to_reference"]["correlation_score"]), 4),
                "reprojectionMedianPx": round(float(cam_report["reprojection_median_px"]), 4),
                "cameraCenterWorldMm": [round(float(x), 3) for x in cam_report["camera_center_world_mm"]],
                "audio": downsample_audio(wav_path),
                "imu": load_imu(imu_path, imu_first),
            }
            tracks[cam]["segments"].append(segment)

    all_segments = [seg for track in tracks.values() for seg in track["segments"]]
    session_end_device = max(seg["deviceStartS"] + seg["durationS"] for seg in all_segments)
    session_end_audio = max(seg["referenceStartS"] - seg["audioOffsetS"] + seg["durationS"] for seg in all_segments)

    return {
        "schema": "ego_exo_v2_session_visualizer_v1",
        "createdFrom": str(OUT_ROOT),
        "referenceCamera": "cam1",
        "session": {
            "label": "v2 single recording session",
            "note": "The folders named take1/take2 in earlier generated outputs are file segments from one recording session.",
            "durationDeviceS": round(float(session_end_device), 4),
            "durationAudioS": round(float(session_end_audio), 4),
        },
        "syncModes": {
            "device": "Uses IMU/device file-start timestamps.",
            "audio": "Uses clap/onset cross-correlation offsets per file segment, with cam1 as reference.",
        },
        "tracks": tracks,
    }


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Ego-Exo v2 Sync Visualizer</title>
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <header class="topbar">
    <div>
      <h1>Ego-Exo v2 Sync Visualizer</h1>
      <p id="statusLine">Loading session...</p>
    </div>
    <div class="toolbar">
      <button id="playBtn" class="primary">Play</button>
      <button id="pauseBtn">Pause</button>
      <button id="frameBackBtn" title="Back one frame">-1f</button>
      <button id="frameForwardBtn" title="Forward one frame">+1f</button>
      <label>Rate
        <select id="rateSelect">
          <option value="0.25">0.25x</option>
          <option value="0.5">0.5x</option>
          <option value="1" selected>1x</option>
          <option value="2">2x</option>
        </select>
      </label>
      <label>Sync
        <select id="syncModeSelect">
          <option value="device">Device/IMU</option>
          <option value="audio" selected>Audio</option>
        </select>
      </label>
    </div>
  </header>

  <main>
    <section class="timebar">
      <div class="timeReadout">
        <span id="timeText">0.000s</span>
        <span id="segmentText"></span>
      </div>
      <input id="timeSlider" type="range" min="0" max="1" step="0.001" value="0">
    </section>

    <section id="videoGrid" class="videoGrid"></section>

    <section class="panel timelinePanel">
      <div class="panelHeader">
        <h2>Audio + IMU Timeline</h2>
        <div class="legend">
          <span class="audioMark"></span>audio
          <span class="onsetMark"></span>onset
          <span class="accMark"></span>accel mag
          <span class="gyroMark"></span>gyro mag
        </div>
      </div>
      <canvas id="timelineCanvas" width="1600" height="720"></canvas>
    </section>

    <section class="panel">
      <div class="panelHeader">
        <h2>Offsets</h2>
        <button id="resetOffsetsBtn">Reset manual offsets</button>
      </div>
      <div id="offsetTable" class="offsetTable"></div>
    </section>
  </main>

  <script src="data/session_data.js"></script>
  <script src="app.js"></script>
</body>
</html>
"""


STYLES_CSS = r""":root {
  color-scheme: light;
  --bg: #f6f7f8;
  --panel: #ffffff;
  --ink: #17191c;
  --muted: #626a73;
  --line: #d6dbe1;
  --accent: #0f766e;
  --accent-strong: #0b5f59;
  --danger: #b91c1c;
  --cam1: #d83b33;
  --cam2: #2374ab;
  --cam3: #2a9d55;
  --cam4: #7c4dbe;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  letter-spacing: 0;
}

.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 20px;
  padding: 16px 20px;
  background: #fff;
  border-bottom: 1px solid var(--line);
  position: sticky;
  top: 0;
  z-index: 10;
}

h1,
h2,
p {
  margin: 0;
}

h1 {
  font-size: 20px;
  line-height: 1.2;
}

h2 {
  font-size: 15px;
}

#statusLine {
  margin-top: 4px;
  color: var(--muted);
  font-size: 13px;
}

.toolbar {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
}

button,
select,
input[type="number"] {
  min-height: 34px;
  border: 1px solid var(--line);
  background: #fff;
  color: var(--ink);
  border-radius: 6px;
  padding: 6px 10px;
  font: inherit;
}

button {
  cursor: pointer;
}

button.primary {
  background: var(--accent);
  border-color: var(--accent);
  color: #fff;
}

button.primary:hover {
  background: var(--accent-strong);
}

label {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: var(--muted);
  font-size: 13px;
}

main {
  padding: 16px;
  display: grid;
  gap: 14px;
}

.timebar,
.panel,
.cameraCard {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
}

.timebar {
  padding: 12px 14px;
}

.timeReadout {
  display: flex;
  align-items: center;
  justify-content: space-between;
  color: var(--muted);
  font-size: 13px;
  margin-bottom: 8px;
}

#timeText {
  color: var(--ink);
  font-variant-numeric: tabular-nums;
  font-weight: 650;
}

#timeSlider {
  width: 100%;
}

.videoGrid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
}

.cameraCard {
  overflow: hidden;
  min-width: 0;
}

.cameraHeader {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  padding: 10px 12px;
  border-bottom: 1px solid var(--line);
}

.cameraTitle {
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 700;
}

.cameraDot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: var(--cam1);
}

.cameraMeta {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
  color: var(--muted);
  font-size: 12px;
  font-variant-numeric: tabular-nums;
}

.videoWrap {
  position: relative;
  aspect-ratio: 16 / 9;
  background: #111;
}

video {
  width: 100%;
  height: 100%;
  display: block;
  object-fit: contain;
  background: #111;
}

.inactiveOverlay {
  display: none;
  position: absolute;
  inset: 0;
  align-items: center;
  justify-content: center;
  color: #fff;
  background: rgba(0, 0, 0, 0.72);
  font-weight: 650;
}

.cameraCard.inactive .inactiveOverlay {
  display: flex;
}

.cameraControls {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, auto));
  gap: 8px;
  padding: 10px 12px 12px;
  align-items: center;
}

.cameraControls input[type="range"] {
  width: 100%;
  min-width: 90px;
}

.manualOffset {
  width: 86px;
  font-variant-numeric: tabular-nums;
}

.panel {
  padding: 12px;
}

.panelHeader {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 10px;
  margin-bottom: 10px;
}

.legend {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
  color: var(--muted);
  font-size: 12px;
}

.legend span {
  display: inline-block;
  width: 18px;
  height: 3px;
  border-radius: 999px;
}

.audioMark {
  background: #8b949e;
}

.onsetMark {
  background: #111827;
}

.accMark {
  background: #16a34a;
}

.gyroMark {
  background: #8b5cf6;
}

.timelinePanel {
  padding-bottom: 10px;
}

#timelineCanvas {
  width: 100%;
  height: 520px;
  display: block;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #fff;
  cursor: crosshair;
}

.offsetTable {
  display: grid;
  grid-template-columns: 110px repeat(2, minmax(150px, 1fr)) 130px 130px;
  gap: 1px;
  overflow: hidden;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: var(--line);
  font-size: 13px;
}

.offsetTable > div {
  background: #fff;
  padding: 8px 10px;
  min-height: 36px;
}

.offsetHead {
  color: var(--muted);
  font-weight: 700;
}

.number {
  font-variant-numeric: tabular-nums;
}

@media (max-width: 950px) {
  .topbar {
    position: static;
    align-items: stretch;
    flex-direction: column;
  }

  .videoGrid {
    grid-template-columns: 1fr;
  }

  .cameraControls {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .offsetTable {
    grid-template-columns: 100px repeat(2, minmax(120px, 1fr));
  }

  .offsetTable .optional {
    display: none;
  }
}
"""


APP_JS = r"""const DATA = window.EGO_EXO_SESSION_DATA;

const COLORS = {
  cam1: getCss("--cam1"),
  cam2: getCss("--cam2"),
  cam3: getCss("--cam3"),
  cam4: getCss("--cam4"),
};

const state = {
  sessionTime: 0,
  playing: false,
  playbackRate: 1,
  syncMode: "audio",
  wallStartS: 0,
  sessionStartS: 0,
  manualOffsets: {},
  trackEls: {},
  muted: {},
};

function getCss(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function fmtS(value, digits = 3) {
  if (!Number.isFinite(value)) return "-";
  return `${value.toFixed(digits)}s`;
}

function sessionDuration() {
  return state.syncMode === "device"
    ? DATA.session.durationDeviceS
    : DATA.session.durationAudioS;
}

function localTimeForSegment(segment, sessionTime, mode = state.syncMode) {
  const manual = state.manualOffsets[segment.cameraId] || 0;
  if (mode === "device") {
    return sessionTime - segment.deviceStartS + manual;
  }
  return sessionTime - segment.referenceStartS + segment.audioOffsetS + manual;
}

function globalTimeForLocal(segment, localT, mode = state.syncMode) {
  const manual = state.manualOffsets[segment.cameraId] || 0;
  if (mode === "device") {
    return segment.deviceStartS + localT - manual;
  }
  return segment.referenceStartS + localT - segment.audioOffsetS - manual;
}

function activeSegment(track, sessionTime) {
  for (const segment of track.segments) {
    const localT = localTimeForSegment(segment, sessionTime);
    if (localT >= 0 && localT <= segment.durationS) {
      return { segment, localT };
    }
  }
  return { segment: null, localT: NaN };
}

function nearestImu(segment, localT) {
  const imu = segment?.imu || [];
  if (!imu.length || !Number.isFinite(localT)) return null;
  let lo = 0;
  let hi = imu.length - 1;
  while (lo < hi) {
    const mid = Math.floor((lo + hi) / 2);
    if (imu[mid][0] < localT) lo = mid + 1;
    else hi = mid;
  }
  return imu[Math.max(0, Math.min(imu.length - 1, lo))];
}

function createVideoGrid() {
  const grid = document.getElementById("videoGrid");
  grid.innerHTML = "";
  for (const [camId, track] of Object.entries(DATA.tracks)) {
    state.manualOffsets[camId] = 0;
    state.muted[camId] = true;

    const card = document.createElement("article");
    card.className = "cameraCard";
    card.dataset.cam = camId;

    const dot = document.createElement("span");
    dot.className = "cameraDot";
    dot.style.background = COLORS[camId];

    const header = document.createElement("div");
    header.className = "cameraHeader";
    header.innerHTML = `
      <div class="cameraTitle"></div>
      <div class="cameraMeta">
        <span class="segmentLabel">-</span>
        <span class="localTime">-</span>
      </div>
    `;
    header.querySelector(".cameraTitle").append(dot, document.createTextNode(track.label));

    const wrap = document.createElement("div");
    wrap.className = "videoWrap";
    const video = document.createElement("video");
    video.playsInline = true;
    video.preload = "metadata";
    video.muted = true;
    video.volume = 0.65;
    const inactive = document.createElement("div");
    inactive.className = "inactiveOverlay";
    inactive.textContent = "No segment at this time";
    wrap.append(video, inactive);

    const controls = document.createElement("div");
    controls.className = "cameraControls";
    controls.innerHTML = `
      <label><input class="muteToggle" type="checkbox"> Audio</label>
      <label>Volume <input class="volumeSlider" type="range" min="0" max="1" step="0.01" value="0.65"></label>
      <label>Offset <input class="manualOffset" type="number" step="0.005" value="0"></label>
      <span class="imuReadout number">IMU -</span>
    `;

    card.append(header, wrap, controls);
    grid.append(card);

    controls.querySelector(".muteToggle").addEventListener("change", (event) => {
      const enabled = event.target.checked;
      state.muted[camId] = !enabled;
      video.muted = !enabled;
    });
    controls.querySelector(".volumeSlider").addEventListener("input", (event) => {
      video.volume = Number(event.target.value);
    });
    controls.querySelector(".manualOffset").addEventListener("change", (event) => {
      state.manualOffsets[camId] = Number(event.target.value) || 0;
      syncAll(true);
      drawTimeline();
      updateOffsetTable();
    });

    state.trackEls[camId] = {
      card,
      video,
      segmentLabel: header.querySelector(".segmentLabel"),
      localTime: header.querySelector(".localTime"),
      imuReadout: controls.querySelector(".imuReadout"),
    };
  }
}

function buildOffsetTable() {
  const table = document.getElementById("offsetTable");
  table.innerHTML = `
    <div class="offsetHead">Camera</div>
    <div class="offsetHead">Segment 1 audio offset</div>
    <div class="offsetHead">Segment 2 audio offset</div>
    <div class="offsetHead optional">Confidence</div>
    <div class="offsetHead optional">Manual</div>
  `;
  for (const [camId, track] of Object.entries(DATA.tracks)) {
    const seg1 = track.segments.find((s) => s.id === "segment1");
    const seg2 = track.segments.find((s) => s.id === "segment2");
    const manual = document.createElement("div");
    manual.className = "optional number";
    manual.dataset.manualFor = camId;
    table.insertAdjacentHTML("beforeend", `
      <div>${track.label}</div>
      <div class="number">${fmtS(seg1.audioOffsetS, 4)}</div>
      <div class="number">${fmtS(seg2.audioOffsetS, 4)}</div>
      <div class="optional">${seg1.audioSyncConfidence} / ${seg2.audioSyncConfidence}</div>
    `);
    table.append(manual);
  }
  updateOffsetTable();
}

function updateOffsetTable() {
  for (const el of document.querySelectorAll("[data-manual-for]")) {
    const cam = el.dataset.manualFor;
    el.textContent = fmtS(state.manualOffsets[cam] || 0, 4);
  }
}

function setSessionTime(timeS, force = true) {
  const maxT = sessionDuration();
  state.sessionTime = Math.max(0, Math.min(maxT, timeS));
  document.getElementById("timeSlider").value = String(state.sessionTime);
  syncAll(force);
  drawTimeline();
}

function syncAll(force = false) {
  const segmentsNow = [];
  for (const [camId, track] of Object.entries(DATA.tracks)) {
    const els = state.trackEls[camId];
    const { segment, localT } = activeSegment(track, state.sessionTime);
    if (!segment) {
      els.card.classList.add("inactive");
      els.segmentLabel.textContent = "-";
      els.localTime.textContent = "-";
      els.imuReadout.textContent = "IMU -";
      els.video.pause();
      continue;
    }

    els.card.classList.remove("inactive");
    if (els.video.dataset.src !== segment.videoUrl) {
      els.video.src = segment.videoUrl;
      els.video.dataset.src = segment.videoUrl;
      els.video.load();
      force = true;
    }
    els.video.playbackRate = state.playbackRate;
    els.video.muted = state.muted[camId];
    const drift = Math.abs(els.video.currentTime - localT);
    if (force || drift > 0.08 || els.video.paused !== !state.playing) {
      try {
        els.video.currentTime = Math.max(0, Math.min(segment.durationS - 0.001, localT));
      } catch (_) {
        // Metadata might not be ready yet; the next sync tick will retry.
      }
    }
    if (state.playing && els.video.paused) {
      els.video.play().catch(() => {
        state.playing = false;
        updatePlayButtons();
      });
    } else if (!state.playing && !els.video.paused) {
      els.video.pause();
    }

    const imu = nearestImu(segment, localT);
    const accMag = imu ? Math.hypot(imu[1], imu[2], imu[3]) : NaN;
    const gyroMag = imu ? Math.hypot(imu[4], imu[5], imu[6]) : NaN;
    els.segmentLabel.textContent = segment.id;
    els.localTime.textContent = `local ${fmtS(localT)}`;
    els.imuReadout.textContent = Number.isFinite(accMag)
      ? `acc ${accMag.toFixed(2)} gyro ${gyroMag.toFixed(3)}`
      : "IMU -";
    segmentsNow.push(segment.id);
  }

  document.getElementById("timeText").textContent = fmtS(state.sessionTime);
  document.getElementById("segmentText").textContent = [...new Set(segmentsNow)].join(" / ");
}

function play() {
  state.playing = true;
  state.sessionStartS = state.sessionTime;
  state.wallStartS = performance.now() / 1000;
  updatePlayButtons();
  syncAll(true);
  requestAnimationFrame(tick);
}

function pause() {
  state.playing = false;
  updatePlayButtons();
  syncAll(false);
}

function tick() {
  if (!state.playing) return;
  const elapsed = (performance.now() / 1000 - state.wallStartS) * state.playbackRate;
  const next = state.sessionStartS + elapsed;
  if (next >= sessionDuration()) {
    setSessionTime(sessionDuration(), true);
    pause();
    return;
  }
  state.sessionTime = next;
  document.getElementById("timeSlider").value = String(state.sessionTime);
  syncAll(false);
  drawTimeline();
  requestAnimationFrame(tick);
}

function updatePlayButtons() {
  document.getElementById("playBtn").disabled = state.playing;
  document.getElementById("pauseBtn").disabled = !state.playing;
}

function resizeCanvas() {
  const canvas = document.getElementById("timelineCanvas");
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(900, Math.floor(rect.width * dpr));
  canvas.height = Math.max(420, Math.floor(rect.height * dpr));
  drawTimeline();
}

function drawTimeline() {
  const canvas = document.getElementById("timelineCanvas");
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const padL = 86;
  const padR = 18;
  const padT = 18;
  const padB = 28;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;
  const cams = Object.keys(DATA.tracks);
  const rowH = plotH / cams.length;
  const maxT = sessionDuration();
  const xForT = (t) => padL + (t / maxT) * plotW;

  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = "#d6dbe1";
  ctx.lineWidth = 1;

  for (let tick = 0; tick <= Math.ceil(maxT / 30); tick++) {
    const t = tick * 30;
    const x = xForT(t);
    ctx.beginPath();
    ctx.moveTo(x, padT);
    ctx.lineTo(x, h - padB);
    ctx.stroke();
    ctx.fillStyle = "#626a73";
    ctx.font = "12px system-ui";
    ctx.fillText(`${t}s`, x + 4, h - 8);
  }

  cams.forEach((camId, index) => {
    const track = DATA.tracks[camId];
    const y0 = padT + index * rowH;
    const mid = y0 + rowH * 0.52;
    const top = y0 + 10;
    const bottom = y0 + rowH - 10;

    ctx.fillStyle = index % 2 === 0 ? "#fbfcfd" : "#ffffff";
    ctx.fillRect(padL, y0, plotW, rowH);
    ctx.strokeStyle = "#d6dbe1";
    ctx.beginPath();
    ctx.moveTo(padL, y0);
    ctx.lineTo(w - padR, y0);
    ctx.stroke();

    ctx.fillStyle = COLORS[camId];
    ctx.font = "bold 14px system-ui";
    ctx.fillText(track.label, 14, y0 + 24);

    for (const segment of track.segments) {
      drawAudio(ctx, segment, xForT, top, mid - 5);
      drawImu(ctx, segment, xForT, mid + 7, bottom);
    }
  });

  const playX = xForT(state.sessionTime);
  ctx.strokeStyle = "#b91c1c";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(playX, padT);
  ctx.lineTo(playX, h - padB);
  ctx.stroke();
}

function drawAudio(ctx, segment, xForT, top, bottom) {
  const rms = segment.audio.rms || [];
  const onset = segment.audio.onset || [];
  const dt = segment.audio.dt || 0.01;
  if (!rms.length) return;
  const height = bottom - top;
  const base = bottom;

  ctx.strokeStyle = "rgba(107, 114, 128, 0.72)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i < rms.length; i += 2) {
    const localT = i * dt;
    const globalT = globalTimeForLocal(segment, localT);
    const x = xForT(globalT);
    const y = base - Math.min(1, rms[i]) * height;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  ctx.strokeStyle = "rgba(17, 24, 39, 0.88)";
  ctx.beginPath();
  for (let i = 0; i < onset.length; i += 2) {
    const value = onset[i];
    if (value < 0.08) continue;
    const localT = i * dt;
    const globalT = globalTimeForLocal(segment, localT);
    const x = xForT(globalT);
    const y = base - Math.min(1, value) * height;
    ctx.moveTo(x, base);
    ctx.lineTo(x, y);
  }
  ctx.stroke();
}

function drawImu(ctx, segment, xForT, top, bottom) {
  const rows = segment.imu || [];
  if (!rows.length) return;
  const height = bottom - top;
  const accBase = 9.81;
  const accScale = 8.0;
  const gyroScale = 0.35;

  ctx.lineWidth = 1.25;
  drawSeries(ctx, rows, xForT, top, height, (row) => {
    const mag = Math.hypot(row[1], row[2], row[3]);
    return Math.max(0, Math.min(1, Math.abs(mag - accBase) / accScale));
  }, "#16a34a", segment);

  drawSeries(ctx, rows, xForT, top, height, (row) => {
    const mag = Math.hypot(row[4], row[5], row[6]);
    return Math.max(0, Math.min(1, mag / gyroScale));
  }, "#8b5cf6", segment);
}

function drawSeries(ctx, rows, xForT, top, height, valueFn, color, segment) {
  ctx.strokeStyle = color;
  ctx.beginPath();
  rows.forEach((row, i) => {
    const globalT = globalTimeForLocal(segment, row[0]);
    const x = xForT(globalT);
    const y = top + height - valueFn(row) * height;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function setupControls() {
  const slider = document.getElementById("timeSlider");
  slider.max = String(sessionDuration());
  slider.addEventListener("input", (event) => {
    const wasPlaying = state.playing;
    if (wasPlaying) pause();
    setSessionTime(Number(event.target.value), true);
    if (wasPlaying) play();
  });

  document.getElementById("playBtn").addEventListener("click", play);
  document.getElementById("pauseBtn").addEventListener("click", pause);
  document.getElementById("frameBackBtn").addEventListener("click", () => setSessionTime(state.sessionTime - 1 / 30, true));
  document.getElementById("frameForwardBtn").addEventListener("click", () => setSessionTime(state.sessionTime + 1 / 30, true));
  document.getElementById("rateSelect").addEventListener("change", (event) => {
    state.playbackRate = Number(event.target.value);
    if (state.playing) {
      state.sessionStartS = state.sessionTime;
      state.wallStartS = performance.now() / 1000;
    }
    syncAll(false);
  });
  document.getElementById("syncModeSelect").addEventListener("change", (event) => {
    state.syncMode = event.target.value;
    slider.max = String(sessionDuration());
    setSessionTime(Math.min(state.sessionTime, sessionDuration()), true);
    updateStatus();
  });
  document.getElementById("resetOffsetsBtn").addEventListener("click", () => {
    for (const camId of Object.keys(state.manualOffsets)) {
      state.manualOffsets[camId] = 0;
      const input = state.trackEls[camId].card.querySelector(".manualOffset");
      input.value = "0";
    }
    syncAll(true);
    drawTimeline();
    updateOffsetTable();
  });

  const canvas = document.getElementById("timelineCanvas");
  canvas.addEventListener("click", (event) => {
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const padL = 86;
    const padR = 18;
    const plotW = rect.width - padL - padR;
    const t = ((x - padL) / plotW) * sessionDuration();
    setSessionTime(t, true);
    if (state.playing) {
      state.sessionStartS = state.sessionTime;
      state.wallStartS = performance.now() / 1000;
    }
  });

  window.addEventListener("resize", resizeCanvas);
}

function updateStatus() {
  const mode = state.syncMode === "audio" ? "audio offsets" : "IMU/device timestamps";
  document.getElementById("statusLine").textContent =
    `${DATA.session.label} | ${mode} | ${fmtS(sessionDuration())} visible timeline`;
}

function firstSharedTime(mode) {
  const starts = [];
  for (const track of Object.values(DATA.tracks)) {
    const firstSegment = track.segments.find((segment) => segment.id === "segment1") || track.segments[0];
    if (!firstSegment) continue;
    const previousMode = state.syncMode;
    state.syncMode = mode;
    starts.push(globalTimeForLocal(firstSegment, 0, mode));
    state.syncMode = previousMode;
  }
  return starts.length ? Math.max(0, Math.max(...starts)) + 0.02 : 0;
}

function init() {
  for (const track of Object.values(DATA.tracks)) {
    for (const segment of track.segments) {
      segment.cameraId = track.id;
    }
  }
  createVideoGrid();
  buildOffsetTable();
  setupControls();
  updateStatus();
  updatePlayButtons();
  resizeCanvas();
  setSessionTime(firstSharedTime(state.syncMode), true);
}

init();
"""


def write_visualizer() -> None:
    data = build_session_data()
    (VIS_ROOT / "data").mkdir(parents=True, exist_ok=True)
    (VIS_ROOT / "index.html").write_text(INDEX_HTML)
    (VIS_ROOT / "styles.css").write_text(STYLES_CSS)
    (VIS_ROOT / "app.js").write_text(APP_JS)
    data_js = "window.EGO_EXO_SESSION_DATA = "
    data_js += json.dumps(data, separators=(",", ":"))
    data_js += ";\n"
    (VIS_ROOT / "data" / "session_data.js").write_text(data_js)
    (VIS_ROOT / "session_data_pretty.json").write_text(json.dumps(data, indent=2))
    print((VIS_ROOT / "index.html").as_uri())
    print(f"Wrote {VIS_ROOT}")


if __name__ == "__main__":
    write_visualizer()
