#!/usr/bin/env python3
"""Build a synced four-camera left/right handpose labeler for the v2 session."""

from __future__ import annotations

import json
from pathlib import Path


EGO_EXO_ROOT = Path("/Users/zikangjiang/dev/ego-exo")
OUT_ROOT = EGO_EXO_ROOT / "visualizations/v2/handpose_labeler"
REPORT_ROOT = EGO_EXO_ROOT / "visualizations/v2"

SEGMENTS = [
    ("segment1", REPORT_ROOT / "take1" / "v2_calibration_and_audio_sync_report.json"),
    ("segment2", REPORT_ROOT / "take2" / "v2_calibration_and_audio_sync_report.json"),
]

KEYPOINT_NAMES = [
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
]

EDGES = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [0, 5],
    [5, 6],
    [6, 7],
    [7, 8],
    [0, 9],
    [9, 10],
    [10, 11],
    [11, 12],
    [0, 13],
    [13, 14],
    [14, 15],
    [15, 16],
    [0, 17],
    [17, 18],
    [18, 19],
    [19, 20],
]


def first_last_imu(report: dict) -> tuple[int, int]:
    starts = []
    ends = []
    for cam in report["extrinsics"]["cameras"].values():
        imu = cam["imu_timestamps"]
        starts.append(int(imu["first_t_us"]))
        ends.append(int(imu["last_t_us"]))
    return min(starts), max(ends)


def build_data() -> dict:
    reports = [(name, json.loads(path.read_text())) for name, path in SEGMENTS]
    first_ref_us = reports[0][1]["extrinsics"]["cameras"]["cam1"]["imu_timestamps"]["first_t_us"]
    tracks = {cam: {"id": cam, "segments": []} for cam in ["cam1", "cam2", "cam3", "cam4"]}

    for segment_name, report in reports:
        ref_start_us = report["extrinsics"]["cameras"]["cam1"]["imu_timestamps"]["first_t_us"]
        ref_session_start_s = (ref_start_us - first_ref_us) / 1_000_000.0
        for cam, cam_report in report["extrinsics"]["cameras"].items():
            audio_report = report["audio_sync"]["cameras"][cam]
            imu_first = int(cam_report["imu_timestamps"]["first_t_us"])
            video_path = Path(cam_report["video"])
            tracks[cam]["segments"].append(
                {
                    "id": segment_name,
                    "videoPath": str(video_path),
                    "videoUrl": "/" + video_path.relative_to(EGO_EXO_ROOT).as_posix(),
                    "durationS": round(float(cam_report["video_duration_s"]), 4),
                    "deviceStartS": round((imu_first - first_ref_us) / 1_000_000.0, 4),
                    "referenceStartS": round(ref_session_start_s, 4),
                    "audioOffsetS": round(float(audio_report["sync_to_reference"]["offset_s"]), 4),
                    "audioSyncConfidence": audio_report["sync_to_reference"]["confidence"],
                    "reprojectionMedianPx": round(float(cam_report["reprojection_median_px"]), 4),
                }
            )

    all_segments = [segment for track in tracks.values() for segment in track["segments"]]
    duration_audio = max(seg["referenceStartS"] - seg["audioOffsetS"] + seg["durationS"] for seg in all_segments)
    duration_device = max(seg["deviceStartS"] + seg["durationS"] for seg in all_segments)
    return {
        "schema": "v2_four_camera_handpose_labeler_data_v1",
        "imageSize": [1920, 1080],
        "fps": 30,
        "referenceCamera": "cam1",
        "session": {
            "label": "v2 single recording session",
            "durationAudioS": round(float(duration_audio), 4),
            "durationDeviceS": round(float(duration_device), 4),
            "note": "segment1/segment2 are file segments from one continuous recording session.",
        },
        "keypointConvention": "MediaPipe_InterHand_21",
        "keypointNames": KEYPOINT_NAMES,
        "edges": EDGES,
        "tracks": tracks,
    }


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>v2 Four-Camera Handpose Labeler</title>
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <header class="topbar">
    <div>
      <h1>v2 Four-Camera Handpose Labeler</h1>
      <p id="statusLine">Loading...</p>
    </div>
    <div class="toolbar">
      <button id="playBtn" class="primary">Play</button>
      <button id="pauseBtn">Pause</button>
      <button id="backFrameBtn">-1f</button>
      <button id="fwdFrameBtn">+1f</button>
      <label>Sync
        <select id="syncModeSelect">
          <option value="audio" selected>Audio</option>
          <option value="device">Device/IMU</option>
        </select>
      </label>
      <label>Time
        <input id="timeInput" type="number" min="0" step="0.033" value="0">
      </label>
      <label>Zoom
        <input id="zoomSlider" type="range" min="1" max="6" step="0.25" value="2">
      </label>
      <label>Dot
        <input id="dotSizeSlider" type="range" min="3" max="12" step="1" value="6">
      </label>
      <label><input id="showLabelsToggle" type="checkbox"> Labels</label>
      <label><input id="showSkeletonToggle" type="checkbox" checked> Skeleton</label>
      <button id="exportBtn">Export JSON</button>
      <button id="copyBtn">Copy</button>
    </div>
  </header>

  <main>
    <section class="timebar">
      <div class="timeReadout">
        <span id="timeText">0.000s</span>
        <span id="frameText">frame 0</span>
      </div>
      <input id="timeSlider" type="range" min="0" max="1" step="0.001" value="0">
    </section>

    <section class="labelbar">
      <div class="activeState">
        <label>Hand
          <select id="handSelect">
            <option value="left" selected>left</option>
            <option value="right">right</option>
          </select>
        </label>
        <label>Keypoint
          <select id="keypointSelect"></select>
        </label>
        <button id="prevKpBtn">Prev</button>
        <button id="nextKpBtn">Next</button>
        <button id="clearActiveBtn">Clear active in all cams</button>
      </div>
      <div id="keypointButtons" class="keypointButtons"></div>
    </section>

    <section id="videoGrid" class="videoGrid"></section>

    <section class="outputPanel">
      <div class="outputHeader">
        <h2>Export</h2>
        <div class="outputActions">
          <button id="loadJsonBtn">Load pasted JSON</button>
          <button id="clearAllBtn">Clear all labels</button>
        </div>
      </div>
      <textarea id="outputJson" spellcheck="false"></textarea>
    </section>
  </main>

  <script src="data/handpose_labeler_data.js"></script>
  <script src="app.js"></script>
</body>
</html>
"""


STYLES_CSS = r""":root {
  color-scheme: light;
  --bg: #f5f6f8;
  --panel: #fff;
  --ink: #17191c;
  --muted: #5f6872;
  --line: #d4dbe3;
  --accent: #0f766e;
  --accent-dark: #0b5f59;
  --left: #e11d48;
  --right: #2563eb;
  --active: #f59e0b;
  --zoom: 2;
  --dot-size: 6px;
}

* { box-sizing: border-box; }

body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  letter-spacing: 0;
}

.topbar {
  position: sticky;
  top: 0;
  z-index: 20;
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 16px;
  padding: 14px 16px;
  background: #fff;
  border-bottom: 1px solid var(--line);
}

h1, h2, p { margin: 0; }
h1 { font-size: 20px; line-height: 1.2; }
h2 { font-size: 15px; }
p { color: var(--muted); font-size: 13px; margin-top: 4px; }

.toolbar,
.activeState,
.keypointButtons,
.cameraControls,
.outputActions {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
}

label {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: var(--muted);
  font-size: 13px;
}

button,
input,
select {
  min-height: 34px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #fff;
  color: var(--ink);
  padding: 6px 9px;
  font: inherit;
}

button { cursor: pointer; }
button.primary {
  background: var(--accent);
  border-color: var(--accent);
  color: #fff;
}
button.primary:hover { background: var(--accent-dark); }
button.active {
  background: #111827;
  border-color: #111827;
  color: #fff;
}
input[type="number"] { width: 94px; font-variant-numeric: tabular-nums; }
input[type="range"] { width: 150px; padding: 0; }

main {
  padding: 14px;
  display: grid;
  gap: 14px;
}

.timebar,
.labelbar,
.cameraCard,
.outputPanel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
}

.timebar,
.labelbar,
.outputPanel {
  padding: 12px;
}

.timeReadout {
  display: flex;
  align-items: center;
  justify-content: space-between;
  color: var(--muted);
  font-size: 13px;
  margin-bottom: 8px;
}

#timeText,
#frameText {
  font-variant-numeric: tabular-nums;
}

#timeText {
  color: var(--ink);
  font-weight: 700;
}

#timeSlider {
  width: 100%;
}

.labelbar {
  display: grid;
  gap: 10px;
}

.keypointButtons button {
  min-height: 28px;
  padding: 4px 7px;
  font-size: 12px;
}

.keypointButtons button.done-left {
  box-shadow: inset 0 -3px 0 var(--left);
}

.keypointButtons button.done-right {
  box-shadow: inset 0 -3px 0 var(--right);
}

.videoGrid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
}

.cameraCard {
  min-width: 0;
  overflow: hidden;
}

.cameraHeader {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 10px 12px;
  border-bottom: 1px solid var(--line);
}

.cameraTitle {
  font-weight: 700;
}

.cameraMeta {
  color: var(--muted);
  font-size: 12px;
  font-variant-numeric: tabular-nums;
}

.cameraControls {
  padding: 10px 12px;
  border-bottom: 1px solid var(--line);
}

.stage {
  position: relative;
  height: 56vh;
  min-height: 420px;
  overflow: auto;
  background: #111;
}

.videoWrap {
  position: relative;
  width: calc(100% * var(--zoom));
  min-width: calc(100% * var(--zoom));
}

video {
  display: block;
  width: 100%;
  height: auto;
  user-select: none;
  background: #000;
}

svg.skeleton {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  z-index: 2;
  overflow: visible;
}

body:not(.show-skeleton) svg.skeleton {
  display: none;
}

.joint {
  position: absolute;
  width: var(--dot-size);
  height: var(--dot-size);
  border: 1.5px solid #fff;
  border-radius: 50%;
  transform: translate(-50%, -50%);
  box-shadow: 0 0 0 1.4px #111, 0 1px 6px rgba(0,0,0,.45);
  pointer-events: none;
  z-index: 3;
}

.joint.left { background: var(--left); }
.joint.right { background: var(--right); }
.joint.active {
  background: var(--active);
  width: calc(var(--dot-size) + 7px);
  height: calc(var(--dot-size) + 7px);
  box-shadow: 0 0 0 2px #fff, 0 0 0 4px #111, 0 2px 10px rgba(0,0,0,.7);
  z-index: 5;
}

.joint span {
  display: none;
  position: absolute;
  left: 10px;
  top: -14px;
  padding: 1px 4px;
  border-radius: 4px;
  background: rgba(0,0,0,.78);
  color: #fff;
  font-size: 10px;
  line-height: 1.2;
  white-space: nowrap;
}

body.show-labels .joint span,
.joint.active span {
  display: block;
}

.inactiveOverlay {
  display: none;
  position: absolute;
  inset: 0;
  align-items: center;
  justify-content: center;
  color: #fff;
  background: rgba(0,0,0,.68);
  font-weight: 700;
  z-index: 6;
}

.cameraCard.inactive .inactiveOverlay {
  display: flex;
}

.outputHeader {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
  margin-bottom: 8px;
}

textarea {
  width: 100%;
  min-height: 260px;
  resize: vertical;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 10px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 12px;
}

@media (max-width: 1100px) {
  .topbar { position: static; display: block; }
  .toolbar { margin-top: 10px; }
  .videoGrid { grid-template-columns: 1fr; }
}
"""


APP_JS = r"""const DATA = window.HANDPOSE_LABELER_DATA;
const CAMS = ["cam1", "cam2", "cam3", "cam4"];
const HANDS = ["left", "right"];

const state = {
  syncMode: "audio",
  sessionTime: 0,
  playing: false,
  playbackRate: 1,
  wallStartS: 0,
  sessionStartS: 0,
  activeHand: "left",
  activeKp: 0,
  labels: { left: {}, right: {} },
  cameraNudges: Object.fromEntries(CAMS.map((cam) => [cam, 0])),
  trackEls: {},
};

function fmtS(value, digits = 3) {
  return `${Number(value || 0).toFixed(digits)}s`;
}

function duration() {
  return state.syncMode === "audio" ? DATA.session.durationAudioS : DATA.session.durationDeviceS;
}

function localTimeForSegment(segment, sessionTime, cam) {
  const nudge = state.cameraNudges[cam] || 0;
  if (state.syncMode === "device") return sessionTime - segment.deviceStartS + nudge;
  return sessionTime - segment.referenceStartS + segment.audioOffsetS + nudge;
}

function globalTimeForLocal(segment, localTime, cam) {
  const nudge = state.cameraNudges[cam] || 0;
  if (state.syncMode === "device") return segment.deviceStartS + localTime - nudge;
  return segment.referenceStartS + localTime - segment.audioOffsetS - nudge;
}

function activeSegment(cam, sessionTime = state.sessionTime) {
  const track = DATA.tracks[cam];
  for (const segment of track.segments) {
    const localT = localTimeForSegment(segment, sessionTime, cam);
    if (localT >= 0 && localT <= segment.durationS) return { segment, localT };
  }
  return { segment: null, localT: NaN };
}

function firstSharedTime() {
  const starts = [];
  for (const cam of CAMS) {
    const segment = DATA.tracks[cam].segments[0];
    starts.push(globalTimeForLocal(segment, 0, cam));
  }
  return Math.max(0, Math.max(...starts)) + 0.02;
}

function keyLabel(index) {
  return `${index} ${DATA.keypointNames[index]}`;
}

function buildKeypointControls() {
  const select = document.getElementById("keypointSelect");
  const buttons = document.getElementById("keypointButtons");
  select.innerHTML = "";
  buttons.innerHTML = "";
  DATA.keypointNames.forEach((name, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = keyLabel(index);
    select.append(option);

    const button = document.createElement("button");
    button.type = "button";
    button.dataset.kp = String(index);
    button.textContent = String(index);
    button.title = keyLabel(index);
    button.addEventListener("click", () => setActiveKp(index));
    buttons.append(button);
  });
}

function buildVideoGrid() {
  const grid = document.getElementById("videoGrid");
  grid.innerHTML = "";
  for (const cam of CAMS) {
    const card = document.createElement("article");
    card.className = "cameraCard";
    card.dataset.cam = cam;
    card.innerHTML = `
      <div class="cameraHeader">
        <div class="cameraTitle">${cam}</div>
        <div class="cameraMeta">
          <span class="segmentText">-</span>
          <span class="localTimeText">-</span>
          <span class="countText">0 labels</span>
        </div>
      </div>
      <div class="cameraControls">
        <button class="nudgeBackBtn">cam -1f</button>
        <button class="nudgeFwdBtn">cam +1f</button>
        <button class="clearCamPointBtn">Clear active</button>
        <button class="clearCamHandBtn">Clear hand in cam</button>
        <span class="nudgeText">nudge 0.000s</span>
      </div>
      <div class="stage">
        <div class="videoWrap">
          <video muted playsinline preload="metadata"></video>
          <svg class="skeleton"></svg>
          <div class="inactiveOverlay">No video segment at this time</div>
        </div>
      </div>
    `;
    grid.append(card);
    const video = card.querySelector("video");
    const wrap = card.querySelector(".videoWrap");

    card.querySelector(".nudgeBackBtn").addEventListener("click", () => nudgeCamera(cam, -1 / DATA.fps));
    card.querySelector(".nudgeFwdBtn").addEventListener("click", () => nudgeCamera(cam, 1 / DATA.fps));
    card.querySelector(".clearCamPointBtn").addEventListener("click", () => {
      clearPoint(cam, state.activeHand, state.activeKp);
    });
    card.querySelector(".clearCamHandBtn").addEventListener("click", () => {
      for (const kp of Object.keys(state.labels[state.activeHand])) {
        delete state.labels[state.activeHand][kp][cam];
      }
      cleanupLabels();
      renderAllLabels();
      updateExport();
    });
    wrap.addEventListener("click", (event) => {
      if (event.target.classList.contains("joint")) return;
      const rect = video.getBoundingClientRect();
      const x = ((event.clientX - rect.left) / rect.width) * DATA.imageSize[0];
      const y = ((event.clientY - rect.top) / rect.height) * DATA.imageSize[1];
      if (x < 0 || y < 0 || x > DATA.imageSize[0] || y > DATA.imageSize[1]) return;
      setPoint(cam, state.activeHand, state.activeKp, x, y);
    });
    state.trackEls[cam] = {
      card,
      video,
      wrap,
      skeleton: card.querySelector("svg.skeleton"),
      segmentText: card.querySelector(".segmentText"),
      localTimeText: card.querySelector(".localTimeText"),
      countText: card.querySelector(".countText"),
      nudgeText: card.querySelector(".nudgeText"),
    };
  }
}

function setPoint(cam, hand, kp, x, y) {
  if (!state.labels[hand][kp]) state.labels[hand][kp] = {};
  state.labels[hand][kp][cam] = {
    x: Math.round(x * 1000) / 1000,
    y: Math.round(y * 1000) / 1000,
  };
  renderAllLabels();
  updateExport();
}

function clearPoint(cam, hand, kp) {
  if (state.labels[hand][kp]) delete state.labels[hand][kp][cam];
  cleanupLabels();
  renderAllLabels();
  updateExport();
}

function cleanupLabels() {
  for (const hand of HANDS) {
    for (const kp of Object.keys(state.labels[hand])) {
      if (!Object.keys(state.labels[hand][kp]).length) delete state.labels[hand][kp];
    }
  }
}

function setActiveKp(kp) {
  state.activeKp = Math.max(0, Math.min(DATA.keypointNames.length - 1, Number(kp) || 0));
  document.getElementById("keypointSelect").value = String(state.activeKp);
  updateKeypointButtons();
  renderAllLabels();
  updateStatus();
}

function updateKeypointButtons() {
  const buttons = document.querySelectorAll("#keypointButtons button");
  buttons.forEach((button) => {
    const kp = Number(button.dataset.kp);
    button.classList.toggle("active", kp === state.activeKp);
    button.classList.toggle("done-left", !!state.labels.left[kp]);
    button.classList.toggle("done-right", !!state.labels.right[kp]);
  });
}

function nudgeCamera(cam, deltaS) {
  state.cameraNudges[cam] = Math.round((state.cameraNudges[cam] + deltaS) * 1000) / 1000;
  syncVideos(true);
  renderAllLabels();
  updateExport();
}

function setSessionTime(timeS, force = true) {
  state.sessionTime = Math.max(0, Math.min(duration(), Number(timeS) || 0));
  document.getElementById("timeSlider").value = String(state.sessionTime);
  document.getElementById("timeInput").value = state.sessionTime.toFixed(3);
  syncVideos(force);
  updateStatus();
  updateExport();
}

function syncVideos(force = false) {
  for (const cam of CAMS) {
    const els = state.trackEls[cam];
    const { segment, localT } = activeSegment(cam);
    if (!segment) {
      els.card.classList.add("inactive");
      els.segmentText.textContent = "-";
      els.localTimeText.textContent = "-";
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
    if (force || Math.abs(els.video.currentTime - localT) > 0.08) {
      try {
        els.video.currentTime = Math.max(0, Math.min(segment.durationS - 0.001, localT));
      } catch (_) {}
    }
    if (state.playing && els.video.paused) {
      els.video.play().catch(() => {
        state.playing = false;
        updatePlayButtons();
      });
    } else if (!state.playing && !els.video.paused) {
      els.video.pause();
    }
    els.segmentText.textContent = segment.id;
    els.localTimeText.textContent = `local ${fmtS(localT)}`;
    els.nudgeText.textContent = `nudge ${fmtS(state.cameraNudges[cam])}`;
  }
  renderAllLabels();
}

function renderAllLabels() {
  for (const cam of CAMS) renderLabelsForCamera(cam);
  updateKeypointButtons();
  updateStatus();
}

function renderLabelsForCamera(cam) {
  const els = state.trackEls[cam];
  els.wrap.querySelectorAll(".joint").forEach((node) => node.remove());
  els.skeleton.innerHTML = "";
  let count = 0;
  for (const hand of HANDS) {
    const points = {};
    for (const [kp, cams] of Object.entries(state.labels[hand])) {
      const point = cams[cam];
      if (!point) continue;
      count += 1;
      points[kp] = point;
      const dot = document.createElement("div");
      dot.className = `joint ${hand} ${hand === state.activeHand && Number(kp) === state.activeKp ? "active" : ""}`;
      dot.style.left = `${(point.x / DATA.imageSize[0]) * 100}%`;
      dot.style.top = `${(point.y / DATA.imageSize[1]) * 100}%`;
      dot.innerHTML = `<span>${hand} ${keyLabel(Number(kp))}</span>`;
      els.wrap.append(dot);
    }
    for (const [a, b] of DATA.edges) {
      const pa = points[a];
      const pb = points[b];
      if (!pa || !pb) continue;
      const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
      line.setAttribute("x1", `${(pa.x / DATA.imageSize[0]) * 100}%`);
      line.setAttribute("y1", `${(pa.y / DATA.imageSize[1]) * 100}%`);
      line.setAttribute("x2", `${(pb.x / DATA.imageSize[0]) * 100}%`);
      line.setAttribute("y2", `${(pb.y / DATA.imageSize[1]) * 100}%`);
      line.setAttribute("stroke", hand === "left" ? "var(--left)" : "var(--right)");
      line.setAttribute("stroke-width", "2.2");
      line.setAttribute("stroke-linecap", "round");
      line.setAttribute("opacity", "0.82");
      els.skeleton.append(line);
    }
  }
  els.countText.textContent = `${count} labels`;
}

function updateStatus() {
  const total = HANDS.reduce((sum, hand) => {
    return sum + Object.values(state.labels[hand]).reduce((s, cams) => s + Object.keys(cams).length, 0);
  }, 0);
  document.getElementById("statusLine").textContent =
    `${DATA.session.label} | ${state.syncMode} sync | ${state.activeHand} ${keyLabel(state.activeKp)} | ${total} 2D labels`;
  document.getElementById("timeText").textContent = fmtS(state.sessionTime);
  document.getElementById("frameText").textContent = `frame ${Math.round(state.sessionTime * DATA.fps)}`;
}

function play() {
  state.playing = true;
  state.sessionStartS = state.sessionTime;
  state.wallStartS = performance.now() / 1000;
  updatePlayButtons();
  syncVideos(true);
  requestAnimationFrame(tick);
}

function pause() {
  state.playing = false;
  updatePlayButtons();
  syncVideos(false);
}

function tick() {
  if (!state.playing) return;
  const elapsed = performance.now() / 1000 - state.wallStartS;
  const next = state.sessionStartS + elapsed;
  if (next >= duration()) {
    setSessionTime(duration(), true);
    pause();
    return;
  }
  state.sessionTime = next;
  document.getElementById("timeSlider").value = String(state.sessionTime);
  document.getElementById("timeInput").value = state.sessionTime.toFixed(3);
  syncVideos(false);
  updateStatus();
  requestAnimationFrame(tick);
}

function updatePlayButtons() {
  document.getElementById("playBtn").disabled = state.playing;
  document.getElementById("pauseBtn").disabled = !state.playing;
}

function exportPayload() {
  const cameras = {};
  for (const cam of CAMS) {
    const { segment, localT } = activeSegment(cam);
    cameras[cam] = {
      segment_id: segment?.id || null,
      video_url: segment?.videoUrl || null,
      local_time_s: Number.isFinite(localT) ? Math.round(localT * 1000) / 1000 : null,
      frame_index_30fps: Number.isFinite(localT) ? Math.round(localT * DATA.fps) : null,
      manual_nudge_s: state.cameraNudges[cam],
    };
  }
  const hands = {};
  for (const hand of HANDS) {
    hands[hand] = {};
    for (const [kp, cams] of Object.entries(state.labels[hand])) {
      const item = {};
      for (const [cam, point] of Object.entries(cams)) {
        item[cam] = [point.x, point.y];
      }
      hands[hand][kp] = item;
    }
  }
  return {
    schema: "multi_camera_hand21_labels_v1",
    sync_mode: state.syncMode,
    session_time_s: Math.round(state.sessionTime * 1000) / 1000,
    image_size: DATA.imageSize,
    fps: DATA.fps,
    keypoint_convention: DATA.keypointConvention,
    keypoint_names: DATA.keypointNames,
    edges: DATA.edges,
    cameras,
    hands,
  };
}

function updateExport() {
  document.getElementById("outputJson").value = JSON.stringify(exportPayload(), null, 2);
}

async function copyExport() {
  updateExport();
  await navigator.clipboard.writeText(document.getElementById("outputJson").value);
}

function loadPastedJson() {
  const text = document.getElementById("outputJson").value.trim();
  if (!text) return;
  const payload = JSON.parse(text);
  state.syncMode = payload.sync_mode || "audio";
  document.getElementById("syncModeSelect").value = state.syncMode;
  state.labels = { left: {}, right: {} };
  for (const hand of HANDS) {
    for (const [kp, cams] of Object.entries(payload.hands?.[hand] || {})) {
      state.labels[hand][kp] = {};
      for (const [cam, xy] of Object.entries(cams)) {
        state.labels[hand][kp][cam] = { x: xy[0], y: xy[1] };
      }
    }
  }
  for (const cam of CAMS) {
    state.cameraNudges[cam] = payload.cameras?.[cam]?.manual_nudge_s || 0;
  }
  document.getElementById("timeSlider").max = String(duration());
  setSessionTime(payload.session_time_s || 0, true);
}

function clearAllLabels() {
  state.labels = { left: {}, right: {} };
  renderAllLabels();
  updateExport();
}

function setup() {
  document.body.classList.add("show-skeleton");
  buildKeypointControls();
  buildVideoGrid();

  document.getElementById("syncModeSelect").addEventListener("change", (event) => {
    state.syncMode = event.target.value;
    document.getElementById("timeSlider").max = String(duration());
    setSessionTime(Math.min(state.sessionTime, duration()), true);
  });
  document.getElementById("handSelect").addEventListener("change", (event) => {
    state.activeHand = event.target.value;
    renderAllLabels();
    updateStatus();
  });
  document.getElementById("keypointSelect").addEventListener("change", (event) => setActiveKp(event.target.value));
  document.getElementById("prevKpBtn").addEventListener("click", () => setActiveKp(state.activeKp - 1));
  document.getElementById("nextKpBtn").addEventListener("click", () => setActiveKp(state.activeKp + 1));
  document.getElementById("clearActiveBtn").addEventListener("click", () => {
    for (const cam of CAMS) clearPoint(cam, state.activeHand, state.activeKp);
  });
  document.getElementById("zoomSlider").addEventListener("input", (event) => {
    document.documentElement.style.setProperty("--zoom", event.target.value);
  });
  document.getElementById("dotSizeSlider").addEventListener("input", (event) => {
    document.documentElement.style.setProperty("--dot-size", `${event.target.value}px`);
  });
  document.getElementById("showLabelsToggle").addEventListener("change", (event) => {
    document.body.classList.toggle("show-labels", event.target.checked);
  });
  document.getElementById("showSkeletonToggle").addEventListener("change", (event) => {
    document.body.classList.toggle("show-skeleton", event.target.checked);
  });
  document.getElementById("timeSlider").addEventListener("input", (event) => setSessionTime(Number(event.target.value), true));
  document.getElementById("timeInput").addEventListener("change", (event) => setSessionTime(Number(event.target.value), true));
  document.getElementById("backFrameBtn").addEventListener("click", () => setSessionTime(state.sessionTime - 1 / DATA.fps, true));
  document.getElementById("fwdFrameBtn").addEventListener("click", () => setSessionTime(state.sessionTime + 1 / DATA.fps, true));
  document.getElementById("playBtn").addEventListener("click", play);
  document.getElementById("pauseBtn").addEventListener("click", pause);
  document.getElementById("exportBtn").addEventListener("click", updateExport);
  document.getElementById("copyBtn").addEventListener("click", copyExport);
  document.getElementById("loadJsonBtn").addEventListener("click", loadPastedJson);
  document.getElementById("clearAllBtn").addEventListener("click", clearAllLabels);

  document.getElementById("timeSlider").max = String(duration());
  setActiveKp(0);
  setSessionTime(firstSharedTime(), true);
  updatePlayButtons();
}

setup();
"""


def write_labeler() -> None:
    data = build_data()
    (OUT_ROOT / "data").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "index.html").write_text(INDEX_HTML)
    (OUT_ROOT / "styles.css").write_text(STYLES_CSS)
    (OUT_ROOT / "app.js").write_text(APP_JS)
    (OUT_ROOT / "data/handpose_labeler_data.js").write_text(
        "window.HANDPOSE_LABELER_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n"
    )
    (OUT_ROOT / "handpose_labeler_data_pretty.json").write_text(json.dumps(data, indent=2))
    print("http://127.0.0.1:8790/visualizations/v2/handpose_labeler/index.html")
    print(f"Wrote {OUT_ROOT}")


if __name__ == "__main__":
    write_labeler()
