const DATA = window.HANDPOSE_LABELER_DATA;
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
