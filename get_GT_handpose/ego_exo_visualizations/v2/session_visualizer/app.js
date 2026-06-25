const DATA = window.EGO_EXO_SESSION_DATA;

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
