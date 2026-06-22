#!/usr/bin/env python3
"""Local browser viewer for EgoVerse hand keypoint visual QA."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import webbrowser
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


def add_egoverse_to_path() -> Path:
    default_repo = Path(__file__).resolve().parents[2] / "EgoVerse"
    repo = Path(os.environ.get("EGOVERSE_REPO", default_repo)).expanduser().resolve()
    if not repo.exists():
        raise SystemExit(
            f"EgoVerse repo not found at {repo}. Set EGOVERSE_REPO or clone it next to handpose_v1."
        )
    sys.path.insert(0, str(repo))
    venv_bin = repo / "emimic" / "bin"
    if venv_bin.exists():
        os.environ["PATH"] = f"{venv_bin}{os.pathsep}{os.environ.get('PATH', '')}"
    return repo


EGOVERSE_REPO = add_egoverse_to_path()

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import simplejpeg  # noqa: E402
import zarr  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

from egomimic.rldb.embodiment.human import Aria, Human, Mecka  # noqa: E402
from egomimic.utils.aws.aws_data_utils import load_env  # noqa: E402
from egomimic.utils.aws.aws_sql import create_default_engine, episode_table_to_df  # noqa: E402
from egomimic.utils.egomimicUtils import INTRINSICS  # noqa: E402


DEFAULT_CACHE_DIR = Path(
    os.environ.get("EGOVERSE_VIEWER_CACHE_DIR", "/data/egoverse_viewer_cache")
)
FRAME_KEYS = {
    "raw-aria": ("left.obs_aria_keypoints", "right.obs_aria_keypoints", Aria),
    "canonical": ("left.obs_keypoints", "right.obs_keypoints", Human),
}
SYNC_LOCKS: dict[str, threading.Lock] = {}
SYNC_LOCKS_GUARD = threading.Lock()
PREFETCH_INFLIGHT: set[str] = set()
PREFETCH_GUARD = threading.Lock()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start a local EgoVerse handpose QA viewer.")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8765")))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--open", action="store_true", help="Open the viewer in a browser.")
    return parser.parse_args()


@lru_cache(maxsize=1)
def episode_df_json() -> list[dict]:
    load_env(required=True)
    df = episode_table_to_df(create_default_engine())
    if "is_deleted" in df.columns:
        df = df[~df["is_deleted"].fillna(False).astype(bool)]
    if "zarr_processed_path" in df.columns:
        df = df[df["zarr_processed_path"].fillna("").astype(str).str.strip() != ""]
    keep = [
        "episode_hash",
        "embodiment",
        "task",
        "lab",
        "scene",
        "operator",
        "num_frames",
        "zarr_processed_path",
    ]
    keep = [c for c in keep if c in df.columns]
    return df[keep].fillna("").to_dict(orient="records")


def row_for_episode(episode_hash: str) -> dict:
    for row in episode_df_json():
        if str(row.get("episode_hash")) == episode_hash:
            return row
    raise KeyError(f"Unknown episode_hash: {episode_hash}")


def filter_rows(rows: list[dict], qs: dict[str, str]) -> list[dict]:
    embodiment = qs.get("embodiment", "").strip().lower()
    task = qs.get("task", "").strip()
    lab = qs.get("lab", "").strip()
    q = qs.get("q", "").strip().lower()
    if embodiment:
        rows = [r for r in rows if str(r.get("embodiment", "")).lower().startswith(embodiment)]
    if task:
        rows = [r for r in rows if str(r.get("task", "")) == task]
    if lab:
        rows = [r for r in rows if str(r.get("lab", "")) == lab]
    if q:
        rows = [r for r in rows if q in json.dumps(r).lower()]
    return rows


def episode_dir(cache_dir: Path, episode_hash: str) -> Path:
    return cache_dir / episode_hash


def episode_ready(path: Path) -> bool:
    return (path / "zarr.json").exists() and (path / "images.front_1" / "zarr.json").exists()


def sync_episode(cache_dir: Path, episode_hash: str) -> Path:
    dest = episode_dir(cache_dir, episode_hash)
    if episode_ready(dest):
        return dest

    with SYNC_LOCKS_GUARD:
        lock = SYNC_LOCKS.setdefault(episode_hash, threading.Lock())

    with lock:
        if episode_ready(dest):
            return dest

        row = row_for_episode(episode_hash)
        src = str(row["zarr_processed_path"]).rstrip("/") + "/*"
        dest.mkdir(parents=True, exist_ok=True)
        load_env(required=True)
        env = os.environ.copy()
        env["AWS_ACCESS_KEY_ID"] = env["R2_ACCESS_KEY_ID"]
        env["AWS_SECRET_ACCESS_KEY"] = env["R2_SECRET_ACCESS_KEY"]
        env["AWS_DEFAULT_REGION"] = "auto"
        env["AWS_REGION"] = "auto"
        endpoint = env["R2_ENDPOINT_URL"]
        with tempfile.NamedTemporaryFile("w", suffix=".s5cmd", delete=False) as f:
            batch_path = Path(f.name)
            f.write(f'sync "{src}" "{dest}/"\n')
        try:
            cmd = [
                "s5cmd",
                "--endpoint-url",
                endpoint,
                "--numworkers",
                "32",
                "run",
                str(batch_path),
            ]
            subprocess.run(
                cmd,
                check=True,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        finally:
            batch_path.unlink(missing_ok=True)
    return dest


def prefetch_episode(cache_dir: Path, episode_hash: str) -> None:
    with PREFETCH_GUARD:
        if episode_hash in PREFETCH_INFLIGHT:
            return
        PREFETCH_INFLIGHT.add(episode_hash)

    def run() -> None:
        try:
            sync_episode(cache_dir, episode_hash)
        except Exception:
            pass
        finally:
            with PREFETCH_GUARD:
                PREFETCH_INFLIGHT.discard(episode_hash)

    threading.Thread(target=run, daemon=True).start()


@lru_cache(maxsize=8)
def open_episode(path_str: str):
    return zarr.open_group(path_str, mode="r")


def pose_world_to_local(points_world: np.ndarray, pose_world: np.ndarray) -> np.ndarray:
    xyz = pose_world[:3]
    quat_wxyz = pose_world[3:7]
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    rot_world_from_local = R.from_quat(quat_xyzw)
    return rot_world_from_local.inv().apply(points_world - xyz)


def hand_points_world_to_local(points_world: np.ndarray, pose_world: np.ndarray) -> np.ndarray:
    points_cam = np.zeros_like(points_world)
    valid = np.isfinite(points_world).all(axis=1) & (np.linalg.norm(points_world, axis=1) > 1e-9)
    if valid.any():
        points_cam[valid] = pose_world_to_local(points_world[valid], pose_world)
    return points_cam


def intrinsics_for_episode(g, episode_hash: str, layout: str) -> np.ndarray:
    camera_intrinsics = g.attrs.get("camera_intrinsics")
    if isinstance(camera_intrinsics, dict):
        try:
            return np.array(
                [
                    [float(camera_intrinsics["fx"]), 0.0, float(camera_intrinsics["cx"]), 0.0],
                    [0.0, float(camera_intrinsics["fy"]), float(camera_intrinsics["cy"]), 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ],
                dtype=np.float64,
            )
        except (KeyError, TypeError, ValueError):
            pass
    if layout == "raw-aria":
        return INTRINSICS["base"]
    embodiment = str(row_for_episode(episode_hash).get("embodiment", "")).lower()
    if embodiment.startswith("mecka"):
        return INTRINSICS["mecka"]
    if embodiment.startswith("scale"):
        return INTRINSICS["scale"]
    return INTRINSICS["base"]


def project_points(points_cam: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    pts_h = np.concatenate([points_cam, np.ones((len(points_cam), 1))], axis=1)
    px = (intrinsics @ pts_h.T).T
    return px / px[:, 2:3]


def draw_handpose(
    image: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    topology_cls,
    intrinsics: np.ndarray,
    dot_radius: int,
    line_thickness: int,
) -> np.ndarray:
    vis = image.copy()
    h, w = vis.shape[:2]
    colors = {
        "thumb": (255, 100, 100),
        "index": (100, 255, 100),
        "middle": (100, 170, 255),
        "ring": (255, 230, 80),
        "pinky": (255, 100, 255),
    }
    dot_colors = {"left": (0, 140, 255), "right": (255, 120, 0)}

    for label, points in (("left", left), ("right", right)):
        px = project_points(points, intrinsics)[:, :2]
        valid = points[:, 2] > 0.01
        valid &= (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h)
        for finger, start, end in topology_cls.FINGER_EDGE_RANGES:
            color = colors[finger]
            for edge_idx in range(start, end):
                i, j = topology_cls.FINGER_EDGES[edge_idx]
                if valid[i] and valid[j]:
                    p1 = tuple(np.round(px[i]).astype(int))
                    p2 = tuple(np.round(px[j]).astype(int))
                    cv2.line(vis, p1, p2, color, line_thickness, cv2.LINE_AA)
        for i in range(21):
            if valid[i]:
                center = tuple(np.round(px[i]).astype(int))
                cv2.circle(vis, center, dot_radius, dot_colors[label], -1, cv2.LINE_AA)
                if dot_radius >= 3:
                    cv2.circle(vis, center, dot_radius, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def frame_pose_data(
    cache_dir: Path,
    episode_hash: str,
    frame: int,
    keypoint_offset: int,
    layout: str,
):
    path = sync_episode(cache_dir, episode_hash)
    g = open_episode(str(path))
    total = int(g.attrs.get("total_frames", g["images.front_1"].shape[0]))
    frame = max(0, min(frame, total - 1))
    keypoint_frame = max(0, min(frame + keypoint_offset, total - 1))
    left_key, right_key, topology_cls = FRAME_KEYS[layout]
    required = [left_key, right_key]
    has_head_pose = "obs_head_pose" in g
    if layout == "raw-aria":
        required.append("obs_head_pose")
    missing = [key for key in required if key not in g]
    if missing:
        raise ValueError(
            f"missing handpose/projection arrays for {episode_hash}: {', '.join(missing)}"
        )
    left_world = np.asarray(g[left_key][keypoint_frame], dtype=np.float64).reshape(21, 3)
    right_world = np.asarray(g[right_key][keypoint_frame], dtype=np.float64).reshape(21, 3)
    if has_head_pose:
        head_world = np.asarray(g["obs_head_pose"][keypoint_frame], dtype=np.float64)
        left_cam = hand_points_world_to_local(left_world, head_world)
        right_cam = hand_points_world_to_local(right_world, head_world)
    else:
        left_cam = left_world
        right_cam = right_world
    intrinsics = intrinsics_for_episode(g, episode_hash, layout)
    return {
        "g": g,
        "total": total,
        "frame": frame,
        "keypoint_frame": keypoint_frame,
        "left_cam": left_cam,
        "right_cam": right_cam,
        "intrinsics": intrinsics,
        "topology_cls": topology_cls,
    }


def render_frame(
    cache_dir: Path,
    episode_hash: str,
    frame: int,
    keypoint_offset: int,
    layout: str,
    dot: int,
    line: int,
) -> bytes:
    pose = frame_pose_data(cache_dir, episode_hash, frame, keypoint_offset, layout)
    g = pose["g"]
    frame = pose["frame"]
    jpeg_value = g["images.front_1"][frame]
    while isinstance(jpeg_value, np.ndarray) and jpeg_value.shape == ():
        jpeg_value = jpeg_value.item()
    if not jpeg_value:
        raise ValueError(f"empty RGB frame {frame} in {episode_hash}")
    try:
        image = simplejpeg.decode_jpeg(jpeg_value, colorspace="RGB")
    except Exception:
        encoded = np.frombuffer(jpeg_value, dtype=np.uint8)
        bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if bgr is None:
            raise
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    overlay = draw_handpose(
        image,
        pose["left_cam"],
        pose["right_cam"],
        pose["topology_cls"],
        pose["intrinsics"],
        dot,
        line,
    )
    return simplejpeg.encode_jpeg(overlay, quality=90, colorspace="RGB")


def pose_json(
    cache_dir: Path,
    episode_hash: str,
    frame: int,
    keypoint_offset: int,
    layout: str,
) -> dict:
    pose = frame_pose_data(cache_dir, episode_hash, frame, keypoint_offset, layout)
    g = pose["g"]
    intrinsics = pose["intrinsics"]
    camera_intrinsics = g.attrs.get("camera_intrinsics")
    width = 640
    height = 480
    if isinstance(camera_intrinsics, dict):
        width = int(camera_intrinsics.get("width", width))
        height = int(camera_intrinsics.get("height", height))
    else:
        feature_shape = g.attrs.get("features", {}).get("images.front_1", {}).get("shape")
        if isinstance(feature_shape, list) and len(feature_shape) >= 2:
            height, width = int(feature_shape[0]), int(feature_shape[1])

    topology_cls = pose["topology_cls"]
    return {
        "episode_hash": episode_hash,
        "frame": pose["frame"],
        "keypoint_frame": pose["keypoint_frame"],
        "total": pose["total"],
        "left": pose["left_cam"].tolist(),
        "right": pose["right_cam"].tolist(),
        "edges": topology_cls.FINGER_EDGES,
        "edge_ranges": topology_cls.FINGER_EDGE_RANGES,
        "intrinsics": intrinsics.tolist(),
        "image_size": {"width": width, "height": height},
    }


def html_page() -> bytes:
    return rb"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>EgoVerse Handpose QA</title>
  <style>
    body { margin: 0; overflow: hidden; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #0b0f14; color: #f9fafb; }
    header { height: 44px; display: flex; align-items: center; padding: 0 14px; background: #111827; font-weight: 650; }
    main { min-width: 0; height: calc(100vh - 44px); display: grid; grid-template-rows: auto minmax(0, 1fr) auto; }
    .bar { min-width: 0; display: grid; grid-template-columns: 130px 220px minmax(120px, 1fr) repeat(4, 104px); gap: 8px; padding: 10px; background: #f8fafc; color: #111827; align-items: end; }
    .playbar { min-width: 0; display: grid; grid-template-columns: 104px 104px 104px minmax(120px, 1fr) 82px 82px; gap: 8px; padding: 10px; background: #f8fafc; color: #111827; align-items: end; }
    label { display: block; font-size: 11px; font-weight: 650; color: #4b5563; margin-bottom: 3px; }
    input, select, button { box-sizing: border-box; width: 100%; height: 32px; border: 1px solid #cbd5e1; border-radius: 5px; padding: 4px 8px; background: white; color: #111827; }
    button { cursor: pointer; font-weight: 650; background: #0f766e; color: white; border: 0; }
    button.secondary { background: #334155; }
    .bar > *, .playbar > * { min-width: 0; }
    #viewerWrap { min-width: 0; min-height: 0; display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 1px; overflow: hidden; background: #1f2937; }
    .pane { min-width: 0; min-height: 0; display: flex; align-items: center; justify-content: center; overflow: hidden; background: #0b0f14; }
    .pane3d { position: relative; }
    .viewBtns { position: absolute; right: 8px; top: 8px; display: flex; gap: 5px; z-index: 2; }
    .viewBtns button { width: auto; height: 26px; padding: 2px 8px; border-radius: 4px; background: rgba(15, 23, 42, 0.84); color: #e2e8f0; border: 1px solid rgba(148, 163, 184, 0.45); font-size: 11px; }
    .viewBtns button:hover { background: rgba(30, 41, 59, 0.94); }
    #viewer { max-width: 100%; max-height: 100%; object-fit: contain; background: #111827; display: block; }
    #viewer3d { width: 100%; height: 100%; display: block; background: #071018; cursor: grab; }
    #viewer3d:active { cursor: grabbing; }
    #status { font-size: 12px; color: #cbd5e1; padding: 0 10px 8px; min-height: 18px; }
    @media (max-width: 900px) {
      .bar { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .playbar { grid-template-columns: repeat(3, minmax(0, 1fr)); }
      .playbar > div:first-of-type { grid-column: 1 / -1; }
      #viewerWrap { grid-template-columns: 1fr; grid-template-rows: minmax(0, 1fr) minmax(0, 1fr); }
    }
    @media (max-width: 560px) {
      header { height: 38px; padding: 0 10px; }
      main { height: calc(100vh - 38px); }
      .bar, .playbar { gap: 6px; padding: 6px; }
      input, select, button { height: 30px; padding: 3px 6px; }
      label { font-size: 10px; }
      #status { font-size: 11px; padding: 0 6px 6px; }
    }
  </style>
</head>
<body>
  <header>EgoVerse Handpose QA</header>
  <main>
    <div class="bar">
      <div><label>Embodiment</label><select id="embodiment"><option value="aria">aria</option><option value="mecka">mecka</option><option value="scale">scale</option><option value="eva">eva</option><option value="">all</option></select></div>
      <div><label>Task</label><select id="task"><option value="">All tasks</option></select></div>
      <div><label>Search</label><input id="search" placeholder="task, episode hash, lab"></div>
      <button id="load">Load</button>
      <button id="prevClip" class="secondary">Prev Clip</button>
      <button id="nextClip">Next Clip</button>
      <button id="nextTask">Next Task</button>
    </div>
    <div id="viewerWrap">
      <div class="pane"><img id="viewer" alt="frame"></div>
      <div class="pane pane3d">
        <div class="viewBtns">
          <button id="viewHome">Home</button>
          <button id="viewFront">Angle</button>
          <button id="viewSide">Side</button>
          <button id="viewTop">Top</button>
        </div>
        <canvas id="viewer3d"></canvas>
      </div>
    </div>
    <div>
      <div class="playbar">
        <button id="play">Play</button>
        <button id="prevFrame" class="secondary">Prev Frame</button>
        <button id="nextFrame" class="secondary">Next Frame</button>
        <div><label>Frame</label><input id="frame" type="range" min="0" max="0" value="30"></div>
        <div><label>Dot</label><input id="dot" type="number" min="1" max="8" value="2"></div>
        <div><label>Lag</label><input id="kpOffset" type="number" min="-10" max="10" value="0"></div>
      </div>
      <div id="status"></div>
    </div>
  </main>
<script>
let episodes = [];
let current = null;
let clipIndex = 0;
let playing = false;
let playTimer = null;
let imageUrl = null;
let requestSeq = 0;
let pose3d = null;
let yaw3d = 0;
let pitch3d = 0;
let zoom3d = 1;
let drag3d = null;
const MAX_AUTO_SKIP = 8;
const $ = (id) => document.getElementById(id);
function setStatus(s) { $("status").textContent = s; }
function selectedEpisode() { return episodes[clipIndex]?.episode_hash; }
function currentLayout() {
  const embodiment = String(current?.embodiment || "");
  return embodiment.startsWith("aria") ? "raw-aria" : "canonical";
}
function setBusy(isBusy) {
  for (const id of ["load", "prevClip", "nextClip", "nextTask", "prevFrame", "nextFrame", "play"]) {
    $(id).disabled = isBusy;
  }
}
function prefetchAround(index) {
  for (const next of [index + 1, index + 2]) {
    const hash = episodes[next]?.episode_hash;
    if (hash) fetch("/api/prefetch?" + new URLSearchParams({episode_hash:hash}).toString()).catch(() => {});
  }
}
async function loadTasks() {
  const currentTask = $("task").value;
  const p = new URLSearchParams({embodiment:$("embodiment").value, q:$("search").value});
  const res = await fetch("/api/tasks?" + p.toString());
  const tasks = await res.json();
  $("task").innerHTML = '<option value="">All tasks</option>';
  for (const t of tasks) {
    const opt = document.createElement("option");
    opt.value = t.task;
    opt.textContent = `${t.task} (${t.count})`;
    $("task").appendChild(opt);
  }
  if ([...$("task").options].some(o => o.value === currentTask)) $("task").value = currentTask;
}
async function loadEpisodes() {
  const seq = ++requestSeq;
  stopPlay();
  setBusy(true);
  setStatus("Loading clips...");
  const p = new URLSearchParams({embodiment:$("embodiment").value, task:$("task").value, limit:"500", offset:"0", q:$("search").value});
  try {
    const res = await fetch("/api/episodes?" + p.toString());
    if (seq !== requestSeq) return;
    episodes = await res.json();
    clipIndex = 0;
    if (!episodes.length) { setStatus("No clips."); return; }
    await loadClip(1, seq);
  } finally {
    if (seq === requestSeq) setBusy(false);
  }
}
async function loadClip(direction = 0, seq = ++requestSeq) {
  setBusy(true);
  const hash = selectedEpisode();
  if (!hash) { setBusy(false); return; }
  const startIndex = clipIndex;
  let skipped = 0;
  try {
    while (seq === requestSeq) {
      current = episodes[clipIndex];
      const total = Math.max(0, Math.round(Number(current.num_frames || 0)) - 1);
      $("frame").max = total;
      $("frame").value = Math.min(30, total);
      setStatus(`Loading ${clipIndex + 1}/${episodes.length} | ${current.task} | ${current.episode_hash}`);
      const loaded = await loadFrame(seq);
      if (loaded.ok) {
        prefetchAround(clipIndex);
        return;
      }
      if (!direction) return;
      skipped += 1;
      if (skipped >= MAX_AUTO_SKIP) {
        setStatus(`Stopped after ${skipped} failed clips. Last error: ${loaded.error || "frame failed"}. Click Next Clip to keep scanning.`);
        return;
      }
      const nextIndex = clipIndex + direction;
      if (nextIndex < 0 || nextIndex >= episodes.length || nextIndex === startIndex) {
        setStatus(`No renderable clip found for ${$("task").value || "current filter"}.`);
        return;
      }
      clipIndex = nextIndex;
    }
  } finally {
    if (seq === requestSeq) setBusy(false);
  }
}
async function loadFrame(seq = requestSeq) {
  const hash = selectedEpisode();
  if (!hash) return {ok:false, error:"no selected clip"};
  const frame = $("frame").value;
  const kpOffset = $("kpOffset").value;
  setStatus(`${clipIndex + 1}/${episodes.length} | ${current.task} | ${hash} | frame ${frame}/${$("frame").max}`);
  const p = new URLSearchParams({episode_hash:hash, frame, kp_offset:kpOffset, layout:currentLayout(), dot:$("dot").value, line:"1", t:Date.now()});
  let res;
  try {
    res = await fetch("/api/frame?" + p.toString());
  } catch (err) {
    const message = String(err);
    setStatus(`${clipIndex + 1}/${episodes.length} | ${current.task} | ${hash} | ${message}`);
    stopPlay();
    return {ok:false, error:message};
  }
  if (seq !== requestSeq) return {ok:false, error:"stale request"};
  if (!res.ok) {
    let message = `Frame failed (${res.status})`;
    try {
      const err = await res.json();
      if (err.error) message = err.error;
    } catch (_) {}
    setStatus(`${clipIndex + 1}/${episodes.length} | ${current.task} | ${hash} | ${message}`);
    stopPlay();
    return {ok:false, error:message};
  }
  const blob = await res.blob();
  if (seq !== requestSeq) return {ok:false, error:"stale request"};
  if (imageUrl) URL.revokeObjectURL(imageUrl);
  imageUrl = URL.createObjectURL(blob);
  $("viewer").src = imageUrl;
  loadPose(seq);
  return {ok:true};
}
async function loadPose(seq = requestSeq) {
  const hash = selectedEpisode();
  if (!hash) return;
  const p = new URLSearchParams({episode_hash:hash, frame:$("frame").value, kp_offset:$("kpOffset").value, layout:currentLayout(), t:Date.now()});
  try {
    const res = await fetch("/api/pose?" + p.toString());
    if (seq !== requestSeq || !res.ok) return;
    pose3d = await res.json();
    draw3d();
  } catch (_) {}
}
function cleanPoints(points) {
  return (points || []).filter(p => p.length === 3 && p.every(Number.isFinite) && Math.hypot(p[0], p[1], p[2]) > 1e-9);
}
function isValidPoint(p) {
  return p && p.length === 3 && p.every(Number.isFinite) && Math.hypot(p[0], p[1], p[2]) > 1e-9;
}
function toViewPoint(p, center) {
  let x = p[0] - center[0];
  let y = p[1] - center[1];
  let z = p[2] - center[2];
  const cy = Math.cos(yaw3d), sy = Math.sin(yaw3d);
  const cp = Math.cos(pitch3d), sp = Math.sin(pitch3d);
  const x1 = cy * x + sy * z;
  const z1 = -sy * x + cy * z;
  const y1 = cp * y - sp * z1;
  const z2 = sp * y + cp * z1;
  return [x1, y1, z2];
}
function screenPoint3d(p, center, scale, ox, oy) {
  const v = toViewPoint(p, center);
  return [ox + v[0] * scale, oy + v[1] * scale, v[2]];
}
function drawLine3d(ctx, a, b, center, color, width, scale, ox, oy) {
  const av = screenPoint3d(a, center, scale, ox, oy);
  const bv = screenPoint3d(b, center, scale, ox, oy);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(av[0], av[1]);
  ctx.lineTo(bv[0], bv[1]);
  ctx.stroke();
}
function drawPoint3d(ctx, p, center, color, radius, scale, ox, oy) {
  const v = screenPoint3d(p, center, scale, ox, oy);
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(v[0], v[1], radius, 0, Math.PI * 2);
  ctx.fill();
}
function drawText3d(ctx, p, center, text, color, scale, ox, oy) {
  if (!isValidPoint(p)) return;
  const v = screenPoint3d(p, center, scale, ox, oy);
  ctx.fillStyle = color;
  ctx.font = "700 13px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
  ctx.fillText(text, v[0] + 7, v[1] - 7);
}
function centroid3d(points) {
  const valid = cleanPoints(points);
  if (!valid.length) return null;
  return valid.reduce((a, p) => [a[0]+p[0], a[1]+p[1], a[2]+p[2]], [0,0,0]).map(v => v / valid.length);
}
function fitScale3d(points, rect) {
  if (!points.length) return 420 * zoom3d;
  const xs = points.map(p => p[0]);
  const ys = points.map(p => p[1]);
  const zs = points.map(p => p[2]);
  const span = Math.max(
    Math.max(...xs) - Math.min(...xs),
    Math.max(...ys) - Math.min(...ys),
    Math.max(...zs) - Math.min(...zs),
    0.08
  );
  const base = Math.min(rect.width, rect.height) * 0.46 / span;
  return Math.max(220, Math.min(1250, base)) * zoom3d;
}
function drawHand3d(ctx, points, center, dotColor, scale, ox, oy) {
  const edges = pose3d?.edges || [];
  const fingerColors = {
    thumb: "#ff6464",
    index: "#64ff64",
    middle: "#64aaff",
    ring: "#ffe650",
    pinky: "#ff64ff",
  };
  const ranges = pose3d?.edge_ranges || [];
  if (ranges.length) {
    for (const [finger, start, end] of ranges) {
      const boneColor = fingerColors[finger] || "#e2e8f0";
      for (let edgeIdx = start; edgeIdx < end; edgeIdx++) {
        const edge = edges[edgeIdx];
        if (!edge) continue;
        const [i, j] = edge;
        if (isValidPoint(points[i]) && isValidPoint(points[j])) drawLine3d(ctx, points[i], points[j], center, boneColor, 2.2, scale, ox, oy);
      }
    }
  } else {
    for (const [i, j] of edges) {
      if (isValidPoint(points[i]) && isValidPoint(points[j])) drawLine3d(ctx, points[i], points[j], center, "#e2e8f0", 2.2, scale, ox, oy);
    }
  }
  for (const p of points) {
    if (isValidPoint(p)) drawPoint3d(ctx, p, center, dotColor, 3, scale, ox, oy);
  }
}
function draw3d() {
  const canvas = $("viewer3d");
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);
  ctx.fillStyle = "#071018";
  ctx.fillRect(0, 0, rect.width, rect.height);
  if (!pose3d) {
    ctx.fillStyle = "#94a3b8";
    ctx.fillText("3D pose", 16, 24);
    return;
  }
  const left = cleanPoints(pose3d.left);
  const right = cleanPoints(pose3d.right);
  const all = left.concat(right);
  const center = all.length
    ? all.reduce((a, p) => [a[0]+p[0], a[1]+p[1], a[2]+p[2]], [0,0,0]).map(v => v / all.length)
    : [0,0,0.55];
  const orbitScale = fitScale3d(all, rect);
  const ox = rect.width / 2;
  const oy = rect.height / 2;
  const gridHalf = 0.18;
  const gridStep = 0.06;
  const gridY = center[1];
  for (let i = -3; i <= 3; i++) {
    const d = i * gridStep;
    drawLine3d(ctx, [center[0]-gridHalf, gridY, center[2]+d], [center[0]+gridHalf, gridY, center[2]+d], center, "#132235", i === 0 ? 1.4 : 1, orbitScale, ox, oy);
    drawLine3d(ctx, [center[0]+d, gridY, center[2]-gridHalf], [center[0]+d, gridY, center[2]+gridHalf], center, "#132235", i === 0 ? 1.4 : 1, orbitScale, ox, oy);
  }

  drawLine3d(ctx, [center[0],center[1],center[2]], [center[0]+0.08,center[1],center[2]], center, "#ef4444", 2, orbitScale, ox, oy);
  drawLine3d(ctx, [center[0],center[1],center[2]], [center[0],center[1]+0.08,center[2]], center, "#22c55e", 2, orbitScale, ox, oy);
  drawLine3d(ctx, [center[0],center[1],center[2]], [center[0],center[1],center[2]+0.08], center, "#38bdf8", 2, orbitScale, ox, oy);
  drawHand3d(ctx, pose3d.left || [], center, "#38bdf8", orbitScale, ox, oy);
  drawHand3d(ctx, pose3d.right || [], center, "#f97316", orbitScale, ox, oy);
  drawText3d(ctx, centroid3d(pose3d.left), center, "L", "#7dd3fc", orbitScale, ox, oy);
  drawText3d(ctx, centroid3d(pose3d.right), center, "R", "#fdba74", orbitScale, ox, oy);
  ctx.fillStyle = "#cbd5e1";
  ctx.font = "12px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
  ctx.fillText("3D handpose only | camera frame", 12, 22);
  ctx.fillStyle = "#94a3b8";
  ctx.fillText("X right | Y down | Z forward", 12, 40);
}
function set3dView(name) {
  const presets = {
    home: [0, 0],
    front: [-0.65, -0.35],
    side: [Math.PI / 2, 0],
    top: [0, -1.35],
  };
  const preset = presets[name] || presets.home;
  yaw3d = preset[0];
  pitch3d = preset[1];
  zoom3d = 1;
  draw3d();
}
function setup3dControls() {
  const canvas = $("viewer3d");
  $("viewHome").onclick = () => set3dView("home");
  $("viewFront").onclick = () => set3dView("front");
  $("viewSide").onclick = () => set3dView("side");
  $("viewTop").onclick = () => set3dView("top");
  canvas.addEventListener("mousedown", (e) => { drag3d = {x:e.clientX, y:e.clientY}; });
  window.addEventListener("mouseup", () => { drag3d = null; });
  window.addEventListener("mousemove", (e) => {
    if (!drag3d) return;
    yaw3d += (e.clientX - drag3d.x) * 0.01;
    pitch3d = Math.max(-1.45, Math.min(1.45, pitch3d + (e.clientY - drag3d.y) * 0.01));
    drag3d = {x:e.clientX, y:e.clientY};
    draw3d();
  });
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    zoom3d = Math.max(0.25, Math.min(4, zoom3d * (e.deltaY > 0 ? 0.9 : 1.1)));
    draw3d();
  }, {passive:false});
  canvas.addEventListener("dblclick", () => {
    set3dView("home");
  });
  window.addEventListener("resize", draw3d);
}
function stepFrame(delta) {
  requestSeq++;
  const next = Math.max(0, Math.min(Number($("frame").max), Number($("frame").value) + delta));
  $("frame").value = next;
  loadFrame(requestSeq);
}
function stopPlay() {
  playing = false;
  $("play").textContent = "Play";
  if (playTimer) clearTimeout(playTimer);
  playTimer = null;
}
function startPlay() {
  playing = true;
  $("play").textContent = "Pause";
  stepFrame(1);
}
$("viewer").onload = () => {
  if (!playing) return;
  if (Number($("frame").value) >= Number($("frame").max)) stopPlay();
  else playTimer = setTimeout(() => stepFrame(1), 40);
};
$("viewer").onerror = () => { setStatus("Frame failed; see terminal for details."); stopPlay(); };
$("load").onclick = async () => { await loadTasks(); await loadEpisodes(); };
$("embodiment").onchange = async () => { await loadTasks(); await loadEpisodes(); };
$("task").onchange = loadEpisodes;
$("prevClip").onclick = async () => { stopPlay(); clipIndex = Math.max(0, clipIndex - 1); await loadClip(-1); };
$("nextClip").onclick = async () => { stopPlay(); clipIndex = Math.min(episodes.length - 1, clipIndex + 1); await loadClip(1); };
$("nextTask").onclick = async () => {
  const task = $("task");
  task.selectedIndex = task.options.length ? (task.selectedIndex + 1) % task.options.length : 0;
  if (task.value === "" && task.options.length > 1) task.selectedIndex = 1;
  await loadEpisodes();
};
$("frame").onchange = () => { requestSeq++; loadFrame(requestSeq); };
$("dot").onchange = () => { requestSeq++; loadFrame(requestSeq); };
$("kpOffset").onchange = () => { requestSeq++; loadFrame(requestSeq); };
$("prevFrame").onclick = () => { stopPlay(); stepFrame(-1); };
$("nextFrame").onclick = () => { stopPlay(); stepFrame(1); };
$("play").onclick = () => { playing ? stopPlay() : startPlay(); };
setup3dControls();
loadTasks().then(loadEpisodes);
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    cache_dir: Path = DEFAULT_CACHE_DIR

    def send_json(self, obj, status=HTTPStatus.OK):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        qs = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        try:
            if parsed.path == "/":
                body = html_page()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif parsed.path == "/favicon.ico":
                self.send_response(HTTPStatus.NO_CONTENT)
                self.end_headers()
            elif parsed.path == "/api/episodes":
                rows = filter_rows(episode_df_json(), qs)
                limit = int(qs.get("limit", "200"))
                offset = max(0, int(qs.get("offset", "0")))
                self.send_json(rows[offset : offset + limit])
            elif parsed.path == "/api/tasks":
                task_qs = {k: v for k, v in qs.items() if k != "task"}
                rows = filter_rows(episode_df_json(), task_qs)
                counts: dict[str, int] = {}
                for row in rows:
                    task = str(row.get("task", "")).strip()
                    if task:
                        counts[task] = counts.get(task, 0) + 1
                tasks = [{"task": task, "count": count} for task, count in sorted(counts.items())]
                self.send_json(tasks)
            elif parsed.path == "/api/prefetch":
                prefetch_episode(self.cache_dir, qs["episode_hash"])
                self.send_json({"ok": True})
            elif parsed.path == "/api/pose":
                self.send_json(
                    pose_json(
                        self.cache_dir,
                        qs["episode_hash"],
                        int(qs.get("frame", "0")),
                        int(qs.get("kp_offset", "0")),
                        qs.get("layout", "raw-aria"),
                    )
                )
            elif parsed.path == "/api/frame":
                body = render_frame(
                    self.cache_dir,
                    qs["episode_hash"],
                    int(qs.get("frame", "0")),
                    int(qs.get("kp_offset", "0")),
                    qs.get("layout", "raw-aria"),
                    max(1, int(qs.get("dot", "2"))),
                    max(1, int(qs.get("line", "1"))),
                )
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    load_env(required=True)
    Handler.cache_dir = cache_dir
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"EgoVerse handpose viewer: {url}")
    print(f"Cache dir: {cache_dir}")
    if args.open:
        webbrowser.open(url)
    server.serve_forever()


if __name__ == "__main__":
    main()
