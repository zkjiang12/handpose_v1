#!/usr/bin/env python3
"""Visualize partial all-available v2 hand triangulation results."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


EGO_EXO_ROOT = Path("/Users/zikangjiang/dev/ego-exo")
CAMERA_IMAGE_WARP_GRID_COLS = 256
CAMERA_FRAME_JPEG_QUALITY = 100
K_DEFAULT = np.array(
    [
        [699.19397931, 0.0, 976.75087121],
        [0.0, 699.60395977, 565.79050329],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
D_DEFAULT = np.array(
    [-0.01803320, 0.06173989, -0.05266772, 0.01903308],
    dtype=np.float64,
).reshape(4, 1)

REPORT_BY_SEGMENT = {
    "segment1": Path("/Users/zikangjiang/dev/ego-exo/visualizations/v2/take1/v2_calibration_and_audio_sync_report.json"),
    "segment2": Path("/Users/zikangjiang/dev/ego-exo/visualizations/v2/take2/v2_calibration_and_audio_sync_report.json"),
}

FINGER_COLORS = {
    "thumb": "#ef4444",
    "index": "#22c55e",
    "middle": "#3b82f6",
    "ring": "#f59e0b",
    "pinky": "#8b5cf6",
    "palm": "#334155",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("all_available_json", type=Path)
    parser.add_argument("--labels-json", type=Path)
    parser.add_argument("--bone-lengths-json", type=Path)
    parser.add_argument("--combo-results-json", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--live-out-dir", type=Path)
    return parser.parse_args()


def resolve_source(path_text: str, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    path = Path(path_text)
    if path.exists():
        return path
    repo_candidate = Path("/Users/zikangjiang/dev/handpose_v1") / path
    if repo_candidate.exists():
        return repo_candidate
    live_candidate = Path("/Users/zikangjiang/dev/ego-exo/visualizations/v2/handpose_labeler/results") / path.name
    if live_candidate.exists():
        return live_candidate
    raise FileNotFoundError(path_text)


def bone_name(a: int, b: int, names: list[str]) -> str:
    start = names[a]
    end = names[b]
    for prefix in ("thumb", "index", "middle", "ring", "pinky"):
        repeated = f"{prefix}_"
        if start.startswith(repeated) and end.startswith(repeated):
            return f"{start}_{end[len(repeated):]}"
    return f"{start}_{end}"


def edge_color(a: int, b: int, names: list[str]) -> str:
    joined = f"{names[a]}_{names[b]}"
    for finger, color in FINGER_COLORS.items():
        if finger in joined:
            return color
    return FINGER_COLORS["palm"]


def world_to_z_up(xyz: np.ndarray) -> np.ndarray:
    return np.array([xyz[0], -xyz[1], -xyz[2]], dtype=float)


def camera_center_world(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    r = cv2.Rodrigues(rvec)[0]
    return (-r.T @ tvec.reshape(3, 1)).reshape(3)


def camera_frustum_world(rvec: np.ndarray, tvec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = cv2.Rodrigues(rvec)[0]
    center = camera_center_world(rvec, tvec)
    corners = np.array([[0, 0], [1919, 0], [1919, 1079], [0, 1079]], dtype=np.float64).reshape(-1, 1, 2)
    normalized = cv2.fisheye.undistortPoints(corners, K_DEFAULT, D_DEFAULT).reshape(-1, 2)
    dirs_cam = np.column_stack([normalized, np.ones(len(normalized))])
    dirs_cam /= np.linalg.norm(dirs_cam, axis=1, keepdims=True)
    dirs_world = (r.T @ dirs_cam.T).T
    optical_axis_world = r.T @ np.array([0.0, 0.0, 1.0])
    optical_axis_world /= np.linalg.norm(optical_axis_world)
    return center, np.vstack([dirs_world, optical_axis_world])


def load_calibration_report(labels: dict) -> dict:
    segment_ids = {cam["segment_id"] for cam in labels["cameras"].values() if cam.get("segment_id")}
    if len(segment_ids) != 1:
        raise ValueError(f"Expected one segment id, got {sorted(segment_ids)}")
    return json.loads(REPORT_BY_SEGMENT[next(iter(segment_ids))].read_text())


def save_camera_keypoint_overlays(
    out_dir: Path,
    labels: dict,
    hand: str,
    keypoint_names: list[str],
    edges: list[list[int]],
    stem: str,
) -> dict[str, dict[str, str | float | int]]:
    out = {}
    colors = {
        "left": (40, 40, 255),
        "right": (255, 90, 30),
    }
    line_color = colors.get(hand, (40, 220, 255))
    for cam, info in sorted(labels["cameras"].items()):
        video_url = info.get("video_url")
        if not video_url:
            continue
        video_path = EGO_EXO_ROOT / video_url.lstrip("/")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            continue
        frame_index = int(info.get("frame_index_30fps") or 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok and info.get("local_time_s") is not None:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(info["local_time_s"]) * 1000.0)
            ok, frame = cap.read()
        cap.release()
        if not ok:
            continue

        points: dict[int, tuple[int, int]] = {}
        for kp_s, cams in labels["hands"].get(hand, {}).items():
            if cam not in cams:
                continue
            x, y = cams[cam]
            points[int(kp_s)] = (int(round(x)), int(round(y)))

        for a, b in edges:
            if a not in points or b not in points:
                continue
            cv2.line(frame, points[a], points[b], line_color, 3, cv2.LINE_AA)
        for idx, point in points.items():
            cv2.circle(frame, point, 7, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, point, 4, line_color, -1, cv2.LINE_AA)
        filename = f"{stem}_{cam}_raw_frame_keypoints.jpg"
        out_path = out_dir / filename
        cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), CAMERA_FRAME_JPEG_QUALITY])
        out[cam] = {
            "path": str(out_path),
            "filename": filename,
            "local_time_s": float(info.get("local_time_s") or 0.0),
            "frame_index_30fps": frame_index,
            "num_keypoints": len(points),
        }
    return out


def camera_plane_points(
    rvec: np.ndarray,
    tvec: np.ndarray,
    uv: np.ndarray,
    plane_distance_mm: float,
) -> np.ndarray:
    r = cv2.Rodrigues(rvec)[0]
    uv_arr = np.asarray(uv, dtype=np.float64).reshape(-1, 1, 2)
    xy = cv2.fisheye.undistortPoints(uv_arr, K_DEFAULT, D_DEFAULT).reshape(-1, 2)
    pts_cam = np.column_stack(
        [
            xy[:, 0] * plane_distance_mm,
            xy[:, 1] * plane_distance_mm,
            np.full(len(xy), plane_distance_mm),
        ]
    )
    pts_world = (r.T @ (pts_cam.T - tvec.reshape(3, 1))).T
    return np.stack([world_to_z_up(point) for point in pts_world])


def camera_ray_display(
    rvec: np.ndarray,
    tvec: np.ndarray,
    uv: list[float] | tuple[float, float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r = cv2.Rodrigues(rvec)[0]
    xy = cv2.fisheye.undistortPoints(np.asarray(uv, dtype=np.float64).reshape(1, 1, 2), K_DEFAULT, D_DEFAULT).reshape(2)
    direction_cam = np.array([xy[0], xy[1], 1.0], dtype=float)
    direction_cam /= np.linalg.norm(direction_cam)
    direction_world = r.T @ direction_cam
    direction_display = world_to_z_up(direction_world)
    direction_display /= np.linalg.norm(direction_display)
    return world_to_z_up(camera_center_world(rvec, tvec)), direction_display


def reconstruct_gt_partial(
    tri_points: dict[str, np.ndarray],
    keypoint_names: list[str],
    edges: list[list[int]],
    gt_lengths: dict[str, float],
) -> dict[str, np.ndarray]:
    if "wrist" not in tri_points:
        raise ValueError("Cannot build GT-length pose without a triangulated wrist")
    model = {"wrist": tri_points["wrist"].copy()}
    pending = [(a, b) for a, b in edges]
    while pending:
        next_pending = []
        progressed = False
        for a, b in pending:
            start = keypoint_names[a]
            end = keypoint_names[b]
            name = bone_name(a, b, keypoint_names)
            if start not in model or start not in tri_points or end not in tri_points or name not in gt_lengths:
                next_pending.append((a, b))
                continue
            direction = tri_points[end] - tri_points[start]
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-9:
                next_pending.append((a, b))
                continue
            model[end] = model[start] + direction / norm * float(gt_lengths[name])
            progressed = True
        if not progressed:
            break
        pending = next_pending
    return model


def project_hand_local(
    tri_points: dict[str, np.ndarray],
    gt_points: dict[str, np.ndarray],
    keypoint_names: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    names = [name for name in keypoint_names if name in tri_points and name in gt_points]
    wrist = tri_points["wrist"]
    tri_arr = np.stack([tri_points[name] - wrist for name in names])
    gt_arr = np.stack([gt_points[name] - wrist for name in names])
    combined = np.vstack([tri_arr, gt_arr])
    _, _, vt = np.linalg.svd(combined - combined.mean(axis=0), full_matrices=False)
    basis = vt[:2].T
    tri_2d = tri_arr @ basis
    gt_2d = gt_arr @ basis

    tip_names = [name for name in ("thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip") if name in names]
    if tip_names:
        tip_indices = [names.index(name) for name in tip_names]
        if tri_2d[tip_indices, 0].mean() < 0:
            tri_2d[:, 0] *= -1
            gt_2d[:, 0] *= -1
    if "thumb_tip" in names and "pinky_tip" in names and tri_2d[names.index("thumb_tip"), 1] < tri_2d[names.index("pinky_tip"), 1]:
        tri_2d[:, 1] *= -1
        gt_2d[:, 1] *= -1
    return ({name: tri_2d[i] for i, name in enumerate(names)}, {name: gt_2d[i] for i, name in enumerate(names)})


def save_2d_comparison(
    path: Path,
    tri_2d: dict[str, np.ndarray],
    gt_2d: dict[str, np.ndarray],
    joint_deltas: dict[str, float],
    keypoint_names: list[str],
    edges: list[list[int]],
    hand: str,
    session_time_s: float,
    missing: list[dict],
) -> None:
    fig, ax = plt.subplots(figsize=(11, 9), constrained_layout=True)
    for a, b in edges:
        start = keypoint_names[a]
        end = keypoint_names[b]
        if start not in tri_2d or end not in tri_2d:
            continue
        gt_a, gt_b = gt_2d[start], gt_2d[end]
        tri_a, tri_b = tri_2d[start], tri_2d[end]
        ax.plot([gt_a[0], gt_b[0]], [gt_a[1], gt_b[1]], color="#2563eb", linewidth=6, alpha=0.25)
        ax.plot([tri_a[0], tri_b[0]], [tri_a[1], tri_b[1]], color=edge_color(a, b, keypoint_names), linewidth=2.8)

    worst = {name for name, _ in sorted(joint_deltas.items(), key=lambda item: item[1], reverse=True)[:7]}
    for i, name in enumerate(keypoint_names):
        if name not in tri_2d:
            continue
        gt = gt_2d[name]
        tri = tri_2d[name]
        ax.plot([gt[0], tri[0]], [gt[1], tri[1]], color="#64748b", linewidth=1.2, linestyle="--", alpha=0.75)
        ax.scatter([gt[0]], [gt[1]], s=54, color="#2563eb", edgecolor="white", linewidth=0.8, zorder=4)
        ax.scatter([tri[0]], [tri[1]], s=42, color="#f97316", edgecolor="#111827", linewidth=0.5, zorder=5)
        label = str(i)
        if name in worst:
            label = f"{i}\n{joint_deltas[name]:.1f}mm"
        ax.text(tri[0] + 2.0, tri[1] + 2.0, label, fontsize=8, color="#111827")

    vals = np.array(list(joint_deltas.values()), dtype=float)
    missing_text = ", ".join(f"{item['index']} {item['name']}" for item in missing) or "none"
    summary = (
        f"{hand} hand @ {session_time_s:.3f}s\n"
        f"partial all-available triangulation\n"
        f"joint delta: median {np.median(vals):.1f} mm, mean {np.mean(vals):.1f} mm\n"
        f"missing: {missing_text}"
    )
    ax.text(
        0.02,
        0.98,
        summary,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.92},
    )
    ax.scatter([], [], s=60, color="#2563eb", label="GT measured lengths, same directions")
    ax.scatter([], [], s=50, color="#f97316", label="triangulated pose")
    ax.plot([], [], color="#64748b", linestyle="--", label="same-joint delta")
    ax.set_title("2D pose comparison in hand-local metric plane")
    ax.set_xlabel("hand-local axis 1 (mm)")
    ax.set_ylabel("hand-local axis 2 (mm)")
    ax.grid(color="#d6dbe1", alpha=0.65)
    ax.axis("equal")
    ax.legend(loc="lower right")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def camera_image_grid_geometry(
    rvec: np.ndarray,
    tvec: np.ndarray,
    width: int,
    height: int,
    plane_distance_mm: float = 220.0,
    grid_cols: int = CAMERA_IMAGE_WARP_GRID_COLS,
) -> dict[str, list]:
    grid_rows = max(2, int(round(grid_cols * height / width)))
    xs = np.linspace(0, width - 1, grid_cols)
    ys = np.linspace(0, height - 1, grid_rows)
    grid_x, grid_y = np.meshgrid(xs, ys)
    uv_grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    pts = camera_plane_points(rvec, tvec, uv_grid, plane_distance_mm)
    uv = np.column_stack([uv_grid[:, 0] / (width - 1), 1.0 - uv_grid[:, 1] / (height - 1)])

    indices = []
    for row in range(grid_rows - 1):
        for col in range(grid_cols - 1):
            v00 = row * grid_cols + col
            v01 = v00 + 1
            v10 = (row + 1) * grid_cols + col
            v11 = v10 + 1
            indices.extend([v00, v10, v11, v00, v11, v01])
    return {
        "vertices": np.round(pts, 4).tolist(),
        "uvs": np.round(uv, 6).tolist(),
        "indices": indices,
        "gridCols": grid_cols,
        "gridRows": grid_rows,
    }


def make_three_line(points: list[np.ndarray], color: str, opacity: float = 1.0) -> dict:
    return {
        "points": [np.round(point, 4).tolist() for point in points],
        "color": color,
        "opacity": opacity,
    }


def save_interactive_scene(
    path: Path,
    all_available: dict,
    labels: dict,
    calibration_report: dict,
    keypoint_names: list[str],
    edges: list[list[int]],
    camera_overlays: dict[str, dict[str, str | float | int]],
) -> None:
    hand = all_available["hand"]
    tri_points = {name: np.array(xyz, dtype=float) for name, xyz in all_available["points_mm"].items()}
    display_points = {name: world_to_z_up(xyz) for name, xyz in tri_points.items()}

    extrinsics = calibration_report["extrinsics"]
    board = extrinsics["board"]
    board_w = board["cols"] * board["square_mm"]
    board_h = board["rows"] * board["square_mm"]
    board_raw = np.array([[0, 0, 0], [board_w, 0, 0], [board_w, board_h, 0], [0, board_h, 0]], dtype=float)
    board_display = np.stack([world_to_z_up(p) for p in board_raw])
    board_lines = [
        make_three_line([board_display[i], board_display[(i + 1) % 4]], "#94a3b8", 0.75)
        for i in range(4)
    ]
    board_surface = np.round(board_display, 4).tolist()

    hand_edges = []
    for a, b in edges:
        start = keypoint_names[a]
        end = keypoint_names[b]
        if start not in display_points or end not in display_points:
            continue
        hand_edges.append(
            {
                **make_three_line(
                    [display_points[start], display_points[end]],
                    edge_color(a, b, keypoint_names),
                    1.0,
                ),
                "width": 1.0 if a != 0 else 1.4,
            }
        )

    hand_points = [
        {
            "index": keypoint_names.index(name),
            "name": name,
            "position": np.round(display_points[name], 4).tolist(),
            "radius": 6.0 if name == "wrist" else 4.2,
        }
        for name in keypoint_names
        if name in display_points
    ]

    camera_colors = {"cam1": "#dc2626", "cam2": "#2563eb", "cam3": "#16a34a", "cam4": "#9333ea"}
    cameras = {}
    bounds_points = [*display_points.values(), *board_display]
    for cam, info in sorted(extrinsics["cameras"].items()):
        rvec = np.array(info["rvec_world_to_cam"], dtype=float).reshape(3, 1)
        tvec = np.array(info["t_world_to_cam_mm"], dtype=float).reshape(3, 1)
        center_raw, dirs_raw = camera_frustum_world(rvec, tvec)
        center = world_to_z_up(center_raw)
        dirs = np.array([world_to_z_up(d) for d in dirs_raw])
        color = camera_colors.get(cam, "#64748b")
        optical_end = center + 240.0 * dirs[4]
        bounds_points.extend([center, optical_end])

        overlay = camera_overlays.get(cam)
        image = None
        projected_points = []
        projected_edges = []
        if overlay:
            overlay_path = Path(str(overlay["path"]))
            image_bgr = cv2.imread(str(overlay_path))
            if image_bgr is not None:
                height, width = image_bgr.shape[:2]
            else:
                width, height = labels.get("image_size", [1920, 1080])
            image = {
                "filename": overlay["filename"],
                "width": int(width),
                "height": int(height),
                "geometry": camera_image_grid_geometry(rvec, tvec, int(width), int(height)),
            }
            bounds_points.extend(np.array(image["geometry"]["vertices"], dtype=float))

            kp_uv = []
            kp_indices = []
            for kp_s, cams in labels["hands"].get(hand, {}).items():
                if cam not in cams:
                    continue
                kp_indices.append(int(kp_s))
                kp_uv.append(cams[cam])
            if kp_uv:
                kp_pts = camera_plane_points(rvec, tvec, np.array(kp_uv, dtype=np.float64), 217.0)
                by_index = {idx: kp_pts[pos] for pos, idx in enumerate(kp_indices)}
                for idx, point in by_index.items():
                    projected_points.append(
                        {
                            "index": idx,
                            "name": keypoint_names[idx],
                            "position": np.round(point, 4).tolist(),
                        }
                    )
                    bounds_points.append(point)
                for a, b in edges:
                    if a not in by_index or b not in by_index:
                        continue
                    projected_edges.append(make_three_line([by_index[a], by_index[b]], "#facc15", 1.0))

        rays = []
        tri_display = {
            name: world_to_z_up(np.array(xyz, dtype=float))
            for name, xyz in all_available["points_mm"].items()
        }
        for kp_s, cams in sorted(labels["hands"].get(hand, {}).items(), key=lambda item: int(item[0])):
            if cam not in cams:
                continue
            kp_index = int(kp_s)
            kp_name = keypoint_names[kp_index]
            origin, direction = camera_ray_display(rvec, tvec, cams[cam])
            target = tri_display.get(kp_name)
            residual_mm = None
            if target is not None:
                depth = float(np.dot(target - origin, direction))
                if depth <= 0:
                    depth = 900.0
                end = origin + direction * max(260.0, depth * 1.08)
                residual_mm = float(np.linalg.norm(origin + direction * depth - target))
            else:
                end = origin + direction * 900.0
            rays.append(
                {
                    "joint": kp_index,
                    "name": kp_name,
                    **make_three_line([origin, end], color, 0.38),
                    "residualMm": None if residual_mm is None else round(residual_mm, 2),
                }
            )
            bounds_points.extend([origin, end])

        cameras[cam] = {
            "color": color,
            "center": np.round(center, 4).tolist(),
            "opticalAxis": make_three_line([center, optical_end], color, 0.9),
            "image": image,
            "projectedPoints": projected_points,
            "projectedEdges": projected_edges,
            "rays": rays,
        }

    bounds_arr = np.stack(bounds_points)
    bounds_min = bounds_arr.min(axis=0)
    bounds_max = bounds_arr.max(axis=0)
    scene_center = (bounds_min + bounds_max) / 2.0
    scene_radius = float(np.linalg.norm(bounds_max - bounds_min) / 2.0)
    hand_arr = np.stack(list(display_points.values()))
    hand_center = hand_arr.mean(axis=0)

    scene_data = {
        "hand": hand,
        "sessionTimeS": float(all_available["session_time_s"]),
        "bounds": {
            "center": np.round(scene_center, 4).tolist(),
            "radius": round(scene_radius, 4),
            "handCenter": np.round(hand_center, 4).tolist(),
        },
        "board": {"surface": board_surface, "lines": board_lines},
        "axes": [
            make_three_line([np.array([0.0, 0.0, 0.0]), np.array([180.0, 0.0, 0.0])], "#ef4444", 0.85),
            make_three_line([np.array([0.0, 0.0, 0.0]), np.array([0.0, 180.0, 0.0])], "#22c55e", 0.85),
            make_three_line([np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 180.0])], "#2563eb", 0.85),
        ],
        "handPoints": hand_points,
        "handEdges": hand_edges,
        "cameras": cameras,
        "missingKeypoints": all_available["missing_keypoints"],
        "textureSource": "native 1920x1080 JPEG camera-frame overlays",
    }
    scene_json = json.dumps(scene_data, separators=(",", ":"))
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{hand.title()} hand session {all_available['session_time_s']:.3f}s</title>
  <script type="importmap">
    {{"imports": {{"three": "https://cdn.jsdelivr.net/npm/three@0.165.0/build/three.module.js", "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.165.0/examples/jsm/"}}}}
  </script>
  <style>
    * {{ box-sizing: border-box; }}
    html, body, #viewer {{ width: 100%; height: 100%; margin: 0; overflow: hidden; }}
    body {{
      background: #0b1020;
      color: #e5e7eb;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }}
    #viewer canvas {{ display: block; width: 100%; height: 100%; }}
    .panel {{
      position: fixed;
      top: 12px;
      left: 12px;
      z-index: 10;
      width: min(420px, calc(100vw - 24px));
      padding: 10px;
      display: grid;
      gap: 8px;
      background: rgba(15, 23, 42, 0.86);
      border: 1px solid rgba(148, 163, 184, 0.35);
      backdrop-filter: blur(10px);
      border-radius: 8px;
      box-shadow: 0 18px 45px rgba(0, 0, 0, 0.25);
    }}
    .row {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }}
    select, button {{
      height: 30px;
      border: 1px solid rgba(148, 163, 184, 0.45);
      background: rgba(30, 41, 59, 0.95);
      color: #f8fafc;
      border-radius: 6px;
      padding: 0 10px;
      font: inherit;
    }}
    button {{ cursor: pointer; }}
    button:hover, select:hover {{ border-color: rgba(226, 232, 240, 0.8); }}
    label {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      min-height: 28px;
      color: #cbd5e1;
      font-size: 13px;
      white-space: nowrap;
    }}
    input[type="range"] {{ width: 118px; accent-color: #38bdf8; }}
    input[type="checkbox"] {{ accent-color: #38bdf8; }}
    .spacer {{ flex: 1 1 auto; }}
    .small {{ font-size: 12px; color: #94a3b8; }}
  </style>
</head>
<body>
  <div id="viewer"></div>
  <section class="panel">
    <div class="row">
      <select id="cameraSelect" aria-label="Camera">
        <option value="all">All cameras</option>
        {''.join(f'<option value="{cam}">{cam}</option>' for cam in sorted(cameras))}
      </select>
      <button type="button" data-view="iso">Iso</button>
      <button type="button" data-view="top">Top</button>
      <button type="button" data-view="side">Side</button>
      <button type="button" data-view="hand">Hand</button>
    </div>
    <div class="row">
      <label><input id="showImages" type="checkbox" checked>Images</label>
      <label><input id="showRays" type="checkbox" checked>Rays</label>
      <label><input id="showProjected" type="checkbox" checked>2D points</label>
      <label><input id="showHand" type="checkbox" checked>3D hand</label>
      <label><input id="showScene" type="checkbox" checked>Scene</label>
    </div>
    <div class="row">
      <label>Image <input id="imageOpacity" type="range" min="0" max="1" step="0.02" value="0.92"></label>
      <label>Rays <input id="rayOpacity" type="range" min="0" max="1" step="0.02" value="0.38"></label>
      <span class="spacer"></span>
      <span class="small">native 1920x1080</span>
    </div>
  </section>
  <script type="module">
    import * as THREE from 'three';
    import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

    const data = {scene_json};
    const root = document.getElementById('viewer');
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b1020);

    const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: false }});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    root.appendChild(renderer.domElement);

    const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 10000);
    camera.up.set(0, 0, 1);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;

    const boundsCenter = new THREE.Vector3(...data.bounds.center);
    const handCenter = new THREE.Vector3(...data.bounds.handCenter);
    const radius = Math.max(data.bounds.radius, 600);
    controls.target.copy(boundsCenter);
    camera.position.set(boundsCenter.x + radius * 0.9, boundsCenter.y - radius * 1.25, boundsCenter.z + radius * 0.7);
    camera.lookAt(boundsCenter);

    const handGroup = new THREE.Group();
    const sceneGroup = new THREE.Group();
    const cameraGroups = {{}};
    scene.add(handGroup, sceneGroup);

    const textureLoader = new THREE.TextureLoader();
    const imageMaterials = [];
    const rayMaterials = [];
    const pointMaterials = [];

    function vec3(point) {{ return new THREE.Vector3(point[0], point[1], point[2]); }}

    function makeLine(points, color, opacity = 1) {{
      const geometry = new THREE.BufferGeometry().setFromPoints(points.map(vec3));
      const material = new THREE.LineBasicMaterial({{
        color,
        transparent: opacity < 1,
        opacity,
        depthWrite: opacity >= 1
      }});
      return new THREE.Line(geometry, material);
    }}

    function makeSphere(position, radiusValue, color, opacity = 1) {{
      const geometry = new THREE.SphereGeometry(radiusValue, 18, 12);
      const material = new THREE.MeshBasicMaterial({{
        color,
        transparent: opacity < 1,
        opacity,
        depthWrite: opacity >= 1
      }});
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.copy(vec3(position));
      return mesh;
    }}

    function makeImagePlane(image) {{
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.Float32BufferAttribute(image.geometry.vertices.flat(), 3));
      geometry.setAttribute('uv', new THREE.Float32BufferAttribute(image.geometry.uvs.flat(), 2));
      geometry.setIndex(image.geometry.indices);
      geometry.computeVertexNormals();
      const texture = textureLoader.load(image.filename);
      texture.colorSpace = THREE.SRGBColorSpace;
      texture.anisotropy = renderer.capabilities.getMaxAnisotropy();
      texture.minFilter = THREE.LinearMipmapLinearFilter;
      texture.magFilter = THREE.LinearFilter;
      const material = new THREE.MeshBasicMaterial({{
        map: texture,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: Number(document.getElementById('imageOpacity').value),
        depthWrite: false
      }});
      imageMaterials.push(material);
      return new THREE.Mesh(geometry, material);
    }}

    data.board.lines.forEach(line => sceneGroup.add(makeLine(line.points, line.color, line.opacity)));
    const boardGeometry = new THREE.BufferGeometry();
    boardGeometry.setAttribute('position', new THREE.Float32BufferAttribute(data.board.surface.flat(), 3));
    boardGeometry.setIndex([0, 1, 2, 0, 2, 3]);
    sceneGroup.add(new THREE.Mesh(boardGeometry, new THREE.MeshBasicMaterial({{
      color: 0x94a3b8,
      transparent: true,
      opacity: 0.08,
      side: THREE.DoubleSide,
      depthWrite: false
    }})));
    data.axes.forEach(line => sceneGroup.add(makeLine(line.points, line.color, line.opacity)));

    data.handEdges.forEach(line => handGroup.add(makeLine(line.points, line.color, line.opacity)));
    data.handPoints.forEach(point => handGroup.add(makeSphere(point.position, point.radius, 0x111827, 1)));

    for (const [cam, camData] of Object.entries(data.cameras)) {{
      const rootGroup = new THREE.Group();
      const imageGroup = new THREE.Group();
      const projectedGroup = new THREE.Group();
      const rayGroup = new THREE.Group();
      const helperGroup = new THREE.Group();
      cameraGroups[cam] = {{ rootGroup, imageGroup, projectedGroup, rayGroup, helperGroup }};
      scene.add(rootGroup);
      rootGroup.add(imageGroup, projectedGroup, rayGroup, helperGroup);

      helperGroup.add(makeSphere(camData.center, 8, camData.color, 1));
      helperGroup.add(makeLine(camData.opticalAxis.points, camData.opticalAxis.color, camData.opticalAxis.opacity));

      if (camData.image) imageGroup.add(makeImagePlane(camData.image));
      camData.projectedEdges.forEach(line => projectedGroup.add(makeLine(line.points, line.color, line.opacity)));
      camData.projectedPoints.forEach(point => projectedGroup.add(makeSphere(point.position, 3.3, 0xffffff, 1)));
      camData.rays.forEach(ray => {{
        const line = makeLine(ray.points, ray.color, Number(document.getElementById('rayOpacity').value));
        line.material.depthWrite = false;
        rayMaterials.push(line.material);
        rayGroup.add(line);
      }});
    }}

    function setOpacity(materials, value) {{
      materials.forEach(material => {{
        material.opacity = value;
        material.transparent = value < 1;
        material.needsUpdate = true;
      }});
    }}

    function updateVisibility() {{
      const selected = document.getElementById('cameraSelect').value;
      const showImages = document.getElementById('showImages').checked;
      const showRays = document.getElementById('showRays').checked;
      const showProjected = document.getElementById('showProjected').checked;
      const showHand = document.getElementById('showHand').checked;
      const showScene = document.getElementById('showScene').checked;
      handGroup.visible = showHand;
      sceneGroup.visible = showScene;
      for (const [cam, groups] of Object.entries(cameraGroups)) {{
        const active = selected === 'all' || selected === cam;
        groups.imageGroup.visible = active && showImages;
        groups.rayGroup.visible = active && showRays;
        groups.projectedGroup.visible = active && showProjected;
        groups.helperGroup.visible = active && showScene;
      }}
    }}

    function setView(name) {{
      camera.up.set(0, 0, 1);
      let target = boundsCenter.clone();
      let position;
      if (name === 'top') {{
        camera.up.set(0, 1, 0);
        position = new THREE.Vector3(boundsCenter.x, boundsCenter.y, boundsCenter.z + radius * 2.0);
      }} else if (name === 'side') {{
        position = new THREE.Vector3(boundsCenter.x + radius * 1.8, boundsCenter.y, boundsCenter.z + radius * 0.25);
      }} else if (name === 'hand') {{
        target = handCenter.clone();
        position = new THREE.Vector3(handCenter.x + 230, handCenter.y - 290, handCenter.z + 210);
      }} else {{
        position = new THREE.Vector3(boundsCenter.x + radius * 0.9, boundsCenter.y - radius * 1.25, boundsCenter.z + radius * 0.7);
      }}
      camera.position.copy(position);
      controls.target.copy(target);
      camera.lookAt(target);
      controls.update();
    }}

    document.querySelectorAll('input, select').forEach(control => control.addEventListener('input', updateVisibility));
    document.getElementById('imageOpacity').addEventListener('input', event => setOpacity(imageMaterials, Number(event.target.value)));
    document.getElementById('rayOpacity').addEventListener('input', event => setOpacity(rayMaterials, Number(event.target.value)));
    document.querySelectorAll('[data-view]').forEach(button => button.addEventListener('click', () => setView(button.dataset.view)));
    updateVisibility();

    window.addEventListener('resize', () => {{
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    }});

    function animate() {{
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }}
    animate();
  </script>
</body>
</html>
"""
    path.write_text(html)


def combo_summary_rows(combo_results: dict, all_available: dict) -> list[dict]:
    rows = []
    hand = all_available["hand"]
    for combo in combo_results["hands"][hand]:
        if combo["num_triangulated_keypoints"] == 0:
            continue
        rows.append(
            {
                "combo": combo["combo"],
                "camera_count": combo["num_cameras"],
                "num_keypoints": combo["num_triangulated_keypoints"],
                "num_bones": len(combo["bones"]),
                "bone_abs_error_mean_mm": combo["bone_abs_error_mean_mm"],
                "bone_abs_error_median_mm": combo["bone_abs_error_median_mm"],
                "bone_abs_error_max_mm": combo["bone_abs_error_max_mm"],
                "reprojection_mean_px": combo["reprojection_mean_px"],
                "reprojection_median_px": combo["reprojection_median_px"],
                "method": "fixed_camera_combo",
            }
        )
    rows.append(
        {
            "combo": "all_available_per_joint",
            "camera_count": "mixed",
            "num_keypoints": all_available["num_triangulated_keypoints"],
            "num_bones": all_available["bone_length_error_mm"]["num_bones_evaluated"],
            "bone_abs_error_mean_mm": all_available["bone_length_error_mm"]["mean_abs"],
            "bone_abs_error_median_mm": all_available["bone_length_error_mm"]["median_abs"],
            "bone_abs_error_max_mm": all_available["bone_length_error_mm"]["max_abs"],
            "reprojection_mean_px": all_available["reprojection_error_px"]["mean"],
            "reprojection_median_px": all_available["reprojection_error_px"]["median"],
            "method": "all_available_per_joint",
        }
    )
    return rows


def save_combo_summary(path_png: Path, path_csv: Path, rows: list[dict], hand: str, session_time_s: float) -> None:
    with path_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    labels = [row["combo"].replace("+", "+\n") for row in rows]
    x = np.arange(len(rows))
    bone_vals = [row["bone_abs_error_median_mm"] for row in rows]
    reproj_vals = [row["reprojection_median_px"] for row in rows]
    colors = []
    for row in rows:
        if row["method"] == "all_available_per_joint":
            colors.append("#111827")
        elif row["camera_count"] == 2:
            colors.append("#2563eb")
        elif row["camera_count"] == 3:
            colors.append("#16a34a")
        else:
            colors.append("#9333ea")

    fig, axes = plt.subplots(2, 1, figsize=(14, 10.5), constrained_layout=True)
    axes[0].bar(x, bone_vals, color=colors, alpha=0.9)
    axes[0].set_ylabel("median abs bone-length error (mm)")
    axes[0].set_title(f"{hand.title()} hand {session_time_s:.3f}s: bone error by camera selection")
    axes[0].grid(axis="y", color="#d6dbe1", alpha=0.75)
    axes[0].set_ylim(0, max(bone_vals) * 1.28)
    for xi, row, val in zip(x, rows, bone_vals):
        axes[0].text(xi, val + 0.35, f"{val:.1f}\n{row['num_keypoints']}kp/{row['num_bones']}b", ha="center", va="bottom", fontsize=8)

    axes[1].bar(x, reproj_vals, color=colors, alpha=0.9)
    axes[1].set_ylabel("median reprojection error (px)")
    axes[1].set_title("Reprojection error by camera selection")
    axes[1].grid(axis="y", color="#d6dbe1", alpha=0.75)
    axes[1].set_ylim(0, max(reproj_vals) * 1.22)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    for xi, val in zip(x, reproj_vals):
        axes[1].text(xi, val + 0.25, f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    handles = [
        plt.Line2D([0], [0], color="#2563eb", linewidth=7, label="2 fixed cameras"),
        plt.Line2D([0], [0], color="#16a34a", linewidth=7, label="3 fixed cameras"),
        plt.Line2D([0], [0], color="#9333ea", linewidth=7, label="4 fixed cameras"),
        plt.Line2D([0], [0], color="#111827", linewidth=7, label="all available per joint"),
    ]
    axes[0].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    fig.savefig(path_png, dpi=180)
    plt.close(fig)


def save_bone_combo_heatmap(
    path: Path,
    combo_results: dict,
    all_available: dict,
    keypoint_names: list[str],
    edges: list[list[int]],
) -> None:
    hand = all_available["hand"]
    combos = [combo for combo in combo_results["hands"][hand] if combo["num_triangulated_keypoints"] > 0]
    bone_names = [bone_name(a, b, keypoint_names) for a, b in edges]
    values = np.full((len(bone_names), len(combos)), np.nan, dtype=float)

    for col, combo in enumerate(combos):
        by_bone = {
            bone["bone"]: bone["abs_error_mm"]
            for bone in combo["bones"]
            if bone.get("abs_error_mm") is not None
        }
        for row, name in enumerate(bone_names):
            if name in by_bone:
                values[row, col] = float(by_bone[name])

    finite = values[np.isfinite(values)]
    vmax = max(1.0, float(np.ceil(finite.max()))) if finite.size else 1.0
    cmap = plt.get_cmap("magma_r").copy()
    cmap.set_bad("#e5e7eb")

    fig_width = max(13.5, len(combos) * 1.2)
    fig_height = max(8.5, len(bone_names) * 0.42)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    image = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=0.0, vmax=vmax, aspect="auto")

    xlabels = [f"{combo['combo']}\n{combo['num_triangulated_keypoints']}/21" for combo in combos]
    ax.set_xticks(np.arange(len(combos)))
    ax.set_xticklabels(xlabels, rotation=38, ha="right")
    ax.set_yticks(np.arange(len(bone_names)))
    ax.set_yticklabels(bone_names)
    ax.set_ylabel("bone")
    ax.set_title(
        f"{hand.title()} Hand {float(all_available['session_time_s']):.3f}s "
        "Absolute Bone-Length Error by Camera Combination"
    )

    ax.set_xticks(np.arange(-0.5, len(combos), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(bone_names), 1), minor=True)
    ax.grid(which="minor", color="#f8fafc", linestyle="-", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            if np.isfinite(value):
                text_color = "white" if value > vmax * 0.55 else "#374151"
                ax.text(col, row, f"{value:.1f}", ha="center", va="center", fontsize=8, color=text_color)
            else:
                ax.text(col, row, "-", ha="center", va="center", fontsize=9, color="#9ca3af")

    cbar = fig.colorbar(image, ax=ax, shrink=0.92, pad=0.02)
    cbar.set_label("absolute error (mm); gray = bone unavailable")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_outputs() -> None:
    args = parse_args()
    all_available = json.loads(args.all_available_json.read_text())
    labels_path = resolve_source(all_available["source_labels"], args.labels_json)
    gt_path = resolve_source(all_available["source_bone_lengths"], args.bone_lengths_json)
    combo_path = args.combo_results_json or args.all_available_json.with_name(
        args.all_available_json.name.replace("_all_available_triangulation.json", "_triangulated_results.json")
    )
    labels = json.loads(labels_path.read_text())
    gt_lengths = json.loads(gt_path.read_text())["bone_lengths_mm"]
    combo_results = json.loads(combo_path.read_text())
    calibration_report = load_calibration_report(labels)
    keypoint_names = labels["keypoint_names"]
    edges = labels["edges"]
    tri_points = {name: np.array(xyz, dtype=float) for name, xyz in all_available["points_mm"].items()}
    gt_points = reconstruct_gt_partial(tri_points, keypoint_names, edges, gt_lengths)
    common_names = [name for name in keypoint_names if name in tri_points and name in gt_points]
    joint_deltas = {name: float(np.linalg.norm(tri_points[name] - gt_points[name])) for name in common_names}
    tri_2d, gt_2d = project_hand_local(tri_points, gt_points, keypoint_names)

    out_dir = args.out_dir or args.all_available_json.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    live_out_dir = args.live_out_dir
    stem = args.all_available_json.stem.replace("_all_available_triangulation", "")
    comparison_2d = out_dir / f"{stem}_all_available_2d_pose_comparison.png"
    interactive = out_dir / f"{stem}_all_available_interactive_3d_scene.html"
    combo_csv = out_dir / f"{stem}_camera_combo_error_summary.csv"
    combo_png = out_dir / f"{stem}_camera_combo_error_summary.png"
    bone_combo_heatmap = out_dir / f"{stem}_bone_error_heatmap_by_camera_combo.png"
    summary_json = out_dir / f"{stem}_visualization_summary.json"
    camera_overlays = save_camera_keypoint_overlays(
        out_dir,
        labels,
        all_available["hand"],
        keypoint_names,
        edges,
        stem,
    )

    save_2d_comparison(
        comparison_2d,
        tri_2d,
        gt_2d,
        joint_deltas,
        keypoint_names,
        edges,
        all_available["hand"],
        float(all_available["session_time_s"]),
        all_available["missing_keypoints"],
    )
    save_interactive_scene(
        interactive,
        all_available,
        labels,
        calibration_report,
        keypoint_names,
        edges,
        camera_overlays,
    )
    rows = combo_summary_rows(combo_results, all_available)
    save_combo_summary(combo_png, combo_csv, rows, all_available["hand"], float(all_available["session_time_s"]))
    save_bone_combo_heatmap(bone_combo_heatmap, combo_results, all_available, keypoint_names, edges)

    payload = {
        "source_all_available": str(args.all_available_json),
        "source_labels": str(labels_path),
        "source_bone_lengths": str(gt_path),
        "source_combo_results": str(combo_path),
        "joint_delta_summary_mm": {
            "mean": float(np.mean(list(joint_deltas.values()))),
            "median": float(np.median(list(joint_deltas.values()))),
            "max": float(np.max(list(joint_deltas.values()))),
        },
        "generated_files": {
            "pose_2d_comparison": str(comparison_2d),
            "interactive_3d_scene": str(interactive),
            "camera_combo_error_summary_csv": str(combo_csv),
            "camera_combo_error_summary_plot": str(combo_png),
            "bone_error_heatmap_by_camera_combo": str(bone_combo_heatmap),
            "camera_frame_overlays": {cam: info["path"] for cam, info in camera_overlays.items()},
            "camera_image_texture_source": "native external 1920x1080 JPEG textures",
            "camera_image_warp_grid_cols": CAMERA_IMAGE_WARP_GRID_COLS,
            "camera_frame_jpeg_quality": CAMERA_FRAME_JPEG_QUALITY,
        },
    }
    summary_json.write_text(json.dumps(payload, indent=2) + "\n")

    if live_out_dir:
        live_out_dir.mkdir(parents=True, exist_ok=True)
        for path in (comparison_2d, interactive, combo_csv, combo_png, bone_combo_heatmap, summary_json):
            shutil.copy2(path, live_out_dir / path.name)
        for info in camera_overlays.values():
            overlay_path = Path(str(info["path"]))
            shutil.copy2(overlay_path, live_out_dir / overlay_path.name)

    print(json.dumps({**payload["generated_files"], "summary_json": str(summary_json)}, indent=2))


if __name__ == "__main__":
    save_outputs()
