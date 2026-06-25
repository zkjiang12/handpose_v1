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
import plotly.graph_objects as go


EGO_EXO_ROOT = Path("/Users/zikangjiang/dev/ego-exo")
CAMERA_IMAGE_TEXTURE_COLS = 512
CAMERA_FRAME_JPEG_QUALITY = 97
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


def draw_label(
    image: np.ndarray,
    text: str,
    xy: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.48
    thickness = 1
    (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = xy
    x0 = max(0, min(image.shape[1] - w - 8, x + 8))
    y0 = max(h + 6, min(image.shape[0] - 4, y - 8))
    cv2.rectangle(image, (x0 - 3, y0 - h - 4), (x0 + w + 3, y0 + baseline + 3), (17, 24, 39), -1)
    cv2.putText(image, text, (x0, y0), font, scale, color, thickness, cv2.LINE_AA)


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
            draw_label(frame, f"{idx} {keypoint_names[idx]}", point, (255, 255, 255))

        cv2.putText(
            frame,
            f"{cam} | local {float(info.get('local_time_s') or 0.0):.3f}s | frame {frame_index}",
            (24, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
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


def bgr_to_hex(rgb: np.ndarray) -> str:
    r, g, b = [int(v) for v in rgb]
    return f"#{r:02x}{g:02x}{b:02x}"


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


def add_keypoint_rays(
    fig: go.Figure,
    labels: dict,
    all_available: dict,
    hand: str,
    cam: str,
    rvec: np.ndarray,
    tvec: np.ndarray,
    keypoint_names: list[str],
    color: str,
) -> None:
    tri_points = {
        name: world_to_z_up(np.array(xyz, dtype=float))
        for name, xyz in all_available["points_mm"].items()
    }
    first_ray_for_camera = True
    for kp_s, cams in sorted(labels["hands"].get(hand, {}).items(), key=lambda item: int(item[0])):
        if cam not in cams:
            continue
        kp_index = int(kp_s)
        kp_name = keypoint_names[kp_index]
        origin, direction = camera_ray_display(rvec, tvec, cams[cam])
        target = tri_points.get(kp_name)
        if target is not None:
            depth = float(np.dot(target - origin, direction))
            if depth <= 0:
                depth = 900.0
            end = origin + direction * max(240.0, depth * 1.08)
            closest = origin + direction * depth
            miss_mm = float(np.linalg.norm(closest - target))
            hover = f"{cam} ray for {kp_index} {kp_name}<br>ray-to-3D residual: {miss_mm:.1f} mm"
        else:
            end = origin + direction * 900.0
            hover = f"{cam} ray for {kp_index} {kp_name}<br>no triangulated 3D joint"
        fig.add_trace(
            go.Scatter3d(
                x=[origin[0], end[0]],
                y=[origin[1], end[1]],
                z=[origin[2], end[2]],
                mode="lines",
                line={"color": color, "width": 2},
                opacity=0.34,
                hovertext=[hover, hover],
                hoverinfo="text",
                legendgroup=f"{cam}-rays",
                name=f"{cam} keypoint rays",
                showlegend=first_ray_for_camera,
            )
        )
        first_ray_for_camera = False


def add_camera_image_plane(
    fig: go.Figure,
    image_path: Path,
    labels: dict,
    hand: str,
    cam: str,
    rvec: np.ndarray,
    tvec: np.ndarray,
    keypoint_names: list[str],
    edges: list[list[int]],
    plane_distance_mm: float = 180.0,
    texture_cols: int = CAMERA_IMAGE_TEXTURE_COLS,
) -> None:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        return
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]
    texture_rows = max(2, int(round(texture_cols * height / width)))
    texture = cv2.resize(image_rgb, (texture_cols, texture_rows), interpolation=cv2.INTER_AREA)

    xs = np.linspace(0, width - 1, texture_cols)
    ys = np.linspace(0, height - 1, texture_rows)
    grid_x, grid_y = np.meshgrid(xs, ys)
    uv_grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    pts = camera_plane_points(rvec, tvec, uv_grid, plane_distance_mm)

    i_idx: list[int] = []
    j_idx: list[int] = []
    k_idx: list[int] = []
    face_colors: list[str] = []
    for row in range(texture_rows - 1):
        for col in range(texture_cols - 1):
            v00 = row * texture_cols + col
            v01 = v00 + 1
            v10 = (row + 1) * texture_cols + col
            v11 = v10 + 1
            color = bgr_to_hex(texture[row, col])
            i_idx.extend([v00, v00])
            j_idx.extend([v10, v11])
            k_idx.extend([v11, v01])
            face_colors.extend([color, color])

    fig.add_trace(
        go.Mesh3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            i=i_idx,
            j=j_idx,
            k=k_idx,
            facecolor=face_colors,
            flatshading=True,
            lighting={"ambient": 1.0, "diffuse": 0.0, "specular": 0.0, "roughness": 1.0},
            hoverinfo="skip",
            opacity=0.92,
            name=f"{cam} raw frame",
            showlegend=False,
        )
    )

    kp_uv = []
    kp_names = []
    kp_indices = []
    for kp_s, cams in labels["hands"].get(hand, {}).items():
        if cam not in cams:
            continue
        kp_indices.append(int(kp_s))
        kp_names.append(keypoint_names[int(kp_s)])
        kp_uv.append(cams[cam])
    if not kp_uv:
        return

    kp_pts = camera_plane_points(rvec, tvec, np.array(kp_uv, dtype=np.float64), plane_distance_mm * 0.992)
    by_index = {idx: kp_pts[pos] for pos, idx in enumerate(kp_indices)}
    for a, b in edges:
        if a not in by_index or b not in by_index:
            continue
        pa = by_index[a]
        pb = by_index[b]
        fig.add_trace(
            go.Scatter3d(
                x=[pa[0], pb[0]],
                y=[pa[1], pb[1]],
                z=[pa[2], pb[2]],
                mode="lines",
                line={"color": "#facc15", "width": 6},
                hoverinfo="skip",
                showlegend=False,
            )
        )
    fig.add_trace(
        go.Scatter3d(
            x=kp_pts[:, 0],
            y=kp_pts[:, 1],
            z=kp_pts[:, 2],
            mode="markers+text",
            marker={"size": 4, "color": "#ffffff", "line": {"color": "#111827", "width": 1}},
            text=[f"{idx} {name}" for idx, name in zip(kp_indices, kp_names)],
            textfont={"size": 10, "color": "#ffffff"},
            textposition="top center",
            hovertext=[f"{cam} {idx} {name}" for idx, name in zip(kp_indices, kp_names)],
            hoverinfo="text",
            name=f"{cam} 2D labels",
            showlegend=False,
        )
    )


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
    pts = np.stack(list(display_points.values()))
    fig = go.Figure()

    extrinsics = calibration_report["extrinsics"]
    board = extrinsics["board"]
    board_w = board["cols"] * board["square_mm"]
    board_h = board["rows"] * board["square_mm"]
    board_raw = np.array([[0, 0, 0], [board_w, 0, 0], [board_w, board_h, 0], [0, board_h, 0]], dtype=float)
    board_display = np.stack([world_to_z_up(p) for p in board_raw])
    fig.add_trace(
        go.Surface(
            x=[[board_display[0, 0], board_display[1, 0]], [board_display[3, 0], board_display[2, 0]]],
            y=[[board_display[0, 1], board_display[1, 1]], [board_display[3, 1], board_display[2, 1]]],
            z=[[0, 0], [0, 0]],
            opacity=0.18,
            colorscale=[[0, "#e2e8f0"], [1, "#e2e8f0"]],
            showscale=False,
            hoverinfo="skip",
            name="ChArUco board plane",
        )
    )

    for a, b in edges:
        start = keypoint_names[a]
        end = keypoint_names[b]
        if start not in display_points or end not in display_points:
            continue
        pa = display_points[start]
        pb = display_points[end]
        fig.add_trace(
            go.Scatter3d(
                x=[pa[0], pb[0]],
                y=[pa[1], pb[1]],
                z=[pa[2], pb[2]],
                mode="lines",
                line={"color": edge_color(a, b, keypoint_names), "width": 10 if a == 0 else 7},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    point_names = [name for name in keypoint_names if name in display_points]
    point_arr = np.stack([display_points[name] for name in point_names])
    fig.add_trace(
        go.Scatter3d(
            x=point_arr[:, 0],
            y=point_arr[:, 1],
            z=point_arr[:, 2],
            mode="markers+text",
            marker={"size": [9 if name == "wrist" else 6 for name in point_names], "color": "#111827"},
            text=[str(keypoint_names.index(name)) for name in point_names],
            textposition="top center",
            hovertext=point_names,
            hoverinfo="text",
            name=f"{hand} hand joints",
        )
    )

    camera_colors = {"cam1": "#dc2626", "cam2": "#2563eb", "cam3": "#16a34a", "cam4": "#9333ea"}
    for cam, info in sorted(extrinsics["cameras"].items()):
        rvec = np.array(info["rvec_world_to_cam"], dtype=float).reshape(3, 1)
        tvec = np.array(info["t_world_to_cam_mm"], dtype=float).reshape(3, 1)
        center_raw, dirs_raw = camera_frustum_world(rvec, tvec)
        center = world_to_z_up(center_raw)
        dirs = np.array([world_to_z_up(d) for d in dirs_raw])
        color = camera_colors.get(cam, "#64748b")
        scale = 180.0
        optical_end = center + scale * 1.25 * dirs[4]
        fig.add_trace(
            go.Scatter3d(
                x=[center[0]],
                y=[center[1]],
                z=[center[2]],
                mode="markers+text",
                marker={"size": 8, "symbol": "diamond", "color": color},
                text=[cam],
                textposition="top center",
                hoverinfo="text",
                name=cam,
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[center[0], optical_end[0]],
                y=[center[1], optical_end[1]],
                z=[center[2], optical_end[2]],
                mode="lines",
                line={"color": color, "width": 5},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        overlay = camera_overlays.get(cam)
        if overlay:
            add_camera_image_plane(
                fig,
                Path(str(overlay["path"])),
                labels,
                hand,
                cam,
                rvec,
                tvec,
                keypoint_names,
                edges,
            )
        add_keypoint_rays(
            fig,
            labels,
            all_available,
            hand,
            cam,
            rvec,
            tvec,
            keypoint_names,
            color,
        )

    axis_origin = np.array([0.0, 0.0, 0.0])
    for label, vec, color in (
        ("+X", np.array([180.0, 0.0, 0.0]), "#ef4444"),
        ("+Y display", np.array([0.0, 180.0, 0.0]), "#22c55e"),
        ("+Z up", np.array([0.0, 0.0, 180.0]), "#2563eb"),
    ):
        end = axis_origin + vec
        fig.add_trace(
            go.Scatter3d(
                x=[axis_origin[0], end[0]],
                y=[axis_origin[1], end[1]],
                z=[axis_origin[2], end[2]],
                mode="lines+text",
                line={"color": color, "width": 7},
                text=["", label],
                textposition="top center",
                hoverinfo="skip",
                showlegend=False,
            )
        )

    missing = ", ".join(f"{item['index']} {item['name']}" for item in all_available["missing_keypoints"]) or "none"
    cameras = {
        "Iso": {"eye": {"x": 1.35, "y": -1.6, "z": 1.0}, "up": {"x": 0, "y": 0, "z": 1}},
        "Top": {"eye": {"x": 0.0, "y": 0.0, "z": 2.35}, "up": {"x": 0, "y": 1, "z": 0}},
        "Side": {"eye": {"x": 2.2, "y": -0.2, "z": 0.55}, "up": {"x": 0, "y": 0, "z": 1}},
        "Camera row": {"eye": {"x": 0.1, "y": -2.2, "z": 0.65}, "up": {"x": 0, "y": 0, "z": 1}},
    }
    fig.update_layout(
        title=f"{hand.title()} hand session {all_available['session_time_s']:.3f}s: interactive 3D scene",
        autosize=True,
        margin={"l": 0, "r": 0, "b": 0, "t": 70},
        paper_bgcolor="#f8fafc",
        scene={
            "xaxis": {"title": "X = raw world X (mm)", "backgroundcolor": "#f8fafc"},
            "yaxis": {"title": "Y = -raw world Y (mm)", "backgroundcolor": "#f8fafc"},
            "zaxis": {"title": "Z up = -raw world Z (mm)", "backgroundcolor": "#f8fafc"},
            "aspectmode": "data",
            "camera": cameras["Iso"],
        },
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "x": 0.02,
                "y": 1.04,
                "buttons": [
                    {"label": name, "method": "relayout", "args": [{"scene.camera": camera}]}
                    for name, camera in cameras.items()
                ],
            }
        ],
        annotations=[
            {
                "text": f"Partial all-available triangulation: {len(display_points)}/21 joints. Missing: {missing}. Drag to rotate.",
                "xref": "paper",
                "yref": "paper",
                "x": 0.01,
                "y": 0.01,
                "showarrow": False,
                "align": "left",
                "font": {"size": 12, "color": "#475569"},
            }
        ],
    )
    plot_html = fig.to_html(
        include_plotlyjs="cdn",
        full_html=False,
        config={"responsive": True},
        default_width="100%",
        default_height="100%",
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{hand.title()} hand session {all_available['session_time_s']:.3f}s</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: #f8fafc;
      color: #172033;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }}
    .scene {{
      min-height: 100vh;
      overflow: hidden;
    }}
    .scene .plotly-graph-div {{
      width: 100vw !important;
      height: 100vh !important;
    }}
  </style>
</head>
<body>
  <main class="scene">{plot_html}</main>
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
            "camera_image_texture_cols": CAMERA_IMAGE_TEXTURE_COLS,
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
