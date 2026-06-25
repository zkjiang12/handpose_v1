#!/usr/bin/env python3
"""Render 3D and 2D comparison visuals for v2 hand triangulation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


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
    parser.add_argument("results_json", type=Path)
    parser.add_argument("--labels-json", type=Path, required=True)
    parser.add_argument("--bone-lengths-json", type=Path, required=True)
    parser.add_argument("--calibration-report-json", type=Path)
    parser.add_argument("--hand", default="right")
    parser.add_argument("--combo", default="cam1+cam2+cam3+cam4")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/Users/zikangjiang/dev/ego-exo/visualizations/v2/handpose_labeler/results"),
    )
    return parser.parse_args()


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


def set_equal_3d_axes(ax: plt.Axes, points: np.ndarray, pad_mm: float = 12.0) -> None:
    mins = points.min(axis=0) - pad_mm
    maxs = points.max(axis=0) + pad_mm
    center = (mins + maxs) / 2.0
    radius = float((maxs - mins).max() / 2.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(max(0.0, center[2] - radius), center[2] + radius)


def world_to_z_up(xyz: np.ndarray) -> np.ndarray:
    """Convert raw ChArUco-world xyz to a display frame where +Z is up."""
    return np.array([xyz[0], xyz[1], -xyz[2]], dtype=float)


def load_calibration_report(labels: dict, report_path: Path | None) -> dict:
    if report_path is not None:
        return json.loads(report_path.read_text())
    segment_ids = {cam["segment_id"] for cam in labels["cameras"].values() if cam.get("segment_id")}
    if len(segment_ids) != 1:
        raise ValueError(f"Cannot infer one calibration report from segments: {sorted(segment_ids)}")
    segment_id = next(iter(segment_ids))
    return json.loads(REPORT_BY_SEGMENT[segment_id].read_text())


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


def reconstruct_gt_pose(
    tri_points: dict[str, np.ndarray],
    keypoint_names: list[str],
    edges: list[list[int]],
    gt_lengths: dict[str, float],
) -> dict[str, np.ndarray]:
    model = {keypoint_names[0]: tri_points[keypoint_names[0]].copy()}
    pending = [(a, b) for a, b in edges]
    while pending:
        next_pending = []
        progressed = False
        for a, b in pending:
            start = keypoint_names[a]
            end = keypoint_names[b]
            if start not in model:
                next_pending.append((a, b))
                continue
            direction = tri_points[end] - tri_points[start]
            norm = float(np.linalg.norm(direction))
            if norm == 0.0:
                raise ValueError(f"Cannot reconstruct zero-length edge {start}->{end}")
            length = gt_lengths[bone_name(a, b, keypoint_names)]
            model[end] = model[start] + direction / norm * length
            progressed = True
        if not progressed:
            missing = ", ".join(f"{keypoint_names[a]}->{keypoint_names[b]}" for a, b in next_pending)
            raise ValueError(f"Could not traverse hand skeleton edges: {missing}")
        pending = next_pending
    return model


def world_to_display(points: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    # World Z is negative above the ChArUco board in these extrinsics.
    return {name: world_to_z_up(xyz) for name, xyz in points.items()}


def project_hand_local(
    tri_points: dict[str, np.ndarray],
    gt_points: dict[str, np.ndarray],
    keypoint_names: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    wrist = tri_points["wrist"]
    tri = np.stack([tri_points[name] - wrist for name in keypoint_names])
    gt = np.stack([gt_points[name] - wrist for name in keypoint_names])
    combined = np.vstack([tri, gt])
    _, _, vt = np.linalg.svd(combined - combined.mean(axis=0), full_matrices=False)
    basis = vt[:2].T
    tri_2d_arr = tri @ basis
    gt_2d_arr = gt @ basis

    fingertip_idxs = [4, 8, 12, 16, 20]
    if tri_2d_arr[fingertip_idxs, 0].mean() < 0:
        tri_2d_arr[:, 0] *= -1
        gt_2d_arr[:, 0] *= -1
    if tri_2d_arr[4, 1] < tri_2d_arr[20, 1]:
        tri_2d_arr[:, 1] *= -1
        gt_2d_arr[:, 1] *= -1

    return (
        {name: tri_2d_arr[i] for i, name in enumerate(keypoint_names)},
        {name: gt_2d_arr[i] for i, name in enumerate(keypoint_names)},
    )


def save_3d_handpose(
    path: Path,
    tri_points: dict[str, np.ndarray],
    keypoint_names: list[str],
    edges: list[list[int]],
    combo: str,
) -> None:
    display = world_to_display(tri_points)
    pts = np.stack([display[name] for name in keypoint_names])

    fig = plt.figure(figsize=(10, 8), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    for a, b in edges:
        pa = display[keypoint_names[a]]
        pb = display[keypoint_names[b]]
        ax.plot(
            [pa[0], pb[0]],
            [pa[1], pb[1]],
            [pa[2], pb[2]],
            color=edge_color(a, b, keypoint_names),
            linewidth=3.0,
            alpha=0.95,
        )
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=42, color="#111827", depthshade=True)
    for i, name in enumerate(keypoint_names):
        p = display[name]
        label = str(i) if i not in (0, 4, 8, 12, 16, 20) else f"{i} {name}"
        ax.text(p[0], p[1], p[2] + 2.0, label, fontsize=8)

    board_z = np.zeros(5)
    min_x, min_y = pts[:, :2].min(axis=0) - 30
    max_x, max_y = pts[:, :2].max(axis=0) + 30
    ax.plot(
        [min_x, max_x, max_x, min_x, min_x],
        [min_y, min_y, max_y, max_y, min_y],
        board_z,
        color="#94a3b8",
        linewidth=1.2,
        linestyle="--",
        label="board plane height = 0",
    )
    ax.scatter([pts[0, 0]], [pts[0, 1]], [pts[0, 2]], s=80, color="#dc2626", label="wrist")
    ax.set_title(f"Right hand 3D triangulated pose ({combo})")
    ax.set_xlabel("world X (mm)")
    ax.set_ylabel("world Y (mm)")
    ax.set_zlabel("height above board (mm)")
    set_equal_3d_axes(ax, np.vstack([pts, [[min_x, min_y, 0], [max_x, max_y, 0]]]))
    ax.view_init(elev=27, azim=-55)
    ax.legend(loc="upper left")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_full_3d_scene(
    path: Path,
    tri_points: dict[str, np.ndarray],
    keypoint_names: list[str],
    edges: list[list[int]],
    labels: dict,
    calibration_report: dict,
    combo: str,
) -> dict:
    extrinsics = calibration_report["extrinsics"]
    board = extrinsics["board"]
    board_width = board["cols"] * board["square_mm"]
    board_height = board["rows"] * board["square_mm"]
    hand_display = world_to_display(tri_points)
    hand_points = np.stack([hand_display[name] for name in keypoint_names])

    fig = plt.figure(figsize=(12, 10), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"Full 3D scene in Z-up frame: board, cameras, and right hand ({combo})")

    board_outline = np.array(
        [
            [0.0, 0.0, 0.0],
            [board_width, 0.0, 0.0],
            [board_width, board_height, 0.0],
            [0.0, board_height, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    ax.plot(board_outline[:, 0], board_outline[:, 1], board_outline[:, 2], color="#111827", linewidth=2.2)
    for x in np.arange(0, board_width + 1e-6, board["square_mm"]):
        ax.plot([x, x], [0, board_height], [0, 0], color="#d1d5db", linewidth=0.55)
    for y in np.arange(0, board_height + 1e-6, board["square_mm"]):
        ax.plot([0, board_width], [y, y], [0, 0], color="#d1d5db", linewidth=0.55)

    ax.scatter([0], [0], [0], color="#dc2626", s=80, label="world origin")
    ax.text(0, 0, 0, " origin", color="#dc2626", fontsize=10)
    axis_len = 160.0
    ax.quiver(0, 0, 0, axis_len, 0, 0, color="#ef4444", linewidth=2.3, arrow_length_ratio=0.12)
    ax.quiver(0, 0, 0, 0, axis_len, 0, color="#22c55e", linewidth=2.3, arrow_length_ratio=0.12)
    ax.quiver(0, 0, 0, 0, 0, axis_len, color="#2563eb", linewidth=2.3, arrow_length_ratio=0.12)
    ax.text(axis_len, 0, 0, "+X", color="#ef4444", fontsize=11)
    ax.text(0, axis_len, 0, "+Y", color="#22c55e", fontsize=11)
    ax.text(0, 0, axis_len, "+Z up", color="#2563eb", fontsize=11)

    camera_colors = {
        "cam1": "#dc2626",
        "cam2": "#2563eb",
        "cam3": "#16a34a",
        "cam4": "#9333ea",
    }
    scene_points = [board_outline, hand_points]
    camera_report = {}
    for cam in sorted(labels["cameras"]):
        if cam not in extrinsics["cameras"]:
            continue
        info = extrinsics["cameras"][cam]
        rvec = np.array(info["rvec_world_to_cam"], dtype=float).reshape(3, 1)
        tvec = np.array(info["t_world_to_cam_mm"], dtype=float).reshape(3, 1)
        center_raw, dirs_raw = camera_frustum_world(rvec, tvec)
        center = world_to_z_up(center_raw)
        dirs = np.array([world_to_z_up(d) for d in dirs_raw])
        color = camera_colors.get(cam, "#64748b")
        ax.scatter([center[0]], [center[1]], [center[2]], color=color, s=95, marker="^")
        ax.text(center[0], center[1], center[2] + 24, cam, color=color, fontsize=10)

        scale = 180.0
        frustum_ends = center[None, :] + scale * dirs[:4]
        optical_end = center + scale * 1.22 * dirs[4]
        for end in frustum_ends:
            ax.plot([center[0], end[0]], [center[1], end[1]], [center[2], end[2]], color=color, linewidth=1.0)
        for i in range(4):
            a = frustum_ends[i]
            b = frustum_ends[(i + 1) % 4]
            ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color=color, linewidth=1.0)
        ax.plot(
            [center[0], optical_end[0]],
            [center[1], optical_end[1]],
            [center[2], optical_end[2]],
            color=color,
            linewidth=2.2,
            linestyle="-",
            alpha=0.9,
        )
        scene_points.extend([center.reshape(1, 3), frustum_ends, optical_end.reshape(1, 3)])
        camera_report[cam] = {
            "raw_world_center_mm": center_raw.tolist(),
            "z_up_center_mm": center.tolist(),
            "z_up_optical_axis_endpoint_mm": optical_end.tolist(),
        }

    for a, b in edges:
        start = keypoint_names[a]
        end = keypoint_names[b]
        pa = hand_display[start]
        pb = hand_display[end]
        ax.plot(
            [pa[0], pb[0]],
            [pa[1], pb[1]],
            [pa[2], pb[2]],
            color=edge_color(a, b, keypoint_names),
            linewidth=3.0,
            alpha=0.98,
        )
    ax.scatter(hand_points[:, 0], hand_points[:, 1], hand_points[:, 2], s=34, color="#111827", label="triangulated hand")
    wrist = hand_display["wrist"]
    ax.scatter([wrist[0]], [wrist[1]], [wrist[2]], s=80, color="#f97316", label="wrist")
    ax.text(wrist[0], wrist[1], wrist[2] + 18, "wrist", color="#9a3412", fontsize=10)

    all_points = np.vstack(scene_points)
    set_equal_3d_axes(ax, all_points, pad_mm=35.0)
    ax.set_xlabel("world X (mm)")
    ax.set_ylabel("world Y (mm)")
    ax.set_zlabel("Z up / height above board (mm)")
    ax.view_init(elev=24, azim=-58)
    ax.legend(loc="upper left")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return {
        "coordinate_frame": "Z-up display frame: [X, Y, Z_up] = [raw_world_X, raw_world_Y, -raw_world_Z]",
        "board_size_mm": [board_width, board_height],
        "cameras": camera_report,
    }


def save_2d_comparison(
    path: Path,
    tri_points_2d: dict[str, np.ndarray],
    gt_points_2d: dict[str, np.ndarray],
    keypoint_names: list[str],
    edges: list[list[int]],
    joint_deltas: dict[str, float],
    combo: str,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 9), constrained_layout=True)
    for a, b in edges:
        start = keypoint_names[a]
        end = keypoint_names[b]
        gt_a, gt_b = gt_points_2d[start], gt_points_2d[end]
        tri_a, tri_b = tri_points_2d[start], tri_points_2d[end]
        color = edge_color(a, b, keypoint_names)
        ax.plot([gt_a[0], gt_b[0]], [gt_a[1], gt_b[1]], color="#2563eb", linewidth=6, alpha=0.28)
        ax.plot([tri_a[0], tri_b[0]], [tri_a[1], tri_b[1]], color=color, linewidth=2.8, alpha=0.95)

    worst_names = {name for name, _ in sorted(joint_deltas.items(), key=lambda item: item[1], reverse=True)[:7]}
    for i, name in enumerate(keypoint_names):
        gt = gt_points_2d[name]
        tri = tri_points_2d[name]
        ax.plot([gt[0], tri[0]], [gt[1], tri[1]], color="#64748b", linewidth=1.2, linestyle="--", alpha=0.75)
        ax.scatter([gt[0]], [gt[1]], s=54, color="#2563eb", edgecolor="white", linewidth=0.8, zorder=4)
        ax.scatter([tri[0]], [tri[1]], s=42, color="#f97316", edgecolor="#111827", linewidth=0.5, zorder=5)
        label = str(i)
        if name in worst_names:
            label = f"{i}\n{joint_deltas[name]:.1f}mm"
        ax.text(tri[0] + 2.0, tri[1] + 2.0, label, fontsize=8, color="#111827")

    vals = np.array(list(joint_deltas.values()), dtype=float)
    summary = (
        f"{combo}\n"
        f"joint delta: median {np.median(vals):.1f} mm\n"
        f"mean {np.mean(vals):.1f} mm, max {np.max(vals):.1f} mm"
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


def main() -> None:
    args = parse_args()
    results = json.loads(args.results_json.read_text())
    labels = json.loads(args.labels_json.read_text())
    gt_lengths = json.loads(args.bone_lengths_json.read_text())["bone_lengths_mm"]
    calibration_report = load_calibration_report(labels, args.calibration_report_json)
    keypoint_names = labels["keypoint_names"]
    edges = labels["edges"]
    combo = next(c for c in results["hands"][args.hand] if c["combo"] == args.combo)
    tri_points = {name: np.array(xyz, dtype=float) for name, xyz in combo["points_mm"].items()}
    gt_points = reconstruct_gt_pose(tri_points, keypoint_names, edges, gt_lengths)

    joint_deltas = {
        name: float(np.linalg.norm(tri_points[name] - gt_points[name]))
        for name in keypoint_names
    }
    tri_2d, gt_2d = project_hand_local(tri_points, gt_points, keypoint_names)

    stem = args.results_json.stem.replace("_triangulated_results", "")
    suffix = args.combo.replace("+", "_")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    hand_3d_path = args.out_dir / f"{stem}_3d_handpose_{suffix}.png"
    full_scene_3d_path = args.out_dir / f"{stem}_full_3d_scene_z_up_{suffix}.png"
    comparison_2d_path = args.out_dir / f"{stem}_2d_pose_comparison_{suffix}.png"
    comparison_json_path = args.out_dir / f"{stem}_pose_comparison_{suffix}.json"

    save_3d_handpose(hand_3d_path, tri_points, keypoint_names, edges, args.combo)
    full_scene = save_full_3d_scene(
        full_scene_3d_path,
        tri_points,
        keypoint_names,
        edges,
        labels,
        calibration_report,
        args.combo,
    )
    save_2d_comparison(comparison_2d_path, tri_2d, gt_2d, keypoint_names, edges, joint_deltas, args.combo)

    payload = {
        "source_results": str(args.results_json),
        "source_labels": str(args.labels_json),
        "source_bone_lengths": str(args.bone_lengths_json),
        "hand": args.hand,
        "combo": args.combo,
        "method": "GT pose reconstructed by preserving each triangulated 3D bone direction and replacing each segment length with the measured right-hand GT length.",
        "full_scene": full_scene,
        "joint_delta_mm_gt_length_pose_vs_triangulated": joint_deltas,
        "joint_delta_summary_mm": {
            "mean": float(np.mean(list(joint_deltas.values()))),
            "median": float(np.median(list(joint_deltas.values()))),
            "max": float(np.max(list(joint_deltas.values()))),
        },
        "generated_files": {
            "handpose_3d": str(hand_3d_path),
            "full_scene_3d_z_up": str(full_scene_3d_path),
            "pose_2d_comparison": str(comparison_2d_path),
        },
    }
    comparison_json_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({**payload["generated_files"], "comparison_json": str(comparison_json_path)}, indent=2))


if __name__ == "__main__":
    main()
