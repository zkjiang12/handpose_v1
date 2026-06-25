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
        frustum = center[None, :] + scale * dirs[:4]
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
        for end in frustum:
            fig.add_trace(
                go.Scatter3d(
                    x=[center[0], end[0]],
                    y=[center[1], end[1]],
                    z=[center[2], end[2]],
                    mode="lines",
                    line={"color": color, "width": 2},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        for i in range(4):
            a = frustum[i]
            b = frustum[(i + 1) % 4]
            fig.add_trace(
                go.Scatter3d(
                    x=[a[0], b[0]],
                    y=[a[1], b[1]],
                    z=[a[2], b[2]],
                    mode="lines",
                    line={"color": color, "width": 2},
                    hoverinfo="skip",
                    showlegend=False,
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
        width=1150,
        height=900,
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
    fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)


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
    summary_json = out_dir / f"{stem}_visualization_summary.json"

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
    save_interactive_scene(interactive, all_available, labels, calibration_report, keypoint_names, edges)
    rows = combo_summary_rows(combo_results, all_available)
    save_combo_summary(combo_png, combo_csv, rows, all_available["hand"], float(all_available["session_time_s"]))

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
        },
    }
    summary_json.write_text(json.dumps(payload, indent=2) + "\n")

    if live_out_dir:
        live_out_dir.mkdir(parents=True, exist_ok=True)
        for path in (comparison_2d, interactive, combo_csv, combo_png, summary_json):
            shutil.copy2(path, live_out_dir / path.name)

    print(json.dumps({**payload["generated_files"], "summary_json": str(summary_json)}, indent=2))


if __name__ == "__main__":
    save_outputs()
