#!/usr/bin/env python3
"""Evaluate v2 ChArUco reprojection and board-corner triangulation accuracy."""

from __future__ import annotations

import csv
import itertools
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

OUT_ROOT = Path("/Users/zikangjiang/dev/ego-exo/visualizations/v2")
EVAL_ROOT = OUT_ROOT / "triangulation_eval"
SEGMENTS = [
    ("segment1", OUT_ROOT / "take1" / "v2_calibration_and_audio_sync_report.json"),
    ("segment2", OUT_ROOT / "take2" / "v2_calibration_and_audio_sync_report.json"),
]


def make_board() -> cv2.aruco.CharucoBoard:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    return cv2.aruco.CharucoBoard((14, 10), 40.0, 20.0, dictionary)


def read_frame(video: Path, time_s: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    cap.set(cv2.CAP_PROP_POS_MSEC, time_s * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {time_s}s from {video}")
    return frame


def detect_charuco(
    frame: np.ndarray,
    board: cv2.aruco.CharucoBoard,
    detector: cv2.aruco.CharucoDetector,
) -> dict:
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(frame)
    if charuco_ids is None or len(charuco_ids) < 12:
        raise RuntimeError("No usable ChArUco detection")

    ids = charuco_ids.flatten().astype(np.int32)
    image_points = charuco_corners.astype(np.float64).reshape(-1, 1, 2)
    object_points = board.getChessboardCorners()[ids].astype(np.float64).reshape(-1, 1, 3)
    norm_points = cv2.fisheye.undistortPoints(image_points, K_DEFAULT, D_DEFAULT)
    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        norm_points,
        np.eye(3, dtype=np.float64),
        None,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise RuntimeError("solvePnP failed")
    pnp_projected, _ = cv2.fisheye.projectPoints(
        object_points,
        rvec,
        tvec,
        K_DEFAULT,
        D_DEFAULT,
    )
    pnp_errors = np.linalg.norm(
        pnp_projected.reshape(-1, 2) - image_points.reshape(-1, 2),
        axis=1,
    )
    return {
        "ids": ids,
        "image_points": image_points.reshape(-1, 2),
        "object_points": object_points.reshape(-1, 3),
        "pnp_rvec": rvec,
        "pnp_tvec": tvec,
        "pnp_reprojection_errors_px": pnp_errors,
    }


def project_errors(
    ids: np.ndarray,
    image_points: np.ndarray,
    board: cv2.aruco.CharucoBoard,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> np.ndarray:
    object_points = board.getChessboardCorners()[ids].astype(np.float64).reshape(-1, 1, 3)
    projected, _ = cv2.fisheye.projectPoints(object_points, rvec, tvec, K_DEFAULT, D_DEFAULT)
    return np.linalg.norm(projected.reshape(-1, 2) - image_points.reshape(-1, 2), axis=1)


def triangulate_multiview(
    observations: dict[str, np.ndarray],
    poses: dict[str, tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    rows = []
    for cam, uv in observations.items():
        rvec, tvec = poses[cam]
        r = cv2.Rodrigues(rvec)[0]
        p = np.hstack([r, tvec.reshape(3, 1)])
        uv_arr = np.array(uv, dtype=np.float64).reshape(1, 1, 2)
        xy = cv2.fisheye.undistortPoints(uv_arr, K_DEFAULT, D_DEFAULT).reshape(2)
        rows.append(xy[0] * p[2] - p[0])
        rows.append(xy[1] * p[2] - p[1])
    _, _, vt = np.linalg.svd(np.stack(rows, axis=0))
    x_h = vt[-1]
    return x_h[:3] / x_h[3]


def summarize_errors(errors: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "p95": float(np.percentile(errors, 95)),
        "max": float(np.max(errors)),
    }


def evaluate_combo(
    combo: tuple[str, ...],
    detections: dict[str, dict],
    poses: dict[str, tuple[np.ndarray, np.ndarray]],
    board: cv2.aruco.CharucoBoard,
) -> dict:
    id_sets = [set(detections[cam]["ids"].tolist()) for cam in combo]
    common_ids = sorted(set.intersection(*id_sets))
    board_points = board.getChessboardCorners().astype(np.float64)
    corner_maps = {
        cam: {
            int(corner_id): detections[cam]["image_points"][i]
            for i, corner_id in enumerate(detections[cam]["ids"])
        }
        for cam in combo
    }
    xyz = []
    expected = []
    for corner_id in common_ids:
        observations = {cam: corner_maps[cam][corner_id] for cam in combo}
        xyz.append(triangulate_multiview(observations, poses))
        expected.append(board_points[corner_id])
    xyz_arr = np.array(xyz)
    expected_arr = np.array(expected)
    errors = np.linalg.norm(xyz_arr - expected_arr, axis=1)

    reproj_errors = []
    for point, corner_id in zip(xyz_arr, common_ids):
        for cam in combo:
            rvec, tvec = poses[cam]
            projected, _ = cv2.fisheye.projectPoints(
                point.reshape(1, 1, 3).astype(np.float64),
                rvec,
                tvec,
                K_DEFAULT,
                D_DEFAULT,
            )
            observed = corner_maps[cam][corner_id]
            reproj_errors.append(float(np.linalg.norm(projected.reshape(2) - observed)))

    stats = summarize_errors(errors)
    return {
        "combo": combo,
        "num_cameras": len(combo),
        "num_common_corners": len(common_ids),
        "corner_ids": common_ids,
        "xyz": xyz_arr,
        "expected": expected_arr,
        "errors_mm": errors,
        "triangulation_mean_mm": stats["mean"],
        "triangulation_median_mm": stats["median"],
        "triangulation_p95_mm": stats["p95"],
        "triangulation_max_mm": stats["max"],
        "reprojected_triangulated_mean_px": float(np.mean(reproj_errors)),
        "reprojected_triangulated_median_px": float(np.median(reproj_errors)),
    }


def board_axes(ax: plt.Axes) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-10, 570)
    ax.set_ylim(410, -10)
    ax.set_xlabel("board X (mm)")
    ax.set_ylabel("board Y (mm)")
    ax.grid(color="#d6dbe1", linewidth=0.5, alpha=0.9)
    outline = np.array([[0, 0], [560, 0], [560, 400], [0, 400], [0, 0]], dtype=float)
    ax.plot(outline[:, 0], outline[:, 1], color="black", linewidth=1.2)


def save_reprojection_heatmap(
    path: Path,
    camera_eval: dict[str, dict],
    board: cv2.aruco.CharucoBoard,
) -> None:
    all_errors = np.concatenate(
        [item["aggregate_reprojection_errors_px"] for item in camera_eval.values()]
    )
    vmax = max(1.0, float(np.percentile(all_errors, 95)))
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for ax, (cam, item) in zip(axes.ravel(), camera_eval.items()):
        ids = item["ids"]
        xy = board.getChessboardCorners()[ids][:, :2]
        errors = item["aggregate_reprojection_errors_px"]
        scatter = ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=errors,
            cmap="magma",
            vmin=0,
            vmax=vmax,
            s=58,
            edgecolor="white",
            linewidth=0.35,
        )
        board_axes(ax)
        ax.set_title(
            f"{cam}: median {np.median(errors):.2f}px, "
            f"p95 {np.percentile(errors, 95):.2f}px"
        )
    fig.colorbar(scatter, ax=axes.ravel().tolist(), label="aggregate-pose reprojection error (px)")
    fig.suptitle("ChArUco Reprojection Error Heatmaps", fontsize=15)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_triangulation_heatmaps(path: Path, combo_results: list[dict]) -> None:
    sorted_results = sorted(combo_results, key=lambda item: (item["num_cameras"], item["combo"]))
    errors_all = np.concatenate([item["errors_mm"] for item in sorted_results])
    vmax = max(2.0, float(np.percentile(errors_all, 95)))
    fig, axes = plt.subplots(4, 3, figsize=(15, 16), constrained_layout=True)
    axes_flat = axes.ravel()
    scatter = None
    for ax, result in zip(axes_flat, sorted_results):
        expected_xy = result["expected"][:, :2]
        scatter = ax.scatter(
            expected_xy[:, 0],
            expected_xy[:, 1],
            c=result["errors_mm"],
            cmap="viridis",
            vmin=0,
            vmax=vmax,
            s=48,
            edgecolor="white",
            linewidth=0.35,
        )
        board_axes(ax)
        ax.set_title(
            f"{'+'.join(result['combo'])}\n"
            f"n={result['num_common_corners']}, med {result['triangulation_median_mm']:.2f}mm, "
            f"p95 {result['triangulation_p95_mm']:.2f}mm"
        )
    for ax in axes_flat[len(sorted_results) :]:
        ax.axis("off")
    if scatter is not None:
        fig.colorbar(scatter, ax=axes_flat.tolist(), label="3D board-corner triangulation error (mm)")
    fig.suptitle("ChArUco Board-Corner Triangulation Error Heatmaps", fontsize=15)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_combo_bar_chart(path: Path, combo_results: list[dict]) -> None:
    sorted_results = sorted(
        combo_results,
        key=lambda item: (item["num_cameras"], item["triangulation_median_mm"], item["combo"]),
    )
    labels = ["+".join(item["combo"]) for item in sorted_results]
    medians = [item["triangulation_median_mm"] for item in sorted_results]
    p95 = [item["triangulation_p95_mm"] for item in sorted_results]
    colors = {
        2: "#2374ab",
        3: "#2a9d55",
        4: "#7c4dbe",
    }
    fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
    x = np.arange(len(labels))
    ax.bar(
        x,
        medians,
        color=[colors[item["num_cameras"]] for item in sorted_results],
        label="median",
    )
    ax.scatter(x, p95, color="black", s=28, zorder=3, label="p95")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("triangulation error (mm)")
    ax.set_title("Board-Corner Triangulation: Median Error by Camera Combination")
    ax.grid(axis="y", color="#d6dbe1", alpha=0.8)
    ax.legend()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def write_combo_csv(path: Path, combo_results: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "combo",
                "num_cameras",
                "num_common_corners",
                "triangulation_mean_mm",
                "triangulation_median_mm",
                "triangulation_p95_mm",
                "triangulation_max_mm",
                "reprojected_triangulated_mean_px",
                "reprojected_triangulated_median_px",
            ],
        )
        writer.writeheader()
        for item in sorted(combo_results, key=lambda x: (x["num_cameras"], x["combo"])):
            writer.writerow(
                {
                    "combo": "+".join(item["combo"]),
                    "num_cameras": item["num_cameras"],
                    "num_common_corners": item["num_common_corners"],
                    "triangulation_mean_mm": item["triangulation_mean_mm"],
                    "triangulation_median_mm": item["triangulation_median_mm"],
                    "triangulation_p95_mm": item["triangulation_p95_mm"],
                    "triangulation_max_mm": item["triangulation_max_mm"],
                    "reprojected_triangulated_mean_px": item["reprojected_triangulated_mean_px"],
                    "reprojected_triangulated_median_px": item["reprojected_triangulated_median_px"],
                }
            )


def evaluate_segment(segment_name: str, report_path: Path) -> dict:
    report = json.loads(report_path.read_text())
    board = make_board()
    detector = cv2.aruco.CharucoDetector(board)
    segment_out = EVAL_ROOT / segment_name
    segment_out.mkdir(parents=True, exist_ok=True)

    cameras = report["extrinsics"]["cameras"]
    poses = {}
    detections = {}
    camera_eval = {}
    for cam, cam_report in cameras.items():
        rvec = np.array(cam_report["rvec_world_to_cam"], dtype=np.float64).reshape(3, 1)
        tvec = np.array(cam_report["t_world_to_cam_mm"], dtype=np.float64).reshape(3, 1)
        poses[cam] = (rvec, tvec)
        frame = read_frame(Path(cam_report["video"]), cam_report["best_visualized_frame_s"])
        detection = detect_charuco(frame, board, detector)
        detections[cam] = detection
        aggregate_errors = project_errors(
            detection["ids"],
            detection["image_points"],
            board,
            rvec,
            tvec,
        )
        camera_eval[cam] = {
            "video": cam_report["video"],
            "best_visualized_frame_s": cam_report["best_visualized_frame_s"],
            "ids": detection["ids"],
            "num_corners": int(len(detection["ids"])),
            "pnp_reprojection_errors_px": detection["pnp_reprojection_errors_px"],
            "aggregate_reprojection_errors_px": aggregate_errors,
        }

    combo_results = []
    camera_names = tuple(sorted(cameras.keys()))
    for combo_size in (2, 3, 4):
        for combo in itertools.combinations(camera_names, combo_size):
            combo_results.append(evaluate_combo(combo, detections, poses, board))

    save_reprojection_heatmap(segment_out / "reprojection_error_heatmaps.png", camera_eval, board)
    save_triangulation_heatmaps(
        segment_out / "triangulation_error_heatmaps_all_combos.png",
        combo_results,
    )
    save_combo_bar_chart(segment_out / "triangulation_combo_median_errors.png", combo_results)
    write_combo_csv(segment_out / "triangulation_combo_summary.csv", combo_results)

    grouped = {}
    for combo_size in (2, 3, 4):
        group = [item for item in combo_results if item["num_cameras"] == combo_size]
        best = min(group, key=lambda item: item["triangulation_median_mm"])
        worst = max(group, key=lambda item: item["triangulation_median_mm"])
        grouped[str(combo_size)] = {
            "num_combinations": len(group),
            "best_combo": "+".join(best["combo"]),
            "best_median_mm": best["triangulation_median_mm"],
            "best_p95_mm": best["triangulation_p95_mm"],
            "worst_combo": "+".join(worst["combo"]),
            "worst_median_mm": worst["triangulation_median_mm"],
            "mean_of_combo_medians_mm": float(np.mean([x["triangulation_median_mm"] for x in group])),
            "median_of_combo_medians_mm": float(np.median([x["triangulation_median_mm"] for x in group])),
        }

    serializable_camera_eval = {}
    for cam, item in camera_eval.items():
        serializable_camera_eval[cam] = {
            "video": item["video"],
            "best_visualized_frame_s": item["best_visualized_frame_s"],
            "num_corners": item["num_corners"],
            "pnp_reprojection_px": summarize_errors(item["pnp_reprojection_errors_px"]),
            "aggregate_pose_reprojection_px": summarize_errors(item["aggregate_reprojection_errors_px"]),
        }

    serializable_combos = []
    for item in sorted(combo_results, key=lambda x: (x["num_cameras"], x["combo"])):
        serializable_combos.append(
            {
                "combo": "+".join(item["combo"]),
                "num_cameras": item["num_cameras"],
                "num_common_corners": item["num_common_corners"],
                "triangulation_mean_mm": item["triangulation_mean_mm"],
                "triangulation_median_mm": item["triangulation_median_mm"],
                "triangulation_p95_mm": item["triangulation_p95_mm"],
                "triangulation_max_mm": item["triangulation_max_mm"],
                "reprojected_triangulated_mean_px": item["reprojected_triangulated_mean_px"],
                "reprojected_triangulated_median_px": item["reprojected_triangulated_median_px"],
            }
        )

    segment_summary = {
        "segment": segment_name,
        "source_report": str(report_path),
        "camera_reprojection": serializable_camera_eval,
        "triangulation_by_camera_count": grouped,
        "triangulation_combos": serializable_combos,
        "generated_files": {
            "combo_csv": str(segment_out / "triangulation_combo_summary.csv"),
            "reprojection_heatmaps": str(segment_out / "reprojection_error_heatmaps.png"),
            "triangulation_heatmaps": str(segment_out / "triangulation_error_heatmaps_all_combos.png"),
            "combo_bar_chart": str(segment_out / "triangulation_combo_median_errors.png"),
        },
    }
    (segment_out / "summary.json").write_text(json.dumps(segment_summary, indent=2))
    return segment_summary


def main() -> None:
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    summaries = [evaluate_segment(name, path) for name, path in SEGMENTS]
    overall = {
        "note": "segment1/segment2 are file segments from one recording session, not separate recording sessions.",
        "summaries": summaries,
    }
    (EVAL_ROOT / "summary.json").write_text(json.dumps(overall, indent=2))
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
