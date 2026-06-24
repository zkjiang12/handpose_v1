#!/usr/bin/env python3
"""Triangulate manually labeled stereo 21-keypoint hand annotations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


KEYPOINTS = [
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

HAND_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]

PALM_EDGES = [(5, 9), (9, 13), (13, 17)]

FINGER_COLORS_BGR = {
    "thumb": (92, 92, 255),
    "index": (89, 166, 35),
    "middle": (247, 129, 47),
    "ring": (34, 153, 210),
    "pinky": (247, 113, 163),
    "palm": (158, 148, 139),
}

FINGER_COLORS_MPL = {
    "thumb": "#ff5c5c",
    "index": "#23a559",
    "middle": "#2f81f7",
    "ring": "#d29922",
    "pinky": "#a371f7",
    "palm": "#8b949e",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Triangulate stereo hand labels and render 3D bone lengths."
    )
    parser.add_argument("--labels-json", type=Path, required=True)
    parser.add_argument(
        "--ego-exo-root",
        type=Path,
        default=Path("/Users/zikangjiang/dev/ego-exo"),
    )
    parser.add_argument(
        "--extrinsics-report",
        type=Path,
        default=Path("/Users/zikangjiang/dev/ego-exo/visualizations/extrinsics_report.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/Users/zikangjiang/dev/ego-exo/visualizations/left_hand_labeling/results"),
    )
    parser.add_argument(
        "--pose-mode",
        choices=("auto", "aggregate", "frame-charuco"),
        default="auto",
        help="auto tries exact-frame ChArUco pose and falls back to aggregate extrinsics.",
    )
    parser.add_argument("--square-mm", type=float, default=40.0)
    parser.add_argument("--marker-mm", type=float, default=20.0)
    parser.add_argument("--board-cols", type=int, default=14)
    parser.add_argument("--board-rows", type=int, default=10)
    return parser.parse_args()


def finger_for_index(index: int) -> str:
    if 1 <= index <= 4:
        return "thumb"
    if 5 <= index <= 8:
        return "index"
    if 9 <= index <= 12:
        return "middle"
    if 13 <= index <= 16:
        return "ring"
    if 17 <= index <= 20:
        return "pinky"
    return "palm"


def edge_name(edge: tuple[int, int]) -> str:
    return f"{KEYPOINTS[edge[0]]}_to_{KEYPOINTS[edge[1]]}"


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def read_frame(video: Path, time_s: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    cap.set(cv2.CAP_PROP_POS_MSEC, float(time_s) * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame at {time_s}s from {video}")
    return frame


def normalize_labels(labels: dict[str, Any]) -> dict[str, Any]:
    if labels.get("image_size") != [1920, 1080]:
        raise ValueError(f"Expected image_size [1920, 1080], got {labels.get('image_size')}")
    if "cameras" not in labels or "points" not in labels:
        raise ValueError("Labels JSON must contain cameras and points.")

    points_by_name: dict[str, dict[str, list[float]]] = {}
    raw_points = labels["points"]
    if isinstance(raw_points, dict):
        for name in KEYPOINTS:
            by_cam = raw_points.get(name, {})
            points_by_name[name] = {
                cam: [float(xy[0]), float(xy[1])]
                for cam, xy in by_cam.items()
                if cam in {"cam1", "cam2"} and xy is not None
            }
    elif isinstance(raw_points, list):
        for index, name in enumerate(KEYPOINTS):
            by_cam = raw_points[index] if index < len(raw_points) else {}
            points_by_name[name] = {
                cam: [float(xy[0]), float(xy[1])]
                for cam, xy in by_cam.items()
                if cam in {"cam1", "cam2"} and xy is not None
            }
    else:
        raise ValueError("Unsupported points format.")

    labels = dict(labels)
    labels["points"] = points_by_name
    labels.setdefault("keypoint_names", KEYPOINTS)
    labels.setdefault("edges", HAND_EDGES)
    labels.setdefault("hand_side", "left")
    labels.setdefault("keypoint_convention", "mediapipe_21")
    return labels


def aggregate_pose_from_report(report: dict[str, Any], cam: str) -> dict[str, Any]:
    data = report["cameras"][cam]
    rvec = np.asarray(data["rvec_world_to_cam"], dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(data["t_world_to_cam_mm"], dtype=np.float64).reshape(3, 1)
    rmat = np.asarray(data["R_world_to_cam"], dtype=np.float64)
    return {
        "rvec": rvec,
        "tvec": tvec,
        "rmat": rmat,
        "source": "aggregate_extrinsics_report",
        "charuco_corners": None,
        "reprojection_mean_px": data.get("reprojection_mean_px"),
        "reprojection_median_px": data.get("reprojection_median_px"),
    }


def solve_frame_charuco_pose(
    frame: np.ndarray,
    k: np.ndarray,
    d: np.ndarray,
    *,
    board_cols: int,
    board_rows: int,
    square_mm: float,
    marker_mm: float,
) -> dict[str, Any] | None:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    board = cv2.aruco.CharucoBoard(
        (board_cols, board_rows),
        square_mm,
        marker_mm,
        dictionary,
    )
    detector = cv2.aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, _, _ = detector.detectBoard(frame)
    if charuco_ids is None or len(charuco_ids) < 30:
        return None

    ids = charuco_ids.flatten().astype(np.int32)
    img_pts = charuco_corners.astype(np.float64).reshape(-1, 1, 2)
    obj_pts = board.getChessboardCorners()[ids].astype(np.float64).reshape(-1, 1, 3)
    norm_pts = cv2.fisheye.undistortPoints(img_pts, k, d)
    ok, rvec, tvec = cv2.solvePnP(
        obj_pts,
        norm_pts,
        np.eye(3, dtype=np.float64),
        None,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    projected, _ = cv2.fisheye.projectPoints(obj_pts, rvec, tvec, k, d)
    errors = np.linalg.norm(projected.reshape(-1, 2) - img_pts.reshape(-1, 2), axis=1)
    if float(np.median(errors)) > 4.0:
        return None
    return {
        "rvec": rvec,
        "tvec": tvec,
        "rmat": cv2.Rodrigues(rvec)[0],
        "source": "exact_frame_charuco",
        "charuco_corners": int(len(charuco_ids)),
        "reprojection_mean_px": float(errors.mean()),
        "reprojection_median_px": float(np.median(errors)),
    }


def choose_poses(
    labels: dict[str, Any],
    report: dict[str, Any],
    frames: dict[str, np.ndarray],
    k: np.ndarray,
    d: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    poses: dict[str, dict[str, Any]] = {}
    for cam in ("cam1", "cam2"):
        aggregate = aggregate_pose_from_report(report, cam)
        if args.pose_mode == "aggregate":
            poses[cam] = aggregate
            continue
        frame_pose = solve_frame_charuco_pose(
            frames[cam],
            k,
            d,
            board_cols=args.board_cols,
            board_rows=args.board_rows,
            square_mm=args.square_mm,
            marker_mm=args.marker_mm,
        )
        if frame_pose is not None:
            poses[cam] = frame_pose
        elif args.pose_mode == "frame-charuco":
            raise RuntimeError(f"Could not solve exact-frame ChArUco pose for {cam}.")
        else:
            poses[cam] = aggregate
    return poses


def camera_projection_matrix(pose: dict[str, Any]) -> np.ndarray:
    return np.hstack([pose["rmat"], pose["tvec"].reshape(3, 1)])


def undistort_norm(xy: list[float], k: np.ndarray, d: np.ndarray) -> np.ndarray:
    pts = np.asarray(xy, dtype=np.float64).reshape(1, 1, 2)
    return cv2.fisheye.undistortPoints(pts, k, d).reshape(2)


def project_world(points_world: np.ndarray, pose: dict[str, Any], k: np.ndarray, d: np.ndarray) -> np.ndarray:
    points = np.asarray(points_world, dtype=np.float64).reshape(-1, 1, 3)
    projected, _ = cv2.fisheye.projectPoints(points, pose["rvec"], pose["tvec"], k, d)
    return projected.reshape(-1, 2)


def camera_center_world(pose: dict[str, Any]) -> np.ndarray:
    return -pose["rmat"].T @ pose["tvec"].reshape(3)


def ray_world(pose: dict[str, Any], xy: list[float], k: np.ndarray, d: np.ndarray) -> np.ndarray:
    norm = undistort_norm(xy, k, d)
    direction_cam = np.array([norm[0], norm[1], 1.0], dtype=np.float64)
    direction_cam /= np.linalg.norm(direction_cam)
    return pose["rmat"].T @ direction_cam


def closest_ray_gap(
    obs: dict[str, list[float]],
    poses: dict[str, dict[str, Any]],
    k: np.ndarray,
    d: np.ndarray,
) -> dict[str, Any]:
    p1 = camera_center_world(poses["cam1"])
    p2 = camera_center_world(poses["cam2"])
    d1 = ray_world(poses["cam1"], obs["cam1"], k, d)
    d2 = ray_world(poses["cam2"], obs["cam2"], k, d)
    d1 /= np.linalg.norm(d1)
    d2 /= np.linalg.norm(d2)
    w0 = p1 - p2
    a = float(d1 @ d1)
    b = float(d1 @ d2)
    c = float(d2 @ d2)
    d0 = float(d1 @ w0)
    e = float(d2 @ w0)
    denom = a * c - b * b
    if abs(denom) < 1e-12:
        s = 0.0
        t = e / c
    else:
        s = (b * e - c * d0) / denom
        t = (a * e - b * d0) / denom
    q1 = p1 + s * d1
    q2 = p2 + t * d2
    return {
        "ray_gap_mm": float(np.linalg.norm(q1 - q2)),
        "closest_point_cam1_ray_world_mm": q1.tolist(),
        "closest_point_cam2_ray_world_mm": q2.tolist(),
        "midpoint_world_mm": ((q1 + q2) / 2.0).tolist(),
    }


def triangulate_labels(
    labels: dict[str, Any],
    poses: dict[str, dict[str, Any]],
    k: np.ndarray,
    d: np.ndarray,
) -> dict[str, Any]:
    projection = {
        cam: camera_projection_matrix(poses[cam])
        for cam in ("cam1", "cam2")
    }
    points_world = np.full((len(KEYPOINTS), 3), np.nan, dtype=np.float64)
    valid = np.zeros(len(KEYPOINTS), dtype=bool)
    reprojection: dict[str, Any] = {}
    ray_consistency: dict[str, Any] = {}

    for index, name in enumerate(KEYPOINTS):
        obs = labels["points"].get(name, {})
        if "cam1" not in obs or "cam2" not in obs:
            continue
        norm1 = undistort_norm(obs["cam1"], k, d).reshape(2, 1)
        norm2 = undistort_norm(obs["cam2"], k, d).reshape(2, 1)
        homog = cv2.triangulatePoints(projection["cam1"], projection["cam2"], norm1, norm2)
        xyz = (homog[:3, 0] / homog[3, 0]).astype(float)
        points_world[index] = xyz
        valid[index] = True

        reprojection[name] = {}
        for cam in ("cam1", "cam2"):
            projected = project_world(xyz.reshape(1, 3), poses[cam], k, d)[0]
            observed = np.asarray(obs[cam], dtype=np.float64)
            reprojection[name][cam] = {
                "observed": observed.tolist(),
                "projected": projected.tolist(),
                "error_px": float(np.linalg.norm(projected - observed)),
            }
        ray_consistency[name] = closest_ray_gap(obs, poses, k, d)

    bone_lengths = {}
    for edge in HAND_EDGES:
        i, j = edge
        if valid[i] and valid[j]:
            bone_lengths[edge_name(edge)] = float(np.linalg.norm(points_world[i] - points_world[j]))

    palm_span_lengths = {}
    for edge in PALM_EDGES:
        i, j = edge
        if valid[i] and valid[j]:
            palm_span_lengths[edge_name(edge)] = float(np.linalg.norm(points_world[i] - points_world[j]))

    reproj_errors = [
        cam_data["error_px"]
        for kp_data in reprojection.values()
        for cam_data in kp_data.values()
    ]
    ray_gaps = [data["ray_gap_mm"] for data in ray_consistency.values()]

    return {
        "points_world_mm": {
            name: points_world[index].tolist() if valid[index] else None
            for index, name in enumerate(KEYPOINTS)
        },
        "valid_keypoints": {
            name: bool(valid[index])
            for index, name in enumerate(KEYPOINTS)
        },
        "num_valid_keypoints": int(valid.sum()),
        "bone_lengths_mm": bone_lengths,
        "palm_span_lengths_mm": palm_span_lengths,
        "reprojection": reprojection,
        "reprojection_summary_px": {
            "mean": float(np.mean(reproj_errors)) if reproj_errors else None,
            "median": float(np.median(reproj_errors)) if reproj_errors else None,
            "max": float(np.max(reproj_errors)) if reproj_errors else None,
        },
        "ray_consistency": ray_consistency,
        "ray_gap_summary_mm": {
            "mean": float(np.mean(ray_gaps)) if ray_gaps else None,
            "median": float(np.median(ray_gaps)) if ray_gaps else None,
            "max": float(np.max(ray_gaps)) if ray_gaps else None,
        },
    }


def put_text(
    image: np.ndarray,
    text: str,
    xy: tuple[int, int],
    *,
    scale: float = 0.48,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        xy,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        xy,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_hand_overlay(
    frame: np.ndarray,
    cam: str,
    labels: dict[str, Any],
    triangulation: dict[str, Any],
    poses: dict[str, dict[str, Any]],
    k: np.ndarray,
    d: np.ndarray,
    *,
    draw_lengths: bool,
) -> np.ndarray:
    image = frame.copy()
    points = np.asarray(
        [
            triangulation["points_world_mm"][name]
            if triangulation["points_world_mm"][name] is not None
            else [np.nan, np.nan, np.nan]
            for name in KEYPOINTS
        ],
        dtype=np.float64,
    )
    valid = np.isfinite(points).all(axis=1)
    projected = np.full((len(KEYPOINTS), 2), np.nan, dtype=np.float64)
    if valid.any():
        projected[valid] = project_world(points[valid], poses[cam], k, d)

    for edge in HAND_EDGES:
        i, j = edge
        if not (valid[i] and valid[j]):
            continue
        color = FINGER_COLORS_BGR[finger_for_index(j)]
        p1_arr = projected[i]
        p2_arr = projected[j]
        p1 = tuple(np.round(p1_arr).astype(int))
        p2 = tuple(np.round(p2_arr).astype(int))
        cv2.line(image, p1, p2, color, 3, cv2.LINE_AA)
        if draw_lengths:
            length = triangulation["bone_lengths_mm"].get(edge_name(edge))
            if length is not None:
                mid = tuple(np.round((p1_arr + p2_arr) / 2.0).astype(int))
                put_text(
                    image,
                    f"{length:.0f}mm",
                    (mid[0] + 4, mid[1] - 4),
                    scale=0.45,
                    color=(255, 255, 255),
                    thickness=1,
                )

    for index, name in enumerate(KEYPOINTS):
        obs = labels["points"].get(name, {}).get(cam)
        if obs is not None:
            center = tuple(np.round(obs).astype(int))
            color = FINGER_COLORS_BGR[finger_for_index(index)]
            cv2.circle(image, center, 6, color, -1, cv2.LINE_AA)
            cv2.circle(image, center, 10, (255, 255, 255), 2, cv2.LINE_AA)
            put_text(
                image,
                str(index),
                (center[0] + 9, center[1] - 9),
                scale=0.42,
                color=(255, 255, 255),
                thickness=1,
            )
        if valid[index]:
            pred = tuple(np.round(projected[index]).astype(int))
            cv2.drawMarker(
                image,
                pred,
                (255, 0, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=16,
                thickness=2,
                line_type=cv2.LINE_AA,
            )

    cam_meta = labels["cameras"][cam]
    summary = triangulation["reprojection_summary_px"]
    text = (
        f"{cam} t={cam_meta['time_s']:.3f}s | dot=clicked | cross=reprojected | "
        f"mean reproj={summary['mean']:.2f}px"
        if summary["mean"] is not None
        else f"{cam} t={cam_meta['time_s']:.3f}s"
    )
    put_text(image, text, (35, 45), scale=0.9, color=(255, 255, 255), thickness=2)
    return image


def crop_around_hand(images: list[np.ndarray], labels: dict[str, Any], triangulation: dict[str, Any]) -> list[np.ndarray]:
    cropped = []
    for image, cam in zip(images, ("cam1", "cam2"), strict=True):
        pts = []
        for name in KEYPOINTS:
            obs = labels["points"].get(name, {}).get(cam)
            if obs is not None:
                pts.append(obs)
            reproj = triangulation["reprojection"].get(name, {}).get(cam, {}).get("projected")
            if reproj is not None:
                pts.append(reproj)
        if not pts:
            cropped.append(image)
            continue
        arr = np.asarray(pts, dtype=np.float64)
        x0 = max(0, int(math.floor(arr[:, 0].min() - 130)))
        y0 = max(0, int(math.floor(arr[:, 1].min() - 130)))
        x1 = min(image.shape[1], int(math.ceil(arr[:, 0].max() + 170)))
        y1 = min(image.shape[0], int(math.ceil(arr[:, 1].max() + 170)))
        crop = image[y0:y1, x0:x1].copy()
        cropped.append(cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC))
    return cropped


def hstack_same_height(images: list[np.ndarray]) -> np.ndarray:
    height = max(image.shape[0] for image in images)
    padded = []
    for image in images:
        if image.shape[0] < height:
            pad = np.zeros((height - image.shape[0], image.shape[1], 3), dtype=image.dtype)
            image = np.vstack([image, pad])
        padded.append(image)
    return np.hstack(padded)


def save_2d_outputs(
    out_dir: Path,
    stem: str,
    frames: dict[str, np.ndarray],
    labels: dict[str, Any],
    triangulation: dict[str, Any],
    poses: dict[str, dict[str, Any]],
    k: np.ndarray,
    d: np.ndarray,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    full_images = []
    crop_images = []
    for cam in ("cam1", "cam2"):
        full = draw_hand_overlay(
            frames[cam],
            cam,
            labels,
            triangulation,
            poses,
            k,
            d,
            draw_lengths=True,
        )
        full_images.append(cv2.resize(full, (960, int(full.shape[0] * 960 / full.shape[1]))))

        crop_src = draw_hand_overlay(
            frames[cam],
            cam,
            labels,
            triangulation,
            poses,
            k,
            d,
            draw_lengths=True,
        )
        crop_images.append(crop_src)

    full_path = out_dir / f"{stem}_hand21_reprojection_lengths.jpg"
    cv2.imwrite(str(full_path), hstack_same_height(full_images))

    crop_path = out_dir / f"{stem}_hand21_zoom_lengths.jpg"
    cv2.imwrite(str(crop_path), hstack_same_height(crop_around_hand(crop_images, labels, triangulation)))
    return {"overlay_full": str(full_path), "overlay_zoom": str(crop_path)}


def set_equal_axes(ax: Any, points: np.ndarray) -> None:
    finite = points[np.isfinite(points).all(axis=1)]
    if len(finite) == 0:
        return
    center = finite.mean(axis=0)
    radius = max(float(np.ptp(finite, axis=0).max()) / 2.0, 40.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def save_3d_output(out_dir: Path, stem: str, triangulation: dict[str, Any]) -> str:
    points = np.asarray(
        [
            triangulation["points_world_mm"][name]
            if triangulation["points_world_mm"][name] is not None
            else [np.nan, np.nan, np.nan]
            for name in KEYPOINTS
        ],
        dtype=np.float64,
    )
    valid = np.isfinite(points).all(axis=1)
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Triangulated left hand, segment lengths in mm")

    for edge in HAND_EDGES:
        i, j = edge
        if not (valid[i] and valid[j]):
            continue
        segment = points[[i, j]]
        color = FINGER_COLORS_MPL[finger_for_index(j)]
        ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color=color, linewidth=2.4)
        length = triangulation["bone_lengths_mm"].get(edge_name(edge))
        if length is not None:
            mid = segment.mean(axis=0)
            ax.text(mid[0], mid[1], mid[2], f"{length:.0f}", fontsize=7, color="black")

    for index, point in enumerate(points):
        if not valid[index]:
            continue
        color = FINGER_COLORS_MPL[finger_for_index(index)]
        ax.scatter([point[0]], [point[1]], [point[2]], color=color, s=28)
        ax.text(point[0], point[1], point[2], str(index), fontsize=7)

    ax.set_xlabel("world X (mm)")
    ax.set_ylabel("world Y (mm)")
    ax.set_zlabel("world Z (mm)")
    set_equal_axes(ax, points)
    ax.view_init(elev=22, azim=-62)
    fig.tight_layout()
    path = out_dir / f"{stem}_hand21_3d_lengths.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def resolve_video_paths(labels: dict[str, Any], report: dict[str, Any], ego_exo_root: Path) -> dict[str, Path]:
    defaults = {
        "cam1": ego_exo_root / "v1-cam1" / "a77033-H-0001.mp4",
        "cam2": ego_exo_root / "v1-cam2" / "a72491-H-0001.mp4",
    }
    paths = {}
    for cam in ("cam1", "cam2"):
        report_path = Path(report["cameras"][cam]["video"])
        paths[cam] = report_path if report_path.exists() else defaults[cam]
    return paths


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    labels = normalize_labels(read_json(args.labels_json))
    report = read_json(args.extrinsics_report)
    k = np.asarray(report["intrinsics"], dtype=np.float64)
    d = np.asarray(report["distortion"], dtype=np.float64).reshape(4, 1)
    videos = resolve_video_paths(labels, report, args.ego_exo_root)

    frames = {
        cam: read_frame(videos[cam], float(labels["cameras"][cam]["time_s"]))
        for cam in ("cam1", "cam2")
    }
    poses = choose_poses(labels, report, frames, k, d, args)
    triangulation = triangulate_labels(labels, poses, k, d)

    stem = args.labels_json.stem
    generated = save_2d_outputs(args.out_dir, stem, frames, labels, triangulation, poses, k, d)
    generated["plot_3d"] = save_3d_output(args.out_dir, stem, triangulation)

    result = {
        "schema": "stereo_hand21_triangulation_v1",
        "input_labels_path": str(args.labels_json),
        "extrinsics_report": str(args.extrinsics_report),
        "pose_mode_requested": args.pose_mode,
        "camera_pose_sources": {
            cam: {
                "source": poses[cam]["source"],
                "charuco_corners": poses[cam]["charuco_corners"],
                "reprojection_mean_px": poses[cam]["reprojection_mean_px"],
                "reprojection_median_px": poses[cam]["reprojection_median_px"],
            }
            for cam in ("cam1", "cam2")
        },
        "hand_side": labels.get("hand_side", "left"),
        "keypoint_convention": labels.get("keypoint_convention", "mediapipe_21"),
        "keypoint_names": KEYPOINTS,
        "bone_edges": [[i, j] for i, j in HAND_EDGES],
        **triangulation,
        "generated_files": generated,
    }
    result_path = args.out_dir / f"{stem}_triangulated_hand21.json"
    with result_path.open("w") as f:
        json.dump(result, f, indent=2)

    summary = {
        "result_path": str(result_path),
        "num_valid_keypoints": result["num_valid_keypoints"],
        "reprojection_summary_px": result["reprojection_summary_px"],
        "ray_gap_summary_mm": result["ray_gap_summary_mm"],
        "generated_files": generated,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
