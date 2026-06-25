#!/usr/bin/env python3
"""Estimate v2 ChArUco extrinsics and audio sync offsets for ego-exo clips."""

from __future__ import annotations

import argparse
import json
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


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


@dataclass(frozen=True)
class CameraClip:
    name: str
    video: Path
    imu: Path


@dataclass(frozen=True)
class Take:
    name: str
    clips: tuple[CameraClip, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/Users/zikangjiang/dev/ego-exo/v2"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/Users/zikangjiang/dev/ego-exo/visualizations/v2"),
    )
    parser.add_argument("--square-mm", type=float, default=40.0)
    parser.add_argument("--marker-mm", type=float, default=20.0)
    parser.add_argument("--board-cols", type=int, default=14)
    parser.add_argument("--board-rows", type=int, default=10)
    parser.add_argument("--dict", type=str, default="DICT_5X5_100")
    parser.add_argument("--sample-step-s", type=int, default=5)
    parser.add_argument("--reference-camera", type=str, default="cam1")
    return parser.parse_args()


def build_takes(data_root: Path) -> list[Take]:
    return [
        Take(
            "take1",
            (
                CameraClip(
                    "cam1",
                    data_root / "camera-1" / "a77064-H-0001.mp4",
                    data_root / "camera-1" / "a77064-H-0001.txt",
                ),
                CameraClip(
                    "cam2",
                    data_root / "camera-2" / "a77033-H-0003.mp4",
                    data_root / "camera-2" / "a77033-H-0003.txt",
                ),
                CameraClip(
                    "cam3",
                    data_root / "camera-3" / "a72491-H-0003.mp4",
                    data_root / "camera-3" / "a72491-H-0003.txt",
                ),
                CameraClip(
                    "cam4",
                    data_root / "camera-4" / "a67244-H-0001.mp4",
                    data_root / "camera-4" / "a67244-H-0001.txt",
                ),
            ),
        ),
        Take(
            "take2",
            (
                CameraClip(
                    "cam1",
                    data_root / "camera-1" / "a77064-H-0002.mp4",
                    data_root / "camera-1" / "a77064-H-0002.txt",
                ),
                CameraClip(
                    "cam2",
                    data_root / "camera-2" / "a77033-H-0004.mp4",
                    data_root / "camera-2" / "a77033-H-0004.txt",
                ),
                CameraClip(
                    "cam3",
                    data_root / "camera-3" / "a72491-H-0004.mp4",
                    data_root / "camera-3" / "a72491-H-0004.txt",
                ),
                CameraClip(
                    "cam4",
                    data_root / "camera-4" / "a67244-H-0002.mp4",
                    data_root / "camera-4" / "a67244-H-0002.txt",
                ),
            ),
        ),
    ]


def aruco_dict(name: str) -> cv2.aruco.Dictionary:
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"Unknown ArUco dictionary: {name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def video_duration_s(video: Path) -> float:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps <= 0:
        raise RuntimeError(f"Could not read fps for {video}")
    return float(frames / fps)


def read_frame(video: Path, time_s: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    cap.set(cv2.CAP_PROP_POS_MSEC, time_s * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame at {time_s}s from {video}")
    return frame


def detect_and_solve(
    frame: np.ndarray,
    board: cv2.aruco.CharucoBoard,
    detector: cv2.aruco.CharucoDetector,
) -> dict | None:
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(frame)
    if charuco_ids is None or len(charuco_ids) < 12:
        return None

    ids = charuco_ids.flatten().astype(np.int32)
    img_pts = charuco_corners.astype(np.float64).reshape(-1, 1, 2)
    obj_pts = board.getChessboardCorners()[ids].astype(np.float64).reshape(-1, 1, 3)

    norm_pts = cv2.fisheye.undistortPoints(img_pts, K_DEFAULT, D_DEFAULT)
    ok, rvec, tvec = cv2.solvePnP(
        obj_pts,
        norm_pts,
        np.eye(3, dtype=np.float64),
        None,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None

    proj, _ = cv2.fisheye.projectPoints(obj_pts, rvec, tvec, K_DEFAULT, D_DEFAULT)
    errors = np.linalg.norm(proj.reshape(-1, 2) - img_pts.reshape(-1, 2), axis=1)

    return {
        "charuco_corners": charuco_corners,
        "charuco_ids": charuco_ids,
        "marker_corners": marker_corners,
        "marker_ids": marker_ids,
        "object_points": obj_pts,
        "image_points": img_pts,
        "projected_points": proj,
        "rvec": rvec,
        "tvec": tvec,
        "reprojection_errors_px": errors,
    }


def project_fisheye(points_world: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    pts = points_world.astype(np.float64).reshape(-1, 1, 3)
    proj, _ = cv2.fisheye.projectPoints(pts, rvec, tvec, K_DEFAULT, D_DEFAULT)
    return proj.reshape(-1, 2)


def draw_axes_and_outline(
    image: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    board_cols: int,
    board_rows: int,
    square_mm: float,
) -> None:
    axis_len = 120.0
    axis_pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [axis_len, 0.0, 0.0],
            [0.0, axis_len, 0.0],
            [0.0, 0.0, axis_len],
        ],
        dtype=np.float64,
    )
    proj = project_fisheye(axis_pts, rvec, tvec).astype(int)
    origin = tuple(proj[0])
    cv2.line(image, origin, tuple(proj[1]), (0, 0, 255), 4)
    cv2.line(image, origin, tuple(proj[2]), (0, 255, 0), 4)
    cv2.line(image, origin, tuple(proj[3]), (255, 0, 0), 4)
    cv2.putText(image, "origin", (origin[0] + 10, origin[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 3)
    cv2.putText(image, "X", tuple(proj[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    cv2.putText(image, "Y", tuple(proj[2]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
    cv2.putText(image, "Z", tuple(proj[3]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)

    width = board_cols * square_mm
    height = board_rows * square_mm
    outline = np.array(
        [
            [0.0, 0.0, 0.0],
            [width, 0.0, 0.0],
            [width, height, 0.0],
            [0.0, height, 0.0],
        ],
        dtype=np.float64,
    )
    outline_proj = project_fisheye(outline, rvec, tvec).astype(int)
    cv2.polylines(image, [outline_proj], True, (255, 255, 0), 3)


def save_detection_overlay(
    out_path: Path,
    frame: np.ndarray,
    solution: dict,
    board_cols: int,
    board_rows: int,
    square_mm: float,
) -> None:
    overlay = frame.copy()
    if solution["marker_ids"] is not None:
        cv2.aruco.drawDetectedMarkers(
            overlay,
            solution["marker_corners"],
            solution["marker_ids"],
            borderColor=(0, 255, 255),
        )
    cv2.aruco.drawDetectedCornersCharuco(
        overlay,
        solution["charuco_corners"],
        solution["charuco_ids"],
        cornerColor=(0, 255, 0),
    )
    draw_axes_and_outline(
        overlay,
        solution["rvec"],
        solution["tvec"],
        board_cols,
        board_rows,
        square_mm,
    )
    n_corners = len(solution["charuco_ids"])
    err = solution["reprojection_errors_px"]
    cv2.putText(
        overlay,
        f"{n_corners} corners; reproj mean {err.mean():.2f}px / med {np.median(err):.2f}px",
        (40, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        3,
    )
    cv2.imwrite(str(out_path), overlay)


def save_reprojection_overlay(out_path: Path, frame: np.ndarray, solution: dict) -> None:
    overlay = frame.copy()
    detected = solution["image_points"].reshape(-1, 2)
    projected = solution["projected_points"].reshape(-1, 2)
    errors = solution["reprojection_errors_px"]
    for (dx, dy), (px, py), err in zip(detected, projected, errors):
        color = (0, 255, 0) if err < 1.0 else (0, 180, 255) if err < 3.0 else (0, 0, 255)
        cv2.circle(overlay, (int(round(dx)), int(round(dy))), 4, color, -1)
        cv2.drawMarker(
            overlay,
            (int(round(px)), int(round(py))),
            (255, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=14,
            thickness=2,
        )
        if err >= 1.0:
            cv2.line(
                overlay,
                (int(round(dx)), int(round(dy))),
                (int(round(px)), int(round(py))),
                (0, 0, 255),
                1,
            )
    cv2.putText(
        overlay,
        "dot = detected ChArUco corner; magenta cross = projected world corner",
        (40, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.88,
        (255, 255, 255),
        3,
    )
    cv2.imwrite(str(out_path), overlay)


def average_rotation(rotations: list[np.ndarray]) -> np.ndarray:
    mean_r = np.mean(np.stack(rotations, axis=0), axis=0)
    u, _, vt = np.linalg.svd(mean_r)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1.0
        r = u @ vt
    return r


def camera_center_world(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    r = cv2.Rodrigues(rvec)[0]
    return -r.T @ tvec.reshape(3)


def aggregate_pose(solutions: list[dict]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    centers = np.array([camera_center_world(s["rvec"], s["tvec"]) for s in solutions])
    median_center = np.median(centers, axis=0)
    distances = np.linalg.norm(centers - median_center[None, :], axis=1)
    threshold = max(80.0, float(np.percentile(distances, 85) * 1.5))
    kept = [s for s, dist in zip(solutions, distances) if dist <= threshold]
    if len(kept) < max(3, len(solutions) // 3):
        kept = solutions

    rotations = [cv2.Rodrigues(s["rvec"])[0] for s in kept]
    translations = [s["tvec"].reshape(3) for s in kept]
    r = average_rotation(rotations)
    t = np.median(np.stack(translations, axis=0), axis=0)
    rvec, _ = cv2.Rodrigues(r)
    return rvec.reshape(3, 1), t.reshape(3, 1), kept


def camera_rays_world(rvec: np.ndarray, tvec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = cv2.Rodrigues(rvec)[0]
    center = camera_center_world(rvec, tvec)
    corners = np.array(
        [[0, 0], [1919, 0], [1919, 1079], [0, 1079]], dtype=np.float64
    ).reshape(-1, 1, 2)
    norm = cv2.fisheye.undistortPoints(corners, K_DEFAULT, D_DEFAULT).reshape(-1, 2)
    dirs_cam = np.column_stack([norm, np.ones(len(norm))])
    dirs_cam /= np.linalg.norm(dirs_cam, axis=1, keepdims=True)
    dirs_world = (r.T @ dirs_cam.T).T
    return center, dirs_world


def set_equal_3d_axes(ax: plt.Axes) -> None:
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    centers = limits.mean(axis=1)
    radius = 0.5 * np.max(limits[:, 1] - limits[:, 0])
    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])


def save_world_scene(
    out_path: Path,
    poses: dict[str, tuple[np.ndarray, np.ndarray]],
    board_cols: int,
    board_rows: int,
    square_mm: float,
) -> dict:
    width = board_cols * square_mm
    height = board_rows * square_mm
    colors = {"cam1": "tab:red", "cam2": "tab:blue", "cam3": "tab:green", "cam4": "tab:purple"}

    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("v2 ChArUco world frame and exo camera poses\n(display height = -world Z)")

    board = np.array(
        [[0, 0, 0], [width, 0, 0], [width, height, 0], [0, height, 0], [0, 0, 0]],
        dtype=float,
    )
    board_display = board.copy()
    board_display[:, 2] *= -1.0
    ax.plot(board_display[:, 0], board_display[:, 1], board_display[:, 2], color="black", linewidth=2)
    ax.scatter(board_display[:, 0], board_display[:, 1], board_display[:, 2], color="black", s=24)
    ax.text(0, 0, 0, "origin / ChArUco board", color="black")

    grid_x = np.arange(0, width + 1e-6, square_mm)
    grid_y = np.arange(0, height + 1e-6, square_mm)
    for x in grid_x:
        ax.plot([x, x], [0, height], [0, 0], color="lightgray", linewidth=0.5)
    for y in grid_y:
        ax.plot([0, width], [y, y], [0, 0], color="lightgray", linewidth=0.5)

    scene_report = {
        "note": "Display uses height=-world_Z so cameras appear above the board. Raw extrinsics are unchanged.",
        "board_size_mm": [width, height],
        "cameras": {},
    }
    for cam, (rvec, tvec) in poses.items():
        center, dirs = camera_rays_world(rvec, tvec)
        center_display = center.copy()
        center_display[2] *= -1.0
        dirs_display = dirs.copy()
        dirs_display[:, 2] *= -1.0
        color = colors.get(cam, "tab:gray")
        ax.scatter([center_display[0]], [center_display[1]], [center_display[2]], color=color, s=90)
        ax.text(center_display[0], center_display[1], center_display[2], cam, color=color)

        scale = 180.0
        ends = center_display[None, :] + scale * dirs_display
        for end in ends:
            ax.plot(
                [center_display[0], end[0]],
                [center_display[1], end[1]],
                [center_display[2], end[2]],
                color=color,
                linewidth=1.15,
            )
        for i in range(4):
            a = ends[i]
            b = ends[(i + 1) % 4]
            ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color=color, linewidth=1.0)

        scene_report["cameras"][cam] = {
            "world_center_mm": center.tolist(),
            "display_x_y_height_mm": center_display.tolist(),
        }

    ax.set_xlabel("world X (mm)")
    ax.set_ylabel("world Y (mm)")
    ax.set_zlabel("display height = -world Z (mm)")
    set_equal_3d_axes(ax)
    ax.view_init(elev=24, azim=-58)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return scene_report


def triangulate_points_multiview(
    observations: dict[str, np.ndarray],
    poses: dict[str, tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    rows = []
    for cam, uv in observations.items():
        rvec, tvec = poses[cam]
        r = cv2.Rodrigues(rvec)[0]
        p = np.hstack([r, tvec.reshape(3, 1)])
        uv_arr = np.array(uv, dtype=np.float64).reshape(-1, 1, 2)
        xy = cv2.fisheye.undistortPoints(uv_arr, K_DEFAULT, D_DEFAULT).reshape(-1, 2)[0]
        rows.append(xy[0] * p[2] - p[0])
        rows.append(xy[1] * p[2] - p[1])
    _, _, vt = np.linalg.svd(np.stack(rows, axis=0))
    x_h = vt[-1]
    return x_h[:3] / x_h[3]


def validate_board_triangulation(
    best_solutions: dict[str, dict],
    poses: dict[str, tuple[np.ndarray, np.ndarray]],
    board: cv2.aruco.CharucoBoard,
) -> dict | None:
    id_sets = []
    corner_maps = {}
    for cam, sol in best_solutions.items():
        ids = sol["charuco_ids"].flatten().astype(int)
        pts = sol["charuco_corners"].reshape(-1, 2)
        corner_maps[cam] = {int(i): pts[j] for j, i in enumerate(ids)}
        id_sets.append(set(ids.tolist()))
    common = sorted(set.intersection(*id_sets))
    if len(common) < 8:
        return None

    expected_all = board.getChessboardCorners().astype(np.float64)
    xyz = []
    expected = []
    per_corner_views = []
    for corner_id in common:
        obs = {cam: cmap[corner_id] for cam, cmap in corner_maps.items()}
        xyz.append(triangulate_points_multiview(obs, poses))
        expected.append(expected_all[corner_id])
        per_corner_views.append(len(obs))
    xyz_arr = np.array(xyz)
    expected_arr = np.array(expected)
    err = np.linalg.norm(xyz_arr - expected_arr, axis=1)
    return {
        "camera_names": sorted(best_solutions.keys()),
        "corner_ids": common,
        "xyz": xyz_arr,
        "expected": expected_arr,
        "errors_mm": err,
        "views_per_corner": per_corner_views,
    }


def save_board_validation(out_path: Path, validation: dict) -> None:
    xyz = validation["xyz"]
    expected = validation["expected"]
    err = validation["errors_mm"]
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(
        "Multiview board-corner triangulation sanity check\n"
        f"mean {err.mean():.2f} mm, median {np.median(err):.2f} mm, max {err.max():.2f} mm"
    )
    ax.scatter(expected[:, 0], expected[:, 1], -expected[:, 2], color="black", s=18, label="known board corner")
    ax.scatter(xyz[:, 0], xyz[:, 1], -xyz[:, 2], color="tab:orange", s=18, label="triangulated")
    for a, b in zip(expected, xyz):
        ax.plot([a[0], b[0]], [a[1], b[1]], [-a[2], -b[2]], color="red", alpha=0.28)
    ax.set_xlabel("world X (mm)")
    ax.set_ylabel("world Y (mm)")
    ax.set_zlabel("display height = -world Z (mm)")
    ax.legend()
    set_equal_3d_axes(ax)
    ax.view_init(elev=35, azim=-65)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_contact_sheet(out_path: Path, image_paths: list[Path], thumb_w: int = 640) -> None:
    images = []
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        h, w = img.shape[:2]
        img = cv2.resize(img, (thumb_w, int(h * thumb_w / w)))
        images.append(img)
    if not images:
        return
    rows = []
    for i in range(0, len(images), 2):
        row = images[i : i + 2]
        if len(row) == 1:
            row.append(np.zeros_like(row[0]))
        if row[0].shape[0] != row[1].shape[0]:
            target_h = min(row[0].shape[0], row[1].shape[0])
            row = [cv2.resize(im, (row[0].shape[1], target_h)) for im in row]
        rows.append(np.hstack(row))
    cv2.imwrite(str(out_path), np.vstack(rows))


def first_last_imu_timestamps_us(path: Path) -> dict | None:
    if not path.exists():
        return None
    first = None
    last = None
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if first is None:
                first = item["t_us"]
            last = item["t_us"]
    if first is None or last is None:
        return None
    return {
        "first_t_us": int(first),
        "last_t_us": int(last),
        "duration_s_from_imu": (last - first) / 1_000_000.0,
    }


def extract_wav(video: Path, wav_path: Path, sample_rate: int = 16000) -> None:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(video),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-acodec",
        "pcm_s16le",
        str(wav_path),
    ]
    subprocess.run(cmd, check=True)


def load_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, sample_rate


def onset_envelope(audio: np.ndarray, sample_rate: int, hop_s: float = 0.005, win_s: float = 0.025) -> tuple[np.ndarray, float]:
    hop = max(1, int(round(hop_s * sample_rate)))
    win = max(hop, int(round(win_s * sample_rate)))
    n = 1 + max(0, (len(audio) - win) // hop)
    if n <= 1:
        return np.zeros(1, dtype=np.float32), hop / sample_rate
    energy = np.empty(n, dtype=np.float64)
    for i in range(n):
        chunk = audio[i * hop : i * hop + win]
        energy[i] = np.sqrt(np.mean(chunk * chunk) + 1e-12)
    log_energy = np.log1p(80.0 * energy)
    onset = np.maximum(np.diff(log_energy, prepend=log_energy[0]), 0.0)
    if onset.std() > 1e-9:
        onset = (onset - np.median(onset)) / (onset.std() + 1e-9)
    onset = np.clip(onset, 0.0, None)
    return onset.astype(np.float32), hop / sample_rate


def estimate_offset_from_envelopes(
    reference: np.ndarray,
    target: np.ndarray,
    hop_s: float,
    max_abs_offset_s: float = 15.0,
) -> dict:
    n = min(len(reference), len(target))
    ref = reference[:n] - np.mean(reference[:n])
    tgt = target[:n] - np.mean(target[:n])
    corr = signal.correlate(tgt, ref, mode="full", method="fft")
    lags = signal.correlation_lags(len(tgt), len(ref), mode="full")
    max_lag = int(round(max_abs_offset_s / hop_s))
    mask = np.abs(lags) <= max_lag
    corr_m = corr[mask]
    lags_m = lags[mask]
    if corr_m.size == 0 or np.allclose(corr_m, 0):
        return {
            "offset_s": 0.0,
            "lag_frames": 0,
            "correlation_score": 0.0,
            "second_best_score": 0.0,
            "confidence": "low",
        }
    best_idx = int(np.argmax(corr_m))
    lag = int(lags_m[best_idx])
    denom = float(np.linalg.norm(ref) * np.linalg.norm(tgt) + 1e-12)
    score = float(corr_m[best_idx] / denom)
    exclusion = max(1, int(round(0.20 / hop_s)))
    tmp = corr_m.copy()
    start = max(0, best_idx - exclusion)
    end = min(len(tmp), best_idx + exclusion + 1)
    tmp[start:end] = -np.inf
    second = float(np.nanmax(tmp) / denom) if np.isfinite(tmp).any() else 0.0
    ratio = score / max(second, 1e-6)
    confidence = "high" if score >= 0.30 and ratio >= 1.15 else "medium" if score >= 0.18 else "low"
    return {
        "offset_s": lag * hop_s,
        "lag_frames": lag,
        "correlation_score": score,
        "second_best_score": second,
        "peak_ratio": ratio,
        "confidence": confidence,
    }


def top_audio_peaks(envelope: np.ndarray, hop_s: float, limit: int = 12) -> list[dict]:
    if envelope.size == 0:
        return []
    distance = max(1, int(round(0.35 / hop_s)))
    min_height = max(float(np.percentile(envelope, 99.2)), float(envelope.mean() + 2.0 * envelope.std()))
    peaks, props = signal.find_peaks(envelope, height=min_height, distance=distance)
    if peaks.size == 0:
        peaks = np.argsort(envelope)[-limit:]
        heights = envelope[peaks]
    else:
        heights = props["peak_heights"]
    order = np.argsort(heights)[::-1][:limit]
    return [
        {"time_s": float(peaks[i] * hop_s), "strength": float(heights[i])}
        for i in order
    ]


def save_audio_sync_plot(
    out_path: Path,
    envelopes: dict[str, np.ndarray],
    hop_s: float,
    sync_report: dict,
    reference_camera: str,
) -> None:
    fig, axes = plt.subplots(len(envelopes), 1, figsize=(12, 2.2 * len(envelopes)), sharex=False)
    if len(envelopes) == 1:
        axes = [axes]
    for ax, (cam, env) in zip(axes, envelopes.items()):
        t = np.arange(len(env)) * hop_s
        offset = 0.0 if cam == reference_camera else sync_report[cam]["offset_s"]
        ax.plot(t - offset, env, linewidth=0.8, color="tab:blue" if cam == reference_camera else "tab:orange")
        ax.set_title(f"{cam} onset envelope aligned to {reference_camera} timeline (subtract offset {offset:+.4f}s)")
        ax.set_ylabel("onset")
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel(f"{reference_camera} timeline seconds")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def process_audio_sync(take: Take, out_dir: Path, reference_camera: str) -> dict:
    audio_dir = out_dir / "audio"
    wavs = {}
    envelopes = {}
    sample_rate = None
    hop_s = None
    report = {
        "reference_camera": reference_camera,
        "offset_convention": (
            "offset_s means target_camera_time = reference_camera_time + offset_s. "
            "To put a target camera event on the reference timeline, use target_time_s - offset_s."
        ),
        "sample_rate_hz": None,
        "cameras": {},
    }

    for clip in take.clips:
        wav_path = audio_dir / f"{clip.name}.wav"
        extract_wav(clip.video, wav_path)
        audio, sr = load_wav_mono(wav_path)
        env, env_hop_s = onset_envelope(audio, sr)
        wavs[clip.name] = wav_path
        envelopes[clip.name] = env
        sample_rate = sr
        hop_s = env_hop_s
        report["cameras"][clip.name] = {
            "video": str(clip.video),
            "wav": str(wav_path),
            "duration_s": len(audio) / sr,
            "top_onset_peaks": top_audio_peaks(env, env_hop_s),
        }

    if reference_camera not in envelopes:
        raise RuntimeError(f"Reference camera {reference_camera} not found in {take.name}")

    report["sample_rate_hz"] = sample_rate
    report["envelope_hop_s"] = hop_s
    ref_env = envelopes[reference_camera]
    sync_by_camera = {}
    for cam, env in envelopes.items():
        if cam == reference_camera:
            sync_by_camera[cam] = {
                "offset_s": 0.0,
                "lag_frames": 0,
                "correlation_score": 1.0,
                "second_best_score": None,
                "confidence": "reference",
            }
        else:
            sync_by_camera[cam] = estimate_offset_from_envelopes(ref_env, env, hop_s)

    for cam, sync in sync_by_camera.items():
        report["cameras"][cam]["sync_to_reference"] = sync

    plot_path = out_dir / "audio_sync_onset_envelopes.png"
    save_audio_sync_plot(plot_path, envelopes, hop_s, sync_by_camera, reference_camera)
    report["audio_sync_plot"] = str(plot_path)
    return report


def process_extrinsics(
    take: Take,
    out_dir: Path,
    board: cv2.aruco.CharucoBoard,
    detector: cv2.aruco.CharucoDetector,
    args: argparse.Namespace,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "take": take.name,
        "intrinsics": {
            "model": "opencv_fisheye_equidistant",
            "resolution": [1920, 1080],
            "K": K_DEFAULT.tolist(),
            "distortion": D_DEFAULT.reshape(-1).tolist(),
        },
        "board": {
            "cols": args.board_cols,
            "rows": args.board_rows,
            "square_mm": args.square_mm,
            "marker_mm": args.marker_mm,
            "dictionary": args.dict,
        },
        "cameras": {},
        "generated_images": [],
    }
    poses = {}
    best_solutions = {}
    overlay_paths = []

    for clip in take.clips:
        duration = video_duration_s(clip.video)
        solutions = []
        detection_attempts = 0
        for time_s in np.arange(0.0, max(0.0, duration - 0.1), args.sample_step_s):
            detection_attempts += 1
            frame = read_frame(clip.video, float(time_s))
            sol = detect_and_solve(frame, board, detector)
            if sol is None:
                continue
            mean_err = float(sol["reprojection_errors_px"].mean())
            median_err = float(np.median(sol["reprojection_errors_px"]))
            n_corners = int(len(sol["charuco_ids"]))
            sol["time_s"] = float(time_s)
            sol["frame"] = frame
            sol["mean_err"] = mean_err
            sol["median_err"] = median_err
            sol["n_corners"] = n_corners
            if n_corners >= 30 and median_err < 3.0 and mean_err < 4.0:
                solutions.append(sol)

        if not solutions:
            raise RuntimeError(f"No usable ChArUco poses for {take.name} {clip.name}")

        rvec, tvec, kept = aggregate_pose(solutions)
        poses[clip.name] = (rvec, tvec)
        best = sorted(kept, key=lambda s: (-s["n_corners"], s["median_err"], s["mean_err"]))[0]
        best_solutions[clip.name] = best

        detection_path = out_dir / f"{clip.name}_01_charuco_detection.jpg"
        reproj_path = out_dir / f"{clip.name}_02_reprojection_overlay.jpg"
        save_detection_overlay(
            detection_path,
            best["frame"],
            best,
            args.board_cols,
            args.board_rows,
            args.square_mm,
        )
        save_reprojection_overlay(reproj_path, best["frame"], best)
        overlay_paths.extend([detection_path, reproj_path])

        all_errors = np.concatenate([s["reprojection_errors_px"] for s in kept])
        imu_times = first_last_imu_timestamps_us(clip.imu)
        center = camera_center_world(rvec, tvec)
        report["cameras"][clip.name] = {
            "video": str(clip.video),
            "imu": str(clip.imu),
            "video_duration_s": duration,
            "sample_step_s": args.sample_step_s,
            "detection_attempts": detection_attempts,
            "num_usable_pose_samples": len(solutions),
            "num_pose_samples_after_outlier_filter": len(kept),
            "best_visualized_frame_s": best["time_s"],
            "best_visualized_num_corners": best["n_corners"],
            "rvec_world_to_cam": rvec.reshape(3).tolist(),
            "R_world_to_cam": cv2.Rodrigues(rvec)[0].tolist(),
            "t_world_to_cam_mm": tvec.reshape(3).tolist(),
            "camera_center_world_mm": center.tolist(),
            "camera_display_x_y_height_mm": [float(center[0]), float(center[1]), float(-center[2])],
            "reprojection_mean_px": float(all_errors.mean()),
            "reprojection_median_px": float(np.median(all_errors)),
            "reprojection_p95_px": float(np.percentile(all_errors, 95)),
            "reprojection_max_px": float(all_errors.max()),
            "imu_timestamps": imu_times,
        }

    scene_path = out_dir / "03_world_scene_board_cameras.png"
    scene_report = save_world_scene(scene_path, poses, args.board_cols, args.board_rows, args.square_mm)
    report["world_scene"] = scene_report
    report["generated_images"].append(str(scene_path))

    validation = validate_board_triangulation(best_solutions, poses, board)
    if validation is not None:
        val_path = out_dir / "04_board_corner_multiview_triangulation.png"
        save_board_validation(val_path, validation)
        report["generated_images"].append(str(val_path))
        report["board_corner_multiview_triangulation"] = {
            "camera_names": validation["camera_names"],
            "num_common_corners": len(validation["corner_ids"]),
            "mean_error_mm": float(validation["errors_mm"].mean()),
            "median_error_mm": float(np.median(validation["errors_mm"])),
            "p95_error_mm": float(np.percentile(validation["errors_mm"], 95)),
            "max_error_mm": float(validation["errors_mm"].max()),
            "note": "This is a sanity check using known ChArUco board geometry, not an independent hand-pose accuracy estimate.",
        }

    contact_path = out_dir / "00_contact_sheet.jpg"
    save_contact_sheet(contact_path, overlay_paths[:8])
    report["generated_images"] = [str(contact_path)] + [str(p) for p in overlay_paths] + report["generated_images"]
    return report


def main() -> None:
    args = parse_args()
    cv2.setLogLevel(0)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dictionary = aruco_dict(args.dict)
    board = cv2.aruco.CharucoBoard(
        (args.board_cols, args.board_rows),
        args.square_mm,
        args.marker_mm,
        dictionary,
    )
    detector = cv2.aruco.CharucoDetector(board)

    summary = {
        "data_root": str(args.data_root),
        "out_dir": str(args.out_dir),
        "takes": {},
    }

    for take in build_takes(args.data_root):
        take_out = args.out_dir / take.name
        take_out.mkdir(parents=True, exist_ok=True)
        extrinsics = process_extrinsics(take, take_out, board, detector, args)
        audio_sync = process_audio_sync(take, take_out, args.reference_camera)
        report = {
            "extrinsics": extrinsics,
            "audio_sync": audio_sync,
        }
        report_path = take_out / "v2_calibration_and_audio_sync_report.json"
        with report_path.open("w") as f:
            json.dump(report, f, indent=2)
        summary["takes"][take.name] = {
            "report": str(report_path),
            "world_scene": extrinsics["generated_images"][-2] if len(extrinsics["generated_images"]) >= 2 else None,
            "audio_sync_plot": audio_sync["audio_sync_plot"],
            "cameras": {
                cam: {
                    "center_world_mm": item["camera_center_world_mm"],
                    "display_x_y_height_mm": item["camera_display_x_y_height_mm"],
                    "reprojection_median_px": item["reprojection_median_px"],
                    "audio_offset_s": audio_sync["cameras"][cam]["sync_to_reference"]["offset_s"],
                    "audio_sync_confidence": audio_sync["cameras"][cam]["sync_to_reference"]["confidence"],
                }
                for cam, item in extrinsics["cameras"].items()
            },
            "board_corner_multiview_triangulation": extrinsics.get("board_corner_multiview_triangulation"),
        }

    summary_path = args.out_dir / "v2_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
