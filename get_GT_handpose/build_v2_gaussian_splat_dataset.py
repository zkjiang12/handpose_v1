#!/usr/bin/env python3
"""Export a calibrated v2 scene package for 3D Gaussian Splat experiments."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np

from visualize_v2_all_available_handpose import (
    D_DEFAULT,
    EGO_EXO_ROOT,
    K_DEFAULT,
    REPORT_BY_SEGMENT,
    camera_center_world,
    camera_frustum_world,
    world_to_z_up,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels-json",
        type=Path,
        default=Path("/Users/zikangjiang/dev/ego-exo/visualizations/v2/handpose_labeler/labels/left_hand_session_171483.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("get_GT_handpose/ego_exo_visualizations/v2/gaussian_splat/left_hand_session_171483"),
    )
    parser.add_argument("--live-out-dir", type=Path)
    parser.add_argument("--undistort-balance", type=float, default=0.8)
    parser.add_argument("--board-sample-mm", type=float, default=4.0)
    parser.add_argument("--full-scene-max-features", type=int, default=14000)
    parser.add_argument("--full-scene-reprojection-threshold-px", type=float, default=4.0)
    parser.add_argument("--full-scene-voxel-mm", type=float, default=8.0)
    parser.add_argument("--dense-stereo-step-px", type=int, default=4)
    parser.add_argument("--dense-stereo-voxel-mm", type=float, default=6.0)
    return parser.parse_args()


def load_report(labels: dict) -> dict:
    segment_ids = {camera["segment_id"] for camera in labels["cameras"].values() if camera.get("segment_id")}
    if len(segment_ids) != 1:
        raise ValueError(f"Expected one segment id, got {sorted(segment_ids)}")
    return json.loads(REPORT_BY_SEGMENT[next(iter(segment_ids))].read_text())


def read_video_frame(video_path: Path, frame_index: int, fallback_time_s: float | None) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok and fallback_time_s is not None:
        cap.set(cv2.CAP_PROP_POS_MSEC, fallback_time_s * 1000.0)
        ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")
    return frame


def undistort_frame(frame: np.ndarray, balance: float) -> tuple[np.ndarray, np.ndarray]:
    height, width = frame.shape[:2]
    new_k = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K_DEFAULT,
        D_DEFAULT,
        (width, height),
        np.eye(3),
        balance=balance,
        new_size=(width, height),
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K_DEFAULT,
        D_DEFAULT,
        np.eye(3),
        new_k,
        (width, height),
        cv2.CV_16SC2,
    )
    return cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT), new_k


def camera_to_world_opengl_meters(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    r_world_to_cam = cv2.Rodrigues(rvec)[0]
    c2w_cv = np.eye(4, dtype=float)
    c2w_cv[:3, :3] = r_world_to_cam.T
    c2w_cv[:3, 3] = (-r_world_to_cam.T @ tvec.reshape(3, 1)).reshape(3) / 1000.0
    cv_to_gl = np.diag([1.0, -1.0, -1.0, 1.0])
    return c2w_cv @ cv_to_gl


def rotation_matrix_to_quaternion_wxyz(r: np.ndarray) -> tuple[float, float, float, float]:
    trace = float(np.trace(r))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r[2, 1] - r[1, 2]) / s
        qy = (r[0, 2] - r[2, 0]) / s
        qz = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        qw = (r[2, 1] - r[1, 2]) / s
        qx = 0.25 * s
        qy = (r[0, 1] + r[1, 0]) / s
        qz = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        qw = (r[0, 2] - r[2, 0]) / s
        qx = (r[0, 1] + r[1, 0]) / s
        qy = 0.25 * s
        qz = (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        qw = (r[1, 0] - r[0, 1]) / s
        qx = (r[0, 2] + r[2, 0]) / s
        qy = (r[1, 2] + r[2, 1]) / s
        qz = 0.25 * s
    return qw, qx, qy, qz


def project_world_to_undistorted_image(point_mm: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, k: np.ndarray) -> tuple[float, float, float]:
    r = cv2.Rodrigues(rvec)[0]
    point_cam = (r @ point_mm.reshape(3, 1) + tvec.reshape(3, 1)).reshape(3)
    if point_cam[2] <= 1e-9:
        return math.nan, math.nan, float(point_cam[2])
    x = k[0, 0] * point_cam[0] / point_cam[2] + k[0, 2]
    y = k[1, 1] * point_cam[1] / point_cam[2] + k[1, 2]
    return float(x), float(y), float(point_cam[2])


def sample_board_splat(
    images: dict[str, np.ndarray],
    cameras: dict[str, dict],
    board: dict,
    k_undistorted: np.ndarray,
    sample_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    board_w = board["cols"] * board["square_mm"]
    board_h = board["rows"] * board["square_mm"]
    xs = np.arange(0.0, board_w + 1e-6, sample_mm)
    ys = np.arange(0.0, board_h + 1e-6, sample_mm)
    points = []
    colors = []
    for y in ys:
        for x in xs:
            point = np.array([x, y, 0.0], dtype=float)
            samples = []
            for cam, info in cameras.items():
                rvec = np.array(info["rvec_world_to_cam"], dtype=float).reshape(3, 1)
                tvec = np.array(info["t_world_to_cam_mm"], dtype=float).reshape(3, 1)
                px, py, depth = project_world_to_undistorted_image(point, rvec, tvec, k_undistorted)
                image = images[cam]
                height, width = image.shape[:2]
                ix = int(round(px))
                iy = int(round(py))
                if depth > 0 and 0 <= ix < width and 0 <= iy < height:
                    samples.append(image[iy, ix])
            if not samples:
                continue
            color_bgr = np.median(np.stack(samples), axis=0)
            points.append(world_to_z_up(point))
            colors.append(color_bgr[::-1])
    return np.array(points, dtype=np.float32), np.array(colors, dtype=np.uint8)


def projection_matrix_world_to_image(camera_info: dict, k: np.ndarray) -> np.ndarray:
    r = cv2.Rodrigues(np.array(camera_info["rvec_world_to_cam"], dtype=float).reshape(3, 1))[0]
    t = np.array(camera_info["t_world_to_cam_mm"], dtype=float).reshape(3, 1)
    return k @ np.hstack([r, t])


def detect_sift_features(images: dict[str, np.ndarray], max_features: int) -> dict[str, tuple[list[cv2.KeyPoint], np.ndarray]]:
    sift = cv2.SIFT_create(nfeatures=max_features, contrastThreshold=0.012, edgeThreshold=12)
    features = {}
    for cam, image in images.items():
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        if descriptors is None:
            descriptors = np.empty((0, 128), dtype=np.float32)
        features[cam] = (keypoints, descriptors)
    return features


def dedupe_voxel(points: np.ndarray, colors: np.ndarray, voxel_mm: float) -> tuple[np.ndarray, np.ndarray]:
    buckets: dict[tuple[int, int, int], list[np.ndarray]] = {}
    color_buckets: dict[tuple[int, int, int], list[np.ndarray]] = {}
    for point, color in zip(points, colors):
        key = tuple(np.round(point / voxel_mm).astype(int).tolist())
        buckets.setdefault(key, []).append(point)
        color_buckets.setdefault(key, []).append(color.astype(float))
    out_points = []
    out_colors = []
    for key in sorted(buckets):
        out_points.append(np.mean(np.stack(buckets[key]), axis=0))
        out_colors.append(np.mean(np.stack(color_buckets[key]), axis=0))
    return np.array(out_points, dtype=np.float32), np.clip(np.array(out_colors), 0, 255).astype(np.uint8)


def triangulate_full_scene_feature_splat(
    images: dict[str, np.ndarray],
    cameras: dict[str, dict],
    k_undistorted: np.ndarray,
    max_features: int,
    reproj_threshold_px: float,
    voxel_mm: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    features = detect_sift_features(images, max_features)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    projection_mats = {
        cam: projection_matrix_world_to_image(info, k_undistorted)
        for cam, info in cameras.items()
    }
    pair_stats = []
    points = []
    colors = []
    cams = sorted(images)
    for i, cam_a in enumerate(cams):
        kp_a, desc_a = features[cam_a]
        if len(desc_a) < 2:
            continue
        for cam_b in cams[i + 1:]:
            kp_b, desc_b = features[cam_b]
            if len(desc_b) < 2:
                continue
            raw_matches = matcher.knnMatch(desc_a, desc_b, k=2)
            matches = [m for m, n in raw_matches if m.distance < 0.72 * n.distance]
            if len(matches) < 12:
                pair_stats.append({"pair": f"{cam_a}+{cam_b}", "matches": len(matches), "accepted": 0})
                continue

            pts_a = np.array([kp_a[m.queryIdx].pt for m in matches], dtype=np.float32)
            pts_b = np.array([kp_b[m.trainIdx].pt for m in matches], dtype=np.float32)
            fundamental, inliers = cv2.findFundamentalMat(pts_a, pts_b, cv2.FM_RANSAC, 2.5, 0.995)
            if inliers is not None:
                keep = inliers.ravel().astype(bool)
                pts_a = pts_a[keep]
                pts_b = pts_b[keep]
            if len(pts_a) < 8:
                pair_stats.append({"pair": f"{cam_a}+{cam_b}", "matches": len(matches), "accepted": 0})
                continue

            hom = cv2.triangulatePoints(projection_mats[cam_a], projection_mats[cam_b], pts_a.T, pts_b.T).T
            valid_w = np.abs(hom[:, 3]) > 1e-9
            hom = hom[valid_w]
            pts_a = pts_a[valid_w]
            pts_b = pts_b[valid_w]
            xyz_world = hom[:, :3] / hom[:, 3:4]
            accepted = 0
            for idx, point_world in enumerate(xyz_world):
                if not np.all(np.isfinite(point_world)):
                    continue
                display = world_to_z_up(point_world)
                if not (-700.0 <= display[0] <= 1400.0 and -1250.0 <= display[1] <= 550.0 and -200.0 <= display[2] <= 1400.0):
                    continue
                px_a, py_a, depth_a = project_world_to_undistorted_image(
                    point_world,
                    np.array(cameras[cam_a]["rvec_world_to_cam"], dtype=float).reshape(3, 1),
                    np.array(cameras[cam_a]["t_world_to_cam_mm"], dtype=float).reshape(3, 1),
                    k_undistorted,
                )
                px_b, py_b, depth_b = project_world_to_undistorted_image(
                    point_world,
                    np.array(cameras[cam_b]["rvec_world_to_cam"], dtype=float).reshape(3, 1),
                    np.array(cameras[cam_b]["t_world_to_cam_mm"], dtype=float).reshape(3, 1),
                    k_undistorted,
                )
                if depth_a <= 0 or depth_b <= 0:
                    continue
                err_a = math.hypot(px_a - float(pts_a[idx, 0]), py_a - float(pts_a[idx, 1]))
                err_b = math.hypot(px_b - float(pts_b[idx, 0]), py_b - float(pts_b[idx, 1]))
                if max(err_a, err_b) > reproj_threshold_px:
                    continue
                image = images[cam_a]
                ix = int(round(pts_a[idx, 0]))
                iy = int(round(pts_a[idx, 1]))
                if not (0 <= ix < image.shape[1] and 0 <= iy < image.shape[0]):
                    continue
                points.append(display)
                colors.append(image[iy, ix, ::-1])
                accepted += 1
            pair_stats.append(
                {
                    "pair": f"{cam_a}+{cam_b}",
                    "matches": len(matches),
                    "after_fundamental": int(len(pts_a)),
                    "accepted": accepted,
                }
            )

    feature_counts = {cam: int(len(features[cam][0])) for cam in cams}
    if not points:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8), {
            "feature_counts": feature_counts,
            "pair_stats": pair_stats,
            "points_before_voxel": 0,
            "points_after_voxel": 0,
        }
    points_arr = np.array(points, dtype=np.float32)
    colors_arr = np.array(colors, dtype=np.uint8)
    deduped_points, deduped_colors = dedupe_voxel(points_arr, colors_arr, voxel_mm)
    return deduped_points, deduped_colors, {
        "feature_counts": feature_counts,
        "pair_stats": pair_stats,
        "points_before_voxel": int(len(points_arr)),
        "points_after_voxel": int(len(deduped_points)),
    }


def dense_stereo_scene_splat(
    images: dict[str, np.ndarray],
    cameras: dict[str, dict],
    k_undistorted: np.ndarray,
    step_px: int,
    voxel_mm: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    pairs = [("cam4", "cam1"), ("cam4", "cam2"), ("cam3", "cam1")]
    height, width = next(iter(images.values())).shape[:2]
    points = []
    colors = []
    pair_stats = []
    for cam_a, cam_b in pairs:
        if cam_a not in images or cam_b not in images:
            continue
        info_a = cameras[cam_a]
        info_b = cameras[cam_b]
        r_a = cv2.Rodrigues(np.array(info_a["rvec_world_to_cam"], dtype=float).reshape(3, 1))[0]
        t_a = np.array(info_a["t_world_to_cam_mm"], dtype=float).reshape(3, 1)
        r_b = cv2.Rodrigues(np.array(info_b["rvec_world_to_cam"], dtype=float).reshape(3, 1))[0]
        t_b = np.array(info_b["t_world_to_cam_mm"], dtype=float).reshape(3, 1)
        r_rel = r_b @ r_a.T
        t_rel = (t_b - r_rel @ t_a).reshape(3)

        r_rect_a, r_rect_b, p_a, p_b, q, _, _ = cv2.stereoRectify(
            k_undistorted,
            np.zeros(5),
            k_undistorted,
            np.zeros(5),
            (width, height),
            r_rel,
            t_rel,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0.4,
        )
        map_ax, map_ay = cv2.initUndistortRectifyMap(k_undistorted, np.zeros(5), r_rect_a, p_a, (width, height), cv2.CV_16SC2)
        map_bx, map_by = cv2.initUndistortRectifyMap(k_undistorted, np.zeros(5), r_rect_b, p_b, (width, height), cv2.CV_16SC2)
        rect_a = cv2.remap(images[cam_a], map_ax, map_ay, cv2.INTER_LINEAR)
        rect_b = cv2.remap(images[cam_b], map_bx, map_by, cv2.INTER_LINEAR)
        gray_a = cv2.cvtColor(rect_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(rect_b, cv2.COLOR_BGR2GRAY)

        min_disp = 768
        num_disp = 768
        block = 5
        sgbm = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=block,
            P1=8 * 3 * block * block,
            P2=32 * 3 * block * block,
            disp12MaxDiff=2,
            uniquenessRatio=7,
            speckleWindowSize=120,
            speckleRange=2,
            preFilterCap=31,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        disparity = sgbm.compute(gray_a, gray_b).astype(np.float32) / 16.0
        points_rect = cv2.reprojectImageTo3D(disparity, q)
        gradient_x = cv2.Sobel(gray_a, cv2.CV_32F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray_a, cv2.CV_32F, 0, 1, ksize=3)
        gradient = cv2.magnitude(gradient_x, gradient_y)

        yy, xx = np.mgrid[0:height:step_px, 0:width:step_px]
        accepted = 0
        sampled = int(xx.size)
        for y, x in zip(yy.ravel(), xx.ravel()):
            disp = float(disparity[y, x])
            if not np.isfinite(disp) or disp <= min_disp + 2 or disp >= min_disp + num_disp - 3:
                continue
            if gradient[y, x] < 5.0:
                continue
            point_rect = points_rect[y, x].astype(float)
            if not np.all(np.isfinite(point_rect)) or np.linalg.norm(point_rect) > 3500.0:
                continue
            point_cam_a = r_rect_a.T @ point_rect.reshape(3, 1)
            point_world = (r_a.T @ (point_cam_a - t_a)).reshape(3)
            display = world_to_z_up(point_world)
            if not (-750.0 <= display[0] <= 1450.0 and -1300.0 <= display[1] <= 600.0 and -180.0 <= display[2] <= 950.0):
                continue
            points.append(display)
            colors.append(rect_a[y, x, ::-1])
            accepted += 1
        pair_stats.append({"pair": f"{cam_a}+{cam_b}", "sampled": sampled, "accepted": accepted})

    if not points:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8), {
            "pair_stats": pair_stats,
            "points_before_voxel": 0,
            "points_after_voxel": 0,
        }
    points_arr = np.array(points, dtype=np.float32)
    colors_arr = np.array(colors, dtype=np.uint8)
    deduped_points, deduped_colors = dedupe_voxel(points_arr, colors_arr, voxel_mm)
    return deduped_points, deduped_colors, {
        "pair_stats": pair_stats,
        "points_before_voxel": int(len(points_arr)),
        "points_after_voxel": int(len(deduped_points)),
    }


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    with path.open("w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for point, color in zip(points, colors):
            f.write(f"{point[0]:.4f} {point[1]:.4f} {point[2]:.4f} {int(color[0])} {int(color[1])} {int(color[2])}\n")


def write_colmap_text(path: Path, labels: dict, report: dict, k_undistorted: np.ndarray, width: int, height: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    cameras_txt = path / "cameras.txt"
    images_txt = path / "images.txt"
    points_txt = path / "points3D.txt"
    cameras_txt.write_text(
        "# Camera list with one line of data per camera:\n"
        "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        f"1 PINHOLE {width} {height} {k_undistorted[0,0]:.10f} {k_undistorted[1,1]:.10f} {k_undistorted[0,2]:.10f} {k_undistorted[1,2]:.10f}\n"
    )
    lines = [
        "# Image list with two lines of data per image:",
        "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "# POINTS2D[] as (X, Y, POINT3D_ID)",
    ]
    for image_id, cam in enumerate(sorted(labels["cameras"]), start=1):
        info = report["extrinsics"]["cameras"][cam]
        r = cv2.Rodrigues(np.array(info["rvec_world_to_cam"], dtype=float).reshape(3, 1))[0]
        t_m = np.array(info["t_world_to_cam_mm"], dtype=float).reshape(3) / 1000.0
        qw, qx, qy, qz = rotation_matrix_to_quaternion_wxyz(r)
        lines.append(
            f"{image_id} {qw:.12f} {qx:.12f} {qy:.12f} {qz:.12f} "
            f"{t_m[0]:.12f} {t_m[1]:.12f} {t_m[2]:.12f} 1 images/{cam}.jpg"
        )
        lines.append("")
    images_txt.write_text("\n".join(lines) + "\n")
    points_txt.write_text("# Empty sparse points. Use board_plane_splat_points.ply as optional initialization/reference geometry.\n")


def write_preview_html(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    report: dict,
    title: str,
    max_preview_points: int = 70000,
) -> None:
    stride = max(1, int(math.ceil(len(points) / max_preview_points)))
    points_small = points[::stride]
    colors_small = colors[::stride]
    camera_payload = {}
    for cam, info in sorted(report["extrinsics"]["cameras"].items()):
        rvec = np.array(info["rvec_world_to_cam"], dtype=float).reshape(3, 1)
        tvec = np.array(info["t_world_to_cam_mm"], dtype=float).reshape(3, 1)
        center_raw, dirs_raw = camera_frustum_world(rvec, tvec)
        center = world_to_z_up(center_raw)
        dirs = np.array([world_to_z_up(d) for d in dirs_raw])
        camera_payload[cam] = {
            "center": np.round(center, 4).tolist(),
            "axis": np.round((center + 230.0 * dirs[4]), 4).tolist(),
        }
    payload = {
        "points": np.round(points_small, 4).tolist(),
        "colors": colors_small.tolist(),
        "cameras": camera_payload,
    }
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <script type="importmap">
    {{"imports": {{"three": "https://cdn.jsdelivr.net/npm/three@0.165.0/build/three.module.js", "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.165.0/examples/jsm/"}}}}
  </script>
  <style>
    html, body, #viewer {{ width: 100%; height: 100%; margin: 0; overflow: hidden; background: #0b1020; }}
    .panel {{
      position: fixed; left: 12px; top: 12px; z-index: 2; padding: 10px 12px;
      color: #e5e7eb; background: rgba(15, 23, 42, 0.88); border: 1px solid rgba(148, 163, 184, 0.35);
      border-radius: 8px; font: 13px Inter, system-ui, sans-serif;
    }}
    input {{ accent-color: #38bdf8; }}
  </style>
</head>
<body>
  <div id="viewer"></div>
  <section class="panel">
    <div>{title}</div>
    <label>Point size <input id="size" type="range" min="1" max="10" step="0.5" value="3"></label>
  </section>
  <script type="module">
    import * as THREE from 'three';
    import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
    const data = {json.dumps(payload, separators=(",", ":"))};
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b1020);
    const camera = new THREE.PerspectiveCamera(45, innerWidth / innerHeight, 0.1, 5000);
    camera.up.set(0, 0, 1);
    camera.position.set(650, -820, 620);
    const renderer = new THREE.WebGLRenderer({{ antialias: true }});
    renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
    renderer.setSize(innerWidth, innerHeight);
    document.getElementById('viewer').appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.target.set(280, -200, 0);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(data.points.flat(), 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(data.colors.flat().map(v => v / 255), 3));
    const material = new THREE.PointsMaterial({{ size: 3, vertexColors: true, transparent: true, opacity: 0.95 }});
    scene.add(new THREE.Points(geometry, material));
    for (const cam of Object.values(data.cameras)) {{
      const center = new THREE.Vector3(...cam.center);
      const axis = new THREE.Vector3(...cam.axis);
      const sphere = new THREE.Mesh(new THREE.SphereGeometry(8, 16, 12), new THREE.MeshBasicMaterial({{ color: 0x38bdf8 }}));
      sphere.position.copy(center);
      scene.add(sphere);
      scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([center, axis]), new THREE.LineBasicMaterial({{ color: 0x38bdf8 }})));
    }}
    document.getElementById('size').addEventListener('input', event => material.size = Number(event.target.value));
    addEventListener('resize', () => {{
      camera.aspect = innerWidth / innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(innerWidth, innerHeight);
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


def write_readme(path: Path, labels: dict) -> None:
    text = f"""# 3D Gaussian Splat Starter Export

This folder is a calibrated starter package for a 3D Gaussian Splat experiment from the v2 four-camera setup.

## What is included

- `images/cam1.jpg` ... `images/cam4.jpg`: synchronized, undistorted still frames at session time `{labels['session_time_s']:.3f}s`.
- `raw_images/cam1.jpg` ... `raw_images/cam4.jpg`: original fisheye frames for reference.
- `transforms.json`: Nerfstudio-style camera transform export using pinhole undistorted intrinsics.
- `colmap_text/`: COLMAP text-format cameras/images export using the known ChArUco extrinsics.
- `board_plane_splat_points.ply`: colored point-splat proxy for the known ChArUco board plane.
- `full_scene_feature_splat_points.ply`: sparse full-scene feature points triangulated from SIFT matches across camera views.
- `dense_stereo_splat_points.ply`: denser, noisier full-scene points from rectified stereo depth across calibrated camera pairs.
- `full_scene_combined_splat_points.ply`: board-plane splat plus sparse feature splat plus dense stereo splat.
- `board_plane_splat_preview.html`: local browser preview of the board-plane splat proxy.
- `full_scene_splat_preview.html`: local browser preview of the combined full-scene splat proxy.

## Important limitation

This is not a trained 3DGS model yet. A real trained splat needs a trainer such as Nerfstudio/gsplat/GraphDECO 3DGS. Those are not installed in this environment, and this machine currently has no CUDA GPU. Also, this capture only has four unique viewpoints, so a trained model will mostly reproduce these views and will not hallucinate unseen scene sides reliably. The full-scene proxy here is a classical multiview/stereo point splat, not learned dense Gaussian optimization.

## Recommended next step

Train on a CUDA machine or cloud instance using `transforms.json` as the camera-pose source. Treat moving hands as dynamic foreground; mask or avoid those frames if you want the static room/table scene.
"""
    path.write_text(text)


def main() -> None:
    args = parse_args()
    labels = json.loads(args.labels_json.read_text())
    report = load_report(labels)

    out_dir = args.out_dir
    images_dir = out_dir / "images"
    raw_dir = out_dir / "raw_images"
    colmap_dir = out_dir / "colmap_text"
    images_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    frames = {}
    undistorted_frames = {}
    k_undistorted = None
    for cam, info in sorted(labels["cameras"].items()):
        video_path = EGO_EXO_ROOT / info["video_url"].lstrip("/")
        raw = read_video_frame(video_path, int(info["frame_index_30fps"]), info.get("local_time_s"))
        undistorted, new_k = undistort_frame(raw, args.undistort_balance)
        if k_undistorted is None:
            k_undistorted = new_k
        frames[cam] = raw
        undistorted_frames[cam] = undistorted
        cv2.imwrite(str(raw_dir / f"{cam}.jpg"), raw, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
        cv2.imwrite(str(images_dir / f"{cam}.jpg"), undistorted, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    assert k_undistorted is not None
    height, width = next(iter(undistorted_frames.values())).shape[:2]
    transforms = {
        "camera_model": "PINHOLE",
        "fl_x": float(k_undistorted[0, 0]),
        "fl_y": float(k_undistorted[1, 1]),
        "cx": float(k_undistorted[0, 2]),
        "cy": float(k_undistorted[1, 2]),
        "w": int(width),
        "h": int(height),
        "source_session_time_s": float(labels["session_time_s"]),
        "source_labels": str(args.labels_json),
        "frames": [],
    }
    for cam in sorted(labels["cameras"]):
        info = report["extrinsics"]["cameras"][cam]
        rvec = np.array(info["rvec_world_to_cam"], dtype=float).reshape(3, 1)
        tvec = np.array(info["t_world_to_cam_mm"], dtype=float).reshape(3, 1)
        transforms["frames"].append(
            {
                "file_path": f"images/{cam}.jpg",
                "camera_name": cam,
                "transform_matrix": camera_to_world_opengl_meters(rvec, tvec).tolist(),
            }
        )
    (out_dir / "transforms.json").write_text(json.dumps(transforms, indent=2) + "\n")
    write_colmap_text(colmap_dir, labels, report, k_undistorted, width, height)

    splat_points, splat_colors = sample_board_splat(
        undistorted_frames,
        report["extrinsics"]["cameras"],
        report["extrinsics"]["board"],
        k_undistorted,
        args.board_sample_mm,
    )
    feature_points, feature_colors, feature_stats = triangulate_full_scene_feature_splat(
        undistorted_frames,
        report["extrinsics"]["cameras"],
        k_undistorted,
        args.full_scene_max_features,
        args.full_scene_reprojection_threshold_px,
        args.full_scene_voxel_mm,
    )
    dense_points, dense_colors, dense_stats = dense_stereo_scene_splat(
        undistorted_frames,
        report["extrinsics"]["cameras"],
        k_undistorted,
        args.dense_stereo_step_px,
        args.dense_stereo_voxel_mm,
    )
    point_sets = [splat_points]
    color_sets = [splat_colors]
    if len(feature_points):
        point_sets.append(feature_points)
        color_sets.append(feature_colors)
    if len(dense_points):
        point_sets.append(dense_points)
        color_sets.append(dense_colors)
    full_scene_points = np.vstack(point_sets)
    full_scene_colors = np.vstack(color_sets)
    write_ply(out_dir / "board_plane_splat_points.ply", splat_points, splat_colors)
    write_ply(out_dir / "full_scene_feature_splat_points.ply", feature_points, feature_colors)
    write_ply(out_dir / "dense_stereo_splat_points.ply", dense_points, dense_colors)
    write_ply(out_dir / "full_scene_combined_splat_points.ply", full_scene_points, full_scene_colors)
    write_preview_html(out_dir / "board_plane_splat_preview.html", splat_points, splat_colors, report, "Board-plane colored splat preview")
    write_preview_html(
        out_dir / "full_scene_splat_preview.html",
        full_scene_points,
        full_scene_colors,
        report,
        "Full-scene sparse splat preview",
    )
    write_readme(out_dir / "README.md", labels)

    summary = {
        "out_dir": str(out_dir),
        "num_images": len(undistorted_frames),
        "image_size": [int(width), int(height)],
        "undistorted_intrinsics": k_undistorted.tolist(),
        "num_board_splat_points": int(len(splat_points)),
        "num_full_scene_feature_splat_points": int(len(feature_points)),
        "num_dense_stereo_splat_points": int(len(dense_points)),
        "num_full_scene_combined_splat_points": int(len(full_scene_points)),
        "full_scene_feature_stats": feature_stats,
        "dense_stereo_stats": dense_stats,
        "limitations": [
            "No trained Gaussian Splat model is produced locally because nerfstudio/gsplat/colmap are not installed.",
            "Only four unique viewpoints are available, so true novel-view reconstruction quality will be limited.",
            "Moving hands are dynamic foreground and should be masked/avoided for static-scene training.",
            "The local full-scene splat is a classical feature/stereo point-splat proxy, not learned dense Gaussian optimization.",
        ],
        "files": {
            "transforms": str(out_dir / "transforms.json"),
            "colmap_text": str(colmap_dir),
            "board_plane_splat_ply": str(out_dir / "board_plane_splat_points.ply"),
            "board_plane_splat_preview": str(out_dir / "board_plane_splat_preview.html"),
            "full_scene_feature_splat_ply": str(out_dir / "full_scene_feature_splat_points.ply"),
            "dense_stereo_splat_ply": str(out_dir / "dense_stereo_splat_points.ply"),
            "full_scene_combined_splat_ply": str(out_dir / "full_scene_combined_splat_points.ply"),
            "full_scene_splat_preview": str(out_dir / "full_scene_splat_preview.html"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    if args.live_out_dir:
        if args.live_out_dir.exists():
            shutil.rmtree(args.live_out_dir)
        shutil.copytree(out_dir, args.live_out_dir)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
