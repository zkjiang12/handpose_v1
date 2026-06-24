#!/usr/bin/env python3
"""Visualize ChArUco-based extrinsic calibration for the ego-exo test clips."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
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


@dataclass(frozen=True)
class CameraInput:
    name: str
    video: Path
    stable_start_s: int
    stable_end_s: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/Users/zikangjiang/dev/ego-exo"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/Users/zikangjiang/dev/ego-exo/visualizations"),
    )
    parser.add_argument("--square-mm", type=float, default=40.0)
    parser.add_argument("--marker-mm", type=float, default=20.0)
    parser.add_argument("--board-cols", type=int, default=14)
    parser.add_argument("--board-rows", type=int, default=10)
    parser.add_argument("--dict", type=str, default="DICT_5X5_100")
    parser.add_argument("--sample-step-s", type=int, default=5)
    return parser.parse_args()


def aruco_dict(name: str) -> cv2.aruco.Dictionary:
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"Unknown ArUco dictionary: {name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


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
    k: np.ndarray,
    d: np.ndarray,
) -> dict | None:
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(frame)
    if charuco_ids is None or len(charuco_ids) < 12:
        return None

    ids = charuco_ids.flatten().astype(np.int32)
    img_pts = charuco_corners.astype(np.float64).reshape(-1, 1, 2)
    obj_pts = board.getChessboardCorners()[ids].astype(np.float64).reshape(-1, 1, 3)

    # The camera model is equidistant/fisheye. Undistort into normalized
    # camera coordinates, then solve PnP with an identity pinhole camera.
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

    proj, _ = cv2.fisheye.projectPoints(obj_pts, rvec, tvec, k, d)
    reproj_err = np.linalg.norm(proj.reshape(-1, 2) - img_pts.reshape(-1, 2), axis=1)

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
        "reprojection_errors_px": reproj_err,
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
    cv2.putText(image, "X", tuple(proj[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.putText(image, "Y", tuple(proj[2]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(image, "Z", tuple(proj[3]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

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
    err = solution["reprojection_errors_px"].mean()
    cv2.putText(
        overlay,
        f"{n_corners} ChArUco corners, mean reproj {err:.2f}px",
        (40, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
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
        color = (0, 255, 0) if err < 2.0 else (0, 128, 255)
        cv2.circle(overlay, (int(round(dx)), int(round(dy))), 4, color, -1)
        cv2.drawMarker(
            overlay,
            (int(round(px)), int(round(py))),
            (255, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=14,
            thickness=2,
        )
        if err >= 2.0:
            cv2.line(
                overlay,
                (int(round(dx)), int(round(dy))),
                (int(round(px)), int(round(py))),
                (0, 0, 255),
                1,
            )
    cv2.putText(
        overlay,
        "green = detected corner, magenta = projected corner",
        (40, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
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


def aggregate_pose(solutions: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    rotations = [cv2.Rodrigues(s["rvec"])[0] for s in solutions]
    translations = [s["tvec"].reshape(3) for s in solutions]
    r = average_rotation(rotations)
    t = np.median(np.stack(translations, axis=0), axis=0)
    rvec, _ = cv2.Rodrigues(r)
    return rvec.reshape(3, 1), t.reshape(3, 1)


def camera_center_world(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    r = cv2.Rodrigues(rvec)[0]
    t = tvec.reshape(3)
    return -r.T @ t


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


def save_3d_scene(
    out_path: Path,
    poses: dict[str, tuple[np.ndarray, np.ndarray]],
    board_cols: int,
    board_rows: int,
    square_mm: float,
) -> None:
    width = board_cols * square_mm
    height = board_rows * square_mm

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("ChArUco world frame and estimated exo camera poses")

    board = np.array(
        [[0, 0, 0], [width, 0, 0], [width, height, 0], [0, height, 0], [0, 0, 0]],
        dtype=float,
    )
    ax.plot(board[:, 0], board[:, 1], board[:, 2], color="black", linewidth=2)
    ax.scatter(board[:, 0], board[:, 1], board[:, 2], color="black", s=20)
    ax.text(0, 0, 0, "world origin / board", color="black")

    grid_x = np.arange(0, width + 1e-6, square_mm)
    grid_y = np.arange(0, height + 1e-6, square_mm)
    for x in grid_x:
        ax.plot([x, x], [0, height], [0, 0], color="lightgray", linewidth=0.5)
    for y in grid_y:
        ax.plot([0, width], [y, y], [0, 0], color="lightgray", linewidth=0.5)

    colors = {"cam1": "tab:red", "cam2": "tab:blue"}
    for cam, (rvec, tvec) in poses.items():
        center, dirs = camera_rays_world(rvec, tvec)
        color = colors.get(cam, "tab:purple")
        ax.scatter([center[0]], [center[1]], [center[2]], color=color, s=80)
        ax.text(center[0], center[1], center[2], cam, color=color)

        scale = 180.0
        ends = center[None, :] + scale * dirs
        for end in ends:
            ax.plot(
                [center[0], end[0]],
                [center[1], end[1]],
                [center[2], end[2]],
                color=color,
                linewidth=1.2,
            )
        for i in range(4):
            a = ends[i]
            b = ends[(i + 1) % 4]
            ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color=color, linewidth=1.0)

        r = cv2.Rodrigues(rvec)[0]
        axes_world = r.T @ np.eye(3) * 90.0
        axis_colors = ["red", "green", "blue"]
        for vec, axis_color in zip(axes_world.T, axis_colors):
            end = center + vec
            ax.plot(
                [center[0], end[0]],
                [center[1], end[1]],
                [center[2], end[2]],
                color=axis_color,
                linewidth=2.0,
            )

    ax.set_xlabel("world X (mm)")
    ax.set_ylabel("world Y (mm)")
    ax.set_zlabel("world Z (mm)")
    set_equal_3d_axes(ax)
    ax.view_init(elev=28, azim=-55)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def triangulate_common_board_corners(
    cam_solutions: dict[str, dict],
    aggregate_poses: dict[str, tuple[np.ndarray, np.ndarray]],
    board: cv2.aruco.CharucoBoard,
) -> dict | None:
    names = sorted(cam_solutions.keys())
    if len(names) < 2:
        return None
    s0, s1 = cam_solutions[names[0]], cam_solutions[names[1]]
    ids0 = s0["charuco_ids"].flatten().astype(int)
    ids1 = s1["charuco_ids"].flatten().astype(int)
    common = sorted(set(ids0.tolist()).intersection(ids1.tolist()))
    if len(common) < 8:
        return None

    idx0 = [np.where(ids0 == i)[0][0] for i in common]
    idx1 = [np.where(ids1 == i)[0][0] for i in common]
    pts0 = s0["charuco_corners"][idx0].astype(np.float64).reshape(-1, 1, 2)
    pts1 = s1["charuco_corners"][idx1].astype(np.float64).reshape(-1, 1, 2)
    norm0 = cv2.fisheye.undistortPoints(pts0, K_DEFAULT, D_DEFAULT).reshape(-1, 2).T
    norm1 = cv2.fisheye.undistortPoints(pts1, K_DEFAULT, D_DEFAULT).reshape(-1, 2).T

    rvec0, tvec0 = aggregate_poses[names[0]]
    rvec1, tvec1 = aggregate_poses[names[1]]
    r0 = cv2.Rodrigues(rvec0)[0]
    r1 = cv2.Rodrigues(rvec1)[0]
    p0 = np.hstack([r0, tvec0.reshape(3, 1)])
    p1 = np.hstack([r1, tvec1.reshape(3, 1)])
    homog = cv2.triangulatePoints(p0, p1, norm0, norm1)
    xyz = (homog[:3] / homog[3]).T
    expected = board.getChessboardCorners()[common].astype(np.float64)
    err = np.linalg.norm(xyz - expected, axis=1)
    return {
        "camera_pair": names,
        "corner_ids": common,
        "xyz": xyz,
        "expected": expected,
        "errors_mm": err,
    }


def save_board_triangulation(out_path: Path, validation: dict) -> None:
    xyz = validation["xyz"]
    expected = validation["expected"]
    err = validation["errors_mm"]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(
        f"Triangulated board corners vs known board corners\n"
        f"mean error {err.mean():.2f} mm, median {np.median(err):.2f} mm"
    )
    ax.scatter(expected[:, 0], expected[:, 1], expected[:, 2], color="black", s=20, label="known")
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], color="tab:orange", s=18, label="triangulated")
    for a, b in zip(expected, xyz):
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color="red", alpha=0.35)
    ax.set_xlabel("world X (mm)")
    ax.set_ylabel("world Y (mm)")
    ax.set_zlabel("world Z (mm)")
    ax.legend()
    set_equal_3d_axes(ax)
    ax.view_init(elev=45, azim=-75)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def make_contact_sheet(out_path: Path, image_paths: list[Path]) -> None:
    images = [cv2.imread(str(p)) for p in image_paths]
    thumbs = []
    for img in images:
        h, w = img.shape[:2]
        target_w = 720
        target_h = int(h * (target_w / w))
        thumbs.append(cv2.resize(img, (target_w, target_h)))
    rows = []
    for i in range(0, len(thumbs), 2):
        row_imgs = thumbs[i : i + 2]
        if len(row_imgs) == 1:
            row_imgs.append(np.zeros_like(row_imgs[0]))
        rows.append(np.hstack(row_imgs))
    sheet = np.vstack(rows)
    cv2.imwrite(str(out_path), sheet)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cv2.setLogLevel(0)

    cameras = [
        CameraInput(
            "cam1",
            args.data_root / "v1-cam1" / "a77033-H-0001.mp4",
            stable_start_s=60,
            stable_end_s=130,
        ),
        CameraInput(
            "cam2",
            args.data_root / "v1-cam2" / "a72491-H-0001.mp4",
            stable_start_s=55,
            stable_end_s=120,
        ),
    ]

    dictionary = aruco_dict(args.dict)
    board = cv2.aruco.CharucoBoard(
        (args.board_cols, args.board_rows),
        args.square_mm,
        args.marker_mm,
        dictionary,
    )
    detector = cv2.aruco.CharucoDetector(board)

    best_solutions: dict[str, dict] = {}
    aggregate_poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    report: dict = {
        "intrinsics": K_DEFAULT.tolist(),
        "distortion_model": "opencv_fisheye_equidistant",
        "distortion": D_DEFAULT.reshape(-1).tolist(),
        "board": {
            "cols": args.board_cols,
            "rows": args.board_rows,
            "square_mm": args.square_mm,
            "marker_mm": args.marker_mm,
            "dictionary": args.dict,
        },
        "cameras": {},
    }

    generated_images: list[Path] = []

    for cam in cameras:
        solutions = []
        for time_s in range(cam.stable_start_s, cam.stable_end_s + 1, args.sample_step_s):
            frame = read_frame(cam.video, time_s)
            sol = detect_and_solve(frame, board, detector, K_DEFAULT, D_DEFAULT)
            if sol is None:
                continue
            mean_err = float(sol["reprojection_errors_px"].mean())
            n_corners = int(len(sol["charuco_ids"]))
            if n_corners >= 40 and mean_err < 3.0:
                sol["time_s"] = time_s
                sol["frame"] = frame
                solutions.append(sol)

        if not solutions:
            raise RuntimeError(f"No usable ChArUco poses found for {cam.name}")

        rvec, tvec = aggregate_pose(solutions)
        aggregate_poses[cam.name] = (rvec, tvec)
        best = sorted(
            solutions,
            key=lambda s: (-len(s["charuco_ids"]), float(s["reprojection_errors_px"].mean())),
        )[0]
        best_solutions[cam.name] = best

        det_path = args.out_dir / f"{cam.name}_01_2d_charuco_detection.jpg"
        reproj_path = args.out_dir / f"{cam.name}_02_reprojection_overlay.jpg"
        save_detection_overlay(
            det_path,
            best["frame"],
            best,
            args.board_cols,
            args.board_rows,
            args.square_mm,
        )
        save_reprojection_overlay(reproj_path, best["frame"], best)
        generated_images.extend([det_path, reproj_path])

        all_errors = np.concatenate([s["reprojection_errors_px"] for s in solutions])
        center = camera_center_world(rvec, tvec)
        report["cameras"][cam.name] = {
            "video": str(cam.video),
            "stable_interval_s": [cam.stable_start_s, cam.stable_end_s],
            "num_pose_samples": len(solutions),
            "best_visualized_frame_s": best["time_s"],
            "rvec_world_to_cam": rvec.reshape(3).tolist(),
            "R_world_to_cam": cv2.Rodrigues(rvec)[0].tolist(),
            "t_world_to_cam_mm": tvec.reshape(3).tolist(),
            "camera_center_world_mm": center.reshape(3).tolist(),
            "reprojection_mean_px": float(all_errors.mean()),
            "reprojection_median_px": float(np.median(all_errors)),
            "reprojection_max_px": float(all_errors.max()),
        }

    scene_path = args.out_dir / "03_3d_board_and_camera_poses.png"
    save_3d_scene(scene_path, aggregate_poses, args.board_cols, args.board_rows, args.square_mm)
    generated_images.append(scene_path)

    validation = triangulate_common_board_corners(best_solutions, aggregate_poses, board)
    if validation is not None:
        tri_path = args.out_dir / "04_triangulated_board_corners_validation.png"
        save_board_triangulation(tri_path, validation)
        generated_images.append(tri_path)
        report["board_corner_triangulation_validation"] = {
            "camera_pair": validation["camera_pair"],
            "num_common_corners": len(validation["corner_ids"]),
            "mean_error_mm": float(validation["errors_mm"].mean()),
            "median_error_mm": float(np.median(validation["errors_mm"])),
            "max_error_mm": float(validation["errors_mm"].max()),
            "note": "Sanity check uses ChArUco board corners, so it is not an independent hand-pose accuracy estimate.",
        }

    contact_path = args.out_dir / "00_process_contact_sheet.jpg"
    make_contact_sheet(contact_path, generated_images[:4])
    report["generated_images"] = [str(contact_path)] + [str(p) for p in generated_images]

    with (args.out_dir / "extrinsics_report.json").open("w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
