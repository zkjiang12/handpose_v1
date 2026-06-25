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


def write_preview_html(path: Path, points: np.ndarray, colors: np.ndarray, report: dict) -> None:
    stride = max(1, int(math.ceil(len(points) / 18000)))
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
  <title>Calibrated Board-Plane Splat Preview</title>
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
    <div>Board-plane colored splat preview</div>
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
- `board_plane_splat_preview.html`: local browser preview of the board-plane splat proxy.

## Important limitation

This is not a trained 3DGS model yet. A real trained splat needs a trainer such as Nerfstudio/gsplat/GraphDECO 3DGS. Those are not installed in this environment, and this machine currently has no CUDA GPU. Also, this capture only has four unique viewpoints, so a trained model will mostly reproduce these views and will not hallucinate unseen scene sides reliably.

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
    write_ply(out_dir / "board_plane_splat_points.ply", splat_points, splat_colors)
    write_preview_html(out_dir / "board_plane_splat_preview.html", splat_points, splat_colors, report)
    write_readme(out_dir / "README.md", labels)

    summary = {
        "out_dir": str(out_dir),
        "num_images": len(undistorted_frames),
        "image_size": [int(width), int(height)],
        "undistorted_intrinsics": k_undistorted.tolist(),
        "num_board_splat_points": int(len(splat_points)),
        "limitations": [
            "No trained Gaussian Splat model is produced locally because nerfstudio/gsplat/colmap are not installed.",
            "Only four unique viewpoints are available, so true novel-view reconstruction quality will be limited.",
            "Moving hands are dynamic foreground and should be masked/avoided for static-scene training.",
        ],
        "files": {
            "transforms": str(out_dir / "transforms.json"),
            "colmap_text": str(colmap_dir),
            "board_plane_splat_ply": str(out_dir / "board_plane_splat_points.ply"),
            "board_plane_splat_preview": str(out_dir / "board_plane_splat_preview.html"),
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
