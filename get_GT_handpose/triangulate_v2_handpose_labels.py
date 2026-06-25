#!/usr/bin/env python3
"""Triangulate exported v2 four-camera hand labels and compare bone lengths."""

from __future__ import annotations

import argparse
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

REPORT_BY_SEGMENT = {
    "segment1": Path("/Users/zikangjiang/dev/ego-exo/visualizations/v2/take1/v2_calibration_and_audio_sync_report.json"),
    "segment2": Path("/Users/zikangjiang/dev/ego-exo/visualizations/v2/take2/v2_calibration_and_audio_sync_report.json"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("labels_json", type=Path)
    parser.add_argument(
        "--bone-lengths-json",
        type=Path,
        help="Optional JSON containing {'bone_lengths_mm': {'wrist_thumb_cmc': 52, ...}}.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/Users/zikangjiang/dev/ego-exo/visualizations/v2/handpose_labeler/results"),
    )
    return parser.parse_args()


def load_poses(labels: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    reports = {name: json.loads(path.read_text()) for name, path in REPORT_BY_SEGMENT.items()}
    poses = {}
    for cam, cam_info in labels["cameras"].items():
        segment_id = cam_info["segment_id"]
        if segment_id is None:
            continue
        cam_report = reports[segment_id]["extrinsics"]["cameras"][cam]
        rvec = np.array(cam_report["rvec_world_to_cam"], dtype=np.float64).reshape(3, 1)
        tvec = np.array(cam_report["t_world_to_cam_mm"], dtype=np.float64).reshape(3, 1)
        poses[cam] = (rvec, tvec)
    return poses


def triangulate_multiview(
    observations: dict[str, list[float]],
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


def reprojection_errors(
    xyz: np.ndarray,
    observations: dict[str, list[float]],
    poses: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, float]:
    errors = {}
    for cam, uv in observations.items():
        rvec, tvec = poses[cam]
        projected, _ = cv2.fisheye.projectPoints(
            xyz.reshape(1, 1, 3).astype(np.float64),
            rvec,
            tvec,
            K_DEFAULT,
            D_DEFAULT,
        )
        errors[cam] = float(np.linalg.norm(projected.reshape(2) - np.array(uv, dtype=np.float64)))
    return errors


def bone_name(a: int, b: int, names: list[str]) -> str:
    return f"{names[a]}_{names[b]}"


def triangulate_combo(
    hand_labels: dict,
    combo: tuple[str, ...],
    poses: dict[str, tuple[np.ndarray, np.ndarray]],
    keypoint_names: list[str],
    edges: list[list[int]],
    gt_bones: dict[str, float],
) -> dict:
    points = {}
    reproj = {}
    for kp_s, cams in hand_labels.items():
        if all(cam in cams and cam in poses for cam in combo):
            observations = {cam: cams[cam] for cam in combo}
            xyz = triangulate_multiview(observations, poses)
            points[kp_s] = xyz
            reproj[kp_s] = reprojection_errors(xyz, observations, poses)

    bones = []
    for a, b in edges:
        a_s = str(a)
        b_s = str(b)
        if a_s not in points or b_s not in points:
            continue
        length = float(np.linalg.norm(points[b_s] - points[a_s]))
        name = bone_name(a, b, keypoint_names)
        gt = gt_bones.get(name)
        bones.append(
            {
                "bone": name,
                "start": a,
                "end": b,
                "length_mm": length,
                "gt_length_mm": gt,
                "error_mm": None if gt is None else length - gt,
                "abs_error_mm": None if gt is None else abs(length - gt),
            }
        )

    reproj_values = [err for item in reproj.values() for err in item.values()]
    abs_bone_errors = [bone["abs_error_mm"] for bone in bones if bone["abs_error_mm"] is not None]
    return {
        "combo": "+".join(combo),
        "num_cameras": len(combo),
        "num_triangulated_keypoints": len(points),
        "points_mm": {
            keypoint_names[int(kp)]: [float(v) for v in xyz]
            for kp, xyz in points.items()
        },
        "reprojection_errors_px": reproj,
        "reprojection_mean_px": None if not reproj_values else float(np.mean(reproj_values)),
        "reprojection_median_px": None if not reproj_values else float(np.median(reproj_values)),
        "bones": bones,
        "bone_abs_error_mean_mm": None if not abs_bone_errors else float(np.mean(abs_bone_errors)),
        "bone_abs_error_median_mm": None if not abs_bone_errors else float(np.median(abs_bone_errors)),
        "bone_abs_error_max_mm": None if not abs_bone_errors else float(np.max(abs_bone_errors)),
    }


def save_bone_csv(path: Path, results: dict) -> None:
    rows = []
    for hand, combos in results["hands"].items():
        for combo in combos:
            for bone in combo["bones"]:
                rows.append(
                    {
                        "hand": hand,
                        "combo": combo["combo"],
                        "num_cameras": combo["num_cameras"],
                        **bone,
                    }
                )
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_summary_plot(path: Path, results: dict) -> None:
    items = []
    for hand, combos in results["hands"].items():
        for combo in combos:
            if combo["bone_abs_error_median_mm"] is None:
                continue
            items.append((hand, combo["combo"], combo["num_cameras"], combo["bone_abs_error_median_mm"]))
    if not items:
        return
    labels = [f"{hand}\\n{combo}" for hand, combo, _, _ in items]
    vals = [v for _, _, _, v in items]
    colors = {2: "#2374ab", 3: "#2a9d55", 4: "#7c4dbe"}
    fig, ax = plt.subplots(figsize=(max(10, len(items) * 0.8), 6), constrained_layout=True)
    ax.bar(np.arange(len(items)), vals, color=[colors[n] for _, _, n, _ in items])
    ax.set_xticks(np.arange(len(items)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("median absolute bone-length error (mm)")
    ax.set_title("Hand Triangulation Bone-Length Error by Camera Combination")
    ax.grid(axis="y", color="#d6dbe1", alpha=0.8)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    labels = json.loads(args.labels_json.read_text())
    gt_bones = {}
    if args.bone_lengths_json:
        gt_bones = json.loads(args.bone_lengths_json.read_text()).get("bone_lengths_mm", {})

    poses = load_poses(labels)
    cameras = sorted(poses.keys())
    keypoint_names = labels["keypoint_names"]
    edges = labels["edges"]

    results = {
        "source_labels": str(args.labels_json),
        "bone_lengths_source": str(args.bone_lengths_json) if args.bone_lengths_json else None,
        "hands": {},
    }
    for hand, hand_labels in labels["hands"].items():
        combos = []
        for combo_size in (2, 3, 4):
            for combo in itertools.combinations(cameras, combo_size):
                combos.append(triangulate_combo(hand_labels, combo, poses, keypoint_names, edges, gt_bones))
        results["hands"][hand] = combos

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / f"{args.labels_json.stem}_triangulated_results.json"
    out_csv = args.out_dir / f"{args.labels_json.stem}_bone_lengths.csv"
    out_plot = args.out_dir / f"{args.labels_json.stem}_bone_length_errors.png"
    out_json.write_text(json.dumps(results, indent=2))
    save_bone_csv(out_csv, results)
    save_summary_plot(out_plot, results)
    print(json.dumps({"results": str(out_json), "bone_csv": str(out_csv), "plot": str(out_plot)}, indent=2))


if __name__ == "__main__":
    main()
