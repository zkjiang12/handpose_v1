#!/usr/bin/env python3
"""Audit MANO-style hand keypoints for geometric plausibility."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import zarr

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from egoverse_handpose_dataset import decode_rgb, valid_joint_mask, world_to_camera  # noqa: E402
from filter_egoverse_handpose_visibility import camera_intrinsics  # noqa: E402


HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)

FINGER_CHAINS = {
    "thumb": (0, 1, 2, 3, 4),
    "index": (0, 5, 6, 7, 8),
    "middle": (0, 9, 10, 11, 12),
    "ring": (0, 13, 14, 15, 16),
    "pinky": (0, 17, 18, 19, 20),
}

MCP_JOINTS = (5, 9, 13, 17)
FINGER_TIP_JOINTS = (4, 8, 12, 16, 20)
SIDE_HAS_KEY = {"left": "has_left", "right": "has_right"}
SIDE_KP_KEY = {"left": "left_keypoints_key", "right": "right_keypoints_key"}
HARD_GEOMETRY_FLAGS = {
    "tiny_bone",
    "huge_bone",
    "tiny_hand_span",
    "huge_hand_span",
    "mcp_order",
    "episode_bone_rms",
    "episode_bone_max",
    "max_speed",
}
SOFT_GEOMETRY_FLAGS = {
    "acute_finger_angle",
    "finger_bone_ratio",
    "edge_length_outlier",
    "hand_scale_outlier",
    "median_speed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit geometry of MANO-style EgoVerse hand keypoints.")
    parser.add_argument("--train-csv", default="outputs/handpose_dataset_visible/train.csv")
    parser.add_argument("--test-csv", default="outputs/handpose_dataset_visible/test.csv")
    parser.add_argument("--out-dir", default="outputs/geometry_audit")
    parser.add_argument("--cache-root", default=os.environ.get("EGOVERSE_CACHE_DIR"))
    parser.add_argument("--min-edge-m", type=float, default=0.003)
    parser.add_argument("--max-edge-m", type=float, default=0.14)
    parser.add_argument("--min-hand-span-m", type=float, default=0.05)
    parser.add_argument("--max-hand-span-m", type=float, default=0.35)
    parser.add_argument("--min-finger-angle-deg", type=float, default=25.0)
    parser.add_argument("--max-edge-robust-z", type=float, default=6.0)
    parser.add_argument("--max-episode-bone-rms-rel", type=float, default=0.15)
    parser.add_argument("--max-episode-bone-rel-dev", type=float, default=0.50)
    parser.add_argument("--max-median-joint-speed-mps", type=float, default=3.0)
    parser.add_argument("--max-any-joint-speed-mps", type=float, default=6.0)
    parser.add_argument("--viz-count", type=int, default=16, help="Number of worst suspicious examples to render.")
    return parser.parse_args()


def str_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def resolve_episode_path(row: dict[str, str], cache_root: str | None) -> Path:
    if cache_root:
        return Path(cache_root).expanduser() / row["episode_hash"]
    return Path(row["episode_path"]).expanduser()


def safe_norm(values: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.linalg.norm(values, axis=axis)


def angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    v1 = a - b
    v2 = c - b
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 <= 1e-9 or n2 <= 1e-9:
        return float("nan")
    cos = float(np.dot(v1, v2) / (n1 * n2))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def robust_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    med = np.nanmedian(values, axis=0)
    mad = np.nanmedian(np.abs(values - med), axis=0)
    return med, np.maximum(mad, 1e-9)


def projection_order_ok(points: np.ndarray, joints: tuple[int, ...]) -> bool:
    selected = points[list(joints)]
    centered = selected - selected.mean(axis=0, keepdims=True)
    if np.linalg.norm(centered) <= 1e-9:
        return False
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    scalars = centered @ axis
    diffs = np.diff(scalars)
    return bool(np.all(diffs >= -1e-6) or np.all(diffs <= 1e-6))


def hand_metrics(points: np.ndarray) -> dict[str, Any]:
    edge_lengths = np.asarray([np.linalg.norm(points[i] - points[j]) for i, j in HAND_EDGES], dtype=np.float64)
    pairwise = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    finger_angles = {}
    min_finger_angle = float("inf")
    for finger, chain in FINGER_CHAINS.items():
        angles = []
        for prev_idx, joint_idx, next_idx in zip(chain[:-2], chain[1:-1], chain[2:]):
            value = angle_deg(points[prev_idx], points[joint_idx], points[next_idx])
            angles.append(value)
            if math.isfinite(value):
                min_finger_angle = min(min_finger_angle, value)
        finger_angles[finger] = angles

    non_thumb_ratio_violations = 0
    for finger in ("index", "middle", "ring", "pinky"):
        chain = FINGER_CHAINS[finger]
        phalanges = np.asarray(
            [
                np.linalg.norm(points[chain[1]] - points[chain[2]]),
                np.linalg.norm(points[chain[2]] - points[chain[3]]),
                np.linalg.norm(points[chain[3]] - points[chain[4]]),
            ],
            dtype=np.float64,
        )
        if phalanges[1] > 1.5 * phalanges[0] or phalanges[2] > 1.5 * phalanges[1]:
            non_thumb_ratio_violations += 1

    mcp_order_ok = projection_order_ok(points, MCP_JOINTS)
    return {
        "edge_lengths": edge_lengths,
        "hand_span_m": float(np.nanmax(pairwise)),
        "wrist_to_tip_max_m": float(np.nanmax(np.linalg.norm(points[list(FINGER_TIP_JOINTS)] - points[0], axis=1))),
        "total_bone_length_m": float(np.nansum(edge_lengths)),
        "min_finger_angle_deg": min_finger_angle if math.isfinite(min_finger_angle) else float("nan"),
        "finger_angles": finger_angles,
        "non_thumb_ratio_violations": non_thumb_ratio_violations,
        "mcp_order_ok": mcp_order_ok,
    }


def collect_samples(csv_paths: dict[str, Path], cache_root: str | None) -> list[dict[str, Any]]:
    groups: dict[str, zarr.Group] = {}
    samples: list[dict[str, Any]] = []
    for split, csv_path in csv_paths.items():
        for row_idx, row in enumerate(load_rows(csv_path)):
            episode_hash = row["episode_hash"]
            episode_path = resolve_episode_path(row, cache_root)
            if episode_hash not in groups:
                groups[episode_hash] = zarr.open_group(str(episode_path), mode="r")
            group = groups[episode_hash]
            frame_idx = int(row["frame_idx"])
            fps = float(group.attrs.get("fps", 30.0) or 30.0)
            for side in ("left", "right"):
                if not str_bool(row.get(SIDE_HAS_KEY[side])) or not row.get(SIDE_KP_KEY[side]):
                    continue
                points = np.asarray(group[row[SIDE_KP_KEY[side]]][frame_idx], dtype=np.float64).reshape(21, 3)
                valid = valid_joint_mask(points)
                if not valid.all():
                    continue
                metrics = hand_metrics(points)
                samples.append(
                    {
                        "split": split,
                        "row_idx": row_idx,
                        "row": row,
                        "episode_hash": episode_hash,
                        "episode_path": episode_path,
                        "frame_idx": frame_idx,
                        "side": side,
                        "fps": fps,
                        "points": points,
                        **metrics,
                    }
                )
    return samples


def add_dataset_robust_flags(samples: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    edge_matrix = np.stack([sample["edge_lengths"] for sample in samples], axis=0)
    log_edges = np.log(np.clip(edge_matrix, 1e-9, None))
    edge_med, edge_mad = robust_stats(log_edges)
    total_lengths = np.asarray([sample["total_bone_length_m"] for sample in samples], dtype=np.float64)
    total_med = float(np.nanmedian(total_lengths))
    total_mad = float(max(np.nanmedian(np.abs(total_lengths - total_med)), 1e-9))

    for sample, log_edge in zip(samples, log_edges):
        edge_z = 0.6745 * (log_edge - edge_med) / edge_mad
        total_z = 0.6745 * (sample["total_bone_length_m"] - total_med) / total_mad
        sample["max_abs_edge_robust_z"] = float(np.nanmax(np.abs(edge_z)))
        sample["total_bone_length_robust_z"] = float(total_z)
        sample["edge_outlier_count"] = int(np.sum(np.abs(edge_z) > args.max_edge_robust_z))

    return {
        "edge_length_median_m": edge_med.tolist(),
        "edge_length_mad_log": edge_mad.tolist(),
        "total_bone_length_median_m": total_med,
        "total_bone_length_mad_m": total_mad,
    }


def add_episode_bone_consistency(samples: list[dict[str, Any]], args: argparse.Namespace) -> None:
    by_sequence: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_sequence[(sample["episode_hash"], sample["side"])].append(sample)
    for seq_samples in by_sequence.values():
        edge_matrix = np.stack([sample["edge_lengths"] for sample in seq_samples], axis=0)
        seq_median = np.nanmedian(edge_matrix, axis=0)
        denom = np.maximum(seq_median, 1e-9)
        for sample in seq_samples:
            rel = np.abs(sample["edge_lengths"] - seq_median) / denom
            sample["episode_bone_rms_rel"] = float(np.sqrt(np.nanmean(rel**2)))
            sample["episode_bone_max_rel_dev"] = float(np.nanmax(rel))


def add_temporal_metrics(samples: list[dict[str, Any]], args: argparse.Namespace) -> None:
    by_sequence: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        sample["median_joint_speed_mps"] = float("nan")
        sample["max_joint_speed_mps"] = float("nan")
        sample["prev_frame_gap"] = 0
        by_sequence[(sample["episode_hash"], sample["side"])].append(sample)

    for seq_samples in by_sequence.values():
        seq_samples.sort(key=lambda sample: sample["frame_idx"])
        for prev, current in zip(seq_samples[:-1], seq_samples[1:]):
            frame_gap = int(current["frame_idx"]) - int(prev["frame_idx"])
            if frame_gap <= 0:
                continue
            dt = frame_gap / float(current["fps"] or 30.0)
            speeds = np.linalg.norm(current["points"] - prev["points"], axis=1) / max(dt, 1e-9)
            current["median_joint_speed_mps"] = float(np.nanmedian(speeds))
            current["max_joint_speed_mps"] = float(np.nanmax(speeds))
            current["prev_frame_gap"] = frame_gap


def flag_samples(samples: list[dict[str, Any]], args: argparse.Namespace) -> None:
    for sample in samples:
        edge_lengths = sample["edge_lengths"]
        flags = []
        if float(np.nanmin(edge_lengths)) < args.min_edge_m:
            flags.append("tiny_bone")
        if float(np.nanmax(edge_lengths)) > args.max_edge_m:
            flags.append("huge_bone")
        if sample["hand_span_m"] < args.min_hand_span_m:
            flags.append("tiny_hand_span")
        if sample["hand_span_m"] > args.max_hand_span_m:
            flags.append("huge_hand_span")
        if sample["min_finger_angle_deg"] < args.min_finger_angle_deg:
            flags.append("acute_finger_angle")
        if sample["non_thumb_ratio_violations"] > 0:
            flags.append("finger_bone_ratio")
        if not sample["mcp_order_ok"]:
            flags.append("mcp_order")
        if sample["edge_outlier_count"] > 0:
            flags.append("edge_length_outlier")
        if abs(sample["total_bone_length_robust_z"]) > args.max_edge_robust_z:
            flags.append("hand_scale_outlier")
        if sample["episode_bone_rms_rel"] > args.max_episode_bone_rms_rel:
            flags.append("episode_bone_rms")
        if sample["episode_bone_max_rel_dev"] > args.max_episode_bone_rel_dev:
            flags.append("episode_bone_max")
        if math.isfinite(sample["median_joint_speed_mps"]) and sample["median_joint_speed_mps"] > args.max_median_joint_speed_mps:
            flags.append("median_speed")
        if math.isfinite(sample["max_joint_speed_mps"]) and sample["max_joint_speed_mps"] > args.max_any_joint_speed_mps:
            flags.append("max_speed")
        sample["flags"] = flags
        sample["flag_count"] = len(flags)
        sample["hard_flags"] = [flag for flag in flags if flag in HARD_GEOMETRY_FLAGS]
        sample["soft_flags"] = [flag for flag in flags if flag in SOFT_GEOMETRY_FLAGS]
        sample["hard_flag_count"] = len(sample["hard_flags"])
        sample["soft_flag_count"] = len(sample["soft_flags"])
        sample["is_suspicious"] = bool(flags)
        sample["hard_geometry_failure"] = bool(sample["hard_flags"])


def summarize(values: list[float]) -> dict[str, float | int] | None:
    arr = np.asarray([value for value in values if math.isfinite(float(value))], dtype=np.float64)
    if arr.size == 0:
        return None
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def sample_to_row(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "split": sample["split"],
        "row_idx": sample["row_idx"],
        "episode_hash": sample["episode_hash"],
        "frame_idx": sample["frame_idx"],
        "side": sample["side"],
        "flag_count": sample["flag_count"],
        "hard_flag_count": sample["hard_flag_count"],
        "soft_flag_count": sample["soft_flag_count"],
        "flags": "|".join(sample["flags"]),
        "hard_flags": "|".join(sample["hard_flags"]),
        "soft_flags": "|".join(sample["soft_flags"]),
        "hand_span_m": sample["hand_span_m"],
        "wrist_to_tip_max_m": sample["wrist_to_tip_max_m"],
        "total_bone_length_m": sample["total_bone_length_m"],
        "total_bone_length_robust_z": sample["total_bone_length_robust_z"],
        "min_edge_m": float(np.nanmin(sample["edge_lengths"])),
        "max_edge_m": float(np.nanmax(sample["edge_lengths"])),
        "max_abs_edge_robust_z": sample["max_abs_edge_robust_z"],
        "edge_outlier_count": sample["edge_outlier_count"],
        "min_finger_angle_deg": sample["min_finger_angle_deg"],
        "non_thumb_ratio_violations": sample["non_thumb_ratio_violations"],
        "mcp_order_ok": sample["mcp_order_ok"],
        "episode_bone_rms_rel": sample["episode_bone_rms_rel"],
        "episode_bone_max_rel_dev": sample["episode_bone_max_rel_dev"],
        "median_joint_speed_mps": sample["median_joint_speed_mps"],
        "max_joint_speed_mps": sample["max_joint_speed_mps"],
        "prev_frame_gap": sample["prev_frame_gap"],
    }


def build_summary(samples: list[dict[str, Any]], robust_summary: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_episode: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    flag_counts = Counter()
    for sample in samples:
        by_split[sample["split"]].append(sample)
        by_episode[(sample["split"], sample["episode_hash"])].append(sample)
        flag_counts.update(sample["flags"])

    split_summary = {}
    for split, split_samples in sorted(by_split.items()):
        split_summary[split] = {
            "hands": len(split_samples),
            "suspicious_hands": sum(sample["is_suspicious"] for sample in split_samples),
            "suspicious_rate": sum(sample["is_suspicious"] for sample in split_samples) / max(1, len(split_samples)),
            "hard_geometry_failure_hands": sum(sample["hard_geometry_failure"] for sample in split_samples),
            "hard_geometry_failure_rate": sum(sample["hard_geometry_failure"] for sample in split_samples) / max(1, len(split_samples)),
            "soft_warning_hands": sum(bool(sample["soft_flags"]) for sample in split_samples),
            "soft_warning_rate": sum(bool(sample["soft_flags"]) for sample in split_samples) / max(1, len(split_samples)),
            "flag_counts": dict(Counter(flag for sample in split_samples for flag in sample["flags"])),
            "hard_flag_counts": dict(Counter(flag for sample in split_samples for flag in sample["hard_flags"])),
            "soft_flag_counts": dict(Counter(flag for sample in split_samples for flag in sample["soft_flags"])),
            "hand_span_m": summarize([sample["hand_span_m"] for sample in split_samples]),
            "total_bone_length_m": summarize([sample["total_bone_length_m"] for sample in split_samples]),
            "episode_bone_rms_rel": summarize([sample["episode_bone_rms_rel"] for sample in split_samples]),
            "max_joint_speed_mps": summarize([sample["max_joint_speed_mps"] for sample in split_samples]),
        }

    episode_summary = []
    for (split, episode_hash), episode_samples in sorted(by_episode.items()):
        episode_summary.append(
            {
                "split": split,
                "episode_hash": episode_hash,
                "hands": len(episode_samples),
                "suspicious_hands": sum(sample["is_suspicious"] for sample in episode_samples),
                "suspicious_rate": sum(sample["is_suspicious"] for sample in episode_samples) / max(1, len(episode_samples)),
                "hard_geometry_failure_hands": sum(sample["hard_geometry_failure"] for sample in episode_samples),
                "hard_geometry_failure_rate": sum(sample["hard_geometry_failure"] for sample in episode_samples) / max(1, len(episode_samples)),
                "flag_counts": dict(Counter(flag for sample in episode_samples for flag in sample["flags"])),
                "episode_bone_rms_rel": summarize([sample["episode_bone_rms_rel"] for sample in episode_samples]),
                "max_joint_speed_mps": summarize([sample["max_joint_speed_mps"] for sample in episode_samples]),
            }
        )

    worst_samples = sorted(samples, key=lambda sample: (sample["flag_count"], sample["episode_bone_rms_rel"], sample["max_abs_edge_robust_z"]), reverse=True)
    return {
        "args": vars(args),
        "mano_available": False,
        "mano_note": (
            "True MANO mesh fitting was not run. This audit checks MANO-order keypoint geometry, "
            "fixed-skeleton consistency, and temporal plausibility."
        ),
        "robust_reference": robust_summary,
        "total_hands": len(samples),
        "suspicious_hands": sum(sample["is_suspicious"] for sample in samples),
        "suspicious_rate": sum(sample["is_suspicious"] for sample in samples) / max(1, len(samples)),
        "hard_geometry_failure_hands": sum(sample["hard_geometry_failure"] for sample in samples),
        "hard_geometry_failure_rate": sum(sample["hard_geometry_failure"] for sample in samples) / max(1, len(samples)),
        "soft_warning_hands": sum(bool(sample["soft_flags"]) for sample in samples),
        "soft_warning_rate": sum(bool(sample["soft_flags"]) for sample in samples) / max(1, len(samples)),
        "flag_counts": dict(flag_counts),
        "hard_flag_counts": dict(Counter(flag for sample in samples for flag in sample["hard_flags"])),
        "soft_flag_counts": dict(Counter(flag for sample in samples for flag in sample["soft_flags"])),
        "split_summary": split_summary,
        "episode_summary": sorted(episode_summary, key=lambda row: row["suspicious_rate"], reverse=True),
        "worst_samples": [sample_to_row(sample) for sample in worst_samples[:100]],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def project_points(points_cam: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    z = np.clip(points_cam[:, 2], 1e-9, None)
    return np.stack(
        [
            intrinsics[0, 0] * points_cam[:, 0] / z + intrinsics[0, 2],
            intrinsics[1, 1] * points_cam[:, 1] / z + intrinsics[1, 2],
        ],
        axis=1,
    )


def draw_hand_overlay(
    image: np.ndarray,
    points_cam: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray,
    color: tuple[int, int, int],
    *,
    radius: int,
    thickness: int,
) -> None:
    h, w = image.shape[:2]
    px = project_points(points_cam, intrinsics)
    valid = mask & (points_cam[:, 2] > 0.01)
    valid &= (px[:, 0] >= 0.0) & (px[:, 0] < w) & (px[:, 1] >= 0.0) & (px[:, 1] < h)
    for i, j in HAND_EDGES:
        if valid[i] and valid[j]:
            p1 = tuple(np.round(px[i]).astype(int))
            p2 = tuple(np.round(px[j]).astype(int))
            cv2.line(image, p1, p2, color, thickness, cv2.LINE_AA)
    for idx in range(21):
        if valid[idx]:
            center = tuple(np.round(px[idx]).astype(int))
            cv2.circle(image, center, radius, color, -1, cv2.LINE_AA)


def add_label(image: np.ndarray, lines: list[str]) -> np.ndarray:
    image = np.ascontiguousarray(image.copy())
    overlay_h = 62
    cv2.rectangle(image, (0, 0), (image.shape[1], overlay_h), (0, 0, 0), -1)
    for idx, line in enumerate(lines[:3]):
        cv2.putText(
            image,
            line[:72],
            (8, 18 + idx * 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return image


def render_overlay_sample(sample: dict[str, Any], groups: dict[str, zarr.Group]) -> np.ndarray:
    group = groups.setdefault(sample["episode_hash"], zarr.open_group(str(sample["episode_path"]), mode="r"))
    row = sample["row"]
    frame_idx = int(sample["frame_idx"])
    image = decode_rgb(group[row["image_key"]][frame_idx])
    image = np.ascontiguousarray(image.copy())
    head_pose = np.asarray(group[row["head_pose_key"]][frame_idx], dtype=np.float64)
    intrinsics, _, _ = camera_intrinsics(group, row["image_key"])
    side_colors = {"left": (60, 180, 255), "right": (255, 150, 50)}
    for side in ("left", "right"):
        key = row.get(SIDE_KP_KEY[side]) or ""
        if not (str_bool(row.get(SIDE_HAS_KEY[side])) and key):
            continue
        points_world = np.asarray(group[key][frame_idx], dtype=np.float64).reshape(21, 3)
        mask = valid_joint_mask(points_world)
        points_cam = np.zeros((21, 3), dtype=np.float64)
        if mask.any():
            points_cam[mask] = world_to_camera(points_world[mask], head_pose)
        color = (255, 40, 40) if side == sample["side"] else side_colors[side]
        draw_hand_overlay(
            image,
            points_cam,
            mask,
            intrinsics,
            color,
            radius=4 if side == sample["side"] else 3,
            thickness=3 if side == sample["side"] else 2,
        )
    return add_label(
        image,
        [
            f"{sample['split']} {sample['episode_hash']} f{sample['frame_idx']} {sample['side']}",
            f"{'|'.join(sample['flags'])}",
            f"span={sample['hand_span_m']*1000:.1f}mm angle={sample['min_finger_angle_deg']:.1f} z={sample['max_abs_edge_robust_z']:.1f}",
        ],
    )


def save_contact_sheet(images: list[np.ndarray], out_path: Path, *, cols: int = 4, thumb_width: int = 360) -> None:
    if not images:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    thumbs = []
    for image in images:
        scale = thumb_width / image.shape[1]
        thumb = cv2.resize(image, (thumb_width, max(1, int(round(image.shape[0] * scale)))), interpolation=cv2.INTER_AREA)
        thumbs.append(thumb)
    cell_h = max(thumb.shape[0] for thumb in thumbs)
    rows = math.ceil(len(thumbs) / cols)
    sheet = np.full((rows * cell_h, cols * thumb_width, 3), 245, dtype=np.uint8)
    for idx, thumb in enumerate(thumbs):
        y = (idx // cols) * cell_h
        x = (idx % cols) * thumb_width
        sheet[y : y + thumb.shape[0], x : x + thumb.shape[1]] = thumb
    cv2.imwrite(str(out_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


def draw_3d_hand(ax: Any, points: np.ndarray, title: str) -> None:
    for i, j in HAND_EDGES:
        segment = points[[i, j]]
        ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color="#2563eb", linewidth=1.8)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color="#dc2626", s=12)
    center = points.mean(axis=0)
    radius = max(float(np.ptp(points, axis=0).max()) / 2.0, 0.05)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=18, azim=-65)


def save_3d_sheet(samples: list[dict[str, Any]], out_path: Path, *, cols: int = 4) -> None:
    if not samples:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = math.ceil(len(samples) / cols)
    fig = plt.figure(figsize=(cols * 3.0, rows * 3.0), dpi=140)
    for idx, sample in enumerate(samples):
        ax = fig.add_subplot(rows, cols, idx + 1, projection="3d")
        title = (
            f"{sample['split']} f{sample['frame_idx']} {sample['side']}\n"
            f"{'|'.join(sample['flags'])[:46]}"
        )
        draw_3d_hand(ax, sample["points"], title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_summary_plot(samples: list[dict[str, Any]], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    train = [sample for sample in samples if sample["split"] == "train"]
    test = [sample for sample in samples if sample["split"] == "test"]
    flags = Counter(flag for sample in samples for flag in sample["flags"])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=140)
    axes[0, 0].hist([sample["hand_span_m"] * 1000 for sample in train], bins=45, alpha=0.65, label="train")
    axes[0, 0].hist([sample["hand_span_m"] * 1000 for sample in test], bins=45, alpha=0.65, label="test")
    axes[0, 0].set_title("Hand span")
    axes[0, 0].set_xlabel("mm")
    axes[0, 0].legend()

    axes[0, 1].hist([sample["total_bone_length_m"] * 1000 for sample in train], bins=45, alpha=0.65, label="train")
    axes[0, 1].hist([sample["total_bone_length_m"] * 1000 for sample in test], bins=45, alpha=0.65, label="test")
    axes[0, 1].set_title("Total MANO-topology bone length")
    axes[0, 1].set_xlabel("mm")
    axes[0, 1].legend()

    axes[1, 0].hist([sample["episode_bone_rms_rel"] for sample in train], bins=45, alpha=0.65, label="train")
    axes[1, 0].hist([sample["episode_bone_rms_rel"] for sample in test], bins=45, alpha=0.65, label="test")
    axes[1, 0].set_title("Within-episode bone-length variation")
    axes[1, 0].set_xlabel("relative RMS")
    axes[1, 0].legend()

    labels, values = zip(*flags.most_common()) if flags else ([], [])
    axes[1, 1].bar(labels, values)
    axes[1, 1].set_title("Geometry flags")
    axes[1, 1].tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_diagnostics(samples: list[dict[str, Any]], out_dir: Path, args: argparse.Namespace) -> None:
    if args.viz_count <= 0:
        return
    suspicious = [sample for sample in samples if sample["is_suspicious"]]
    worst = sorted(
        suspicious,
        key=lambda sample: (
            sample["flag_count"],
            sample["max_abs_edge_robust_z"],
            -sample["min_finger_angle_deg"],
            sample["episode_bone_rms_rel"],
        ),
        reverse=True,
    )[: args.viz_count]
    worst_test = [sample for sample in worst if sample["split"] == "test"]
    if len(worst_test) < args.viz_count:
        worst_test = sorted(
            [sample for sample in suspicious if sample["split"] == "test"],
            key=lambda sample: (
                sample["flag_count"],
                sample["max_abs_edge_robust_z"],
                -sample["min_finger_angle_deg"],
                sample["episode_bone_rms_rel"],
            ),
            reverse=True,
        )[: args.viz_count]

    groups: dict[str, zarr.Group] = {}
    save_summary_plot(samples, out_dir / "hand_geometry_summary.png")
    save_contact_sheet([render_overlay_sample(sample, groups) for sample in worst], out_dir / "worst_geometry_overlays.png")
    save_3d_sheet(worst, out_dir / "worst_geometry_3d.png")
    save_contact_sheet([render_overlay_sample(sample, groups) for sample in worst_test], out_dir / "worst_test_geometry_overlays.png")
    save_3d_sheet(worst_test, out_dir / "worst_test_geometry_3d.png")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = collect_samples(
        {"train": Path(args.train_csv), "test": Path(args.test_csv)},
        cache_root=args.cache_root,
    )
    if not samples:
        raise SystemExit("No valid hand samples found.")
    robust_summary = add_dataset_robust_flags(samples, args)
    add_episode_bone_consistency(samples, args)
    add_temporal_metrics(samples, args)
    flag_samples(samples, args)
    rows = [sample_to_row(sample) for sample in samples]
    write_csv(out_dir / "hand_geometry_samples.csv", rows)
    summary = build_summary(samples, robust_summary, args)
    (out_dir / "hand_geometry_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    save_diagnostics(samples, out_dir, args)
    print(json.dumps(
        {
            "total_hands": summary["total_hands"],
            "suspicious_hands": summary["suspicious_hands"],
            "suspicious_rate": summary["suspicious_rate"],
            "hard_geometry_failure_hands": summary["hard_geometry_failure_hands"],
            "hard_geometry_failure_rate": summary["hard_geometry_failure_rate"],
            "soft_warning_hands": summary["soft_warning_hands"],
            "soft_warning_rate": summary["soft_warning_rate"],
            "flag_counts": summary["flag_counts"],
            "hard_flag_counts": summary["hard_flag_counts"],
            "soft_flag_counts": summary["soft_flag_counts"],
            "split_summary": summary["split_summary"],
        },
        indent=2,
    ))
    print(f"Wrote geometry audit to {out_dir}")


if __name__ == "__main__":
    main()
