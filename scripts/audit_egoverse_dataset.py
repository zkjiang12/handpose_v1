#!/usr/bin/env python3
"""Audit locally cached EgoVerse Zarr episodes.

This script is intentionally read-only. It scans cached episode directories,
summarizes the fields that physically exist in each Zarr group, and writes
per-episode plus per-source reports that can be used before training.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import zarr

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None

try:
    import simplejpeg
except ImportError:  # pragma: no cover - optional dependency
    simplejpeg = None

try:
    from scipy.spatial.transform import Rotation as R
except ImportError:  # pragma: no cover - optional dependency
    R = None


HAND_KEYS = (
    "left.obs_keypoints",
    "right.obs_keypoints",
    "left.obs_aria_keypoints",
    "right.obs_aria_keypoints",
)
REQUIRED_TRAIN_KEYS = ("images.front_1", "obs_head_pose")
INTERESTING_KEYS = (
    "images.front_1",
    "images.left_wrist",
    "images.right_wrist",
    "annotations",
    "obs_head_pose",
    "obs_eye_gaze",
    "obs_rgb_timestamps_ns",
    "left.obs_keypoints",
    "right.obs_keypoints",
    "left.obs_aria_keypoints",
    "right.obs_aria_keypoints",
    "left.obs_wrist_pose",
    "right.obs_wrist_pose",
    "left.obs_ee_pose",
    "right.obs_ee_pose",
    "left.obs_gripper",
    "right.obs_gripper",
    "left.obs_joints",
    "right.obs_joints",
    "left.cmd_ee_pose",
    "right.cmd_ee_pose",
    "left.cmd_gripper",
    "right.cmd_gripper",
    "left.cmd_joints",
    "right.cmd_joints",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit local EgoVerse Zarr caches and write schema reports."
    )
    parser.add_argument(
        "--cache-dir",
        action="append",
        default=[],
        help=(
            "Directory containing cached episode folders. May be passed multiple "
            "times. Defaults to common local EgoVerse cache directories."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "dataset_audit"),
        help="Directory where CSV/JSON reports will be written.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional cap for quick debugging.",
    )
    parser.add_argument(
        "--sample-frames",
        type=int,
        default=200,
        help="Max frames per episode to sample for keypoint validity/range stats.",
    )
    parser.add_argument(
        "--skip-image-decode",
        action="store_true",
        help="Skip JPEG decoding used to infer image height/width.",
    )
    return parser.parse_args()


def default_cache_dirs() -> list[Path]:
    return [
        Path("/Users/zikangjiang/data/egoverse_viewer_cache"),
        Path("/Users/zikangjiang/data/egoverse_keypoint_cache"),
    ]


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def discover_episode_dirs(cache_dirs: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    episodes: list[Path] = []
    for cache_dir in cache_dirs:
        if not cache_dir.exists():
            continue
        for zarr_json in sorted(cache_dir.glob("*/zarr.json")):
            episode_dir = zarr_json.parent.resolve()
            if episode_dir in seen:
                continue
            seen.add(episode_dir)
            episodes.append(episode_dir)
    return episodes


def source_for(attrs: dict[str, Any], keys: set[str]) -> str:
    embodiment = str(attrs.get("embodiment", "")).lower()
    if "mecka" in embodiment:
        return "mecka"
    if embodiment.startswith("scale") or embodiment == "scale":
        return "scale"
    if "eva" in embodiment:
        return "eva"
    if embodiment.startswith("human") or any("obs_aria_keypoints" in k for k in keys):
        return "aria/human"
    return embodiment or "unknown"


def arr_shape_dtype(group: zarr.Group, key: str) -> tuple[str, str, int]:
    arr = group[key]
    shape = tuple(int(v) for v in arr.shape)
    first_dim = int(shape[0]) if shape else 0
    return json.dumps(shape), str(arr.dtype), first_dim


def decode_image_shape(group: zarr.Group, max_index: int) -> tuple[int | None, int | None]:
    if simplejpeg is None and cv2 is None:
        return None, None
    candidates = [0, 1, 5, 10, 30, min(max_index - 1, 100)]
    for idx in candidates:
        if idx < 0 or idx >= max_index:
            continue
        value = group["images.front_1"][idx]
        while isinstance(value, np.ndarray) and value.shape == ():
            value = value.item()
        if not value:
            continue
        if simplejpeg is not None:
            try:
                image = simplejpeg.decode_jpeg(value, colorspace="RGB")
                return int(image.shape[0]), int(image.shape[1])
            except Exception:
                pass
        if cv2 is not None:
            try:
                encoded = np.frombuffer(value, dtype=np.uint8)
                image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                if image is not None:
                    return int(image.shape[0]), int(image.shape[1])
            except Exception:
                pass
    return None, None


def keypoint_stats(group: zarr.Group, key: str, sample_frames: int) -> dict[str, float | int | None]:
    if key not in group:
        return {
            f"{key}.valid_fraction": None,
            f"{key}.zero_fraction": None,
            f"{key}.finite_fraction": None,
            f"{key}.coord_min": None,
            f"{key}.coord_max": None,
            f"{key}.norm_median": None,
        }
    arr = group[key]
    n = min(int(arr.shape[0]), sample_frames)
    if n <= 0:
        return {
            f"{key}.valid_fraction": 0.0,
            f"{key}.zero_fraction": None,
            f"{key}.finite_fraction": None,
            f"{key}.coord_min": None,
            f"{key}.coord_max": None,
            f"{key}.norm_median": None,
        }
    values = np.asarray(arr[:n], dtype=np.float64).reshape(n, 21, 3)
    finite = np.isfinite(values).all(axis=2)
    zero = np.linalg.norm(np.nan_to_num(values), axis=2) <= 1e-9
    valid = finite & ~zero
    valid_values = values[valid]
    if valid_values.size == 0:
        coord_min = coord_max = norm_median = None
    else:
        coord_min = float(np.nanmin(valid_values))
        coord_max = float(np.nanmax(valid_values))
        norm_median = float(np.nanmedian(np.linalg.norm(valid_values, axis=1)))
    return {
        f"{key}.valid_fraction": float(valid.mean()),
        f"{key}.zero_fraction": float(zero.mean()),
        f"{key}.finite_fraction": float(finite.mean()),
        f"{key}.coord_min": coord_min,
        f"{key}.coord_max": coord_max,
        f"{key}.norm_median": norm_median,
    }


def pose_world_to_local(points_world: np.ndarray, pose_world: np.ndarray) -> np.ndarray:
    if R is None:
        raise RuntimeError("scipy is required for camera-frame keypoint stats")
    xyz = pose_world[:3]
    quat_wxyz = pose_world[3:7]
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    rot_world_from_local = R.from_quat(quat_xyzw)
    return rot_world_from_local.inv().apply(points_world - xyz)


def camera_frame_norm_median(
    group: zarr.Group, key: str, sample_frames: int
) -> float | None:
    if R is None or key not in group or "obs_head_pose" not in group:
        return None
    n = min(int(group[key].shape[0]), int(group["obs_head_pose"].shape[0]), sample_frames)
    if n <= 0:
        return None
    keypoints = np.asarray(group[key][:n], dtype=np.float64).reshape(n, 21, 3)
    head_pose = np.asarray(group["obs_head_pose"][:n], dtype=np.float64)
    local_points: list[np.ndarray] = []
    for i in range(n):
        points = keypoints[i]
        valid = np.isfinite(points).all(axis=1) & (np.linalg.norm(points, axis=1) > 1e-9)
        if valid.any():
            local_points.append(pose_world_to_local(points[valid], head_pose[i]))
    if not local_points:
        return None
    local = np.concatenate(local_points, axis=0)
    return float(np.nanmedian(np.linalg.norm(local, axis=1)))


def warning_list(
    keys: set[str],
    frame_lengths: dict[str, int],
    image_height: int | None,
    image_width: int | None,
    kp_stats: dict[str, Any],
) -> list[str]:
    warnings_out: list[str] = []
    for key in REQUIRED_TRAIN_KEYS:
        if key not in keys:
            warnings_out.append(f"missing {key}")
    if "left.obs_keypoints" not in keys and "right.obs_keypoints" not in keys:
        warnings_out.append("missing canonical hand keypoints")
    if "images.front_1" in frame_lengths:
        image_len = frame_lengths["images.front_1"]
        for key in ("left.obs_keypoints", "right.obs_keypoints", "obs_head_pose"):
            if key in frame_lengths and frame_lengths[key] != image_len:
                warnings_out.append(f"{key} length {frame_lengths[key]} != image length {image_len}")
    for key in ("left.obs_keypoints", "right.obs_keypoints"):
        valid_fraction = kp_stats.get(f"{key}.valid_fraction")
        if valid_fraction is not None and valid_fraction < 0.5:
            warnings_out.append(f"{key} low valid fraction {valid_fraction:.3f}")
    if image_height is not None and image_width is not None:
        if image_width != 640:
            warnings_out.append(f"unexpected image width {image_width}")
        if image_height not in (360, 480):
            warnings_out.append(f"unexpected image height {image_height}")
    return warnings_out


def audit_episode(path: Path, sample_frames: int, decode_images: bool) -> dict[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        group = zarr.open_group(str(path), mode="r")
        keys = set(group.keys())
    attrs = dict(group.attrs)
    source = source_for(attrs, keys)
    frame_lengths: dict[str, int] = {}
    shape_dtype: dict[str, str] = {}
    dtype_by_key: dict[str, str] = {}

    for key in sorted(keys):
        shape, dtype, first_dim = arr_shape_dtype(group, key)
        frame_lengths[key] = first_dim
        shape_dtype[key] = shape
        dtype_by_key[key] = dtype

    image_frames = frame_lengths.get("images.front_1")
    image_height = image_width = None
    if decode_images and image_frames:
        image_height, image_width = decode_image_shape(group, image_frames)

    kp_stats: dict[str, Any] = {}
    for key in HAND_KEYS:
        kp_stats.update(keypoint_stats(group, key, sample_frames))

    left_cam_norm = camera_frame_norm_median(group, "left.obs_keypoints", sample_frames)
    right_cam_norm = camera_frame_norm_median(group, "right.obs_keypoints", sample_frames)

    warnings_out = warning_list(keys, frame_lengths, image_height, image_width, kp_stats)
    trainable_handpose = (
        "images.front_1" in keys
        and "obs_head_pose" in keys
        and ("left.obs_keypoints" in keys or "right.obs_keypoints" in keys)
    )

    row: dict[str, Any] = {
        "episode_hash": path.name,
        "path": str(path),
        "source": source,
        "embodiment": attrs.get("embodiment", ""),
        "task": attrs.get("task", ""),
        "lab": attrs.get("lab", ""),
        "scene": attrs.get("scene", ""),
        "operator": attrs.get("operator", ""),
        "fps": attrs.get("fps", ""),
        "total_frames_attr": attrs.get("total_frames", ""),
        "image_frames": image_frames,
        "image_height": image_height,
        "image_width": image_width,
        "annotation_count": frame_lengths.get("annotations"),
        "field_count": len(keys),
        "fields": "|".join(sorted(keys)),
        "trainable_handpose": trainable_handpose,
        "has_front_image": "images.front_1" in keys,
        "has_head_pose": "obs_head_pose" in keys,
        "has_left_keypoints": "left.obs_keypoints" in keys,
        "has_right_keypoints": "right.obs_keypoints" in keys,
        "has_left_aria_keypoints": "left.obs_aria_keypoints" in keys,
        "has_right_aria_keypoints": "right.obs_aria_keypoints" in keys,
        "has_camera_intrinsics_attr": isinstance(attrs.get("camera_intrinsics"), dict),
        "left.obs_keypoints.camera_norm_median": left_cam_norm,
        "right.obs_keypoints.camera_norm_median": right_cam_norm,
        "warnings": "|".join(warnings_out),
    }
    row.update(kp_stats)
    for key in INTERESTING_KEYS:
        row[f"{key}.shape"] = shape_dtype.get(key)
        row[f"{key}.dtype"] = dtype_by_key.get(key)
        row[f"{key}.frames"] = frame_lengths.get(key)
    return json_safe(row)


def summarize_by_source(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["source"])].append(row)

    summaries: list[dict[str, Any]] = []
    for source, source_rows in sorted(grouped.items()):
        image_frames = [int(r["image_frames"] or 0) for r in source_rows]
        warning_counter: Counter[str] = Counter()
        image_shapes: Counter[str] = Counter()
        for row in source_rows:
            for warning in str(row.get("warnings") or "").split("|"):
                if warning:
                    warning_counter[warning] += 1
            if row.get("image_height") and row.get("image_width"):
                image_shapes[f"{row['image_height']}x{row['image_width']}"] += 1

        def count_true(key: str) -> int:
            return sum(1 for row in source_rows if bool(row.get(key)))

        summaries.append(
            {
                "source": source,
                "episodes": len(source_rows),
                "frames_total": sum(image_frames),
                "frames_min": min(image_frames) if image_frames else 0,
                "frames_median": int(np.median(image_frames)) if image_frames else 0,
                "frames_max": max(image_frames) if image_frames else 0,
                "trainable_handpose_episodes": count_true("trainable_handpose"),
                "has_front_image": count_true("has_front_image"),
                "has_head_pose": count_true("has_head_pose"),
                "has_left_keypoints": count_true("has_left_keypoints"),
                "has_right_keypoints": count_true("has_right_keypoints"),
                "has_left_aria_keypoints": count_true("has_left_aria_keypoints"),
                "has_right_aria_keypoints": count_true("has_right_aria_keypoints"),
                "has_camera_intrinsics_attr": count_true("has_camera_intrinsics_attr"),
                "image_shapes": json.dumps(dict(image_shapes), sort_keys=True),
                "top_warnings": json.dumps(dict(warning_counter.most_common(8))),
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(json_safe(value), indent=2, sort_keys=True) + "\n")


def print_summary(summaries: list[dict[str, Any]], out_dir: Path) -> None:
    print(f"Wrote reports to {out_dir}")
    print()
    print(
        "source, episodes, frames, trainable, left_kp, right_kp, head_pose, image_shapes"
    )
    for row in summaries:
        print(
            f"{row['source']}, {row['episodes']}, {row['frames_total']}, "
            f"{row['trainable_handpose_episodes']}, {row['has_left_keypoints']}, "
            f"{row['has_right_keypoints']}, {row['has_head_pose']}, {row['image_shapes']}"
        )


def main() -> None:
    args = parse_args()
    cache_dirs = [Path(p).expanduser().resolve() for p in args.cache_dir]
    if not cache_dirs:
        cache_dirs = default_cache_dirs()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    episode_dirs = discover_episode_dirs(cache_dirs)
    if args.max_episodes is not None:
        episode_dirs = episode_dirs[: args.max_episodes]
    if not episode_dirs:
        raise SystemExit(
            "No cached EgoVerse episodes found. Pass --cache-dir or download episodes first."
        )

    rows: list[dict[str, Any]] = []
    for i, episode_dir in enumerate(episode_dirs, start=1):
        try:
            rows.append(
                audit_episode(
                    episode_dir,
                    sample_frames=max(0, args.sample_frames),
                    decode_images=not args.skip_image_decode,
                )
            )
        except Exception as exc:
            rows.append(
                {
                    "episode_hash": episode_dir.name,
                    "path": str(episode_dir),
                    "source": "error",
                    "warnings": f"audit failed: {exc}",
                    "trainable_handpose": False,
                }
            )
        if i % 25 == 0:
            print(f"Audited {i}/{len(episode_dirs)} episodes...")

    summaries = summarize_by_source(rows)
    write_csv(out_dir / "episodes.csv", rows)
    write_csv(out_dir / "source_summary.csv", summaries)
    write_json(out_dir / "episodes.json", rows)
    write_json(out_dir / "source_summary.json", summaries)
    print_summary(summaries, out_dir)


if __name__ == "__main__":
    main()
