#!/usr/bin/env python3
"""Filter EgoVerse hand-pose manifests by projected hand visibility."""

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

import numpy as np
import zarr

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from egoverse_handpose_dataset import valid_joint_mask, world_to_camera  # noqa: E402


FALLBACK_WIDTH = 640
FALLBACK_HEIGHT = 480
FALLBACK_FOCAL = 266.50860444


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create train/test manifests that only supervise hands with enough "
            "3D joints projected into the RGB image."
        )
    )
    parser.add_argument("--train-csv", default="outputs/handpose_dataset/train.csv")
    parser.add_argument("--test-csv", default="outputs/handpose_dataset/test.csv")
    parser.add_argument("--out-dir", default="outputs/handpose_dataset_visible")
    parser.add_argument("--cache-root", default=os.environ.get("EGOVERSE_CACHE_DIR"))
    parser.add_argument("--min-visible-ratio", type=float, default=0.5)
    parser.add_argument("--min-visible-joints", type=int, default=1)
    parser.add_argument("--min-depth-m", type=float, default=0.01)
    return parser.parse_args()


def str_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def resolve_episode_path(row: dict[str, str], cache_root: str | None) -> Path:
    if cache_root:
        return Path(cache_root).expanduser() / row["episode_hash"]
    return Path(row["episode_path"]).expanduser()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def camera_intrinsics(group: zarr.Group, image_key: str) -> tuple[np.ndarray, int, int]:
    intrinsics = group.attrs.get("intrinsics")
    if isinstance(intrinsics, dict):
        value = intrinsics.get("front_1") or intrinsics.get(image_key) or intrinsics.get(image_key.split(".")[-1])
        if value is not None:
            matrix = np.asarray(value, dtype=np.float64)
            if matrix.shape[0] >= 3 and matrix.shape[1] >= 3:
                width, height = image_size(group, image_key)
                return matrix[:3, :3], width, height

    camera_intrinsics_attr = group.attrs.get("camera_intrinsics")
    if isinstance(camera_intrinsics_attr, dict):
        try:
            width = int(camera_intrinsics_attr.get("width", FALLBACK_WIDTH))
            height = int(camera_intrinsics_attr.get("height", FALLBACK_HEIGHT))
            return (
                np.array(
                    [
                        [float(camera_intrinsics_attr["fx"]), 0.0, float(camera_intrinsics_attr["cx"])],
                        [0.0, float(camera_intrinsics_attr["fy"]), float(camera_intrinsics_attr["cy"])],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float64,
                ),
                width,
                height,
            )
        except (KeyError, TypeError, ValueError):
            pass

    width, height = image_size(group, image_key)
    return (
        np.array(
            [
                [FALLBACK_FOCAL, 0.0, width / 2.0],
                [0.0, FALLBACK_FOCAL, height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        width,
        height,
    )


def image_size(group: zarr.Group, image_key: str) -> tuple[int, int]:
    features = group.attrs.get("features")
    if isinstance(features, dict):
        feature = features.get(image_key)
        if isinstance(feature, dict):
            shape = feature.get("shape")
            if isinstance(shape, list) and len(shape) >= 2:
                return int(shape[1]), int(shape[0])
    return FALLBACK_WIDTH, FALLBACK_HEIGHT


def projected_hand_stats(
    group: zarr.Group,
    row: dict[str, str],
    *,
    hand_key: str,
    frame_idx: int,
    head_pose: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
    min_depth_m: float,
) -> dict[str, float | int]:
    points_world = np.asarray(group[hand_key][frame_idx], dtype=np.float64).reshape(21, 3)
    valid = valid_joint_mask(points_world)
    points_cam = np.zeros((21, 3), dtype=np.float64)
    if valid.any():
        points_cam[valid] = world_to_camera(points_world[valid], head_pose)

    z = np.clip(points_cam[:, 2], 1e-9, None)
    px = intrinsics[0, 0] * points_cam[:, 0] / z + intrinsics[0, 2]
    py = intrinsics[1, 1] * points_cam[:, 1] / z + intrinsics[1, 2]
    visible = (
        valid
        & (points_cam[:, 2] > min_depth_m)
        & (px >= 0.0)
        & (px < width)
        & (py >= 0.0)
        & (py < height)
    )
    valid_joints = int(valid.sum())
    visible_joints = int(visible.sum())
    visible_ratio = visible_joints / valid_joints if valid_joints else 0.0
    if visible_joints >= 2:
        bbox_area = float((px[visible].max() - px[visible].min()) * (py[visible].max() - py[visible].min()))
    else:
        bbox_area = 0.0
    return {
        "valid_joints": valid_joints,
        "visible_joints": visible_joints,
        "visible_ratio": visible_ratio,
        "bbox_area_px2": bbox_area,
    }


def keep_hand(stats: dict[str, float | int], *, min_visible_ratio: float, min_visible_joints: int) -> bool:
    valid_joints = int(stats["valid_joints"])
    visible_joints = int(stats["visible_joints"])
    if valid_joints <= 0:
        return False
    required_by_ratio = math.ceil(valid_joints * min_visible_ratio)
    return visible_joints >= max(min_visible_joints, required_by_ratio)


def filter_split(
    rows: list[dict[str, str]],
    *,
    split_name: str,
    cache_root: str | None,
    min_visible_ratio: float,
    min_visible_joints: int,
    min_depth_m: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    groups: dict[str, zarr.Group] = {}
    output_rows: list[dict[str, Any]] = []
    hand_audit: list[dict[str, Any]] = []
    per_episode: dict[str, Counter] = defaultdict(Counter)
    totals = Counter()

    for row_idx, row in enumerate(rows):
        episode_hash = row["episode_hash"]
        episode_path = resolve_episode_path(row, cache_root)
        if episode_hash not in groups:
            groups[episode_hash] = zarr.open_group(str(episode_path), mode="r")
        group = groups[episode_hash]
        frame_idx = int(row["frame_idx"])
        head_pose = np.asarray(group[row["head_pose_key"]][frame_idx], dtype=np.float64)
        intrinsics, width, height = camera_intrinsics(group, row["image_key"])
        filtered = dict(row)
        row_kept_hands = 0
        row_labeled_hands = 0
        row_zero_visible_hands = 0

        for side, has_key, key_col in (
            ("left", "has_left", "left_keypoints_key"),
            ("right", "has_right", "right_keypoints_key"),
        ):
            stats = {
                "valid_joints": 0,
                "visible_joints": 0,
                "visible_ratio": 0.0,
                "bbox_area_px2": 0.0,
            }
            labeled = str_bool(row.get(has_key)) and bool(row.get(key_col))
            kept = False
            if labeled:
                row_labeled_hands += 1
                stats = projected_hand_stats(
                    group,
                    row,
                    hand_key=row[key_col],
                    frame_idx=frame_idx,
                    head_pose=head_pose,
                    intrinsics=intrinsics,
                    width=width,
                    height=height,
                    min_depth_m=min_depth_m,
                )
                kept = keep_hand(
                    stats,
                    min_visible_ratio=min_visible_ratio,
                    min_visible_joints=min_visible_joints,
                )
                if int(stats["visible_joints"]) == 0:
                    row_zero_visible_hands += 1
                totals["input_labeled_hands"] += 1
                totals["kept_hands" if kept else "dropped_hands"] += 1
                totals["zero_visible_labeled_hands"] += int(int(stats["visible_joints"]) == 0)
                per_episode[episode_hash]["input_labeled_hands"] += 1
                per_episode[episode_hash]["kept_hands" if kept else "dropped_hands"] += 1
                per_episode[episode_hash]["zero_visible_labeled_hands"] += int(int(stats["visible_joints"]) == 0)

            filtered[has_key] = "True" if kept else "False"
            filtered[f"{side}_visible_joints"] = int(stats["visible_joints"])
            filtered[f"{side}_visible_ratio"] = f"{float(stats['visible_ratio']):.6f}"
            filtered[f"{side}_visible_bbox_area_px2"] = f"{float(stats['bbox_area_px2']):.3f}"
            filtered[f"{side}_visibility_keep"] = "True" if kept else "False"
            row_kept_hands += int(kept)

            hand_audit.append(
                {
                    "split": split_name,
                    "row_idx": row_idx,
                    "episode_hash": episode_hash,
                    "frame_idx": frame_idx,
                    "side": side,
                    "labeled": labeled,
                    "kept": kept,
                    **stats,
                }
            )

        totals["input_rows"] += 1
        per_episode[episode_hash]["input_rows"] += 1
        if row_zero_visible_hands > 0:
            totals["rows_with_zero_visible_hand"] += 1
            per_episode[episode_hash]["rows_with_zero_visible_hand"] += 1
        if row_kept_hands > 0:
            output_rows.append(filtered)
            totals["kept_rows"] += 1
            per_episode[episode_hash]["kept_rows"] += 1
        else:
            totals["dropped_rows"] += 1
            per_episode[episode_hash]["dropped_rows"] += 1
        if row_labeled_hands:
            per_episode[episode_hash]["input_rows_with_labels"] += 1

    episode_summary = []
    for episode_hash, counts in sorted(per_episode.items()):
        input_rows = max(1, counts["input_rows"])
        input_hands = max(1, counts["input_labeled_hands"])
        episode_summary.append(
            {
                "episode_hash": episode_hash,
                **dict(counts),
                "dropped_row_rate": counts["dropped_rows"] / input_rows,
                "dropped_hand_rate": counts["dropped_hands"] / input_hands,
                "zero_visible_row_rate": counts["rows_with_zero_visible_hand"] / input_rows,
                "zero_visible_hand_rate": counts["zero_visible_labeled_hands"] / input_hands,
            }
        )

    summary = {
        **dict(totals),
        "episodes": len(per_episode),
        "kept_row_rate": totals["kept_rows"] / max(1, totals["input_rows"]),
        "dropped_row_rate": totals["dropped_rows"] / max(1, totals["input_rows"]),
        "kept_hand_rate": totals["kept_hands"] / max(1, totals["input_labeled_hands"]),
        "dropped_hand_rate": totals["dropped_hands"] / max(1, totals["input_labeled_hands"]),
        "zero_visible_row_rate": totals["rows_with_zero_visible_hand"] / max(1, totals["input_rows"]),
        "zero_visible_hand_rate": totals["zero_visible_labeled_hands"] / max(1, totals["input_labeled_hands"]),
        "episode_summary": sorted(episode_summary, key=lambda row: row["dropped_hand_rate"], reverse=True),
    }
    return output_rows, summary, hand_audit


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.min_visible_ratio <= 1.0:
        raise SystemExit("--min-visible-ratio must be in [0, 1]")
    if args.min_visible_joints < 1:
        raise SystemExit("--min-visible-joints must be >= 1")
    if args.min_depth_m < 0:
        raise SystemExit("--min-depth-m must be >= 0")

    train_rows = load_rows(Path(args.train_csv))
    test_rows = load_rows(Path(args.test_csv))
    if not train_rows or not test_rows:
        raise SystemExit("Train and test CSVs must both be non-empty.")

    train_filtered, train_summary, train_audit = filter_split(
        train_rows,
        split_name="train",
        cache_root=args.cache_root,
        min_visible_ratio=args.min_visible_ratio,
        min_visible_joints=args.min_visible_joints,
        min_depth_m=args.min_depth_m,
    )
    test_filtered, test_summary, test_audit = filter_split(
        test_rows,
        split_name="test",
        cache_root=args.cache_root,
        min_visible_ratio=args.min_visible_ratio,
        min_visible_joints=args.min_visible_joints,
        min_depth_m=args.min_depth_m,
    )
    if not train_filtered or not test_filtered:
        raise SystemExit("Visibility filtering produced an empty train or test split.")

    out_dir = Path(args.out_dir)
    new_fields = [
        "left_visible_joints",
        "left_visible_ratio",
        "left_visible_bbox_area_px2",
        "left_visibility_keep",
        "right_visible_joints",
        "right_visible_ratio",
        "right_visible_bbox_area_px2",
        "right_visibility_keep",
    ]
    fieldnames = list(train_rows[0].keys()) + [field for field in new_fields if field not in train_rows[0]]
    write_csv(out_dir / "train.csv", train_filtered, fieldnames)
    write_csv(out_dir / "test.csv", test_filtered, fieldnames)
    write_csv(out_dir / "manifest.csv", train_filtered + test_filtered, fieldnames)

    audit_fields = [
        "split",
        "row_idx",
        "episode_hash",
        "frame_idx",
        "side",
        "labeled",
        "kept",
        "valid_joints",
        "visible_joints",
        "visible_ratio",
        "bbox_area_px2",
    ]
    write_csv(out_dir / "hand_visibility_audit.csv", train_audit + test_audit, audit_fields)

    summary = {
        "args": vars(args),
        "train": train_summary,
        "test": test_summary,
    }
    (out_dir / "visibility_filter_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("train", "test")}, indent=2))
    print(f"Wrote visibility-filtered manifests to {out_dir}")


if __name__ == "__main__":
    main()
