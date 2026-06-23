#!/usr/bin/env python3
"""Build frame-level train/test manifests for EgoVerse hand-pose training."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import zarr


LEFT_KEY = "left.obs_keypoints"
RIGHT_KEY = "right.obs_keypoints"
IMAGE_KEY = "images.front_1"
HEAD_POSE_KEY = "obs_head_pose"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Aria frame-level hand-pose manifests from audit output."
    )
    parser.add_argument("--audit-csv", default="outputs/dataset_audit/episodes.csv")
    parser.add_argument("--out-dir", default="outputs/handpose_dataset")
    parser.add_argument("--source", default="aria/human")
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-valid-joints", type=int, default=15)
    parser.add_argument("--max-episodes", type=int, default=None)
    return parser.parse_args()


def str_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_audit_rows(path: Path, source: str) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    selected = []
    for row in rows:
        if row.get("source") != source:
            continue
        if not str_bool(row.get("has_front_image")):
            continue
        if not str_bool(row.get("has_head_pose")):
            continue
        if not (str_bool(row.get("has_left_keypoints")) or str_bool(row.get("has_right_keypoints"))):
            continue
        selected.append(row)
    return selected


def valid_joint_mask(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(21, 3)
    finite = np.isfinite(points).all(axis=1)
    nonzero = np.linalg.norm(np.nan_to_num(points), axis=1) > 1e-9
    return finite & nonzero


def valid_joint_counts(group: zarr.Group, key: str, frame_indices: list[int]) -> dict[int, int]:
    if not frame_indices:
        return {}
    arr = group[key]
    usable = [idx for idx in frame_indices if idx < arr.shape[0]]
    if not usable:
        return {}
    values = np.asarray(arr.get_orthogonal_selection((usable, slice(None))), dtype=np.float64)
    values = values.reshape(len(usable), 21, 3)
    finite = np.isfinite(values).all(axis=2)
    nonzero = np.linalg.norm(np.nan_to_num(values), axis=2) > 1e-9
    counts = (finite & nonzero).sum(axis=1)
    return {idx: int(count) for idx, count in zip(usable, counts)}


def has_nonempty_image(group: zarr.Group, frame_idx: int) -> bool:
    value = group[IMAGE_KEY][frame_idx]
    while isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    return bool(value)


def build_rows(
    audit_rows: list[dict[str, str]],
    *,
    frame_stride: int,
    min_valid_joints: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for audit_row in audit_rows:
        episode_path = Path(audit_row["path"])
        group = zarr.open_group(str(episode_path), mode="r")
        keys = set(group.keys())
        image_frames = int(group[IMAGE_KEY].shape[0])
        head_frames = int(group[HEAD_POSE_KEY].shape[0])
        total_frames = min(image_frames, head_frames)
        frame_indices = list(range(0, total_frames, frame_stride))
        left_counts = valid_joint_counts(group, LEFT_KEY, frame_indices) if LEFT_KEY in keys else {}
        right_counts = valid_joint_counts(group, RIGHT_KEY, frame_indices) if RIGHT_KEY in keys else {}

        for frame_idx in frame_indices:
            if not has_nonempty_image(group, frame_idx):
                continue
            left_valid_joints = left_counts.get(frame_idx, 0)
            right_valid_joints = right_counts.get(frame_idx, 0)
            has_left = left_valid_joints >= min_valid_joints
            has_right = right_valid_joints >= min_valid_joints
            if not (has_left or has_right):
                continue
            rows.append(
                {
                    "episode_hash": audit_row["episode_hash"],
                    "episode_path": str(episode_path),
                    "source": audit_row["source"],
                    "split": "",
                    "frame_idx": frame_idx,
                    "frame_stride": frame_stride,
                    "image_key": IMAGE_KEY,
                    "head_pose_key": HEAD_POSE_KEY,
                    "left_keypoints_key": LEFT_KEY if LEFT_KEY in keys else "",
                    "right_keypoints_key": RIGHT_KEY if RIGHT_KEY in keys else "",
                    "has_left": has_left,
                    "has_right": has_right,
                    "left_valid_joints": left_valid_joints,
                    "right_valid_joints": right_valid_joints,
                    "min_valid_joints": min_valid_joints,
                    "image_height": audit_row.get("image_height", ""),
                    "image_width": audit_row.get("image_width", ""),
                }
            )
    return rows


def assign_episode_splits(
    rows: list[dict[str, Any]], *, test_fraction: float, seed: int
) -> tuple[list[dict[str, Any]], set[str], set[str]]:
    episodes = sorted({str(row["episode_hash"]) for row in rows})
    rng = random.Random(seed)
    rng.shuffle(episodes)
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("--test-fraction must be between 0 and 1")
    n_test = max(1, int(round(len(episodes) * test_fraction)))
    n_test = min(n_test, len(episodes) - 1)
    test_episodes = set(episodes[:n_test])
    train_episodes = set(episodes[n_test:])
    for row in rows:
        row["split"] = "test" if row["episode_hash"] in test_episodes else "train"
    return rows, train_episodes, test_episodes


def summarize(rows: list[dict[str, Any]], train_episodes: set[str], test_episodes: set[str]) -> dict[str, Any]:
    by_split = defaultdict(list)
    for row in rows:
        by_split[row["split"]].append(row)

    def split_summary(split_rows: list[dict[str, Any]]) -> dict[str, Any]:
        episodes = {row["episode_hash"] for row in split_rows}
        hand_counts = Counter(
            "both" if row["has_left"] and row["has_right"] else "left_only" if row["has_left"] else "right_only"
            for row in split_rows
        )
        return {
            "episodes": len(episodes),
            "frames": len(split_rows),
            "left_frames": sum(1 for row in split_rows if row["has_left"]),
            "right_frames": sum(1 for row in split_rows if row["has_right"]),
            "both_hand_frames": hand_counts.get("both", 0),
            "left_only_frames": hand_counts.get("left_only", 0),
            "right_only_frames": hand_counts.get("right_only", 0),
        }

    return {
        "total": split_summary(rows),
        "train": split_summary(by_split["train"]),
        "test": split_summary(by_split["test"]),
        "train_episodes": sorted(train_episodes),
        "test_episodes": sorted(test_episodes),
        "episode_overlap": sorted(train_episodes & test_episodes),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.frame_stride <= 0:
        raise SystemExit("--frame-stride must be positive")
    if not 1 <= args.min_valid_joints <= 21:
        raise SystemExit("--min-valid-joints must be in [1, 21]")

    audit_rows = load_audit_rows(Path(args.audit_csv), args.source)
    if args.max_episodes is not None:
        audit_rows = audit_rows[: args.max_episodes]
    if not audit_rows:
        raise SystemExit("No matching trainable episodes found in audit CSV.")

    rows = build_rows(
        audit_rows,
        frame_stride=args.frame_stride,
        min_valid_joints=args.min_valid_joints,
    )
    if not rows:
        raise SystemExit("No valid frame rows found after filtering.")
    rows, train_episodes, test_episodes = assign_episode_splits(
        rows, test_fraction=args.test_fraction, seed=args.seed
    )
    summary = summarize(rows, train_episodes, test_episodes)
    if summary["episode_overlap"]:
        raise SystemExit(f"Train/test episode overlap: {summary['episode_overlap']}")

    out_dir = Path(args.out_dir)
    write_csv(out_dir / "manifest.csv", rows)
    write_csv(out_dir / "train.csv", [row for row in rows if row["split"] == "train"])
    write_csv(out_dir / "test.csv", [row for row in rows if row["split"] == "test"])
    (out_dir / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Wrote {len(rows)} frame rows to {out_dir}")
    print(json.dumps({k: summary[k] for k in ("total", "train", "test")}, indent=2))


if __name__ == "__main__":
    main()
