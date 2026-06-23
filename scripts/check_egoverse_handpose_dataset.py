#!/usr/bin/env python3
"""Sanity checks for EgoVerse hand-pose train/test manifests."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from egoverse_handpose_dataset import EgoVerseHandPoseDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check EgoVerse hand-pose manifests.")
    parser.add_argument("--train-csv", default="outputs/handpose_dataset/train.csv")
    parser.add_argument("--test-csv", default="outputs/handpose_dataset/test.csv")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-rows", type=int, default=64)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def summarize_rows(rows: list[dict[str, str]]) -> dict[str, int]:
    episodes = {row["episode_hash"] for row in rows}
    hands = Counter()
    for row in rows:
        left = row.get("has_left") == "True"
        right = row.get("has_right") == "True"
        if left and right:
            hands["both"] += 1
        elif left:
            hands["left_only"] += 1
        elif right:
            hands["right_only"] += 1
    return {
        "episodes": len(episodes),
        "frames": len(rows),
        "both_hand_frames": hands["both"],
        "left_only_frames": hands["left_only"],
        "right_only_frames": hands["right_only"],
    }


def check_no_aria_keypoints(rows: list[dict[str, str]]) -> None:
    for row in rows:
        values = " ".join(str(v) for v in row.values())
        if "obs_aria_keypoints" in values:
            raise AssertionError(f"Aria-native keypoint key leaked into manifest: {row}")


def check_loader(csv_path: Path, args: argparse.Namespace) -> dict[str, object]:
    dataset = EgoVerseHandPoseDataset(
        csv_path,
        image_size=args.image_size,
        max_rows=args.max_rows,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    batch = next(iter(loader))
    if tuple(batch["image"].shape[1:]) != (3, args.image_size, args.image_size):
        raise AssertionError(f"Unexpected image shape: {tuple(batch['image'].shape)}")
    if tuple(batch["keypoints"].shape[1:]) != (2, 21, 3):
        raise AssertionError(f"Unexpected keypoint shape: {tuple(batch['keypoints'].shape)}")
    if tuple(batch["valid_mask"].shape[1:]) != (2, 21):
        raise AssertionError(f"Unexpected mask shape: {tuple(batch['valid_mask'].shape)}")
    if not torch.any(batch["valid_mask"]):
        raise AssertionError("Loaded batch has no valid joints.")
    return {
        "csv": str(csv_path),
        "loaded_rows": len(dataset),
        "batch_image_shape": list(batch["image"].shape),
        "batch_keypoints_shape": list(batch["keypoints"].shape),
        "batch_valid_joints": int(batch["valid_mask"].sum().item()),
    }


def main() -> None:
    args = parse_args()
    train_path = Path(args.train_csv)
    test_path = Path(args.test_csv)
    train_rows = read_rows(train_path)
    test_rows = read_rows(test_path)
    if not train_rows or not test_rows:
        raise SystemExit("Train and test CSVs must both be non-empty.")

    train_episodes = {row["episode_hash"] for row in train_rows}
    test_episodes = {row["episode_hash"] for row in test_rows}
    overlap = train_episodes & test_episodes
    if overlap:
        raise SystemExit(f"Train/test episode overlap: {sorted(overlap)}")

    check_no_aria_keypoints(train_rows + test_rows)
    result = {
        "train": summarize_rows(train_rows),
        "test": summarize_rows(test_rows),
        "loader_train": check_loader(train_path, args),
        "loader_test": check_loader(test_path, args),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
