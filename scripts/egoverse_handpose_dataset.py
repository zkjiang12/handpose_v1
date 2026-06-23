#!/usr/bin/env python3
"""PyTorch dataset for frame-level EgoVerse hand-pose manifests."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import simplejpeg
import torch
import torch.nn.functional as F
import zarr
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def str_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def valid_joint_mask(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(21, 3)
    finite = np.isfinite(points).all(axis=1)
    nonzero = np.linalg.norm(np.nan_to_num(points), axis=1) > 1e-9
    return finite & nonzero


def world_to_camera(points_world: np.ndarray, head_pose: np.ndarray) -> np.ndarray:
    xyz = np.asarray(head_pose[:3], dtype=np.float64)
    qw, qx, qy, qz = np.asarray(head_pose[3:7], dtype=np.float64)
    rot_world_from_camera = R.from_quat([qx, qy, qz, qw])
    return rot_world_from_camera.inv().apply(np.asarray(points_world, dtype=np.float64) - xyz)


def decode_rgb(jpeg_value: Any) -> np.ndarray:
    while isinstance(jpeg_value, np.ndarray) and jpeg_value.shape == ():
        jpeg_value = jpeg_value.item()
    if not jpeg_value:
        raise ValueError("empty JPEG value")
    try:
        return simplejpeg.decode_jpeg(jpeg_value, colorspace="RGB")
    except Exception:
        encoded = np.frombuffer(jpeg_value, dtype=np.uint8)
        bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if bgr is None:
            raise
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class EgoVerseHandPoseDataset(Dataset):
    """Frame-level Aria hand-pose dataset backed by Zarr episodes."""

    def __init__(
        self,
        csv_path: str | Path,
        image_size: int = 224,
        normalize: bool = True,
        max_rows: int | None = None,
        cache_root: str | Path | None = None,
    ):
        self.csv_path = Path(csv_path)
        with self.csv_path.open(newline="") as f:
            self.rows = list(csv.DictReader(f))
        if max_rows is not None:
            self.rows = self.rows[:max_rows]
        if not self.rows:
            raise ValueError(f"No rows found in {self.csv_path}")
        self.image_size = int(image_size)
        self.normalize = normalize
        env_cache_root = os.environ.get("EGOVERSE_CACHE_DIR")
        self.cache_root = Path(cache_root or env_cache_root).expanduser() if (cache_root or env_cache_root) else None
        self._groups: dict[str, zarr.Group] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve_episode_path(self, row: dict[str, str]) -> Path:
        if self.cache_root is not None:
            episode_hash = row.get("episode_hash")
            if not episode_hash:
                raise KeyError("Manifest row is missing episode_hash; cannot resolve against EGOVERSE_CACHE_DIR.")
            episode_path = self.cache_root / episode_hash
        else:
            episode_path = Path(row["episode_path"]).expanduser()
        if not episode_path.exists():
            cache_hint = f" EGOVERSE_CACHE_DIR={self.cache_root}" if self.cache_root is not None else ""
            raise FileNotFoundError(f"EgoVerse episode directory not found: {episode_path}.{cache_hint}")
        return episode_path

    def _group(self, episode_path: Path) -> zarr.Group:
        episode_key = str(episode_path)
        if episode_key not in self._groups:
            self._groups[episode_key] = zarr.open_group(episode_key, mode="r")
        return self._groups[episode_key]

    def _image_tensor(self, group: zarr.Group, frame_idx: int, image_key: str) -> torch.Tensor:
        image = decode_rgb(group[image_key][frame_idx])
        tensor = torch.from_numpy(image).permute(2, 0, 1).to(torch.float32) / 255.0
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        if self.normalize:
            tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
        return tensor

    def _load_hand(
        self,
        group: zarr.Group,
        frame_idx: int,
        key: str,
        head_pose: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        points_world = np.asarray(group[key][frame_idx], dtype=np.float64).reshape(21, 3)
        joint_mask = valid_joint_mask(points_world)
        points_cam = np.zeros((21, 3), dtype=np.float32)
        if joint_mask.any():
            points_cam[joint_mask] = world_to_camera(points_world[joint_mask], head_pose).astype(np.float32)
        return points_cam, joint_mask

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        frame_idx = int(row["frame_idx"])
        group = self._group(self._resolve_episode_path(row))
        head_pose = np.asarray(group[row["head_pose_key"]][frame_idx], dtype=np.float64)

        keypoints = np.zeros((2, 21, 3), dtype=np.float32)
        valid_mask = np.zeros((2, 21), dtype=bool)

        left_key = row.get("left_keypoints_key") or ""
        if str_bool(row.get("has_left")) and left_key:
            keypoints[0], valid_mask[0] = self._load_hand(group, frame_idx, left_key, head_pose)

        right_key = row.get("right_keypoints_key") or ""
        if str_bool(row.get("has_right")) and right_key:
            keypoints[1], valid_mask[1] = self._load_hand(group, frame_idx, right_key, head_pose)

        return {
            "image": self._image_tensor(group, frame_idx, row["image_key"]),
            "keypoints": torch.from_numpy(keypoints),
            "valid_mask": torch.from_numpy(valid_mask),
            "row_idx": idx,
            "episode_hash": row["episode_hash"],
            "frame_idx": frame_idx,
            "source": row["source"],
            "split": row["split"],
        }
