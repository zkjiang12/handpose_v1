#!/usr/bin/env python3
"""Render EgoVerse hand-keypoint overlays for a small episode sample."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def add_egoverse_to_path() -> Path:
    default_repo = Path(__file__).resolve().parents[2] / "EgoVerse"
    repo = Path(os.environ.get("EGOVERSE_REPO", default_repo)).expanduser().resolve()
    if not repo.exists():
        raise SystemExit(
            f"EgoVerse repo not found at {repo}. Set EGOVERSE_REPO or clone it next to handpose_v1."
        )
    sys.path.insert(0, str(repo))
    return repo


EGOVERSE_REPO = add_egoverse_to_path()
EGOVERSE_VENV_BIN = EGOVERSE_REPO / "emimic" / "bin"
if EGOVERSE_VENV_BIN.exists():
    os.environ["PATH"] = f"{EGOVERSE_VENV_BIN}{os.pathsep}{os.environ.get('PATH', '')}"

import imageio.v3 as iio  # noqa: E402
import imageio_ffmpeg  # noqa: E402
import mediapy as mpy  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from egomimic.rldb.embodiment.human import Aria, Mecka  # noqa: E402
from egomimic.rldb.embodiment import embodiment as embodiment_mod  # noqa: E402
from egomimic.rldb.filters import DatasetFilter  # noqa: E402
from egomimic.rldb.zarr.zarr_dataset_multi import (  # noqa: E402
    MultiDataset,
    S3EpisodeResolver,
)
from egomimic.utils.aws.aws_data_utils import load_env  # noqa: E402
from egomimic.utils.aws.aws_sql import create_default_engine, episode_table_to_df  # noqa: E402
from egomimic.utils.egomimicUtils import INTRINSICS, cam_frame_to_cam_pixels  # noqa: E402


EMBODIMENTS = {
    "aria": Aria,
    "mecka": Mecka,
}

# Some current Aria Zarr metadata uses the generic label "human_bimanual",
# while EgoVerse's enum expects source-specific labels such as "aria_bimanual".
# The viewer only needs a stable numeric ID to let EgoVerse's dataset loader
# finish constructing batches.
embodiment_mod.EMBODIMENT._member_map_.setdefault(
    "HUMAN_BIMANUAL", embodiment_mod.EMBODIMENT.ARIA_BIMANUAL
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download/cache one EgoVerse episode and render hand keypoint overlays."
    )
    parser.add_argument("--embodiment", choices=EMBODIMENTS, default="aria")
    parser.add_argument(
        "--keypoint-layout",
        choices=("raw-aria", "canonical"),
        default="raw-aria",
        help=(
            "For Aria, raw-aria uses left/right.obs_aria_keypoints with Aria's "
            "native topology. canonical uses left/right.obs_keypoints."
        ),
    )
    parser.add_argument("--episode-hash", default=None)
    parser.add_argument("--task", default=None, help="Optional SQL task filter.")
    parser.add_argument("--lab", default=None, help="Optional SQL lab filter.")
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument(
        "--cache-dir",
        default="/Users/zikangjiang/data/egoverse_keypoint_cache",
        help="Local directory for downloaded Zarr episode cache.",
    )
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "egoverse_keypoints.mp4"),
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the rendered MP4 with the system default video player.",
    )
    parser.add_argument("--dot-radius", type=int, default=2)
    parser.add_argument("--line-thickness", type=int, default=1)
    return parser.parse_args()


def keymap_for(embodiment_cls, keypoint_layout: str):
    try:
        key_map = embodiment_cls.get_keymap(keymap_mode="keypoints")
    except TypeError:
        key_map = embodiment_cls.get_keymap(mode="keypoints")
    if embodiment_cls is Aria and keypoint_layout == "raw-aria":
        for side in ("left", "right"):
            key_map[f"{side}.action_keypoints"]["zarr_key"] = (
                f"{side}.obs_aria_keypoints"
            )
            key_map[f"{side}.obs_keypoints"]["zarr_key"] = (
                f"{side}.obs_aria_keypoints"
            )
    return key_map


def choose_episode(args: argparse.Namespace) -> str:
    if args.episode_hash:
        return args.episode_hash

    engine = create_default_engine()
    df = episode_table_to_df(engine)
    mask = df["embodiment"].astype(str).str.startswith(args.embodiment, na=False)

    if "is_deleted" in df.columns:
        mask &= ~df["is_deleted"].fillna(False).astype(bool)
    if "zarr_processed_path" in df.columns:
        mask &= df["zarr_processed_path"].fillna("").astype(str).str.strip() != ""
    if args.task:
        mask &= df["task"].astype(str) == args.task
    if args.lab:
        mask &= df["lab"].astype(str) == args.lab

    matches = df.loc[mask].copy()
    if matches.empty:
        raise SystemExit(
            "No matching episode found. Try removing --task/--lab or pass --episode-hash."
        )

    row = matches.iloc[0]
    print(
        "Selected episode:",
        row["episode_hash"],
        "| embodiment:",
        row.get("embodiment", ""),
        "| task:",
        row.get("task", ""),
        "| lab:",
        row.get("lab", ""),
        "| candidates:",
        len(matches),
    )
    return str(row["episode_hash"])


def describe_batch(batch: dict) -> None:
    print("Batch keys:")
    for key, value in sorted(batch.items()):
        shape = tuple(value.shape) if hasattr(value, "shape") else type(value).__name__
        print(f"  {key}: {shape}")


def render_keypoints(embodiment_cls, batch, dot_radius: int, line_thickness: int):
    image = batch[embodiment_cls.VIZ_IMAGE_KEY][0]
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = image.permute(1, 2, 0)
    image = image.detach().cpu().numpy()
    if image.dtype != np.uint8:
        image = (image * 255.0).clip(0, 255).astype(np.uint8)

    actions = batch["actions_keypoints"][0, 0].detach().cpu().numpy()
    if actions.shape[-1] == 138:
        left = actions[6 : 6 + 63].reshape(21, 3)
        right = actions[75 : 75 + 63].reshape(21, 3)
    else:
        left = actions[:63].reshape(21, 3)
        right = actions[63:126].reshape(21, 3)

    intrinsics = INTRINSICS[embodiment_cls.VIZ_INTRINSICS_KEY]
    vis = image.copy()
    h, w = vis.shape[:2]
    colors = {
        "thumb": (255, 100, 100),
        "index": (100, 255, 100),
        "middle": (100, 170, 255),
        "ring": (255, 230, 80),
        "pinky": (255, 100, 255),
    }
    dots = {"left": (0, 140, 255), "right": (255, 120, 0)}
    for hand, points in (("left", left), ("right", right)):
        px = cam_frame_to_cam_pixels(points, intrinsics)[:, :2]
        valid = points[:, 2] > 0.01
        valid &= (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h)
        for finger, start, end in embodiment_cls.FINGER_EDGE_RANGES:
            for edge_idx in range(start, end):
                i, j = embodiment_cls.FINGER_EDGES[edge_idx]
                if valid[i] and valid[j]:
                    p1 = tuple(np.round(px[i]).astype(int))
                    p2 = tuple(np.round(px[j]).astype(int))
                    import cv2

                    cv2.line(vis, p1, p2, colors[finger], line_thickness, cv2.LINE_AA)
        for i in range(21):
            if valid[i]:
                center = tuple(np.round(px[i]).astype(int))
                import cv2

                cv2.circle(vis, center, dot_radius, dots[hand], -1, cv2.LINE_AA)
                if dot_radius >= 3:
                    cv2.circle(vis, center, dot_radius, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def main() -> None:
    args = parse_args()
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    load_env(required=True)
    mpy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())

    embodiment_cls = EMBODIMENTS[args.embodiment]
    episode_hash = choose_episode(args)

    resolver = S3EpisodeResolver(
        str(cache_dir),
        key_map=keymap_for(embodiment_cls, args.keypoint_layout),
        transform_list=embodiment_cls.get_transform_list(mode="keypoints_headframe_ypr"),
    )
    filters = DatasetFilter(
        filter_lambdas=[f"lambda row: row['episode_hash'] == {episode_hash!r}"]
    )
    dataset = MultiDataset._from_resolver(
        resolver,
        filters=filters,
        sync_from_s3=True,
        mode="total",
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    frames = []
    for i, batch in enumerate(loader):
        if i == 0:
            describe_batch(batch)
        vis = render_keypoints(
            embodiment_cls,
            batch,
            dot_radius=max(1, args.dot_radius),
            line_thickness=max(1, args.line_thickness),
        )
        frames.append(vis)
        if len(frames) >= args.max_frames:
            break

    if not frames:
        raise SystemExit("No frames rendered from the selected episode.")

    mpy.write_video(str(out), frames, fps=30)
    first_png = out.with_suffix(".first_frame.png")
    iio.imwrite(first_png, frames[0])
    print(f"Wrote video: {out}")
    print(f"Wrote first-frame preview: {first_png}")
    print(f"EgoVerse repo: {EGOVERSE_REPO}")
    print(f"Cache dir: {cache_dir}")
    if args.open:
        subprocess.run(["open", str(out)], check=False)


if __name__ == "__main__":
    main()
