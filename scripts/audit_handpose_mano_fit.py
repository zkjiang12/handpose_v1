#!/usr/bin/env python3
"""Fit MANO to EgoVerse hand keypoints and report residual plausibility."""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import os
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from audit_handpose_geometry import (  # noqa: E402
    HAND_EDGES,
    add_label,
    collect_samples,
    draw_hand_overlay,
    save_contact_sheet,
)
from egoverse_handpose_dataset import decode_rgb, world_to_camera  # noqa: E402
from filter_egoverse_handpose_visibility import camera_intrinsics  # noqa: E402


MANO_TIP_NAMES = ("thumb", "index", "middle", "ring", "pinky")

# smplx MANO returns 16 joints in this order:
# wrist, index1-3, middle1-3, pinky1-3, ring1-3, thumb1-3.
# Append five fingertip vertices, then remap to the dataset convention:
# wrist, thumb1-4, index1-4, middle1-4, ring1-4, pinky1-4.
MANO21_REMAP = (0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit MANO to filtered EgoVerse handpose labels.")
    parser.add_argument("--train-csv", default="outputs/handpose_dataset_visible/train.csv")
    parser.add_argument("--test-csv", default="outputs/handpose_dataset_visible/test.csv")
    parser.add_argument("--out-dir", default="outputs/mano_fit_audit")
    parser.add_argument("--mano-model-root", default="models")
    parser.add_argument("--cache-root", default=os.environ.get("EGOVERSE_CACHE_DIR"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--iters", type=int, default=220)
    parser.add_argument("--lr", type=float, default=0.015)
    parser.add_argument("--pose-reg", type=float, default=1e-6)
    parser.add_argument("--shape-reg", type=float, default=1e-6)
    parser.add_argument("--max-train-hands", type=int, default=0, help="0 fits all train hands.")
    parser.add_argument("--max-test-hands", type=int, default=0, help="0 fits all test hands.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--failure-mpjpe-mm", type=float, default=15.0)
    parser.add_argument("--viz-count", type=int, default=16)
    return parser.parse_args()


def chumpy_compatibility_patch() -> None:
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]
    for name, value in {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "unicode": str,
        "str": str,
    }.items():
        if not hasattr(np, name):
            setattr(np, name, value)


def import_smplx() -> tuple[Any, dict[str, int]]:
    chumpy_compatibility_patch()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import smplx  # type: ignore
        from smplx.vertex_ids import vertex_ids  # type: ignore

    return smplx, vertex_ids["mano"]


def pick_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available.")
    return device


def sample_subset(samples: list[dict[str, Any]], max_count: int, seed: int) -> list[dict[str, Any]]:
    if max_count <= 0 or len(samples) <= max_count:
        return samples

    rng = np.random.default_rng(seed)
    by_episode: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_episode[(sample["split"], sample["episode_hash"], sample["side"])].append(sample)

    selected: list[dict[str, Any]] = []
    for seq_samples in by_episode.values():
        seq_samples = sorted(seq_samples, key=lambda sample: sample["frame_idx"])
        selected.append(seq_samples[len(seq_samples) // 2])

    if len(selected) >= max_count:
        idx = rng.choice(len(selected), size=max_count, replace=False)
        return [selected[int(i)] for i in sorted(idx)]

    selected_ids = {(sample["split"], sample["row_idx"], sample["side"]) for sample in selected}
    remaining = [sample for sample in samples if (sample["split"], sample["row_idx"], sample["side"]) not in selected_ids]
    take = min(len(remaining), max_count - len(selected))
    if take:
        idx = rng.choice(len(remaining), size=take, replace=False)
        selected.extend(remaining[int(i)] for i in idx)
    return sorted(selected, key=lambda sample: (sample["split"], sample["episode_hash"], sample["frame_idx"], sample["side"]))


def mano21(
    model: torch.nn.Module,
    tip_vertex_ids: dict[str, int],
    global_orient: torch.Tensor,
    hand_pose: torch.Tensor,
    betas: torch.Tensor,
    transl: torch.Tensor,
) -> torch.Tensor:
    output = model(
        global_orient=global_orient,
        hand_pose=hand_pose,
        betas=betas,
        transl=transl,
        return_verts=True,
    )
    tips = output.vertices[:, [tip_vertex_ids[name] for name in MANO_TIP_NAMES], :]
    joints21 = torch.cat([output.joints, tips], dim=1)
    return joints21[:, MANO21_REMAP]


def kabsch_initialize(rest: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rotations = []
    translations = []
    rigid_mpjpe = []
    for rest_points, target_points in zip(rest, target):
        rest_center = rest_points.mean(axis=0)
        target_center = target_points.mean(axis=0)
        rest_centered = rest_points - rest_center
        target_centered = target_points - target_center
        u, _, vt = np.linalg.svd(rest_centered.T @ target_centered)
        rotation = vt.T @ u.T
        if np.linalg.det(rotation) < 0:
            vt[-1] *= -1
            rotation = vt.T @ u.T
        translation = target_center - rotation @ rest_center
        aligned = (rotation @ rest_points.T).T + translation
        rigid_mpjpe.append(float(np.linalg.norm(aligned - target_points, axis=1).mean() * 1000.0))
        rotations.append(R.from_matrix(rotation).as_rotvec())
        translations.append(translation)
    return (
        np.asarray(rotations, dtype=np.float32),
        np.asarray(translations, dtype=np.float32),
        np.asarray(rigid_mpjpe, dtype=np.float32),
    )


def fit_batch(
    samples: list[dict[str, Any]],
    *,
    side: str,
    args: argparse.Namespace,
    device: torch.device,
    smplx: Any,
    tip_vertex_ids: dict[str, int],
) -> list[dict[str, Any]]:
    batch_size = len(samples)
    model = smplx.create(
        args.mano_model_root,
        model_type="mano",
        is_rhand=(side == "right"),
        use_pca=False,
        flat_hand_mean=True,
        batch_size=batch_size,
    ).to(device)

    target_np = np.stack([sample["points"] for sample in samples]).astype(np.float32)
    with torch.no_grad():
        zeros3 = torch.zeros((batch_size, 3), device=device)
        zeros45 = torch.zeros((batch_size, 45), device=device)
        zeros10 = torch.zeros((batch_size, 10), device=device)
        rest = mano21(model, tip_vertex_ids, zeros3, zeros45, zeros10, zeros3).detach().cpu().numpy()

    init_global_orient, init_transl, rigid_mpjpe = kabsch_initialize(rest, target_np)
    global_orient = torch.nn.Parameter(torch.tensor(init_global_orient, dtype=torch.float32, device=device))
    transl = torch.nn.Parameter(torch.tensor(init_transl, dtype=torch.float32, device=device))
    hand_pose = torch.nn.Parameter(torch.zeros((batch_size, 45), dtype=torch.float32, device=device))
    betas = torch.nn.Parameter(torch.zeros((batch_size, 10), dtype=torch.float32, device=device))
    target = torch.tensor(target_np, dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam([global_orient, transl, hand_pose, betas], lr=args.lr)
    last_loss = float("nan")
    for _ in range(args.iters):
        optimizer.zero_grad(set_to_none=True)
        pred = mano21(model, tip_vertex_ids, global_orient, hand_pose, betas, transl)
        diff = pred - target
        data_loss = diff.square().sum(dim=-1).mean(dim=-1).mean()
        loss = data_loss + args.pose_reg * hand_pose.square().mean() + args.shape_reg * betas.square().mean()
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())

    with torch.no_grad():
        pred = mano21(model, tip_vertex_ids, global_orient, hand_pose, betas, transl)
        diff = pred - target
        per_joint_dist = torch.linalg.norm(diff, dim=-1)
        mpjpe = per_joint_dist.mean(dim=-1).detach().cpu().numpy() * 1000.0
        rmse = torch.sqrt(diff.square().sum(dim=-1).mean(dim=-1)).detach().cpu().numpy() * 1000.0
        max_joint_error = per_joint_dist.max(dim=-1).values.detach().cpu().numpy() * 1000.0
        pred_np = pred.detach().cpu().numpy()
        pose_l2 = torch.linalg.norm(hand_pose, dim=-1).detach().cpu().numpy()
        beta_l2 = torch.linalg.norm(betas, dim=-1).detach().cpu().numpy()
        max_abs_pose = torch.max(torch.abs(hand_pose), dim=-1).values.detach().cpu().numpy()

    rows = []
    for idx, sample in enumerate(samples):
        rows.append(
            {
                "split": sample["split"],
                "row_idx": sample["row_idx"],
                "episode_hash": sample["episode_hash"],
                "frame_idx": sample["frame_idx"],
                "side": sample["side"],
                "mano_mpjpe_mm": float(mpjpe[idx]),
                "mano_rmse_mm": float(rmse[idx]),
                "mano_max_joint_error_mm": float(max_joint_error[idx]),
                "rigid_mpjpe_mm": float(rigid_mpjpe[idx]),
                "rigid_to_mano_improvement_mm": float(rigid_mpjpe[idx] - mpjpe[idx]),
                "pose_l2": float(pose_l2[idx]),
                "beta_l2": float(beta_l2[idx]),
                "max_abs_pose_rad": float(max_abs_pose[idx]),
                "mano_fit_failure": bool(mpjpe[idx] > args.failure_mpjpe_mm),
                "optimizer_last_loss": last_loss,
                "points": sample["points"],
                "mano_points": pred_np[idx],
                "sample": sample,
            }
        )
    return rows


def fit_samples(samples: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    smplx, tip_vertex_ids = import_smplx()
    device = pick_device(args.device)
    print(f"Using device: {device}")
    results: list[dict[str, Any]] = []
    for split in ("train", "test"):
        for side in ("left", "right"):
            side_samples = [sample for sample in samples if sample["split"] == split and sample["side"] == side]
            if not side_samples:
                continue
            total = len(side_samples)
            for start in range(0, total, args.batch_size):
                batch = side_samples[start : start + args.batch_size]
                t0 = time.time()
                batch_results = fit_batch(
                    batch,
                    side=side,
                    args=args,
                    device=device,
                    smplx=smplx,
                    tip_vertex_ids=tip_vertex_ids,
                )
                results.extend(batch_results)
                done = min(start + len(batch), total)
                median_mpjpe = np.median([row["mano_mpjpe_mm"] for row in batch_results])
                print(
                    f"{split} {side}: {done}/{total} "
                    f"batch_median={median_mpjpe:.2f}mm elapsed={time.time() - t0:.1f}s",
                    flush=True,
                )
    return results


def serializable_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "split": row["split"],
        "row_idx": row["row_idx"],
        "episode_hash": row["episode_hash"],
        "frame_idx": row["frame_idx"],
        "side": row["side"],
        "mano_mpjpe_mm": row["mano_mpjpe_mm"],
        "mano_rmse_mm": row["mano_rmse_mm"],
        "mano_max_joint_error_mm": row["mano_max_joint_error_mm"],
        "rigid_mpjpe_mm": row["rigid_mpjpe_mm"],
        "rigid_to_mano_improvement_mm": row["rigid_to_mano_improvement_mm"],
        "pose_l2": row["pose_l2"],
        "beta_l2": row["beta_l2"],
        "max_abs_pose_rad": row["max_abs_pose_rad"],
        "mano_fit_failure": row["mano_fit_failure"],
        "optimizer_last_loss": row["optimizer_last_loss"],
    }


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
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


def build_summary(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_episode: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_side: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_split[row["split"]].append(row)
        by_episode[(row["split"], row["episode_hash"])].append(row)
        by_side[(row["split"], row["side"])].append(row)

    split_summary = {}
    for split, split_rows in sorted(by_split.items()):
        failures = [row for row in split_rows if row["mano_fit_failure"]]
        split_summary[split] = {
            "hands": len(split_rows),
            "mano_fit_failures": len(failures),
            "mano_fit_failure_rate": len(failures) / max(1, len(split_rows)),
            "mano_mpjpe_mm": summarize([row["mano_mpjpe_mm"] for row in split_rows]),
            "mano_rmse_mm": summarize([row["mano_rmse_mm"] for row in split_rows]),
            "rigid_mpjpe_mm": summarize([row["rigid_mpjpe_mm"] for row in split_rows]),
            "pose_l2": summarize([row["pose_l2"] for row in split_rows]),
            "beta_l2": summarize([row["beta_l2"] for row in split_rows]),
        }

    side_summary = {}
    for (split, side), side_rows in sorted(by_side.items()):
        failures = [row for row in side_rows if row["mano_fit_failure"]]
        side_summary[f"{split}:{side}"] = {
            "hands": len(side_rows),
            "mano_fit_failures": len(failures),
            "mano_fit_failure_rate": len(failures) / max(1, len(side_rows)),
            "mano_mpjpe_mm": summarize([row["mano_mpjpe_mm"] for row in side_rows]),
        }

    episode_summary = []
    for (split, episode_hash), episode_rows in sorted(by_episode.items()):
        failures = [row for row in episode_rows if row["mano_fit_failure"]]
        episode_summary.append(
            {
                "split": split,
                "episode_hash": episode_hash,
                "hands": len(episode_rows),
                "mano_fit_failures": len(failures),
                "mano_fit_failure_rate": len(failures) / max(1, len(episode_rows)),
                "mano_mpjpe_mm": summarize([row["mano_mpjpe_mm"] for row in episode_rows]),
            }
        )

    failures = [row for row in rows if row["mano_fit_failure"]]
    worst = sorted(rows, key=lambda row: row["mano_mpjpe_mm"], reverse=True)
    return {
        "args": vars(args),
        "mano_model_available": True,
        "total_hands": len(rows),
        "mano_fit_failures": len(failures),
        "mano_fit_failure_rate": len(failures) / max(1, len(rows)),
        "failure_threshold_mpjpe_mm": args.failure_mpjpe_mm,
        "mano_mpjpe_mm": summarize([row["mano_mpjpe_mm"] for row in rows]),
        "mano_rmse_mm": summarize([row["mano_rmse_mm"] for row in rows]),
        "rigid_mpjpe_mm": summarize([row["rigid_mpjpe_mm"] for row in rows]),
        "split_summary": split_summary,
        "side_summary": side_summary,
        "episode_summary": sorted(
            episode_summary,
            key=lambda row: (row["mano_fit_failure_rate"], row["mano_mpjpe_mm"]["median"] if row["mano_mpjpe_mm"] else 0.0),
            reverse=True,
        ),
        "worst_samples": [serializable_row(row) for row in worst[:100]],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serial_rows = [serializable_row(row) for row in rows]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(serial_rows[0].keys()))
        writer.writeheader()
        writer.writerows(serial_rows)


def render_mano_overlay(row: dict[str, Any], groups: dict[str, Any]) -> np.ndarray:
    sample = row["sample"]
    group = groups.setdefault(sample["episode_hash"], __import__("zarr").open_group(str(sample["episode_path"]), mode="r"))
    source_row = sample["row"]
    frame_idx = int(sample["frame_idx"])
    image = np.ascontiguousarray(decode_rgb(group[source_row["image_key"]][frame_idx]).copy())
    head_pose = np.asarray(group[source_row["head_pose_key"]][frame_idx], dtype=np.float64)
    intrinsics, _, _ = camera_intrinsics(group, source_row["image_key"])
    side = sample["side"]
    mask = np.ones(21, dtype=bool)
    target_cam = world_to_camera(row["points"], head_pose)
    mano_cam = world_to_camera(row["mano_points"], head_pose)
    draw_hand_overlay(image, target_cam, mask, intrinsics, (40, 160, 255), radius=4, thickness=3)
    draw_hand_overlay(image, mano_cam, mask, intrinsics, (255, 50, 50), radius=3, thickness=2)

    return add_label(
        image,
        [
            f"{row['split']} {row['episode_hash']} f{row['frame_idx']} {side}",
            f"MANO MPJPE={row['mano_mpjpe_mm']:.1f}mm rigid={row['rigid_mpjpe_mm']:.1f}mm",
            "blue=label red=MANO fit",
        ],
    )


def set_equal_3d_axes(ax: Any, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float((maxs - mins).max()) / 2.0, 0.05)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def draw_hand_3d(ax: Any, points: np.ndarray, color: str, label: str) -> None:
    plotted = False
    for i, j in HAND_EDGES:
        segment = points[[i, j]]
        ax.plot(
            segment[:, 0],
            segment[:, 1],
            segment[:, 2],
            color=color,
            linewidth=1.6,
            label=label if not plotted else None,
        )
        plotted = True
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=color, s=8)


def save_3d_sheet(rows: list[dict[str, Any]], path: Path, *, cols: int = 4) -> None:
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plot_rows = math.ceil(len(rows) / cols)
    fig = plt.figure(figsize=(cols * 3.0, plot_rows * 3.0), dpi=140)
    for idx, row in enumerate(rows):
        ax = fig.add_subplot(plot_rows, cols, idx + 1, projection="3d")
        draw_hand_3d(ax, row["points"], "#2563eb", "label")
        draw_hand_3d(ax, row["mano_points"], "#dc2626", "MANO")
        points = np.concatenate([row["points"], row["mano_points"]], axis=0)
        set_equal_3d_axes(ax, points)
        ax.set_title(f"{row['split']} f{row['frame_idx']} {row['side']}\n{row['mano_mpjpe_mm']:.1f}mm", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.view_init(elev=18, azim=-65)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles[:2], labels[:2], loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_summary_plot(rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train = [row for row in rows if row["split"] == "train"]
    test = [row for row in rows if row["split"] == "test"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=140)

    axes[0, 0].hist([row["mano_mpjpe_mm"] for row in train], bins=60, alpha=0.65, label="train")
    axes[0, 0].hist([row["mano_mpjpe_mm"] for row in test], bins=60, alpha=0.65, label="test")
    axes[0, 0].set_title("MANO fit MPJPE")
    axes[0, 0].set_xlabel("mm")
    axes[0, 0].legend()

    axes[0, 1].scatter(
        [row["rigid_mpjpe_mm"] for row in rows],
        [row["mano_mpjpe_mm"] for row in rows],
        s=4,
        alpha=0.35,
    )
    axes[0, 1].set_title("Rigid vs full MANO fit")
    axes[0, 1].set_xlabel("rigid MPJPE mm")
    axes[0, 1].set_ylabel("MANO MPJPE mm")

    axes[1, 0].hist([row["pose_l2"] for row in train], bins=60, alpha=0.65, label="train")
    axes[1, 0].hist([row["pose_l2"] for row in test], bins=60, alpha=0.65, label="test")
    axes[1, 0].set_title("Optimized MANO pose norm")
    axes[1, 0].set_xlabel("L2 norm")
    axes[1, 0].legend()

    failures = Counter()
    for row in rows:
        if row["mano_fit_failure"]:
            failures[row["split"]] += 1
    axes[1, 1].bar(list(failures.keys()), list(failures.values()))
    axes[1, 1].set_title("MANO fit failures")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_diagnostics(rows: list[dict[str, Any]], out_dir: Path, args: argparse.Namespace) -> None:
    if args.viz_count <= 0:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    save_summary_plot(rows, out_dir / "mano_fit_summary.png")
    worst = sorted(rows, key=lambda row: row["mano_mpjpe_mm"], reverse=True)[: args.viz_count]
    worst_test = sorted(
        [row for row in rows if row["split"] == "test"],
        key=lambda row: row["mano_mpjpe_mm"],
        reverse=True,
    )[: args.viz_count]
    groups: dict[str, Any] = {}
    save_contact_sheet([render_mano_overlay(row, groups) for row in worst], out_dir / "worst_mano_fit_overlays.png")
    save_3d_sheet(worst, out_dir / "worst_mano_fit_3d.png")
    save_contact_sheet([render_mano_overlay(row, groups) for row in worst_test], out_dir / "worst_test_mano_fit_overlays.png")
    save_3d_sheet(worst_test, out_dir / "worst_test_mano_fit_3d.png")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = collect_samples(
        {"train": Path(args.train_csv), "test": Path(args.test_csv)},
        cache_root=args.cache_root,
    )
    train_samples = sample_subset([sample for sample in samples if sample["split"] == "train"], args.max_train_hands, args.seed)
    test_samples = sample_subset([sample for sample in samples if sample["split"] == "test"], args.max_test_hands, args.seed + 1)
    selected = train_samples + test_samples
    if not selected:
        raise SystemExit("No samples selected for MANO fitting.")

    rows = fit_samples(selected, args)
    write_csv(out_dir / "mano_fit_samples.csv", rows)
    summary = build_summary(rows, args)
    (out_dir / "mano_fit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    save_diagnostics(rows, out_dir, args)
    print(json.dumps(
        {
            "total_hands": summary["total_hands"],
            "mano_fit_failures": summary["mano_fit_failures"],
            "mano_fit_failure_rate": summary["mano_fit_failure_rate"],
            "split_summary": summary["split_summary"],
        },
        indent=2,
    ))
    print(f"Wrote MANO fit audit to {out_dir}")


if __name__ == "__main__":
    main()
