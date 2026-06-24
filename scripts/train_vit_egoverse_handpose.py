#!/usr/bin/env python3
"""Simple ViT baseline for EgoVerse hand-pose regression."""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, default_collate
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from egoverse_handpose_dataset import EgoVerseHandPoseDataset, IMAGENET_MEAN, IMAGENET_STD  # noqa: E402

VIZ_FILE_RE = re.compile(r"^epoch_(?P<epoch>\d+)(?:_step_(?P<step>\d+))?_sample_(?P<sample>\d+)_")
VIZ_SAMPLE_RE = re.compile(r"sample_(\d+)")
EPOCH_FILE_RE = re.compile(r"^epoch_(?P<epoch>\d+)")

HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)


class ViTHandPose(nn.Module):
    def __init__(
        self,
        model_name: str,
        pretrained: bool,
        *,
        backbone_source: str = "auto",
        freeze_backbone: bool = False,
        head_type: str = "linear",
        head_hidden_dims: tuple[int, ...] = (1024, 512),
        head_dropout: float = 0.0,
    ):
        super().__init__()
        self.backbone_source = self._resolve_backbone_source(model_name, backbone_source)
        self.backbone = self._create_backbone(model_name, pretrained, self.backbone_source)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.head = self._create_head(
            self._backbone_num_features(self.backbone),
            head_type=head_type,
            hidden_dims=head_hidden_dims,
            dropout=head_dropout,
        )

    @staticmethod
    def _resolve_backbone_source(model_name: str, backbone_source: str) -> str:
        if backbone_source != "auto":
            return backbone_source
        return "dinov3" if model_name.startswith("dinov3_") else "timm"

    @staticmethod
    def _create_backbone(model_name: str, pretrained: bool, backbone_source: str) -> nn.Module:
        if backbone_source == "timm":
            return timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        if backbone_source == "dinov3":
            return torch.hub.load(
                repo_or_dir="facebookresearch/dinov3",
                model=model_name,
                pretrained=pretrained,
            )
        raise ValueError(f"Unsupported backbone source: {backbone_source}")

    @staticmethod
    def _backbone_num_features(backbone: nn.Module) -> int:
        for attr in ("num_features", "embed_dim"):
            value = getattr(backbone, attr, None)
            if value is not None:
                return int(value)
        raise AttributeError("Backbone does not expose num_features or embed_dim")

    @staticmethod
    def _create_head(
        input_dim: int,
        *,
        head_type: str,
        hidden_dims: tuple[int, ...],
        dropout: float,
    ) -> nn.Module:
        output_dim = 2 * 21 * 3
        if head_type == "linear":
            return nn.Linear(input_dim, output_dim)
        if head_type == "mlp":
            layers: list[nn.Module] = []
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, output_dim))
            return nn.Sequential(*layers)
        raise ValueError(f"Unsupported head type: {head_type}")

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(image)).view(-1, 2, 21, 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a basic ViT on EgoVerse hand pose.")
    parser.add_argument("--train-csv", default="outputs/handpose_dataset/train.csv")
    parser.add_argument("--test-csv", default="outputs/handpose_dataset/test.csv")
    parser.add_argument("--out-dir", default=None, help="Exact run output directory. If omitted, creates runs-root/run_###.")
    parser.add_argument("--runs-root", default="outputs/vit_runs", help="Parent directory for auto-numbered runs.")
    parser.add_argument("--run-prefix", default="run", help="Prefix for auto-numbered run folders.")
    parser.add_argument("--model-name", default="vit_tiny_patch16_224")
    parser.add_argument("--backbone-source", choices=("auto", "timm", "dinov3"), default="auto")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--head-type", choices=("linear", "mlp"), default="linear")
    parser.add_argument("--head-hidden-dims", default="1024,512", help="Comma-separated MLP hidden dims.")
    parser.add_argument("--head-dropout", type=float, default=0.0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-eval-steps", type=int, default=None)
    parser.add_argument("--overfit-batches", type=int, default=None)
    parser.add_argument("--viz-every", type=int, default=0, help="Save prediction overlays every N epochs; 0 disables.")
    parser.add_argument("--viz-per-epoch", type=int, default=0, help="Save prediction overlays N times during each viz epoch; 0 saves at epoch end only.")
    parser.add_argument("--viz-samples", type=int, default=4)
    parser.add_argument("--ranked-viz-every", type=int, default=0, help="After test eval, render best/worst test samples every N epochs; 0 disables.")
    parser.add_argument("--ranked-viz-percentile", type=float, default=10.0, help="Percentile bucket for ranked visualizations, e.g. 10 renders bottom/top 10%% by MPJPE.")
    parser.add_argument("--ranked-viz-max-samples", type=int, default=10, help="Max samples to render per ranked bucket; 0 disables the cap.")
    parser.add_argument("--ranked-viz-max-per-episode", type=int, default=2, help="Max ranked samples per episode per bucket; 0 disables the cap.")
    parser.add_argument("--ranked-viz-min-frame-gap", type=int, default=60, help="Minimum frame_idx gap between ranked samples from the same episode.")
    parser.add_argument("--gt-radius", type=int, default=1)
    parser.add_argument("--pred-radius", type=int, default=2)
    parser.add_argument("--plot-every", type=int, default=1, help="Update loss chart every N epochs; 0 disables.")
    parser.add_argument("--log-every-steps", type=int, default=0, help="Print/write train metrics every N batches; 0 disables.")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True, help="Show tqdm progress bars.")
    parser.add_argument("--open-live-plot", action="store_true", help="Open a local auto-refreshing metrics HTML page.")
    parser.add_argument("--save-every", type=int, default=10, help="Save epoch checkpoint every N epochs; 0 disables.")
    parser.add_argument(
        "--keep-checkpoints",
        type=int,
        default=1,
        help="Number of periodic epoch checkpoints to retain; negative keeps all.",
    )
    parser.add_argument("--resume", default=None, help="Resume from a checkpoint such as /runs/name/last.pt.")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Build data/model and exit without training.")
    return parser.parse_args()


def parse_head_hidden_dims(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    dims = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if any(dim <= 0 for dim in dims):
        raise ValueError(f"--head-hidden-dims must contain positive integers: {value}")
    return dims


def prune_epoch_checkpoints(checkpoint_dir: Path, keep: int) -> None:
    if keep < 0:
        return
    checkpoints = sorted(checkpoint_dir.glob("epoch_*.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
    for checkpoint in checkpoints[keep:]:
        checkpoint.unlink()


def allocate_numbered_out_dir(args: argparse.Namespace) -> Path:
    runs_root = Path(args.runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.run_prefix}_"
    for run_id in range(1, 100000):
        candidate = runs_root / f"{args.run_prefix}_{run_id:03d}"
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError(f"Could not allocate a numbered run under {runs_root}")


def preview_numbered_out_dir(args: argparse.Namespace) -> Path:
    runs_root = Path(args.runs_root)
    prefix = f"{args.run_prefix}_"
    existing = []
    if runs_root.exists():
        for path in runs_root.iterdir():
            if not path.is_dir():
                continue
            name = path.name
            if name.startswith(prefix) and name[len(prefix):].isdigit():
                existing.append(int(name[len(prefix):]))
    return runs_root / f"{args.run_prefix}_{max(existing, default=0) + 1:03d}"


def resolve_out_dir(args: argparse.Namespace, distributed: bool, rank: int, *, allocate: bool) -> Path:
    if args.out_dir:
        return Path(args.out_dir)
    if args.resume:
        return Path(args.resume).expanduser().resolve().parent

    out_dir = (allocate_numbered_out_dir(args) if allocate else preview_numbered_out_dir(args)) if rank == 0 else None
    if distributed:
        shared = [str(out_dir) if out_dir is not None else None]
        dist.broadcast_object_list(shared, src=0)
        out_dir = Path(shared[0])
    if out_dir is None:
        raise RuntimeError("Only rank 0 can allocate an output directory")
    return out_dir


def setup_distributed(enabled: bool) -> tuple[bool, int, int, int]:
    if not enabled:
        return False, 0, 1, 0
    if "RANK" not in os.environ:
        raise RuntimeError("--distributed requires torchrun environment variables")
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return True, rank, world_size, local_rank


def is_rank0(rank: int) -> bool:
    return rank == 0


def model_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    if isinstance(model, DistributedDataParallel):
        return model.module.state_dict()
    return model.state_dict()


def load_model_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    target = model.module if isinstance(model, DistributedDataParallel) else model
    target.load_state_dict(state)


def read_metrics_history(metrics_path: Path) -> list[dict]:
    if not metrics_path.exists():
        return []
    history = []
    with metrics_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                history.append(json.loads(line))
    return history


def read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if limit is not None:
        return rows[-limit:]
    return rows


def masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.unsqueeze(-1).expand_as(pred)
    if not torch.any(expanded):
        return pred.sum() * 0.0
    return nn.functional.smooth_l1_loss(pred[expanded], target[expanded])


@torch.no_grad()
def mpjpe_mm(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    distances = torch.linalg.norm(pred - target, dim=-1)
    if not torch.any(mask):
        return distances.sum() * 0.0
    return distances[mask].mean() * 1000.0


@torch.no_grad()
def per_sample_mpjpe_mm(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    distances = torch.linalg.norm(pred - target, dim=-1)
    flat_distances = distances.flatten(1)
    flat_mask = mask.flatten(1)
    valid_counts = flat_mask.sum(dim=1)
    sample_errors = torch.full(
        (pred.shape[0],),
        float("nan"),
        dtype=distances.dtype,
        device=distances.device,
    )
    valid_samples = valid_counts > 0
    if torch.any(valid_samples):
        sample_errors[valid_samples] = (
            (flat_distances * flat_mask).sum(dim=1)[valid_samples] / valid_counts[valid_samples]
        ) * 1000.0
    return sample_errors


def project_points(points_cam: np.ndarray, image_size: int) -> np.ndarray:
    fx = 133.25430222 * 2 * (image_size / 640.0)
    fy = 133.25430222 * 2 * (image_size / 480.0)
    cx = image_size / 2.0
    cy = image_size / 2.0
    z = np.clip(points_cam[:, 2], 1e-6, None)
    return np.stack([fx * points_cam[:, 0] / z + cx, fy * points_cam[:, 1] / z + cy], axis=1)


def projected_valid_points(
    points_cam: np.ndarray,
    mask: np.ndarray,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    px = project_points(points_cam, image_size)
    h = w = image_size
    valid = np.asarray(mask, dtype=bool).copy()
    valid &= points_cam[:, 2] > 0.01
    valid &= (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h)
    return px, valid


def draw_hand(
    image: np.ndarray,
    points_cam: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    radius: int,
    thickness: int,
) -> None:
    px, valid = projected_valid_points(points_cam, mask, image.shape[0])
    for i, j in HAND_EDGES:
        if valid[i] and valid[j]:
            p1 = tuple(np.round(px[i]).astype(int))
            p2 = tuple(np.round(px[j]).astype(int))
            cv2.line(image, p1, p2, color, thickness, cv2.LINE_AA)
    for i, valid in enumerate(mask):
        if not valid:
            continue
        if points_cam[i, 2] > 0.01:
            x, y = np.round(px[i]).astype(int)
            h, w = image.shape[:2]
            if not (0 <= x < w and 0 <= y < h):
                continue
            cv2.circle(image, (x, y), radius, color, -1, cv2.LINE_AA)


def prune_duplicate_visualizations(viz_dir: Path) -> None:
    candidates: dict[tuple[int, str], list[tuple[float, Path]]] = {}
    for path in viz_dir.glob("*.png"):
        match = VIZ_FILE_RE.match(path.stem)
        if not match:
            continue
        epoch = int(match.group("epoch"))
        sample = match.group("sample")
        step_text = match.group("step")
        step = float("inf") if step_text is None else int(step_text)
        candidates.setdefault((epoch, sample), []).append((step, path))

    for paths in candidates.values():
        if len(paths) <= 1:
            continue
        keep = max(paths, key=lambda item: (item[0], item[1].stat().st_mtime))[1]
        for _, path in paths:
            if path == keep:
                continue
            path.unlink(missing_ok=True)


def build_grouped_gallery(out_dir: Path, image_dir: Path) -> str:
    images = sorted(image_dir.glob("*.png"))
    grouped: dict[str, list[Path]] = {}
    for path in images:
        match = VIZ_SAMPLE_RE.search(path.stem)
        sample_id = match.group(1) if match else "unknown"
        grouped.setdefault(sample_id, []).append(path)
    gallery = "\n".join(
        f"""    <section class="sample-row">
      <h3>Test sample {html.escape(sample_id)}</h3>
      <div class="sample-strip">
{''.join(
        f'''        <figure>
          <img src="{html.escape(str(path.relative_to(out_dir)))}" alt="{html.escape(path.stem)}">
          <figcaption>{html.escape(path.stem)}</figcaption>
        </figure>
'''
        for path in paths
    )}      </div>
    </section>"""
        for sample_id, paths in sorted(grouped.items())
    )
    return gallery or "    <p>No images written yet.</p>"


def image_figure_html(out_dir: Path, path: Path) -> str:
    return f"""        <figure>
          <img src="{html.escape(str(path.relative_to(out_dir)))}" alt="{html.escape(path.stem)}">
          <figcaption>{html.escape(path.stem)}</figcaption>
        </figure>
"""


def epoch_from_path(path: Path) -> int | None:
    match = EPOCH_FILE_RE.match(path.stem)
    return int(match.group("epoch")) if match else None


def latest_epoch_images(paths: list[Path]) -> tuple[int | None, list[Path]]:
    epoch_paths = [(epoch_from_path(path), path) for path in paths]
    epochs = [epoch for epoch, _ in epoch_paths if epoch is not None]
    if not epochs:
        return None, paths
    latest_epoch = max(epochs)
    return latest_epoch, [path for epoch, path in epoch_paths if epoch == latest_epoch]


def build_ranked_gallery(out_dir: Path, image_root: Path) -> str:
    if not image_root.exists():
        return "    <p>No ranked visualizations written yet.</p>"

    sections = []
    for bucket_dir in sorted(path for path in image_root.iterdir() if path.is_dir()):
        images = sorted(bucket_dir.glob("*.png"))
        if not images:
            continue
        latest_epoch, images = latest_epoch_images(images)
        title = bucket_dir.name.replace("_", " ")
        if latest_epoch is not None:
            title = f"{title} - latest epoch {latest_epoch}"
        sections.append(
            f"""    <section class="sample-row">
      <h3>{html.escape(title)}</h3>
      <div class="sample-strip">
{''.join(image_figure_html(out_dir, path) for path in images)}
      </div>
    </section>"""
        )
    return "\n".join(sections) or "    <p>No ranked visualizations written yet.</p>"


def write_live_metrics_html(out_dir: Path) -> None:
    run_title = html.escape(out_dir.name.replace("_", " ").title())
    prune_duplicate_visualizations(out_dir / "viz")
    prune_duplicate_visualizations(out_dir / "viz3d")
    overlay_items = build_grouped_gallery(out_dir, out_dir / "viz")
    pose3d_items = build_grouped_gallery(out_dir, out_dir / "viz3d")
    ranked_overlay_items = build_ranked_gallery(out_dir, out_dir / "ranked_viz")
    ranked_pose3d_items = build_ranked_gallery(out_dir, out_dir / "ranked_viz3d")

    page = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="3">
  <title>{run_title} - EgoVerse ViT Metrics</title>
  <style>
    body {{ margin: 24px; font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: #f8fafc; color: #111827; }}
    h1, h2 {{ margin: 0 0 12px; }}
    h3 {{ margin: 0 0 4px; font-size: 13px; color: #334155; }}
    section {{ margin-top: 28px; }}
    .chart {{ max-width: min(1100px, 100%); border: 1px solid #cbd5e1; background: white; }}
    .gallery {{ display: grid; gap: 14px; max-width: 100%; }}
    .sample-row {{ margin-top: 0; }}
    .sample-strip {{ display: flex; gap: 0; overflow-x: auto; padding-bottom: 2px; }}
    figure {{ margin: 0; padding: 3px; border: 1px solid #cbd5e1; background: white; }}
    figure + figure {{ border-left: 0; }}
    figure img {{ width: 160px; display: block; image-rendering: auto; }}
    figcaption {{ margin-top: 3px; width: 160px; max-height: 22px; overflow: hidden; font-size: 8px; line-height: 1.15; color: #475569; overflow-wrap: anywhere; }}
  </style>
</head>
<body>
  <h1>{run_title}</h1>
  <p>Auto-refreshes every 3 seconds; the file is rewritten on train-step logs, visualizations, and epoch summaries.</p>
  <section>
    <h2>Loss / MPJPE</h2>
    <img class="chart" src="loss_curve.png" alt="Loss curve">
  </section>
  <section>
    <h2>Overlay Images</h2>
    <p>Fixed examples from the test split; each row is the same sample across epochs and steps.</p>
    <div class="gallery">
{overlay_items}
    </div>
  </section>
  <section>
    <h2>3D Hand Poses</h2>
    <p>Green/blue are ground truth hands; red/magenta are current model predictions.</p>
    <div class="gallery">
{pose3d_items}
    </div>
  </section>
  <section>
    <h2>Ranked Test Overlays</h2>
    <p>Bottom buckets are lowest MPJPE examples; top buckets are highest MPJPE examples from epoch-end test inference.</p>
    <div class="gallery">
{ranked_overlay_items}
    </div>
  </section>
  <section>
    <h2>Ranked Test 3D Poses</h2>
    <p>3D renderings for the same ranked examples.</p>
    <div class="gallery">
{ranked_pose3d_items}
    </div>
  </section>
</body>
</html>
"""
    (out_dir / "metrics_live.html").write_text(page)


def maybe_open_live_plot(out_dir: Path, enabled: bool) -> None:
    if not enabled:
        return
    html_path = (out_dir / "metrics_live.html").resolve()
    try:
        subprocess.Popen(["open", str(html_path)])
    except Exception:
        print(f"Open metrics live page manually: {html_path}")


def update_metric_plot(history: list[dict], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    test_loss = [row["test"]["loss"] for row in history]
    test_mpjpe = [row["test"]["mpjpe_mm"] for row in history]
    train_steps = read_jsonl(out_dir / "train_steps.jsonl")
    train_x = []
    train_loss = []
    train_mpjpe = []
    for row in train_steps:
        total_steps = max(1, int(row.get("total_steps", 1)))
        epoch = int(row.get("epoch", 1))
        step = int(row.get("step", 0))
        train_x.append(epoch - 1 + step / total_steps)
        train_loss.append(float(row.get("running_loss", row.get("loss", 0.0))))
        train_mpjpe.append(float(row.get("running_mpjpe_mm", row.get("mpjpe_mm", 0.0))))

    epoch_train_loss = [row["train"]["loss"] for row in history]
    epoch_train_mpjpe = [row["train"]["mpjpe_mm"] for row in history]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=140)
    if train_x:
        axes[0, 0].plot(train_x, train_loss, linewidth=1.8, label="train running")
    elif epochs:
        axes[0, 0].plot(epochs, epoch_train_loss, marker="o", label="train")
    axes[0, 0].set_title("Train SmoothL1 Loss")
    axes[0, 0].set_xlabel("epoch")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    if train_x:
        axes[0, 1].plot(train_x, train_mpjpe, linewidth=1.8, label="train running")
    elif epochs:
        axes[0, 1].plot(epochs, epoch_train_mpjpe, marker="o", label="train")
    axes[0, 1].set_title("Train MPJPE (mm)")
    axes[0, 1].set_xlabel("epoch")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(epochs, test_loss, marker="o", linewidth=1.8, label="test epoch")
    axes[1, 0].set_title("Test SmoothL1 Loss")
    axes[1, 0].set_xlabel("epoch")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(epochs, test_mpjpe, marker="o", linewidth=1.8, label="test epoch")
    axes[1, 1].set_title("Test MPJPE (mm)")
    axes[1, 1].set_xlabel("epoch")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "loss_curve.png")
    plt.close(fig)


def set_equal_3d_axes(ax, points: np.ndarray) -> None:
    finite_points = points[np.isfinite(points).all(axis=1)]
    if finite_points.size == 0:
        return
    mins = finite_points.min(axis=0)
    maxs = finite_points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float((maxs - mins).max()) / 2.0, 0.05)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def draw_hand_3d(ax, points: np.ndarray, mask: np.ndarray, color: str, label: str) -> None:
    plotted_label = False
    for i, j in HAND_EDGES:
        if not (mask[i] and mask[j]):
            continue
        segment = points[[i, j]]
        ax.plot(
            segment[:, 0],
            segment[:, 1],
            segment[:, 2],
            color=color,
            linewidth=1.8,
            label=label if not plotted_label else None,
        )
        plotted_label = True
    valid_points = points[mask]
    if valid_points.size:
        ax.scatter(valid_points[:, 0], valid_points[:, 1], valid_points[:, 2], color=color, s=10)


def save_3d_hand_pose_plot(
    target: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray,
    out_path: Path,
    *,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(4, 4), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    draw_hand_3d(ax, target[0], mask[0], "#22c55e", "GT left")
    draw_hand_3d(ax, target[1], mask[1], "#38bdf8", "GT right")
    draw_hand_3d(ax, pred[0], mask[0], "#ef4444", "Pred left")
    draw_hand_3d(ax, pred[1], mask[1], "#d946ef", "Pred right")
    valid_target = target[mask]
    valid_pred = pred[mask]
    points = np.concatenate([valid_target, valid_pred], axis=0) if valid_target.size or valid_pred.size else target.reshape(-1, 3)
    set_equal_3d_axes(ax, points)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=18, azim=-65)
    ax.legend(fontsize=6, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def make_viz_batch(dataset: EgoVerseHandPoseDataset, sample_count: int) -> dict:
    if sample_count <= 0:
        raise ValueError("--viz-samples must be positive when --viz-every is enabled")

    episode_to_indices: dict[str, list[int]] = {}
    for idx, row in enumerate(dataset.rows):
        episode_to_indices.setdefault(row["episode_hash"], []).append(idx)

    episodes = sorted(episode_to_indices)
    if not episodes:
        raise ValueError("Cannot build visualization batch from an empty dataset")

    if len(episodes) >= sample_count:
        positions = np.linspace(0, len(episodes) - 1, sample_count, dtype=int)
        selected_indices = []
        for pos in positions:
            indices = episode_to_indices[episodes[int(pos)]]
            selected_indices.append(indices[len(indices) // 2])
    else:
        positions = np.linspace(0, len(dataset) - 1, sample_count, dtype=int)
        selected_indices = [int(pos) for pos in positions]

    return default_collate([dataset[idx] for idx in selected_indices])


@torch.no_grad()
def save_visualizations(
    model: nn.Module,
    batch: dict,
    device: torch.device,
    out_dir: Path,
    *,
    epoch: int,
    step: int | None = None,
    max_samples: int,
    gt_radius: int,
    pred_radius: int,
    pose3d_dir: Path | None = None,
    sample_labels: list[str] | None = None,
    title_labels: list[str] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pose3d_dir = pose3d_dir or out_dir.parent / "viz3d"
    pose3d_dir.mkdir(parents=True, exist_ok=True)
    was_training = model.training
    model.eval()
    image = batch["image"].to(device)
    target = batch["keypoints"].to(device)
    mask = batch["valid_mask"].to(device)
    pred = model(image)
    if isinstance(pred, tuple):
        pred = pred[0]

    image_cpu = batch["image"].detach().cpu()
    pred_cpu = pred.detach().cpu().numpy()
    target_cpu = target.detach().cpu().numpy()
    mask_cpu = mask.detach().cpu().numpy()
    n = min(max_samples, image_cpu.shape[0])
    for i in range(n):
        rgb = image_cpu[i] * IMAGENET_STD + IMAGENET_MEAN
        rgb = (rgb.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        vis = np.ascontiguousarray(rgb.copy())
        draw_hand(vis, target_cpu[i, 0], mask_cpu[i, 0], (40, 220, 40), gt_radius, 1)
        draw_hand(vis, target_cpu[i, 1], mask_cpu[i, 1], (40, 160, 255), gt_radius, 1)
        draw_hand(vis, pred_cpu[i, 0], mask_cpu[i, 0], (255, 40, 40), pred_radius, 1)
        draw_hand(vis, pred_cpu[i, 1], mask_cpu[i, 1], (255, 80, 220), pred_radius, 1)
        episode = batch["episode_hash"][i]
        frame = int(batch["frame_idx"][i])
        step_label = f"_step_{step:04d}" if step is not None else ""
        sample_label = sample_labels[i] if sample_labels is not None else f"sample_{i:02d}"
        title_label = title_labels[i] if title_labels is not None else f"sample {i}"
        out = out_dir / f"epoch_{epoch:03d}{step_label}_{sample_label}_{episode}_{frame}.png"
        cv2.imwrite(str(out), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        pose3d_out = pose3d_dir / out.name
        save_3d_hand_pose_plot(
            target_cpu[i],
            pred_cpu[i],
            mask_cpu[i],
            pose3d_out,
            title=f"epoch {epoch} {title_label}",
        )
    if was_training:
        model.train()


def filename_float(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def percentile_bucket_label(percentile: float) -> str:
    return f"{percentile:g}".replace(".", "p") + "pct"


def ranked_bucket_size(total_samples: int, percentile: float, max_samples: int) -> int:
    if total_samples <= 0:
        return 0
    size = max(1, math.ceil(total_samples * (percentile / 100.0)))
    if max_samples > 0:
        size = min(size, max_samples)
    return min(size, total_samples)


def candidate_respects_frame_gap(row: dict, selected_rows: list[dict], min_frame_gap: int) -> bool:
    if min_frame_gap <= 0:
        return True
    episode = row["episode_hash"]
    frame_idx = int(row["frame_idx"])
    for selected in selected_rows:
        if selected["episode_hash"] != episode:
            continue
        if abs(frame_idx - int(selected["frame_idx"])) < min_frame_gap:
            return False
    return True


def select_diverse_ranked_metrics(
    sorted_metrics: list[dict],
    *,
    size: int,
    max_per_episode: int,
    min_frame_gap: int,
) -> list[dict]:
    if size <= 0:
        return []
    if max_per_episode < 0:
        raise ValueError("max_per_episode must be >= 0")
    if min_frame_gap < 0:
        raise ValueError("min_frame_gap must be >= 0")

    selected: list[dict] = []
    selected_by_episode: dict[str, int] = {}
    max_rounds = max_per_episode if max_per_episode > 0 else size
    for round_idx in range(max_rounds):
        made_progress = False
        for row in sorted_metrics:
            if len(selected) >= size:
                return selected
            episode = row["episode_hash"]
            if selected_by_episode.get(episode, 0) != round_idx:
                continue
            if not candidate_respects_frame_gap(row, selected, min_frame_gap):
                continue
            selected.append(row)
            selected_by_episode[episode] = round_idx + 1
            made_progress = True
        if not made_progress:
            break
    return selected


def select_ranked_sample_metrics(
    sample_metrics: list[dict],
    *,
    percentile: float,
    max_samples: int,
    max_per_episode: int,
    min_frame_gap: int,
) -> dict[str, list[dict]]:
    valid_metrics = [
        row for row in sample_metrics
        if math.isfinite(float(row["mpjpe_mm"]))
    ]
    if not valid_metrics:
        return {}
    sorted_metrics = sorted(valid_metrics, key=lambda row: float(row["mpjpe_mm"]))
    size = ranked_bucket_size(len(sorted_metrics), percentile, max_samples)
    if size <= 0:
        return {}
    label = percentile_bucket_label(percentile)
    best = select_diverse_ranked_metrics(
        sorted_metrics,
        size=size,
        max_per_episode=max_per_episode,
        min_frame_gap=min_frame_gap,
    )
    worst = select_diverse_ranked_metrics(
        list(reversed(sorted_metrics)),
        size=size,
        max_per_episode=max_per_episode,
        min_frame_gap=min_frame_gap,
    )
    return {
        f"bottom_{label}_best": best,
        f"top_{label}_worst": worst,
    }


def append_ranked_sample_index(
    out_dir: Path,
    epoch: int,
    ranked: dict[str, list[dict]],
    *,
    percentile: float,
    max_per_episode: int,
    min_frame_gap: int,
) -> None:
    index_path = out_dir / "ranked_viz" / "ranked_samples.jsonl"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("a") as f:
        for bucket, rows in ranked.items():
            for rank, row in enumerate(rows, start=1):
                payload = {
                    "epoch": epoch,
                    "bucket": bucket,
                    "rank": rank,
                    "percentile": percentile,
                    "max_per_episode": max_per_episode,
                    "min_frame_gap": min_frame_gap,
                    **row,
                }
                f.write(json.dumps(payload) + "\n")


@torch.no_grad()
def save_ranked_visualizations(
    model: nn.Module,
    dataset: EgoVerseHandPoseDataset,
    sample_metrics: list[dict],
    device: torch.device,
    out_dir: Path,
    *,
    epoch: int,
    percentile: float,
    max_samples: int,
    max_per_episode: int,
    min_frame_gap: int,
    gt_radius: int,
    pred_radius: int,
) -> None:
    ranked = select_ranked_sample_metrics(
        sample_metrics,
        percentile=percentile,
        max_samples=max_samples,
        max_per_episode=max_per_episode,
        min_frame_gap=min_frame_gap,
    )
    if not ranked:
        return

    append_ranked_sample_index(
        out_dir,
        epoch,
        ranked,
        percentile=percentile,
        max_per_episode=max_per_episode,
        min_frame_gap=min_frame_gap,
    )
    for bucket, rows in ranked.items():
        batch = default_collate([dataset[int(row["row_idx"])] for row in rows])
        sample_labels = [
            f"rank_{rank:02d}_sample_{int(row['row_idx']):06d}_mpjpe_{filename_float(float(row['mpjpe_mm']))}mm"
            for rank, row in enumerate(rows, start=1)
        ]
        title_labels = [
            f"{bucket.replace('_', ' ')} rank {rank} MPJPE {float(row['mpjpe_mm']):.2f} mm"
            for rank, row in enumerate(rows, start=1)
        ]
        save_visualizations(
            model,
            batch,
            device,
            out_dir / "ranked_viz" / bucket,
            epoch=epoch,
            max_samples=len(rows),
            gt_radius=gt_radius,
            pred_radius=pred_radius,
            pose3d_dir=out_dir / "ranked_viz3d" / bucket,
            sample_labels=sample_labels,
            title_labels=title_labels,
        )


def make_loader(
    csv_path: str,
    args: argparse.Namespace,
    *,
    train: bool,
    distributed: bool,
    max_rows: int | None = None,
) -> tuple[EgoVerseHandPoseDataset, DataLoader, DistributedSampler | None]:
    dataset = EgoVerseHandPoseDataset(csv_path, image_size=args.image_size, max_rows=max_rows)
    sampler = DistributedSampler(dataset, shuffle=train) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=train and sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    return dataset, loader, sampler


def evenly_spaced_steps(total_steps: int, count: int) -> set[int]:
    if total_steps <= 0 or count <= 0:
        return set()
    count = min(count, total_steps)
    return {max(1, int(round(step))) for step in np.linspace(0, total_steps, count + 1)[1:]}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    epoch: int,
    max_steps: int | None,
    log_every_steps: int,
    step_metrics_path: Path | None,
    progress: bool,
    metrics_history: list[dict] | None = None,
    viz_steps: set[int] | None = None,
    viz_callback: Callable[[int], None] | None = None,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_mpjpe = 0.0
    window_loss = 0.0
    window_mpjpe = 0.0
    window_steps = 0
    steps = 0
    total_steps = len(loader)
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)
    iterator = loader
    if progress:
        iterator = tqdm(loader, total=total_steps, desc=f"epoch {epoch} train", dynamic_ncols=True)
    for batch in iterator:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["keypoints"].to(device, non_blocking=True)
        mask = batch["valid_mask"].to(device, non_blocking=True)
        pred = model(image)
        loss = masked_smooth_l1(pred, target, mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        batch_loss = float(loss.detach().cpu())
        batch_mpjpe = float(mpjpe_mm(pred.detach(), target, mask).cpu())
        total_loss += batch_loss
        total_mpjpe += batch_mpjpe
        steps += 1
        window_loss += batch_loss
        window_mpjpe += batch_mpjpe
        window_steps += 1
        if progress:
            iterator.set_postfix(loss=f"{total_loss / max(1, steps):.4f}", mpjpe=f"{total_mpjpe / max(1, steps):.1f}")
        should_log = log_every_steps > 0 and (steps % log_every_steps == 0 or steps == total_steps)
        if should_log and step_metrics_path is not None:
            row = {
                "epoch": epoch,
                "step": steps,
                "total_steps": total_steps,
                "loss": window_loss / max(1, window_steps),
                "mpjpe_mm": window_mpjpe / max(1, window_steps),
                "running_loss": total_loss / max(1, steps),
                "running_mpjpe_mm": total_mpjpe / max(1, steps),
            }
            print(json.dumps({"train_step": row}), flush=True)
            with step_metrics_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            if metrics_history is not None:
                update_metric_plot(metrics_history, step_metrics_path.parent)
            write_live_metrics_html(step_metrics_path.parent)
            window_loss = 0.0
            window_mpjpe = 0.0
            window_steps = 0
        if viz_steps is not None and viz_callback is not None and steps in viz_steps:
            viz_callback(steps)
        if max_steps is not None and steps >= max_steps:
            break
    return {"loss": total_loss / max(1, steps), "mpjpe_mm": total_mpjpe / max(1, steps), "steps": steps}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    max_steps: int | None,
    collect_sample_metrics: bool = False,
) -> tuple[dict[str, float], list[dict]]:
    model.eval()
    total_loss = 0.0
    total_mpjpe = 0.0
    steps = 0
    sample_metrics: list[dict] = []
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["keypoints"].to(device, non_blocking=True)
        mask = batch["valid_mask"].to(device, non_blocking=True)
        pred = model(image)
        total_loss += float(masked_smooth_l1(pred, target, mask).cpu())
        total_mpjpe += float(mpjpe_mm(pred, target, mask).cpu())
        if collect_sample_metrics:
            sample_errors = per_sample_mpjpe_mm(pred, target, mask).detach().cpu().numpy()
            row_indices = batch["row_idx"].detach().cpu().numpy()
            for i, sample_mpjpe in enumerate(sample_errors):
                sample_metrics.append(
                    {
                        "row_idx": int(row_indices[i]),
                        "episode_hash": batch["episode_hash"][i],
                        "frame_idx": int(batch["frame_idx"][i]),
                        "mpjpe_mm": float(sample_mpjpe),
                    }
                )
        steps += 1
        if max_steps is not None and steps >= max_steps:
            break
    metrics = {"loss": total_loss / max(1, steps), "mpjpe_mm": total_mpjpe / max(1, steps), "steps": steps}
    return metrics, sample_metrics


def main() -> None:
    args = parse_args()
    if args.ranked_viz_every < 0:
        raise ValueError("--ranked-viz-every must be >= 0")
    if args.ranked_viz_every > 0 and not (0.0 < args.ranked_viz_percentile <= 50.0):
        raise ValueError("--ranked-viz-percentile must be > 0 and <= 50")
    if args.ranked_viz_max_samples < 0:
        raise ValueError("--ranked-viz-max-samples must be >= 0")
    if args.ranked_viz_max_per_episode < 0:
        raise ValueError("--ranked-viz-max-per-episode must be >= 0")
    if args.ranked_viz_min_frame_gap < 0:
        raise ValueError("--ranked-viz-min-frame-gap must be >= 0")
    head_hidden_dims = parse_head_hidden_dims(args.head_hidden_dims)
    args.parsed_head_hidden_dims = list(head_hidden_dims)
    distributed, rank, _, local_rank = setup_distributed(args.distributed)
    out_dir = resolve_out_dir(args, distributed=distributed, rank=rank, allocate=not args.dry_run)
    args.resolved_out_dir = str(out_dir)
    device = torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    )
    max_rows = args.batch_size * args.overfit_batches if args.overfit_batches else None
    train_dataset, train_loader, train_sampler = make_loader(
        args.train_csv, args, train=True, distributed=distributed, max_rows=max_rows
    )
    test_dataset, test_loader, _ = make_loader(
        args.test_csv, args, train=False, distributed=distributed, max_rows=max_rows
    )
    model = ViTHandPose(
        args.model_name,
        pretrained=args.pretrained,
        backbone_source=args.backbone_source,
        freeze_backbone=args.freeze_backbone,
        head_type=args.head_type,
        head_hidden_dims=head_hidden_dims,
        head_dropout=args.head_dropout,
    ).to(device)
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank] if torch.cuda.is_available() else None)

    if args.dry_run:
        if is_rank0(rank):
            print(
                json.dumps(
                    {
                        "device": str(device),
                        "train_rows": len(train_dataset),
                        "test_rows": len(test_dataset),
                        "model_name": args.model_name,
                        "backbone_source": args.backbone_source,
                        "pretrained": args.pretrained,
                        "freeze_backbone": args.freeze_backbone,
                        "head_type": args.head_type,
                        "head_hidden_dims": args.parsed_head_hidden_dims,
                        "head_dropout": args.head_dropout,
                        "image_size": args.image_size,
                        "out_dir": str(out_dir),
                    },
                    indent=2,
                )
            )
        if distributed:
            dist.destroy_process_group()
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch = 0
    if args.resume:
        checkpoint_path = Path(args.resume).expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        load_model_state(model, checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", checkpoint.get("metrics", {}).get("epoch", 0)))
        if is_rank0(rank):
            print(f"Resumed from {checkpoint_path} at epoch {start_epoch}")

    if is_rank0(rank):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (out_dir / "viz").mkdir(parents=True, exist_ok=True)
        (out_dir / "viz3d").mkdir(parents=True, exist_ok=True)
        (out_dir / "ranked_viz").mkdir(parents=True, exist_ok=True)
        (out_dir / "ranked_viz3d").mkdir(parents=True, exist_ok=True)
        config_path = out_dir / "config.json"
        if not args.resume or not config_path.exists():
            config_path.write_text(json.dumps(vars(args), indent=2) + "\n")
        update_metric_plot(read_metrics_history(out_dir / "metrics.jsonl"), out_dir)
        write_live_metrics_html(out_dir)
        maybe_open_live_plot(out_dir, args.open_live_plot)
        metrics_path = out_dir / "metrics.jsonl"
        step_metrics_path = out_dir / "train_steps.jsonl"
    else:
        metrics_path = None
        step_metrics_path = None

    viz_batch = None
    if args.viz_every > 0 and is_rank0(rank):
        viz_batch = make_viz_batch(test_dataset, args.viz_samples)

    metrics_history = read_metrics_history(metrics_path) if metrics_path is not None else []
    best_test_mpjpe = min(
        (float(row["test"]["mpjpe_mm"]) for row in metrics_history if "test" in row and "mpjpe_mm" in row["test"]),
        default=float("inf"),
    )
    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        should_viz_epoch = (
            viz_batch is not None
            and args.viz_every > 0
            and (epoch + 1) % args.viz_every == 0
            and is_rank0(rank)
        )
        should_ranked_viz_epoch = (
            args.ranked_viz_every > 0
            and (epoch + 1) % args.ranked_viz_every == 0
            and is_rank0(rank)
        )
        viz_steps = None
        viz_callback = None
        if should_viz_epoch and args.viz_per_epoch > 0:
            total_train_steps = len(train_loader)
            if args.max_steps is not None:
                total_train_steps = min(total_train_steps, args.max_steps)
            viz_steps = evenly_spaced_steps(total_train_steps, args.viz_per_epoch)

            def viz_callback(step: int, *, current_epoch: int = epoch + 1) -> None:
                save_visualizations(
                    model.module if isinstance(model, DistributedDataParallel) else model,
                    viz_batch,
                    device,
                    out_dir / "viz",
                    epoch=current_epoch,
                    step=step,
                    max_samples=args.viz_samples,
                    gt_radius=args.gt_radius,
                    pred_radius=args.pred_radius,
                )
                write_live_metrics_html(out_dir)

        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch=epoch + 1,
            max_steps=args.max_steps,
            log_every_steps=args.log_every_steps,
            step_metrics_path=step_metrics_path,
            progress=args.progress and is_rank0(rank),
            metrics_history=metrics_history,
            viz_steps=viz_steps,
            viz_callback=viz_callback,
        )
        test_metrics, test_sample_metrics = evaluate(
            model,
            test_loader,
            device,
            max_steps=args.max_eval_steps,
            collect_sample_metrics=should_ranked_viz_epoch,
        )
        if is_rank0(rank):
            metrics = {"epoch": epoch + 1, "train": train_metrics, "test": test_metrics}
            metrics_history.append(metrics)
            print(json.dumps(metrics, indent=2))
            payload = {
                "model": model_state_dict(model),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "metrics": metrics,
                "args": vars(args),
            }
            torch.save(payload, out_dir / "last.pt")
            if float(test_metrics["mpjpe_mm"]) < best_test_mpjpe:
                best_test_mpjpe = float(test_metrics["mpjpe_mm"])
                torch.save(payload, out_dir / "best.pt")
            if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
                checkpoint_dir = out_dir / "checkpoints"
                torch.save(payload, checkpoint_dir / f"epoch_{epoch + 1:03d}.pt")
                prune_epoch_checkpoints(checkpoint_dir, args.keep_checkpoints)
            if metrics_path is not None:
                with metrics_path.open("a") as f:
                    f.write(json.dumps(metrics) + "\n")
            if args.plot_every > 0 and (epoch + 1) % args.plot_every == 0:
                update_metric_plot(metrics_history, out_dir)
            if should_viz_epoch and args.viz_per_epoch <= 0:
                save_visualizations(
                    model.module if isinstance(model, DistributedDataParallel) else model,
                    viz_batch,
                    device,
                    out_dir / "viz",
                    epoch=epoch + 1,
                    max_samples=args.viz_samples,
                    gt_radius=args.gt_radius,
                    pred_radius=args.pred_radius,
                )
            if should_ranked_viz_epoch:
                save_ranked_visualizations(
                    model.module if isinstance(model, DistributedDataParallel) else model,
                    test_dataset,
                    test_sample_metrics,
                    device,
                    out_dir,
                    epoch=epoch + 1,
                    percentile=args.ranked_viz_percentile,
                    max_samples=args.ranked_viz_max_samples,
                    max_per_episode=args.ranked_viz_max_per_episode,
                    min_frame_gap=args.ranked_viz_min_frame_gap,
                    gt_radius=args.gt_radius,
                    pred_radius=args.pred_radius,
                )
            write_live_metrics_html(out_dir)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
