#!/usr/bin/env python3
"""Simple ViT baseline for EgoVerse hand-pose regression."""

from __future__ import annotations

import argparse
import html
import json
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

VIZ_SAMPLE_RE = re.compile(r"sample_(\d+)")

HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)


class ViTHandPose(nn.Module):
    def __init__(self, model_name: str, pretrained: bool):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.head = nn.Linear(self.backbone.num_features, 2 * 21 * 3)

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
    parser.add_argument("--pretrained", action="store_true")
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
    parser.add_argument("--gt-radius", type=int, default=1)
    parser.add_argument("--pred-radius", type=int, default=2)
    parser.add_argument("--plot-every", type=int, default=1, help="Update loss chart every N epochs; 0 disables.")
    parser.add_argument("--log-every-steps", type=int, default=0, help="Print/write train metrics every N batches; 0 disables.")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True, help="Show tqdm progress bars.")
    parser.add_argument("--open-live-plot", action="store_true", help="Open a local auto-refreshing metrics HTML page.")
    parser.add_argument("--save-every", type=int, default=1, help="Save epoch checkpoint every N epochs; 0 disables.")
    parser.add_argument("--resume", default=None, help="Resume from a checkpoint such as /runs/name/last.pt.")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Build data/model and exit without training.")
    return parser.parse_args()


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


def write_live_metrics_html(out_dir: Path) -> None:
    run_title = html.escape(out_dir.name.replace("_", " ").title())
    viz_images = sorted((out_dir / "viz").glob("*.png"))
    grouped_viz: dict[str, list[Path]] = {}
    for path in viz_images:
        match = VIZ_SAMPLE_RE.search(path.stem)
        sample_id = match.group(1) if match else "unknown"
        grouped_viz.setdefault(sample_id, []).append(path)
    gallery_items = "\n".join(
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
        for sample_id, paths in sorted(grouped_viz.items())
    )
    if not gallery_items:
        gallery_items = "    <p>No overlay images written yet.</p>"

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
{gallery_items}
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
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
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
        out = out_dir / f"epoch_{epoch:03d}{step_label}_sample_{i:02d}_{episode}_{frame}.png"
        cv2.imwrite(str(out), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    if was_training:
        model.train()


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
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_mpjpe = 0.0
    steps = 0
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["keypoints"].to(device, non_blocking=True)
        mask = batch["valid_mask"].to(device, non_blocking=True)
        pred = model(image)
        total_loss += float(masked_smooth_l1(pred, target, mask).cpu())
        total_mpjpe += float(mpjpe_mm(pred, target, mask).cpu())
        steps += 1
        if max_steps is not None and steps >= max_steps:
            break
    return {"loss": total_loss / max(1, steps), "mpjpe_mm": total_mpjpe / max(1, steps), "steps": steps}


def main() -> None:
    args = parse_args()
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
    model = ViTHandPose(args.model_name, pretrained=args.pretrained).to(device)
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
                        "pretrained": args.pretrained,
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
    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        should_viz_epoch = (
            viz_batch is not None
            and args.viz_every > 0
            and (epoch + 1) % args.viz_every == 0
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
        test_metrics = evaluate(model, test_loader, device, max_steps=args.max_eval_steps)
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
            if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
                torch.save(payload, out_dir / "checkpoints" / f"epoch_{epoch + 1:03d}.pt")
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
            write_live_metrics_html(out_dir)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
