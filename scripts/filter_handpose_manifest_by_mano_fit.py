#!/usr/bin/env python3
"""Filter hand-pose manifests using MANO fit residuals."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SIDE_HAS_KEY = {"left": "has_left", "right": "has_right"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drop hand labels whose MANO fit residual is too high.")
    parser.add_argument("--train-csv", default="outputs/handpose_dataset_visible/train.csv")
    parser.add_argument("--test-csv", default="outputs/handpose_dataset_visible/test.csv")
    parser.add_argument("--mano-fit-csv", default="outputs/mano_fit_audit/mano_fit_samples.csv")
    parser.add_argument("--out-dir", default="outputs/handpose_dataset_visible_mano")
    parser.add_argument("--max-mano-mpjpe-mm", type=float, default=15.0)
    return parser.parse_args()


def str_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_failed_hands(path: Path, max_mpjpe_mm: float) -> tuple[set[tuple[str, int, str]], dict[str, Any]]:
    failed: set[tuple[str, int, str]] = set()
    totals = Counter()
    by_split = defaultdict(Counter)
    for row in load_rows(path):
        split = row["split"]
        side = row["side"]
        row_idx = int(row["row_idx"])
        mpjpe = float(row["mano_mpjpe_mm"])
        is_failure = mpjpe > max_mpjpe_mm
        totals["hands"] += 1
        totals["failed_hands"] += int(is_failure)
        by_split[split]["hands"] += 1
        by_split[split]["failed_hands"] += int(is_failure)
        if is_failure:
            failed.add((split, row_idx, side))
    return failed, {"totals": dict(totals), "by_split": {split: dict(counts) for split, counts in by_split.items()}}


def filter_split(
    rows: list[dict[str, str]],
    *,
    split: str,
    failed_hands: set[tuple[str, int, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    kept_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    totals = Counter()
    per_episode = defaultdict(Counter)

    for row_idx, row in enumerate(rows):
        output = dict(row)
        labeled_before = 0
        labeled_after = 0
        failed_sides = []
        for side, has_key in SIDE_HAS_KEY.items():
            was_labeled = str_bool(row.get(has_key))
            labeled_before += int(was_labeled)
            failed = (split, row_idx, side) in failed_hands
            if was_labeled and failed:
                output[has_key] = "False"
                failed_sides.append(side)
            labeled_after += int(str_bool(output.get(has_key)))

        keep_row = labeled_after > 0
        if keep_row:
            kept_rows.append(output)

        episode_hash = row["episode_hash"]
        totals["input_rows"] += 1
        totals["kept_rows"] += int(keep_row)
        totals["dropped_rows"] += int(not keep_row)
        totals["input_labeled_hands"] += labeled_before
        totals["kept_labeled_hands"] += labeled_after
        totals["dropped_labeled_hands"] += labeled_before - labeled_after
        totals["rows_with_mano_failure"] += int(bool(failed_sides))
        per_episode[episode_hash]["input_rows"] += 1
        per_episode[episode_hash]["kept_rows"] += int(keep_row)
        per_episode[episode_hash]["dropped_rows"] += int(not keep_row)
        per_episode[episode_hash]["dropped_labeled_hands"] += labeled_before - labeled_after

        audit_rows.append(
            {
                "split": split,
                "row_idx": row_idx,
                "episode_hash": episode_hash,
                "frame_idx": row["frame_idx"],
                "failed_sides": "|".join(failed_sides),
                "labeled_hands_before": labeled_before,
                "labeled_hands_after": labeled_after,
                "kept_row": keep_row,
            }
        )

    return (
        kept_rows,
        {
            "split": split,
            **dict(totals),
            "per_episode": {episode: dict(counts) for episode, counts in sorted(per_episode.items())},
        },
        audit_rows,
    )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    train_rows = load_rows(Path(args.train_csv))
    test_rows = load_rows(Path(args.test_csv))
    failed_hands, mano_summary = load_failed_hands(Path(args.mano_fit_csv), args.max_mano_mpjpe_mm)

    train_out, train_summary, train_audit = filter_split(train_rows, split="train", failed_hands=failed_hands)
    test_out, test_summary, test_audit = filter_split(test_rows, split="test", failed_hands=failed_hands)

    if not train_rows or not test_rows:
        raise SystemExit("Input train/test CSVs must be non-empty.")
    write_rows(out_dir / "train.csv", train_out, list(train_rows[0].keys()))
    write_rows(out_dir / "test.csv", test_out, list(test_rows[0].keys()))
    write_rows(
        out_dir / "mano_filter_audit.csv",
        train_audit + test_audit,
        [
            "split",
            "row_idx",
            "episode_hash",
            "frame_idx",
            "failed_sides",
            "labeled_hands_before",
            "labeled_hands_after",
            "kept_row",
        ],
    )
    summary = {
        "args": vars(args),
        "mano_fit_input": mano_summary,
        "train": train_summary,
        "test": test_summary,
    }
    (out_dir / "mano_filter_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Wrote MANO-filtered manifests to {out_dir}")


if __name__ == "__main__":
    main()
