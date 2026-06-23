#!/usr/bin/env python3
"""Cache EgoVerse episodes referenced by training manifests."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from egoverse_handpose_viewer import sync_episode  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache EgoVerse episodes referenced by manifest CSVs.")
    parser.add_argument("csv", nargs="+", help="Manifest CSV paths.")
    parser.add_argument("--cache-dir", default="/data/egoverse_cache")
    parser.add_argument("--max-rows", type=int, default=None, help="Only inspect the first N rows per CSV.")
    parser.add_argument("--max-episodes", type=int, default=None, help="Only sync the first N unique episodes.")
    return parser.parse_args()


def episode_hashes(csv_paths: list[str], max_rows: int | None) -> list[str]:
    seen: set[str] = set()
    hashes: list[str] = []
    for csv_path in csv_paths:
        with Path(csv_path).open(newline="") as f:
            for row_idx, row in enumerate(csv.DictReader(f)):
                if max_rows is not None and row_idx >= max_rows:
                    break
                episode_hash = row.get("episode_hash", "").strip()
                if episode_hash and episode_hash not in seen:
                    seen.add(episode_hash)
                    hashes.append(episode_hash)
    return hashes


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    hashes = episode_hashes(args.csv, args.max_rows)
    if args.max_episodes is not None:
        hashes = hashes[: args.max_episodes]
    if not hashes:
        raise SystemExit("No episode_hash values found in the provided manifest CSVs.")

    for idx, episode_hash in enumerate(hashes, start=1):
        print(f"[{idx}/{len(hashes)}] caching {episode_hash}", flush=True)
        path = sync_episode(cache_dir, episode_hash)
        print(f"cached {episode_hash} at {path}", flush=True)


if __name__ == "__main__":
    main()
