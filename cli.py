#!/usr/bin/env python3
"""Kuku Promo Finder Studio — Command Line Batch Runner (`cli.py`).

Usage:
  python cli.py
  python cli.py --provider semantic_reranker --output-dir raw_outputs
  python cli.py --provider gemini --model gemini-2.5-flash
  python cli.py --provider heuristic
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

from src.benchmark import run_full_pipeline_and_export
from src.candidates import build_all_candidates
from src.parser import load_episodes_from_zip, load_synthetic_demo_episodes

DEFAULT_ZIP = Path("I Married a Broken Soldier. He was a Duke-scripts (2).zip")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Kuku Promo Finder Studio — 50-Episode Microdrama Promo Extraction Pipeline"
    )
    parser.add_argument(
        "--zip",
        type=str,
        default=str(DEFAULT_ZIP),
        help="Path to screenplay .zip archive (falls back to synthetic demo if absent)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="semantic_reranker",
        choices=["semantic_reranker", "heuristic", "gemini", "openai", "anthropic", "cli"],
        help="Stage-2 scoring provider (default: semantic_reranker)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional explicit model identifier override",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="raw_outputs",
        help="Output directory for generated JSON, CSV, and Markdown deliverables",
    )
    args = parser.parse_args()

    t0 = time.time()
    zip_path = Path(args.zip)
    if zip_path.exists():
        print(f"[1/3] Loading 50-episode screenplay pack from: {zip_path}")
        episodes = load_episodes_from_zip(zip_path)
    else:
        print(f"[1/3] Archive {zip_path} not found; loading synthetic demo fixture.")
        episodes = load_synthetic_demo_episodes()

    total_cues = sum(len(ep.cues) for ep in episodes)
    total_dur_min = sum(ep.total_duration_sec for ep in episodes) / 60.0
    print(
        f"      Parsed {len(episodes)} episodes | {sum(len(e.scenes) for e in episodes)} scenes | "
        f"{total_cues} concurrent cues | {total_dur_min:.1f} mins total runtime."
    )

    print("[2/3] Running Stage-1 dialogue-safe 20.0s-60.0s window generator + NMS + multi-segment stitcher...")
    candidates, lore_tokens = build_all_candidates(
        episodes, max_single_per_episode=3, include_stitched=True
    )
    print(
        f"      Generated {len(candidates)} deduplicated candidates across {len(episodes)} episodes "
        f"({len(lore_tokens)} dynamic speaker lore tokens tracked)."
    )

    out_dir = Path(args.output_dir)
    print(f"[3/3] Running Stage-2 judge (`{args.provider}`) and exporting deliverables to `{out_dir}/`...")
    results = run_full_pipeline_and_export(
        episodes=episodes,
        raw_candidates=candidates,
        provider=args.provider,
        model_name=args.model,
        out_dir=out_dir,
    )

    top20 = results["top20"]
    dt = time.time() - t0
    print(f"\n=== PIPELINE COMPLETE in {dt:.2f}s ===")
    print(f"Top-5 Ranked Promos across {len(episodes)} episodes:")
    for idx, c in enumerate(top20[:5], start=1):
        print(
            f"  #{idx:02d} | {c.timestamp_range_display:<38} | {c.duration_sec:5.1f}s | "
            f"Score: {c.final_score:5.2f} | Type: {c.promo_type} | Payoff: {c.payoff_source}"
        )
    print(f"\nDeliverables written to `{out_dir.resolve()}`:")
    for p in sorted(out_dir.iterdir()):
        print(f"  - {p.name} ({p.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
