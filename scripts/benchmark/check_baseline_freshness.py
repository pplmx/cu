#!/usr/bin/env python3
"""
Check baseline freshness and report staleness.

Usage:
    python scripts/benchmark/check_baseline_freshness.py --max-age 30
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser(description="Check baseline freshness")
    parser.add_argument(
        "--baselines-dir",
        type=Path,
        default=Path(__file__).parent / "baselines",
        help="Directory containing baselines",
    )
    parser.add_argument(
        "--max-age",
        type=int,
        default=30,
        help="Maximum age in days before warning",
    )
    parser.add_argument(
        "--fail",
        action="store_true",
        help="Exit with error if baselines are stale",
    )
    return parser.parse_args()


def check_baseline_freshness(baselines_dir: Path, max_age: int, fail: bool):
    """Check if baselines are fresh enough."""
    if not baselines_dir.exists():
        print(f"No baselines directory found: {baselines_dir}")
        if fail:
            return 1
        return 0

    stale_count = 0
    fresh_count = 0
    now = datetime.now()
    max_age_delta = timedelta(days=max_age)

    for version_dir in baselines_dir.iterdir():
        if not version_dir.is_dir():
            continue

        metadata_file = version_dir / "metadata.json"
        if not metadata_file.exists():
            print(f"⚠ {version_dir.name}: No metadata.json found")
            stale_count += 1
            continue

        try:
            import json
            with open(metadata_file) as f:
                metadata = json.load(f)

            date_str = metadata.get("date", "")
            if not date_str:
                print(f"⚠ {version_dir.name}: No date in metadata")
                stale_count += 1
                continue

            baseline_date = datetime.fromisoformat(date_str)
            age = now - baseline_date

            if age > max_age_delta:
                print(f"⚠ {version_dir.name}: Stale ({age.days} days old, max {max_age})")
                stale_count += 1
            else:
                print(f"✓ {version_dir.name}: Fresh ({age.days} days old)")
                fresh_count += 1

        except Exception as e:
            print(f"⚠ {version_dir.name}: Error reading metadata: {e}")
            stale_count += 1

    print(f"\nBaselines: {fresh_count} fresh, {stale_count} stale")

    if stale_count > 0:
        print(f"\n⚠ {stale_count} baseline(s) older than {max_age} days")
        if fail:
            return 1

    return 0


def main():
    args = get_args()
    return check_baseline_freshness(args.baselines_dir, args.max_age, args.fail)


if __name__ == "__main__":
    import sys
    sys.exit(main())
