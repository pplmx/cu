#!/usr/bin/env python3
"""
Update baseline results for regression comparison.

Usage:
    python scripts/benchmark/update_baseline.py --results results/latest --version v1.7.0
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser(description="Update benchmark baselines")
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Directory containing benchmark results",
    )
    parser.add_argument(
        "--version",
        type=str,
        required=True,
        help="Baseline version tag (e.g., 'v1.7.0', 'main')",
    )
    parser.add_argument(
        "--baselines-dir",
        type=Path,
        default=Path(__file__).parent / "baselines",
        help="Directory to store baselines",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing baseline",
    )
    return parser.parse_args()


def get_device_info():
    """Get GPU device information."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if lines:
                parts = lines[0].split(", ")
                return {
                    "gpu": parts[0].strip() if len(parts) > 0 else "Unknown",
                    "driver": parts[1].strip() if len(parts) > 1 else "Unknown",
                    "memory": parts[2].strip() if len(parts) > 2 else "Unknown",
                }
    except Exception:
        pass
    return {"gpu": "Unknown", "driver": "Unknown", "memory": "Unknown"}


def get_cuda_version():
    """Get CUDA version."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "release" in line:
                    parts = line.split("release")[1].strip().split(",")
                    if parts:
                        return parts[0].strip()
    except Exception:
        pass
    return "Unknown"


def update_baseline(results_dir: Path, version: str, baselines_dir: Path, force: bool):
    """Update baseline with new results."""
    baselines_dir = baselines_dir / version
    baselines_dir.mkdir(parents=True, exist_ok=True)

    device_info = get_device_info()
    cuda_version = get_cuda_version()

    metadata = {
        "version": version,
        "date": datetime.now().isoformat(),
        "gpu": device_info["gpu"],
        "driver": device_info["driver"],
        "memory": device_info["memory"],
        "cuda": cuda_version,
    }

    metadata_file = baselines_dir / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Updated metadata: {metadata_file}")

    result_files = list(results_dir.glob("*.json"))
    if not result_files:
        print(f"No result files found in {results_dir}")
        return False

    for result_file in result_files:
        baseline_file = baselines_dir / result_file.name

        if baseline_file.exists() and not force:
            print(f"Baseline {baseline_file.name} exists (use --force to overwrite)")
            continue

        try:
            with open(result_file) as f:
                data = json.load(f)

            if "context" not in data:
                data["context"] = {}
            data["context"]["baseline_version"] = version
            data["context"]["baseline_date"] = datetime.now().isoformat()

            with open(baseline_file, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Updated baseline: {baseline_file}")

        except Exception as e:
            print(f"Error processing {result_file}: {e}")
            return False

    print(f"\nBaseline '{version}' updated successfully!")
    print(f"Location: {baselines_dir}")
    return True


def main():
    args = get_args()

    if not args.results.exists():
        print(f"Results directory not found: {args.results}")
        return 1

    if not update_baseline(args.results, args.version, args.baselines_dir, args.force):
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
