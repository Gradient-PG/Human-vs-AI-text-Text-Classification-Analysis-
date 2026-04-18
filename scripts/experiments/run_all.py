#!/usr/bin/env python
"""
Run the full experiment sequence respecting dependencies.

Usage:
    uv run scripts/experiments/run_all.py --generator gpt4
    uv run scripts/experiments/run_all.py --generator gpt4 --skip characterize
    uv run scripts/experiments/run_all.py --generator gpt4 --only sparse_probe ablation

Experiment order (dependencies read from YAML configs via source_experiment):
    1. sparse_probe           (no dependencies)
    2. ablation               (depends on sparse_probe)
    3. patching               (depends on sparse_probe)
    4. confound               (depends on sparse_probe)
    5. auc_comparison         (depends on sparse_probe)
    6. characterize           (depends on sparse_probe)

If an experiment fails, only its dependents are skipped.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

EXPERIMENT_ORDER = [
    "sparse_probe",
    "ablation",
    "patching",
    # "confound",
    "auc_comparison",
    "characterize",
]


def _read_dependency(config_dir: Path, experiment: str) -> str | None:
    """Read source_experiment from a YAML config (if it exists)."""
    config_file = config_dir / f"{experiment}.yaml"
    if not config_file.exists():
        return None
    with open(config_file) as f:
        raw = yaml.safe_load(f) or {}
    return raw.get("source_experiment")


def _collect_transitive_deps(
    experiment: str,
    dep_map: dict[str, str | None],
) -> set[str]:
    """Return all transitive dependencies of *experiment*."""
    deps: set[str] = set()
    current = dep_map.get(experiment)
    while current is not None:
        deps.add(current)
        current = dep_map.get(current)
    return deps


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all experiments in dependency order."
    )
    parser.add_argument(
        "--generator",
        type=str,
        default="gpt4",
        help="Generator to run on (default: gpt4).",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="config/experiments",
        help="Directory containing per-experiment YAML configs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/experiments",
        help="Base output directory.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run identifier (e.g. timestamp). Auto-generated if not provided.",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        help="Experiments to skip.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Run only these experiments (in dependency order).",
    )
    args = parser.parse_args()

    config_dir = Path(args.config_dir)

    experiments = list(EXPERIMENT_ORDER)
    if args.only:
        experiments = [e for e in experiments if e in args.only]
    experiments = [e for e in experiments if e not in args.skip]

    if not experiments:
        print("No experiments to run.")
        return

    dep_map = {exp: _read_dependency(config_dir, exp) for exp in EXPERIMENT_ORDER}

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Generator:   {args.generator}")
    print(f"Run ID:      {run_id}")
    print(f"Experiments: {', '.join(experiments)}")
    print()

    script = Path(__file__).parent / "run_experiment.py"

    failed: set[str] = set()
    skipped: set[str] = set()

    t0 = time.time()
    for exp in experiments:
        deps = _collect_transitive_deps(exp, dep_map)
        blocked_by = deps & failed
        if blocked_by:
            print(f"\n{'='*60}")
            print(f"  SKIP {exp} (dependency failed: {', '.join(sorted(blocked_by))})")
            print(f"{'='*60}")
            skipped.add(exp)
            continue

        print(f"\n{'='*60}")
        print(f"  {exp}")
        print(f"{'='*60}\n")

        cmd = [
            sys.executable, str(script),
            exp,
            "--generator", args.generator,
            "--run-id", run_id,
        ]

        config_file = config_dir / f"{exp}.yaml"
        if config_file.exists():
            cmd.extend(["--config", str(config_file)])

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\n  FAILED: {exp} (exit code {result.returncode})")
            failed.add(exp)

    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    succeeded = set(experiments) - failed - skipped
    if succeeded:
        print(f"  Succeeded: {', '.join(sorted(succeeded))}")
    if failed:
        print(f"  Failed:    {', '.join(sorted(failed))}")
    if skipped:
        print(f"  Skipped:   {', '.join(sorted(skipped))}")
    print(f"  Elapsed:   {elapsed:.1f}s")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
