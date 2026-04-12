#!/usr/bin/env python
"""
Run the full experiment sequence respecting dependencies.

Usage:
    uv run scripts/experiments/run_all.py --generator gpt4
    uv run scripts/experiments/run_all.py --generator gpt4 --skip mlp_probe
    uv run scripts/experiments/run_all.py --generator gpt4 --only sparse_probe ablation

Experiment order (dependencies →):
    1. sparse_probe           (no dependencies)
    2. ablation               (depends on 1)
    3. patching               (depends on 1)
    4. confound               (depends on 1)
    5. auc_comparison         (depends on 1)
    6. characterize           (depends on 1, multi-generator)
    7. mlp_probe              (depends on 1, conditional)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

EXPERIMENT_ORDER = [
    "sparse_probe",
    "ablation",
    "patching",
    "confound",
    "auc_comparison",
    "characterize",
]


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
    parser.add_argument(
        "--include-mlp",
        action="store_true",
        help="Include the conditional MLP probe experiment.",
    )
    args = parser.parse_args()

    experiments = list(EXPERIMENT_ORDER)
    if args.include_mlp:
        experiments.append("mlp_probe")

    if args.only:
        experiments = [e for e in experiments if e in args.only]

    experiments = [e for e in experiments if e not in args.skip]

    if not experiments:
        print("No experiments to run.")
        return

    print(f"Generator: {args.generator}")
    print(f"Experiments: {', '.join(experiments)}")
    print()

    script = Path(__file__).parent / "run_experiment.py"
    source_dir = Path(args.output_dir) / "sparse_probe" / args.generator

    t0 = time.time()
    for exp in experiments:
        print(f"\n{'='*60}")
        print(f"  {exp}")
        print(f"{'='*60}\n")

        cmd = [
            sys.executable, str(script),
            exp,
            "--generator", args.generator,
        ]

        config_file = Path(args.config_dir) / f"{exp}.yaml"
        if config_file.exists():
            cmd.extend(["--config", str(config_file)])

        if exp != "sparse_probe":
            cmd.extend(["--source-dir", str(source_dir)])

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\nERROR: {exp} failed with exit code {result.returncode}")
            print("Stopping sequence.")
            sys.exit(result.returncode)

    elapsed = time.time() - t0
    print(f"\nAll experiments completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
