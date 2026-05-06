#!/usr/bin/env python
"""
Sample-size sensitivity for sparse_probe.

Runs ``sparse_probe`` at several ``max_samples`` values for one generator and
writes a summary comparing accuracy, selected-set size, stable-neuron count,
and Jaccard overlap of stable sets between adjacent sample sizes.

This is a standalone supplementary experiment: it calls the existing
run_experiment.py once per sample size via subprocess, then aggregates the
resulting aggregate.json files. It does NOT modify the main pipeline.

Usage:
    uv run scripts/experiments/run_sample_size_sensitivity.py --generator gpt4
    uv run scripts/experiments/run_sample_size_sensitivity.py \\
        --generator mistral --sample-sizes 1000 2000 5000 \\
        --base-config config/experiments/sparse_probe.yaml

    uv run scripts/experiments/run_sample_size_sensitivity.py --generator gpt4 --sample-sizes 200 500 1000 2000 5000 10000
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_base_config(path: Path) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    if raw.get("experiment") != "sparse_probe":
        raise ValueError(
            f"Base config at {path} must be a sparse_probe config, "
            f"got experiment={raw.get('experiment')!r}"
        )
    return raw


def _write_override_config(
    base: dict, max_samples: int | None, target_path: Path,
) -> None:
    override = dict(base)
    if max_samples is None:
        override["max_samples"] = None
    else:
        override["max_samples"] = int(max_samples)
    with open(target_path, "w") as f:
        yaml.safe_dump(override, f, sort_keys=False)


def _run_one(
    generator: str, max_samples: int | None, run_id: str,
    base_config: dict, output_dir: Path,
) -> int:
    size_label = "all" if max_samples is None else str(max_samples)
    child_run_id = f"{run_id}/n{size_label}"

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=f"_sparse_probe_n{size_label}.yaml",
        delete=False, dir=str(output_dir),
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        _write_override_config(base_config, max_samples, tmp_path)

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "experiments" / "run_experiment.py"),
            "sparse_probe",
            "--generator", generator,
            "--run-id", child_run_id,
            "--config", str(tmp_path),
        ]
        print(f"\n{'='*60}\n  n={size_label}\n{'='*60}\n")
        result = subprocess.run(cmd)
        return result.returncode
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


def _load_aggregate(
    base: dict, run_id: str, generator: str, max_samples: int | None,
) -> dict | None:
    size_label = "all" if max_samples is None else str(max_samples)
    agg_path = (
        Path(base.get("output_dir", "results/experiments"))
        / f"{run_id}" / f"n{size_label}"
        / "sparse_probe" / generator / "aggregate.json"
    )
    if not agg_path.exists():
        print(f"  Warning: missing aggregate at {agg_path}")
        return None
    with open(agg_path) as f:
        return json.load(f)


def _jaccard(a: list, b: list) -> float:
    set_a = {tuple(x) for x in a}
    set_b = {tuple(x) for x in b}
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / max(1, len(set_a | set_b))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run sparse_probe at multiple max_samples values and summarise "
            "stability of findings."
        )
    )
    parser.add_argument(
        "--generator", type=str, default="gpt4",
        help="Generator to run on.",
    )
    parser.add_argument(
        "--sample-sizes", nargs="+", default=["200", "500", "1000", "2000", "5000"],
        help=(
            "Sample sizes to sweep. Use integer values, or 'all' for no "
            "subsampling."
        ),
    )
    parser.add_argument(
        "--base-config", type=str,
        default="config/experiments/sparse_probe.yaml",
        help="Base sparse_probe YAML to override.",
    )
    parser.add_argument(
        "--run-id", type=str, default=None,
        help="Parent run identifier. Auto-generated if not provided.",
    )
    args = parser.parse_args()

    sizes: list[int | None] = []
    for s in args.sample_sizes:
        if str(s).lower() == "all":
            sizes.append(None)
        else:
            sizes.append(int(s))

    base_config_path = Path(args.base_config)
    if not base_config_path.is_absolute():
        base_config_path = PROJECT_ROOT / base_config_path
    base_config = _load_base_config(base_config_path)

    parent_run_id = args.run_id or (
        "samplesize_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_root = (
        Path(base_config.get("output_dir", "results/experiments"))
        / parent_run_id
    )
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Generator:     {args.generator}")
    print(f"Parent run-id: {parent_run_id}")
    print(f"Sample sizes:  {[('all' if s is None else s) for s in sizes]}")
    print(f"Output root:   {output_root}")

    t0 = time.time()
    failed: list[str] = []
    for n in sizes:
        rc = _run_one(
            args.generator, n, parent_run_id, base_config, output_root,
        )
        if rc != 0:
            label = "all" if n is None else str(n)
            failed.append(label)
            print(f"  FAILED n={label} (exit {rc})")

    summary: dict = {
        "generator": args.generator,
        "parent_run_id": parent_run_id,
        "sample_sizes": ["all" if n is None else int(n) for n in sizes],
        "failed": failed,
        "runs": {},
    }

    for n in sizes:
        label = "all" if n is None else str(n)
        agg = _load_aggregate(base_config, parent_run_id, args.generator, n)
        if agg is None:
            continue
        metrics = agg.get("aggregate_metrics", {})
        stable = agg.get("stable_neurons", [])
        summary["runs"][label] = {
            "accuracy_mean": metrics.get("accuracy_mean"),
            "accuracy_std": metrics.get("accuracy_std"),
            "n_selected_mean": metrics.get("n_selected_mean"),
            "n_selected_std": metrics.get("n_selected_std"),
            "n_stable_neurons": agg.get("n_stable_neurons"),
            "stable_neurons": stable,
        }

    jaccard: dict[str, float] = {}
    labels_in_order = [
        ("all" if n is None else str(n)) for n in sizes
        if ("all" if n is None else str(n)) in summary["runs"]
    ]
    for i in range(len(labels_in_order) - 1):
        a_label = labels_in_order[i]
        b_label = labels_in_order[i + 1]
        jaccard[f"{a_label}_vs_{b_label}"] = _jaccard(
            summary["runs"][a_label]["stable_neurons"],
            summary["runs"][b_label]["stable_neurons"],
        )
    summary["jaccard_adjacent"] = jaccard

    for run_entry in summary["runs"].values():
        run_entry.pop("stable_neurons", None)

    summary_path = output_root / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    elapsed = time.time() - t0

    print(f"\n{'='*60}\n  Summary\n{'='*60}")
    print(f"  Wrote {summary_path}")
    print(f"  Elapsed: {elapsed:.1f}s")
    if failed:
        print(f"  Failed sizes: {', '.join(failed)}")

    print(f"\n  {'n':>8}  {'acc_mean':>10}  {'n_sel_mean':>12}  {'stable':>8}")
    for label in labels_in_order:
        r = summary["runs"][label]
        acc = r.get("accuracy_mean")
        ns = r.get("n_selected_mean")
        st = r.get("n_stable_neurons")
        acc_str = f"{acc:.4f}" if isinstance(acc, (int, float)) else "-"
        ns_str = f"{ns:.1f}" if isinstance(ns, (int, float)) else "-"
        st_str = str(st) if st is not None else "-"
        print(f"  {label:>8}  {acc_str:>10}  {ns_str:>12}  {st_str:>8}")

    if jaccard:
        print("\n  Jaccard(stable_neurons) between adjacent sample sizes:")
        for key, val in jaccard.items():
            print(f"    {key:>20}:  {val:.3f}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
