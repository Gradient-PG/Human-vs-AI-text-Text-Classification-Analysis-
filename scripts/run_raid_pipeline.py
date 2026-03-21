#!/usr/bin/env python
"""
Run the full analysis pipeline (tokenize -> extract -> analyze) for each
RAID AI model vs human, producing separate results per model.

Usage:
    # All 11 AI models vs human (1000 samples each)
    uv run scripts/run_raid_pipeline.py

    # Specific models only
    uv run scripts/run_raid_pipeline.py --models gpt4 chatgpt mistral-chat

    # More samples
    uv run scripts/run_raid_pipeline.py --samples 5000

    # Restrict to specific domains
    uv run scripts/run_raid_pipeline.py --models gpt4 --domains news wiki

Output structure:
    data/processed/raid_{model}/          tokenized dataset
    results/activations_raid_{model}/     activations + neuron stats CSVs
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from utils.raid_loader import ALL_RAID_MODELS, slug


def _run(cmd: list[str], label: str) -> bool:
    """Run a subprocess, print its output live, return True on success."""
    print(f"\n{'-' * 60}")
    print(f"  {label}")
    print(f"  $ {' '.join(cmd)}")
    print(f"{'-' * 60}\n")

    result = subprocess.run(cmd, cwd=str(Path(__file__).parent.parent))
    return result.returncode == 0


def run_pipeline_for_model(
    model: str,
    *,
    samples: int,
    batch_size: int,
    extract_batch_size: int,
    seed: int,
    domains: list[str] | None,
) -> bool:
    model_slug = slug(model)
    dataset_name = f"raid_{model_slug}"
    tokenized_path = f"data/processed/{dataset_name}"
    activations_dir = f"results/activations_raid_{model_slug}"

    print("\n" + "=" * 60)
    print(f"  PIPELINE: {model} vs human")
    print(f"  Samples:     {samples}")
    print(f"  Dataset:     {tokenized_path}")
    print(f"  Activations: {activations_dir}")
    print("=" * 60)

    t0 = time.time()

    tokenize_cmd = [
        sys.executable, "scripts/tokenize_raid.py",
        "--model", model,
        "--max-samples", str(samples),
        "--dataset-name", dataset_name,
        "--batch-size", str(batch_size),
        "--seed", str(seed),
    ]
    if domains:
        tokenize_cmd.extend(["--domains"] + domains)

    if not _run(tokenize_cmd, f"Step 1/3  Tokenize ({model} vs human)"):
        print(f"FAILED at tokenization for {model}")
        return False

    extract_cmd = [
        sys.executable, "scripts/extract_activations.py",
        "--tokenized-path", tokenized_path,
        "--output", activations_dir,
        "--samples", str(samples),
        "--batch-size", str(extract_batch_size),
        "--split", "train",
    ]
    if not _run(extract_cmd, f"Step 2/3  Extract activations ({model})"):
        print(f"FAILED at extraction for {model}")
        return False

    analyze_cmd = [
        sys.executable, "scripts/analyze_activations.py",
        "--input", activations_dir,
    ]
    if not _run(analyze_cmd, f"Step 3/3  Analyze neurons ({model})"):
        print(f"FAILED at analysis for {model}")
        return False

    elapsed = time.time() - t0
    print(f"\n  {model} vs human completed in {elapsed:.0f}s")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run full RAID pipeline for each AI model vs human"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=f"AI models to analyse (default: all {len(ALL_RAID_MODELS)}). "
        f"Choices: {', '.join(ALL_RAID_MODELS)}",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Total samples per model, balanced 50/50 AI vs human, "
        "stratified across domains (default: 1000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Tokenization batch size (default: 1000)",
    )
    parser.add_argument(
        "--extract-batch-size",
        type=int,
        default=64,
        help="BERT forward-pass batch size during extraction (default: 64). "
        "Lower this if you run out of memory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=None,
        help="Restrict RAID domains (default: all 6)",
    )

    args = parser.parse_args()
    models = args.models or ALL_RAID_MODELS

    invalid = [m for m in models if m not in ALL_RAID_MODELS]
    if invalid:
        print(f"Unknown model(s): {invalid}")
        print(f"Valid choices: {ALL_RAID_MODELS}")
        sys.exit(1)

    print("=" * 60)
    print("  RAID MULTI-MODEL PIPELINE")
    print("=" * 60)
    print(f"  Models:           {models}")
    print(f"  Samples/model:    {args.samples:,}")
    print(f"  Domains:          {args.domains or 'all'}")
    print(f"  Seed:             {args.seed}")

    t_total = time.time()
    results: dict[str, bool] = {}

    for model in models:
        ok = run_pipeline_for_model(
            model,
            samples=args.samples,
            batch_size=args.batch_size,
            extract_batch_size=args.extract_batch_size,
            seed=args.seed,
            domains=args.domains,
        )
        results[model] = ok

    elapsed_total = time.time() - t_total

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for model, ok in results.items():
        model_slug = slug(model)
        status = "OK" if ok else "FAILED"
        print(f"  {model:20s}  {status:>6s}    results/activations_raid_{model_slug}/")

    n_ok = sum(results.values())
    print(f"\n  {n_ok}/{len(results)} models completed in {elapsed_total:.0f}s")

    if n_ok < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
