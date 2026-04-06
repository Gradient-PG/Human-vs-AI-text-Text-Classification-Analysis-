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

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

from raid_pipeline.raid_loader import ALL_RAID_MODELS, RAIDConfig, load_raid, slug
from raid_pipeline.dataset_tokenizer import DatasetTokenizer
from raid_pipeline.activation_extractor import ActivationExtractor
from raid_pipeline.model_loader import load_bert_model
from raid_analysis.data.activations import load_activations
from raid_analysis.data.neuron_stats import (
    compute_neuron_statistics,
    identify_discriminative_neurons,
)


def run_pipeline_for_model(
    model: str,
    *,
    samples: int,
    batch_size: int,
    extract_batch_size: int,
    seed: int,
    domains: list[str] | None,
    encoder,
    device: str,
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

    # Step 1: Tokenize
    print(f"\n{'-' * 60}")
    print(f"  Step 1/3  Tokenize ({model} vs human)")
    print(f"{'-' * 60}\n")

    config = RAIDConfig(
        model=model,
        domains=domains,
        max_samples=samples,
        seed=seed,
    )
    ds = load_raid(config)

    tokenizer = DatasetTokenizer(
        output_dir="data/processed",
        max_length=512,
        random_state=seed,
        text_column="text",
        extra_columns=["domain", "source_model"],
    )
    tokenizer.tokenize_and_save(ds, batch_size=batch_size, dataset_name=dataset_name)

    # Step 2: Extract activations
    print(f"\n{'-' * 60}")
    print(f"  Step 2/3  Extract activations ({model})")
    print(f"{'-' * 60}\n")

    extractor = ActivationExtractor(
        encoder=encoder,
        layers=list(range(1, 13)),
        device=device,
    )
    extractor.extract_and_save(
        tokenized_dataset_path=tokenized_path,
        output_dir=activations_dir,
        batch_size=extract_batch_size,
        max_samples=samples,
        split="train",
    )

    # Step 3: Neuron statistics
    print(f"\n{'-' * 60}")
    print(f"  Step 3/3  Analyze neurons ({model})")
    print(f"{'-' * 60}\n")

    activations, labels, metadata = load_activations(activations_dir)
    output_path = Path(activations_dir)

    for layer_idx in metadata["layers"]:
        stats_df = compute_neuron_statistics(activations[layer_idx], labels)
        stats_df, _ = identify_discriminative_neurons(stats_df)
        csv_path = output_path / f"layer_{layer_idx}_neuron_stats.csv"
        stats_df.to_csv(csv_path, index=False)
        n_disc = stats_df["discriminative"].sum()
        print(f"  Layer {layer_idx:2d}: {n_disc:3d} discriminative → {csv_path.name}")

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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Loading BERT model once (device: {device}) ...")
    encoder, _tokenizer = load_bert_model(device=device)

    t_total = time.time()
    results: dict[str, bool] = {}

    for model in models:
        try:
            ok = run_pipeline_for_model(
                model,
                samples=args.samples,
                batch_size=args.batch_size,
                extract_batch_size=args.extract_batch_size,
                seed=args.seed,
                domains=args.domains,
                encoder=encoder,
                device=device,
            )
        except Exception as exc:
            print(f"\n  FAILED: {model} — {exc}")
            ok = False
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
