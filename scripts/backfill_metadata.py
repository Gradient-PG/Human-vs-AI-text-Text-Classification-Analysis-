#!/usr/bin/env python
"""
TEMPORARY SCRIPT TO BACKFILL METADATA FOR EXISTING ACTIVATIONS
WILL BE REMOVED AFTER ALL ACTIVATIONS HAVE METADATA

Backfill sample_metadata.npz for activation directories that were created
before the extractor was updated to save metadata automatically.

Replays the exact same shuffle used in ActivationExtractor.extract_activations
(seed=42 for both class subsets and final shuffle) so the metadata rows are
index-aligned with the saved .npy files.

Usage:
    # All 11 generators
    uv run scripts/backfill_metadata.py

    # Specific generators
    uv run scripts/backfill_metadata.py --models gpt4 chatgpt

    # Preview without writing
    uv run scripts/backfill_metadata.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from raid_pipeline.raid_loader import ALL_RAID_MODELS, slug
from raid_analysis.data.metadata import compute_metadata_from_dataset, save_metadata


def replay_shuffle(dataset, max_samples: int):
    """Replay the exact shuffle ActivationExtractor uses to pick samples.

    Mirrors activation_extractor.py lines 65–74:
        ai_subset  = ai_samples.shuffle(seed=42).select(range(n))
        human_subset = human_samples.shuffle(seed=42).select(range(n))
        dataset = concatenate_datasets([ai_subset, human_subset]).shuffle(seed=42)
    """
    from datasets import concatenate_datasets

    ai_samples = dataset.filter(lambda x: x["label"] == 1)
    human_samples = dataset.filter(lambda x: x["label"] == 0)

    samples_per_class = max_samples // 2

    ai_subset = ai_samples.shuffle(seed=42).select(
        range(min(samples_per_class, len(ai_samples)))
    )
    human_subset = human_samples.shuffle(seed=42).select(
        range(min(samples_per_class, len(human_samples)))
    )

    return concatenate_datasets([ai_subset, human_subset]).shuffle(seed=42)


def backfill_one(model: str, *, dry_run: bool = False) -> bool:
    model_slug = slug(model)
    activations_dir = project_root / "results" / f"activations_raid_{model_slug}"
    tokenized_path = project_root / "data" / "processed" / f"raid_{model_slug}"
    metadata_path = activations_dir / "sample_metadata.npz"
    metadata_json = activations_dir / "metadata.json"

    if not activations_dir.exists():
        print(f"  [{model}] SKIP — activations directory not found: {activations_dir}")
        return False

    if metadata_path.exists():
        print(f"  [{model}] SKIP — sample_metadata.npz already exists")
        return True

    if not tokenized_path.exists():
        print(f"  [{model}] ERROR — tokenized dataset not found: {tokenized_path}")
        return False

    if not metadata_json.exists():
        print(f"  [{model}] ERROR — metadata.json not found: {metadata_json}")
        return False

    with open(metadata_json) as f:
        meta = json.load(f)
    n_samples = meta["n_samples"]

    print(f"  [{model}] loading tokenized dataset from {tokenized_path} ...")
    from datasets import load_from_disk
    dataset_dict = load_from_disk(str(tokenized_path))
    dataset = dataset_dict["train"]

    print(f"  [{model}] replaying shuffle (n_samples={n_samples}) ...")
    aligned = replay_shuffle(dataset, max_samples=n_samples)

    if len(aligned) != n_samples:
        print(
            f"  [{model}] WARNING — replayed {len(aligned)} rows but "
            f"activations have {n_samples}. Counts may mismatch."
        )

    sample_metadata = compute_metadata_from_dataset(aligned)

    if dry_run:
        print(
            f"  [{model}] DRY RUN — would write {metadata_path} "
            f"({len(sample_metadata)} samples, "
            f"domains={sample_metadata.domain_names})"
        )
        return True

    save_metadata(sample_metadata, activations_dir)
    print(
        f"  [{model}] wrote {metadata_path} "
        f"({len(sample_metadata)} samples, "
        f"domains={sample_metadata.domain_names})"
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill sample_metadata.npz for existing activation directories."
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help=f"Models to backfill (default: all {len(ALL_RAID_MODELS)}). "
             f"Choices: {', '.join(ALL_RAID_MODELS)}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written without actually writing.",
    )
    args = parser.parse_args()

    models = args.models or ALL_RAID_MODELS
    invalid = [m for m in models if m not in ALL_RAID_MODELS]
    if invalid:
        print(f"Unknown model(s): {invalid}")
        print(f"Valid: {ALL_RAID_MODELS}")
        sys.exit(1)

    print(f"Backfilling sample_metadata.npz for {len(models)} model(s) ...")
    if args.dry_run:
        print("(dry run — no files will be written)\n")

    ok = 0
    for model in models:
        if backfill_one(model, dry_run=args.dry_run):
            ok += 1

    print(f"\n{ok}/{len(models)} completed.")
    if ok < len(models):
        sys.exit(1)


if __name__ == "__main__":
    main()
