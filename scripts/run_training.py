#!/usr/bin/env python
"""
Run training pipeline from config with wandb logging.

Usage:
    python scripts/run_training.py baseline
    python scripts/run_training.py baseline --force-retokenize
    python scripts/run_training.py baseline --wandb-project my-project
    python scripts/run_training.py baseline --no-save
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.training_pipeline import TrainingPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Run text classification training with automatic caching"
    )
    parser.add_argument(
        "experiment", type=str, help="Experiment config name (e.g., baseline)"
    )
    parser.add_argument(
        "--force-retokenize",
        action="store_true",
        help="Force re-tokenization even if cached",
    )
    parser.add_argument(
        "--force-reencode", action="store_true", help="Force re-encoding even if cached"
    )
    parser.add_argument(
        "--wandb-project", type=str, default=None, help="Weights & Biases project name"
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Weights & Biases entity (username/team)",
    )
    parser.add_argument(
        "--wandb-run-name", type=str, default=None, help="Weights & Biases run name"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save trained model to disk",
    )
    args = parser.parse_args()

    # Initialize wandb if project specified
    wandb_logger = None
    if args.wandb_project:
        try:
            import wandb

            wandb_logger = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name,
                config={},  # Will be updated with full config in pipeline
            )
            print(f"Weights & Biases logging enabled: {args.wandb_project}")
        except ImportError:
            print("⚠ wandb not installed. Install with: pip install wandb")
            print("  Continuing without wandb logging...")

    pipeline = TrainingPipeline(args.experiment, wandb_logger=wandb_logger)
    pipeline.run(
        force_retokenize=args.force_retokenize, 
        force_reencode=args.force_reencode,
        save_model=not args.no_save
    )

    if wandb_logger:
        wandb_logger.finish()


if __name__ == "__main__":
    main()
