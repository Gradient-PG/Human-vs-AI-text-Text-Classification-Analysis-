#!/usr/bin/env python
"""
Extract BERT CLS token activations from test set for interpretability analysis.

Usage:
    python scripts/extract_activations.py
    python scripts/extract_activations.py --layers 3 6 9 12 --samples 1000
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import torch
from transformers import AutoModel
from utils.activation_extractor import ActivationExtractor


def main():
    parser = argparse.ArgumentParser(
        description="Extract BERT layer activations for analysis"
    )
    parser.add_argument(
        "--tokenized-path",
        type=str,
        default="data/processed/AI_Human/tokenized/bert-base-uncased",
        help="Path to tokenized dataset",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="bert-base-uncased",
        help="Encoder model name",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[3, 6, 9, 12],
        help="Layer indices to extract (1-12 for BERT-base)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of samples to analyze (balanced by class)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for extraction",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/activations",
        help="Output directory for activations",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("BERT ACTIVATION EXTRACTION")
    print("=" * 60)
    print(f"  Encoder: {args.encoder}")
    print(f"  Layers: {args.layers}")
    print(f"  Samples: {args.samples}")
    print(f"  Output: {args.output}\n")
    
    # Load encoder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading encoder model on {device}...")
    encoder = AutoModel.from_pretrained(args.encoder)
    
    # Create extractor
    extractor = ActivationExtractor(
        encoder=encoder,
        layers=args.layers,
        device=device
    )
    
    # Extract and save (always uses test split)
    extractor.extract_and_save(
        tokenized_dataset_path=args.tokenized_path,
        output_dir=args.output,
        batch_size=args.batch_size,
        max_samples=args.samples,
        split="test"
    )
    
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"\nNext step: python scripts/analyze_activations.py")


if __name__ == "__main__":
    main()


