"""
Tokenize raw text data and save as HuggingFace Dataset.

This script loads raw CSV data, performs train/test split, tokenizes the text,
and saves the tokenized datasets to disk for later encoding.

Usage:
    python scripts/tokenize_dataset.py

Input:
    - data/raw/AI_Human.csv

Output:
    - data/processed/tokenized_dataset/ (tokenized HuggingFace Dataset)
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.dataset_tokenizer import DatasetTokenizer
from datasets import load_dataset
from utils import hidden


def main():
    # Configuration
    output_dir = "data/processed"
    tokenizer_name = "distilbert-base-uncased"
    max_length = 512
    test_size = 0.2
    batch_size = 1000
    random_state = 42

    tokenizer = DatasetTokenizer(
        tokenizer_name=tokenizer_name,
        output_dir=output_dir,
        max_length=max_length,
        test_size=test_size,
        random_state=random_state,
    )

    ds = load_dataset(
        "NicolaiSivesind/human-vs-machine",
        "wiki_labeled",
        token=hidden.hf_api_key,
    )

    tokenizer.tokenize_and_save(ds, batch_size=batch_size)


if __name__ == "__main__":
    main()
