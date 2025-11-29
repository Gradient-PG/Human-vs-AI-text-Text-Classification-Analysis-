"""
Encode tokenized dataset using a specified encoder model.

This script loads a tokenized dataset, encodes it using a transformer model
(e.g., BERT), and saves the encoded embeddings to disk for later training.

Usage:
    python scripts/encode_dataset.py
    
Input:
    - data/processed/tokenized_dataset/ (tokenized HuggingFace Dataset)
    
Output:
    - data/processed/encoded_dataset/ (encoded embeddings + labels)
"""

import sys
from pathlib import Path
import torch
from transformers import BertModel

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.dataset_encoder import DatasetEncoder


def main():
    # Configuration
    tokenized_dataset_path = "data/processed/tokenized_dataset"
    encoded_output_dir = "data/processed/encoded_dataset"
    encoder_name = "bert-base-uncased"
    batch_size = 64
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading encoder: {encoder_name}")
    encoder = BertModel.from_pretrained(encoder_name)
    
    dataset_encoder = DatasetEncoder(
        encoder=encoder,
        output_dir=encoded_output_dir,
        device=device
    )
    
    dataset_encoder.encode_and_save(
        tokenized_dataset_path=tokenized_dataset_path,
        batch_size=batch_size
    )


if __name__ == "__main__":
    main()

