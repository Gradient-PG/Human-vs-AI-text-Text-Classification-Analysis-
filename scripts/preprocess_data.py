"""
Complete preprocessing pipeline script.

This script orchestrates the full preprocessing workflow:
1. Load raw CSV data
2. Split into train/test sets
3. Tokenize texts using Hugging Face tokenizer
4. Save processed data for model training

Usage:
    python scripts/preprocess_data.py
    
Input:
    - data/raw/AI_Human.csv (or your CSV filename)
    
Output:
    - data/processed/train_encodings.pt
    - data/processed/test_encodings.pt
    - data/processed/train_labels.npy
    - data/processed/test_labels.npy
    - data/processed/train.csv (optional, for reference)
    - data/processed/test.csv (optional, for reference)
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.text_preprocessing import (
    load_raw_data,
    preprocess_text,
    create_train_test_split,
    tokenize_texts,
    save_processed_data
)


def main():
    """
    Main preprocessing pipeline.
    
    Steps to implement:
    1. Define paths (raw CSV, output directory)
    2. Load raw data using load_raw_data()
    3. Print basic info (shape, columns, label distribution)
    4. Create train/test split using create_train_test_split()
    5. Extract texts and labels from train/test DataFrames
    6. Tokenize texts using tokenize_texts()
    7. Save processed data using save_processed_data()
    8. Optionally save train/test CSVs for reference
    9. Print summary statistics
    
    Configuration suggestions:
    - text_column = 'text' (adjust based on your CSV)
    - label_column = 'generated' (0 = human, 1 = AI, adjust as needed)
    - test_size = 0.2
    - tokenizer = 'distilbert-base-uncased'
    - max_length = 512
    
    Hint: Use try/except to handle errors gracefully and print helpful messages
    """
    
    print("="*60)
    print("Starting preprocessing pipeline...")
    print("="*60)
    
    # TODO: Implement the pipeline
    # Step 1: Define paths
    
    # Step 2: Load raw data
    
    # Step 3: Print basic info
    
    # Step 4: Create train/test split
    
    # Step 5: Extract texts and labels
    
    # Step 6: Tokenize
    
    # Step 7: Save processed data
    
    # Step 8: (Optional) Save CSV files
    
    # Step 9: Print summary
    
    print("="*60)
    print("Preprocessing complete!")
    print("="*60)


if __name__ == "__main__":
    main()

