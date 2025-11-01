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
import pandas as pd

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.text_preprocessing import (
    preprocess_text,
    create_train_test_split,
    tokenize_texts,
    save_processed_data
)
from scripts.load_dataset import print_dataset_stats


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
    
    # Step 0: Define paths
    raw_data_path = 'data/raw/AI_Human.csv'
    processed_data_path = 'data/processed'

    # Step 1: Load raw data
    print("1. Loading raw data...")
    df = pd.read_csv(raw_data_path)
    df = df[:30000]

    print_dataset_stats(df)
  
    # Step 2: Preprocess text
    print("2. Preprocessing text...")

    df['text'] = preprocess_text(df, text_column='text')

    # Step 3: Create train/test split
    print("3. Creating train/test split...")

    train_df, test_df = create_train_test_split(df, text_column='text', label_column='generated', test_size=0.2, random_state=0)

    # Step 4: Extract texts and labels
    print("4. Extracting texts and labels...")

    print("\nTrain DataFrame:")
    print(train_df)
    print("\nTest DataFrame:")
    print(test_df)

    train_texts = train_df['text'].tolist()
    test_texts = test_df['text'].tolist()
    train_labels = train_df['generated'].tolist()
    test_labels = test_df['generated'].tolist()
      
    # Step 5: Tokenize
    print("5. Tokenizing texts...")

    max_length = 512
    batch_size = 1000

    train_encodings = tokenize_texts(train_texts, max_length=max_length, batch_size=batch_size)
    test_encodings = tokenize_texts(test_texts, max_length=max_length, batch_size=batch_size)

    # Step 6: Save processed data
    print("6. Saving processed data...")

    save_processed_data(
        train_encodings = train_encodings,
        test_encodings = test_encodings,
        train_labels = train_labels,
        test_labels = test_labels,
        output_dir = processed_data_path
    )
    
    # Step 7: Print summary
    print(f"7. Preprocessing complete! Files were saved to {processed_data_path}")


if __name__ == "__main__":
    main()

