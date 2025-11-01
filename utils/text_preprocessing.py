"""
Text preprocessing utilities for the Human vs AI text classification project.

This module provides functions for:
- Loading raw data
- Basic text cleaning
- Tokenization using Hugging Face transformers
- Train/test splitting
"""

import pandas as pd
import numpy as np
import torch
from typing import List, Tuple, Dict, Any
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from tqdm import tqdm
from datasets import Dataset, DatasetDict, load_from_disk
import os


def preprocess_text(df: pd.DataFrame, text_column: str = 'text') -> List[str]:
    """
    Basic text preprocessing (optional cleaning).
    
    Args:
        df: DataFrame containing text data
        text_column: Name of the column with text
        
    Returns:
        List of preprocessed text strings
        
    Hint: You might want to:
        - Convert to lowercase (optional, tokenizer may handle this)
        - Remove extra whitespace
        - Handle NaN values
    """
    # TODO: Replace newlines and other special characters with spaces

    df[text_column] = df[text_column].apply(lambda x: x.lower())
    df[text_column] = df[text_column].apply(lambda x: x.strip())

    return df[text_column].tolist()


def create_train_test_split(
    df: pd.DataFrame,
    text_column: str = 'text',
    label_column: str = 'label',
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test sets.
    
    Args:
        df: Full DataFrame
        text_column: Name of text column
        label_column: Name of label column
        test_size: Fraction for test set (0.0 to 1.0)
        random_state: Random seed for reproducibility
        
    Returns:
        train_df, test_df tuple
    """
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[label_column])
    return train_df, test_df


def tokenize_and_create_dataset(
    texts: List[str],
    labels: List[int],
    tokenizer_name: str = 'distilbert-base-uncased',
    max_length: int = 512,
    padding: str = 'max_length',
    truncation: bool = True,
    batch_size: int = 1000
) -> Dataset:
    """
    Tokenize texts and create a HuggingFace Dataset with labels.
    
    Args:
        texts: List of text strings to tokenize
        labels: List of integer labels
        tokenizer_name: Name of the HF tokenizer to use
        max_length: Maximum sequence length
        padding: Padding strategy ('max_length' or 'longest')
        truncation: Whether to truncate sequences exceeding max_length
        batch_size: Number of texts to process at once
        
    Returns:
        HuggingFace Dataset with tokenized texts and labels
    """
    print(f"   Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Check if fast tokenizer is available
    if hasattr(tokenizer, 'is_fast') and tokenizer.is_fast:
        print(f"   ++ Using fast tokenizer (Rust-based)")
    else:
        print(f"   -- Using slow tokenizer")
    
    print(f"   Creating dataset from {len(texts)} texts...")
    dataset = Dataset.from_dict({"text": texts, "labels": labels})
    
    # Define tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding=padding,
            truncation=truncation,
            max_length=max_length
        )
    
    print(f"   Tokenizing in batches of {batch_size}...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=batch_size,
        remove_columns=["text"],
        desc="Tokenizing"
    )
    
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    return tokenized_dataset


def save_dataset(
    dataset_dict: DatasetDict,
    output_dir: str = 'data/processed'
) -> None:
    """
    Save HuggingFace DatasetDict to disk using memory-mapped format.
    
    Args:
        dataset_dict: DatasetDict containing train/test splits
        output_dir: Directory to save processed data
    """
    dataset_path = os.path.join(output_dir, 'tokenized_dataset')
    dataset_dict.save_to_disk(dataset_path)
    print(f"   Dataset saved to {dataset_path}")


def load_dataset(
    data_dir: str = 'data/processed'
) -> DatasetDict:
    """
    Load HuggingFace DatasetDict from disk.
    
    Args:
        data_dir: Directory containing processed data
        
    Returns:
        DatasetDict with train and test splits
    """
    dataset_path = os.path.join(data_dir, 'tokenized_dataset')
    dataset = load_from_disk(dataset_path)
    print(f"   Dataset loaded from {dataset_path}")
    print(f"   Train samples: {len(dataset['train'])}, Test samples: {len(dataset['test'])}")
    return dataset

