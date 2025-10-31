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


def tokenize_texts(
    texts: List[str],
    tokenizer_name: str = 'distilbert-base-uncased',
    max_length: int = 512,
    padding: str = 'max_length',
    truncation: bool = True
) -> Dict[str, Any]:
    """
    Tokenize text using Hugging Face tokenizer.
    
    Args:
        texts: List of text strings to tokenize
        tokenizer_name: Name of the HF tokenizer to use
        max_length: Maximum sequence length
        padding: Padding strategy ('max_length' or 'longest')
        truncation: Whether to truncate sequences exceeding max_length
        
    Returns:
        Dictionary with 'input_ids', 'attention_mask', etc.
        
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    encodings = tokenizer(texts, max_length=max_length, padding=padding, truncation=truncation, return_tensors='pt')
    return encodings


def save_processed_data(
    train_encodings: Dict[str, Any],
    test_encodings: Dict[str, Any],
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    output_dir: str = 'data/processed'
) -> None:
    """
    Save tokenized data to disk for later use.
    
    Args:
        train_encodings: Tokenized training texts
        test_encodings: Tokenized test texts
        train_labels: Training labels
        test_labels: Test labels
        output_dir: Directory to save processed data
    """
    import os

    torch.save(train_encodings, os.path.join(output_dir, 'train_encodings.pt'))
    torch.save(test_encodings, os.path.join(output_dir, 'test_encodings.pt'))
    np.save(os.path.join(output_dir, 'train_labels.npy'), train_labels)
    np.save(os.path.join(output_dir, 'test_labels.npy'), test_labels)


def load_processed_data(
    data_dir: str = 'data/processed'
) -> Tuple[Dict[str, Any], Dict[str, Any], np.ndarray, np.ndarray]:
    """
    Load previously saved processed data.
    
    Args:
        data_dir: Directory containing processed data
        
    Returns:
        train_encodings, test_encodings, train_labels, test_labels
        
    Hint: Use torch.load() or np.load() to load the saved files
    """
    # TODO: Implement
    pass

