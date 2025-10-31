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


def load_raw_data(path: str) -> pd.DataFrame:
    """
    Load raw CSV data.
    
    Args:
        path: Path to the CSV file
        
    Returns:
        DataFrame with text and labels
        
    Hint: Simple pd.read_csv() wrapper for consistency
    """
    # TODO: Implement
    pass


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
        
    Note: Don't do too much - the tokenizer will handle most things!
    """
    # TODO: Implement
    pass


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
        
    Hint: Use sklearn.model_selection.train_test_split with stratify parameter
    """
    # TODO: Implement
    pass


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
        
    Hint:
        1. Load tokenizer: AutoTokenizer.from_pretrained(tokenizer_name)
        2. Call tokenizer(texts, ...) with appropriate parameters
        3. Return the tokenizer output as dict
    """
    # TODO: Implement
    pass


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
        
    Hint:
        Use torch.save() or np.save() to save the data
        Save as: train_encodings.pt, test_encodings.pt, 
                 train_labels.npy, test_labels.npy
    """
    # TODO: Implement
    pass


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

