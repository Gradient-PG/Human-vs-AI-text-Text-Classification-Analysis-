"""
PyTorch Dataset and DataLoader utilities for text classification.

This module provides:
- TextDataset class for wrapping tokenized texts and labels
- Helper functions to create train/test DataLoaders
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, Any
import numpy as np


class TextDataset(Dataset):
    """
    PyTorch Dataset for text classification.
    
    Wraps tokenized encodings (input_ids, attention_mask, etc.) 
    and corresponding labels.
    """
    
    def __init__(self, encodings: Dict[str, Any], labels: np.ndarray):
        """
        Initialize the dataset.
        
        Args:
            encodings: Dictionary with keys like 'input_ids', 'attention_mask'
                      Each value should be a list or array of shape (num_samples, seq_len)
            labels: Array of labels, shape (num_samples,)
            
        Hint: Store encodings and labels as instance variables
        """
        # TODO: Implement
        pass
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Get a single item from the dataset.
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            Tuple of (input_dict, label) where:
                - input_dict contains tensors for 'input_ids', 'attention_mask', etc.
                - label is a single label tensor
                
        Hint: 
            - Extract idx-th element from each encoding key
            - Convert to torch.Tensor if not already
            - Return as ({key: tensor}, label_tensor)
        """
        # TODO: Implement
        pass
    
    def __len__(self) -> int:
        """
        Return the total number of samples.
        
        Hint: Return length of labels array
        """
        # TODO: Implement
        pass


def get_dataloaders(
    train_encodings: Dict[str, Any],
    test_encodings: Dict[str, Any],
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    batch_size: int = 32,
    shuffle_train: bool = True,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders for training and testing.
    
    Args:
        train_encodings: Tokenized training data
        test_encodings: Tokenized test data
        train_labels: Training labels
        test_labels: Test labels
        batch_size: Batch size for DataLoader
        shuffle_train: Whether to shuffle training data
        num_workers: Number of workers for data loading (0 = main thread)
        
    Returns:
        train_loader, test_loader tuple
        
    Hint:
        1. Create TextDataset instances for train and test
        2. Wrap each in a DataLoader with appropriate parameters
        3. Remember to shuffle training data but not test data
    """
    # TODO: Implement
    pass


def get_single_dataloader(
    encodings: Dict[str, Any],
    labels: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0
) -> DataLoader:
    """
    Create a single DataLoader (useful for inference or validation).
    
    Args:
        encodings: Tokenized data
        labels: Corresponding labels
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of workers
        
    Returns:
        DataLoader instance
        
    Hint: Similar to get_dataloaders but for a single dataset
    """
    # TODO: Implement
    pass

