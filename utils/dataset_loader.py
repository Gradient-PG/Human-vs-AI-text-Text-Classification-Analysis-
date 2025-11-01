"""
PyTorch Dataset and DataLoader utilities for text classification.

This module provides:
- TextDataset class for wrapping tokenized texts and labels
- Helper functions to create train/test DataLoaders
- Functions to load datasets from saved files
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, Any, Optional, Union
from pathlib import Path
import numpy as np


class TextDataset(Dataset):
    """
    PyTorch Dataset for text classification.
    
    Can be initialized with:
    1. Pre-loaded encodings and labels (in-memory)
    2. Paths to saved files (loads dynamically)
    """
    
    def __init__(
        self, 
        encodings: Optional[Dict[str, torch.Tensor]] = None,
        labels: Optional[Union[np.ndarray, torch.Tensor]] = None,
        encodings_path: Optional[str] = None,
        labels_path: Optional[str] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            encodings: Dictionary with tensors like 'input_ids', 'attention_mask'
                      Shape: {key: Tensor[num_samples, seq_len]}
            labels: Array/Tensor of labels, shape (num_samples,)
            encodings_path: Path to saved encodings file (.pt)
            labels_path: Path to saved labels file (.npy or .pt)
        """

        if encodings_path and labels_path:
            self.encodings = torch.load(encodings_path)
            self.labels = torch.load(labels_path)
        elif encodings and labels:
            self.encodings = encodings
            self.labels = labels
        else:
            raise ValueError("Either encodings_path and labels_path or encodings and labels must be provided")
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Get a single item from the dataset.
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            Tuple of (input_dict, label) where:
                - input_dict: {key: tensor[seq_len]} for this sample
                - label: scalar tensor for this sample
        """
        return self.encodings[idx], self.labels[idx]
    
    def __len__(self) -> int:
        """
        Return the total number of samples.
        
        Hint: Return len(self.labels) or self.labels.shape[0]
        """
        return len(self.labels)

def get_dataloaders(
    train_dataset: TextDataset,
    test_dataset: TextDataset,
    batch_size: int = 32,
    shuffle_train: bool = True,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders for training and testing.
    
    Can work with either in-memory data or file paths.
    
    Args:
        train_dataset: Training dataset
        test_dataset: Test dataset
        batch_size: Batch size for DataLoader
        shuffle_train: Whether to shuffle training data
        num_workers: Number of workers for data loading (0 = main thread)
        
    Returns:
        train_loader, test_loader tuple        
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader
