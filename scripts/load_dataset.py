"""
Simple script to load and inspect the raw dataset from Kaggle.

Usage:
    python scripts/load_dataset.py

Expected input:
    - data/raw/AI_Human.csv (or similar CSV from Kaggle)
    
Output:
    - Prints basic dataset statistics to console
"""

import pandas as pd
from pathlib import Path


def load_raw_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load the raw CSV dataset.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        DataFrame with the loaded data
        
    Hint: Use pandas.read_csv()
    """
    # TODO: Implement loading logic
    pass


def print_dataset_stats(df: pd.DataFrame) -> None:
    """
    Print basic statistics about the dataset.
    
    Args:
        df: The loaded DataFrame
        
    Should print:
        - Shape (rows, columns)
        - Column names
        - Data types
        - Missing values count
        - Label distribution (value_counts for the label column)
        - First few examples
        
    Hint: Use df.info(), df.describe(), df['label_column'].value_counts()
    """
    # TODO: Implement stats printing
    pass


def main():
    """
    Main entry point - load data and display stats.
    
    Hint: 
    1. Define the path to your raw CSV (e.g., 'data/raw/AI_Human.csv')
    2. Call load_raw_dataset()
    3. Call print_dataset_stats()
    """
    # TODO: Implement main logic
    pass


if __name__ == "__main__":
    main()

