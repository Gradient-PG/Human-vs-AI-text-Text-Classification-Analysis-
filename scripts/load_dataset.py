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
        
    """
    print(df.info())
    print(df['generated'].value_counts())
    print(df.head())


def main():
    """
    Main entry point - load data and display stats.
    """
    df = pd.read_csv('data/raw/AI_Human.csv')
    print_dataset_stats(df)

if __name__ == "__main__":
    main()
