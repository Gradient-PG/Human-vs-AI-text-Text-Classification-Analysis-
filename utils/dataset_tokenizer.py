"""
Dataset tokenizer utility for tokenizing raw text data.
Handles train/test splitting and saves tokenized datasets to disk.
"""

import pandas as pd
from pathlib import Path
from typing import Tuple
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict


class DatasetTokenizer:
    """
    Tokenizes raw text data and saves as HuggingFace Dataset.

    Args:
        tokenizer_name: Name of the HuggingFace tokenizer to use
        output_dir: Directory where tokenized datasets will be saved
        max_length: Maximum sequence length for tokenization
        test_size: Fraction of data to use for test set
        random_state: Random seed for reproducibility
    """

    def __init__(
        self,
        tokenizer_name: str = "distilbert-base-uncased",
        output_dir: str = "data/processed",
        max_length: int = 512,
        test_size: float = 0.2,
        random_state: int = 42
    ):
        self.tokenizer_name = tokenizer_name
        self.output_dir = Path(output_dir)
        self.max_length = max_length
        self.test_size = test_size
        self.random_state = random_state

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def _preprocess_text(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Basic text preprocessing."""
        df[text_column] = df[text_column].apply(lambda x: x.lower().strip())
        return df

    def _split_data(
        self,
        df: pd.DataFrame,
        label_column: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets."""
        train_df, test_df = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=df[label_column]
        )
        return train_df, test_df

    def _tokenize_dataset(
        self,
        texts: list,
        labels: list,
        batch_size: int = 1000,
        desc: str = "Tokenizing"
    ) -> Dataset:
        """Tokenize texts and create HuggingFace Dataset."""
        # Create dataset
        dataset = Dataset.from_dict({"text": texts, "labels": labels})

        # Define tokenization function
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=self.max_length
            )

        # Tokenize in batches
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=batch_size,
            remove_columns=["text"],
            desc=desc
        )

        tokenized_dataset.set_format(
            type='torch',
            columns=['input_ids', 'attention_mask', 'labels']
        )

        return tokenized_dataset

    def tokenize_and_save(
        self,
        csv_path: str,
        text_column: str = "text",
        label_column: str = "generated",
        batch_size: int = 1000
    ):
        """
        Load CSV, tokenize, and save to disk.

        Args:
            csv_path: Path to raw CSV file
            text_column: Name of the text column
            label_column: Name of the label column
            batch_size: Batch size for tokenization
        """
        print(f"Loading data from {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} samples")

        df = self._preprocess_text(df, text_column)

        # Train/test split
        train_df, test_df = self._split_data(df, label_column)
        print(f"Train: {len(train_df)}, Test: {len(test_df)}")

        # Tokenize train/test
        train_dataset = self._tokenize_dataset(
            train_df[text_column].tolist(),
            train_df[label_column].tolist(),
            batch_size=batch_size,
            desc="Tokenizing train"
        )

        test_dataset = self._tokenize_dataset(
            test_df[text_column].tolist(),
            test_df[label_column].tolist(),
            batch_size=batch_size,
            desc="Tokenizing test"
        )

        dataset_dict = DatasetDict({
            'train': train_dataset,
            'test': test_dataset
        })

        output_path = self.output_dir / "tokenized_dataset"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_dict.save_to_disk(str(output_path))

        print(f"Saved to {output_path}")
        return output_path

