"""
Dataset tokenizer utility for tokenizing raw text data.
Handles train/test splitting and saves tokenized datasets to disk.
"""

from pathlib import Path
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
import pandas as pd


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
        tokenizer_name: str = "bert-base-uncased",
        output_dir: str = "data/processed",
        max_length: int = 512,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        self.tokenizer_name = tokenizer_name
        self.output_dir = Path(output_dir)
        self.max_length = max_length
        self.test_size = test_size
        self.random_state = random_state

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def _preprocess_dataset(self, ds):
        def preprocessing_function(row):
            row["text"] = row["text"].lower().strip()
            if row["label"] == 0 and len(row["text"]) > 800:
                row["text"] = row["text"][:-400]
            return row

        cleaned = ds.map(lambda x: preprocessing_function(x), batch_size=1000)
        return cleaned

    def _tokenize_dataset(
        self,
        dataset: Dataset,
        batch_size: int = 1000,
        desc: str = "Tokenizing",
    ) -> Dataset:
        """Tokenize texts and create HuggingFace Dataset."""

        # Define tokenization function
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )

        # Tokenize in batches
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=batch_size,
            remove_columns=["text"],
            desc=desc,
        )

        tokenized_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "label", "title_id"]
        )

        return tokenized_dataset

    def add_title_ids(self, ds):
        """
        Converts titles to their ids, in order to detect duplicates further in the pipeline. Requires entire datasetdict,
        to create a global id across test train and validation
        """
        train_titles = (
            pd.Series(ds["train"]["title"]).drop_duplicates().reset_index(drop=True)
        )
        test_titles = (
            pd.Series(ds["test"]["title"]).drop_duplicates().reset_index(drop=True)
        )
        validation_titles = (
            pd.Series(ds["validation"]["title"])
            .drop_duplicates()
            .reset_index(drop=True)
        )

        titles = pd.concat([train_titles, test_titles, validation_titles])

        title_id_map = {v: k for k, v in titles.items()}

        def map_title_to_id(example):
            example["title_id"] = title_id_map[example["title"]]
            return example

        ds = ds.map(map_title_to_id)
        return ds

    def tokenize_and_save(
        self,
        hf_base_dataset: Dataset,
        batch_size: int = 1000,
    ):
        """
        Load CSV, tokenize, and save to disk.

        Args:
            csv_path: Path to raw CSV file
            text_column: Name of the text column
            label_column: Name of the label column
            batch_size: Batch size for tokenization
        """
        print(f"Loaded {len(hf_base_dataset)} samples")

        hf_base_dataset = self.add_title_ids(hf_base_dataset)

        train_ds = self._preprocess_dataset(hf_base_dataset["train"])
        test_ds = self._preprocess_dataset(hf_base_dataset["test"])
        valid_ds = self._preprocess_dataset(hf_base_dataset["validation"])

        # Tokenize train/test
        train_tokens_dataset = self._tokenize_dataset(
            train_ds,
            batch_size=batch_size,
            desc="Tokenizing train",
        )

        test_tokens_dataset = self._tokenize_dataset(
            test_ds,
            batch_size=batch_size,
            desc="Tokenizing test",
        )

        valid_tokens_ds = self._tokenize_dataset(
            valid_ds,
            batch_size=batch_size,
            desc="Tokenizing validation",
        )

        dataset_dict = DatasetDict(
            {
                "train": train_tokens_dataset,
                "test": test_tokens_dataset,
                "validation": valid_tokens_ds,
            }
        )

        output_path = self.output_dir / "tokenized_dataset"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_dict.save_to_disk(str(output_path))

        print(f"Saved to {output_path}")
        return output_path
