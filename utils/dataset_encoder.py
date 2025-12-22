"""
Dataset encoder utility for encoding tokenized datasets using any encoder model.
Saves encoded embeddings to disk for later use in training.
"""

import torch
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict, load_from_disk
from pathlib import Path
from tqdm import tqdm
import numpy as np


class DatasetEncoder:
    """
    Encodes tokenized datasets using a specified encoder model and saves to disk.
    Uses CLS token pooling for BERT-family models, mean pooling otherwise.

    Args:
        encoder: The encoder model (e.g., BertModel, any model with forward pass)
        output_dir: Directory where encoded datasets will be saved
        device: Device to run encoding on ('cuda' or 'cpu')
    """

    def __init__(self, encoder, output_dir: str, device: str = "cuda"):
        self.encoder = encoder
        self.output_dir = Path(output_dir)
        self.device = device
        self.encoder.to(device)
        self.encoder.eval()

        # Detect if encoder is BERT-family (has CLS token)
        self.use_cls = self._is_bert_family(encoder)
        pooling_type = "CLS token" if self.use_cls else "mean"
        print(f"Pooling strategy: {pooling_type}")

    def _is_bert_family(self, encoder) -> bool:
        """Check if encoder is from BERT family (supports CLS token)."""
        model_name = encoder.__class__.__name__.lower()
        bert_variants = ["bert", "roberta", "distilbert", "albert", "electra"]
        return any(variant in model_name for variant in bert_variants)

    def encode_dataset(
        self,
        dataset: Dataset,
        batch_size: int = 64,
        desc: str = "Encoding",
    ) -> Dataset:
        """
        Encode a single dataset and return as HuggingFace Dataset.

        Args:
            dataset: Tokenized dataset with 'input_ids', 'attention_mask', 'label'
            batch_size: Batch size for encoding
            desc: Description for progress bar

        Returns:
            Dataset with 'embeddings' and 'label'
        """
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_embeddings = []
        all_labels = []
        all_title_ids = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=desc):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"]
                title_ids = batch["title_id"]

                # Get encoder output
                output = self.encoder(
                    input_ids=input_ids, attention_mask=attention_mask
                )

                # Apply pooling: (batch_size, seq_len, hidden_size) -> (batch_size, hidden_size)
                if self.use_cls:
                    # CLS token pooling (first token)
                    embeddings = output.last_hidden_state[:, 0, :]
                else:
                    # Mean pooling over sequence length
                    embeddings = self._mean_pooling(
                        output.last_hidden_state, attention_mask
                    )

                all_embeddings.append(embeddings.cpu().numpy())
                all_labels.append(labels.numpy())
                all_title_ids.append(title_ids.numpy())

        # Concatenate all batches
        embeddings_array = np.vstack(all_embeddings)
        labels_array = np.concatenate(all_labels)
        title_ids_array = np.concat(all_title_ids)

        # Create new dataset
        encoded_dataset = Dataset.from_dict(
            {
                "embeddings": embeddings_array,
                "label": labels_array,
                "title_ids": title_ids_array,
            }
        )

        # Set format to numpy for efficient loading
        encoded_dataset.set_format(type="numpy", columns=["embeddings", "label"])

        return encoded_dataset

    def _mean_pooling(self, token_embeddings, attention_mask):
        """
        Perform mean pooling over sequence length, considering attention mask.

        Args:
            token_embeddings: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len)

        Returns:
            pooled: (batch_size, hidden_size)
        """
        # Expand attention mask to match embeddings shape
        mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )

        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)

        # Average
        return sum_embeddings / sum_mask

    def encode_and_save(self, tokenized_dataset_path: str, batch_size: int = 64):
        """
        Load tokenized dataset, encode it, and save to output_dir.

        Args:
            tokenized_dataset_path: Path to tokenized dataset
            batch_size: Batch size for encoding
        """
        print(f"Loading tokenized dataset from {tokenized_dataset_path}...")
        dataset_dict = load_from_disk(tokenized_dataset_path)

        encoded_dict = {}

        if "train" in dataset_dict:
            print("\nEncoding train split...")
            encoded_dict["train"] = self.encode_dataset(
                dataset_dict["train"], batch_size=batch_size, desc="Encoding train"
            )

        if "test" in dataset_dict:
            print("\nEncoding test split...")
            encoded_dict["test"] = self.encode_dataset(
                dataset_dict["test"], batch_size=batch_size, desc="Encoding test"
            )

        if "validation" in dataset_dict:
            print("\nEncoding test split...")
            encoded_dict["validation"] = self.encode_dataset(
                dataset_dict["validation"],
                batch_size=batch_size,
                desc="Encoding validation",
            )

        encoded_dataset_dict = DatasetDict(encoded_dict)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        encoded_dataset_dict.save_to_disk(str(self.output_dir))

        print(f"Encoded and saved to {self.output_dir}")
