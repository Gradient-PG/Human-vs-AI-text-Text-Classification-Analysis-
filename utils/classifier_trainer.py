"""
Generic classifier trainer that works with any head model (sklearn).
Trains on pre-encoded datasets loaded from disk.
"""

import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_from_disk
from pathlib import Path
from tqdm import tqdm
from typing import Union
import time


# TODO: Add pytorch support
class ClassifierTrainer:
    """
    Generic trainer for classification heads (sklearn models).

    Args:
        head: The classifier model (sklearn)
        model_save_path: Path where trained model will be saved (optional, None for no saving)
        wandb_logger: Weights & Biases logger (optional)
    """

    def __init__(self, head, model_save_path: str = None, wandb_logger=None):
        self.head = head
        self.model_save_path = Path(model_save_path) if model_save_path else None
        self.wandb = wandb_logger
        self.is_sklearn_partial = hasattr(head, "partial_fit")
        self.is_sklearn_regular = hasattr(head, "fit")

    def load_encoded_dataset(self, encoded_dataset_path: str):
        """
        Load pre-encoded dataset from disk.

        Args:
            encoded_dataset_path: Path to encoded dataset directory
        """
        print(f"Loading encoded dataset from {encoded_dataset_path}...")
        dataset_dict = load_from_disk(encoded_dataset_path)

        self.train_dataset = dataset_dict["train"]
        self.test_dataset = dataset_dict["test"]

        print(f"Train: {len(self.train_dataset)}, Test: {len(self.test_dataset)}")

    def _evaluate_sklearn(self, dataset, batch_size: int) -> float:
        """Evaluate sklearn model on dataset."""
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        total_correct = 0
        total_samples = 0

        for batch in dataloader:
            X = batch["embeddings"].numpy()
            y = batch["labels"].numpy()

            y_pred = self.head.predict(X)
            total_correct += (y_pred == y).sum()
            total_samples += len(y)

        return total_correct / total_samples

    def train(
        self,
        epochs: int = 1,
        batch_size: int = 64,
        eval_every: int = 100,
    ):
        """
        Train the classifier head on encoded data.

        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            eval_every: Evaluate every N batches
        """
        if not hasattr(self, "train_dataset"):
            raise ValueError("Must call load_encoded_dataset() before train()")

        if self.is_sklearn_partial:
            self._train_sklearn_partial(epochs, batch_size, eval_every)
        elif self.is_sklearn_regular:
            self._train_sklearn_regular()
        else:
            raise NotImplementedError("PyTorch training not yet implemented")

    def _train_sklearn_regular(self):
        """Train sklearn model with partial_fit."""

        print("\nConverting training data to numpy...")
        X = self.train_dataset["embeddings"][0 : len(self.train_dataset)]

        print("\nConverting test data to numpy...")
        y = self.train_dataset["labels"][0 : len(self.train_dataset)]

        print("\nTraining...")
        start = time.time()
        print(start)
        self.head.fit(X, y)
        print(start - time.time())

        print("\nEvaluation...")
        final_train_score = self._evaluate_sklearn(self.train_dataset, 64)
        final_test_score = self._evaluate_sklearn(self.test_dataset, 64)

        print(
            f"Train accuracy: {final_train_score:.4f}, Test accuracy: {final_test_score:.4f}"
        )

        # Log final metrics to wandb
        if self.wandb:
            self.wandb.log(
                {
                    "final_train_acc": final_train_score,
                    "final_test_acc": final_test_score,
                }
            )

        # Save model only if path provided
        if self.model_save_path:
            self.save_model()

    def _train_sklearn_partial(self, epochs: int, batch_size: int, eval_every: int):
        """Train sklearn model with partial_fit."""
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        global_step = 0

        for epoch in range(epochs):
            pbar = tqdm(
                enumerate(train_dataloader),
                total=len(train_dataloader),
                desc=f"Epoch {epoch+1}/{epochs}",
            )

            for i, batch in pbar:
                X = batch["embeddings"].numpy()
                y = batch["labels"].numpy()

                # Train step
                self.head.partial_fit(X, y, classes=[0, 1])
                global_step += 1

                # Periodic evaluation
                if i % eval_every == 0:
                    train_score = self.head.score(X, y)
                    test_score = self._evaluate_sklearn(self.test_dataset, batch_size)

                    pbar.set_postfix(
                        {
                            "train_acc": f"{train_score:.4f}",
                            "test_acc": f"{test_score:.4f}",
                        }
                    )

                    # Log to wandb
                    if self.wandb:
                        self.wandb.log(
                            {
                                "train_acc": train_score,
                                "test_acc": test_score,
                                "epoch": epoch + 1,
                                "step": global_step,
                            }
                        )

        # Final evaluation
        print("\nFinal evaluation...")
        final_train_score = self._evaluate_sklearn(self.train_dataset, batch_size)
        final_test_score = self._evaluate_sklearn(self.test_dataset, batch_size)

        print(
            f"Train accuracy: {final_train_score:.4f}, Test accuracy: {final_test_score:.4f}"
        )

        # Log final metrics to wandb
        if self.wandb:
            self.wandb.log(
                {
                    "final_train_acc": final_train_score,
                    "final_test_acc": final_test_score,
                }
            )

        # Save model only if path provided
        if self.model_save_path:
            self.save_model()

    def save_model(self):
        """Save trained model to disk."""
        self.model_save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.model_save_path, "wb") as f:
            pickle.dump(self.head, f)

        print(f"Model saved to {self.model_save_path}")
