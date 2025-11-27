from sklearn import linear_model
import torch
from tqdm import tqdm
from datasets import load_from_disk
from transformers import BertModel
import pickle
import os
import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from utils.encoded_dataset_loader import get_encoded_dataloader

torch.manual_seed(42)

DEVICE = "cuda"
model_dir = "models"
optuna = True


def save_training_run(
    training_run_name: str,
    final_model: str,
    optuna_params: str = None,
):
    model_path = os.path.join(model_dir, training_run_name)
    with open(model_path, "wb") as wfile:
        pickle.dump(final_model, wfile)

    # append optuna params and current time to the csv


def test_model(test_dataloader, model, batch_size):
    """
    Iterates over entire test_dataloader and returns average score
    """

    total_right = 0
    for X, y in test_dataloader:
        X = X.squeeze().numpy()
        y = y.squeeze().numpy()
        y_pred = model.predict(X)
        total_right += (y_pred & y).sum()

    return total_right / len(test_dataloader.dataset)


def run_training(
    train_dataloader,
    test_dataloader,
    batch_size,
    model,
    training_run_name: str = None,
    epochs: int = 1,
):
    for epoch in range(epochs):
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for i, (X, y) in pbar:
            X = X.squeeze().numpy()
            y = y.squeeze().numpy()

            model.partial_fit(X, y, classes=[0, 1])

            if i % 2 == 0:
                train_score = model.score(X, y)
                test_score = test_model(test_dataloader, model, batch_size)

            pbar.set_description(
                f"train_score = {train_score}, test_score={test_score}"
            )

        print(f"Epoch {epoch} finished")

    save_training_run(training_run_name, model)


def main():
    model = linear_model.SGDClassifier(loss="hinge", random_state=42)
    # model = linear_model.PassiveAggressiveClassifier()

    encoder: BertModel = BertModel.from_pretrained("bert-base-uncased").to(DEVICE)

    batch_size = 64

    train_dataset = load_from_disk(
        "/home/microslaw/projects/Human-vs-AI-text-Text-Classification-Analysis-/data/processed/tokenized_dataset/train"
    )

    test_dataset = load_from_disk(
        "/home/microslaw/projects/Human-vs-AI-text-Text-Classification-Analysis-/data/processed/tokenized_dataset/test"
    )

    encoded_train_dataloader = get_encoded_dataloader(
        train_dataset,
        encoder,
        batch_size,
        device=DEVICE,
    )
    encoded_test_dataloader = get_encoded_dataloader(
        test_dataset,
        encoder,
        batch_size,
        device=DEVICE,
        subset_len=batch_size * 5,
    )

    run_training(
        train_dataloader=encoded_train_dataloader,
        test_dataloader=encoded_test_dataloader,
        batch_size=batch_size,
        model=model,
        training_run_name="test1",
    )


if __name__ == "__main__":
    main()
