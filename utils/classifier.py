import pickle
import torch
import numpy as np
from pathlib import Path
import plotly.express as px
import warnings


class AiHumanPredictor:
    def __init__(self, tokenizer, encoder, model_path: str = None, device: str = "cpu"):
        self.tokenizer = tokenizer
        self.encoder = encoder.to(device)
        self.encoder.eval()
        self.model_save_path = Path(model_path) if model_path else None
        self.device = device
        self.signed_weights_head = None
        self.feature_weights: bool = None
        self.feature_bias: float = None

        self._load_head()

    def forward(self, text) -> float:
        """
        Predicts whether the text was human-written or ai-generated.
        Returns 0 if human and 1 if AI
        """
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            encoder_output = self.encoder(
                tokens["input_ids"], attention_mask=tokens["attention_mask"]
            )
        embeddings = encoder_output.last_hidden_state[:, 0, :]
        prediction = self.head.predict(embeddings.reshape(1, -1).cpu().numpy())[0]
        return prediction

    def _load_head(self):
        """Load trained model from disk."""
        with open(self.model_save_path, "rb") as f:
            head_candidate = pickle.load(f)

            if hasattr(head_candidate, "coef_"):
                self.feature_weights = head_candidate.coef_.ravel()
                self.feature_bias = head_candidate.intercept_
                self.signed_weights_head = True
            elif hasattr(head_candidate, "feature_importances_"):
                self.feature_weights = head_candidate.feature_importances_.ravel()
                warnings.warn(
                    f"Warning: Model of type {type(head_candidate).__name__} does not have signed feature weights"
                )
            else:
                raise TypeError(
                    "Head needs to have either 'coef_' or 'feature_importances_' attribute"
                )

        self.head = head_candidate
        print(f"Head model loaded from {self.model_save_path} successfully")

