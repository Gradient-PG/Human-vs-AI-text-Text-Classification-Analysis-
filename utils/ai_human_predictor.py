import pickle
import torch
import numpy as np
from pathlib import Path
import plotly.express as px
import warnings


class AiHumanPredictor:
    def __init__(self, tokenizer, encoder, model_path: str = None, device: str = "cpu"):
        """
        Predicts whether the text was human-written or ai-generated.
        Returns 0 if human and 1 if AI
        """
        self.tokenizer = tokenizer
        self.encoder = encoder.to(device)
        self.encoder.eval()
        self.model_save_path = Path(model_path) if model_path else None
        self.device = device
        self.signed_weights_head = None
        self.feature_weights: bool = None

    def forward(self, text) -> float:
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
        prediction = self.head.predict(embeddings.reshape(1, -1).cpu().numpy())
        return prediction

    def load_head(self):
        """Save trained model to disk."""
        with open(self.model_save_path, "rb") as f:
            head_candidate = pickle.load(f)

            if hasattr(head_candidate, "coef_"):
                self.feature_weights = head_candidate.coef_.ravel()
                self.signed_weights_head = True
            elif hasattr(head_candidate, "feature_importances_"):
                self.feature_weights = head_candidate.feature_importances_.ravel()
                raise warnings.warn(
                    f"Warning: Model of type {type(head_candidate).__name__} does not have signed feature weights"
                )
            else:
                raise TypeError(
                    "Head needs to have either 'coef_' or 'feature_importances_' attribute"
                )

        self.head = head_candidate
        print(f"Head model loaded from {self.model_save_path} successfully")

    def get_token_importance_by_gradients(self, text):
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        )
        input_ids = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device)

        self.encoder.zero_grad()
        # self.encoder.train() ideally should be used, but activates dropout which randomizes outputs

        embeddings = self.encoder.embeddings.word_embeddings(
            input_ids
        )
        embeddings = embeddings.detach()
        embeddings.requires_grad_(True)
        out = self.encoder(inputs_embeds=embeddings, attention_mask=mask)
        hs = out.last_hidden_state[0]
        obj = hs[0, :].sum()
        obj.backward()
        grads = embeddings.grad[0]
        self.encoder.zero_grad()
        self.encoder.eval()

        inv_vocab = {v: k for k, v in self.tokenizer.vocab.items()}
        tokens = [inv_vocab[item.item()] for item in input_ids.ravel()]
        importance = (grads.cpu().numpy() * self.feature_weights).mean(axis=1)

        return tokens, importance

    def visualize_token_importance(self, text, file_path: str = None):
        """
        Generates a plot describing which tokens contributed to model's decision.
        Will save html plot in file_path. If file_path = None, will return the figure
        """
        tokens, importance = self.get_token_importance_by_gradients(text)

        fig = px.imshow(
            importance.reshape(16, -1),
            color_continuous_midpoint=0,
            color_continuous_scale="RdBu",
        )
        fig.data[0].text = np.array(tokens).reshape(16, -1)
        fig.data[0].texttemplate = "%{text}"

        if file_path is None:
            return fig

        fig.write_html(file_path)
