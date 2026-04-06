"""Load frozen BERT model for activation extraction and experiment use."""

from __future__ import annotations

import torch
from transformers import AutoModel, AutoTokenizer


def load_bert_model(
    model_name: str = "bert-base-uncased",
    device: str | None = None,
) -> tuple[AutoModel, AutoTokenizer]:
    """
    Load a frozen BERT model and its tokenizer.

    The model is set to eval mode with ``output_hidden_states=True`` so that
    all 12 layer hidden states are accessible in the forward pass output.

    Args:
        model_name: HuggingFace model identifier.
        device: ``"cuda"``, ``"cpu"``, or ``None`` (auto-detect).

    Returns:
        (model, tokenizer) — model already on ``device`` in eval mode.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model.to(device)
    model.eval()

    return model, tokenizer
