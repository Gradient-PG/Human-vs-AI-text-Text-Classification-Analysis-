"""
Utility modules for the Human vs AI Text Interpretability project.
"""

from pathlib import Path

from .activation_extractor import ActivationExtractor
from .dataset_tokenizer import DatasetTokenizer
from .model import load_bert_model
from .raid_loader import ALL_DOMAINS, ALL_RAID_MODELS, RAIDConfig, load_raid, slug

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def project_root() -> Path:
    """Return the repository root (one level above ``utils/``)."""
    return _PROJECT_ROOT


__all__ = [
    "ALL_DOMAINS",
    "ALL_RAID_MODELS",
    "ActivationExtractor",
    "DatasetTokenizer",
    "RAIDConfig",
    "load_bert_model",
    "load_raid",
    "project_root",
    "slug",
]
