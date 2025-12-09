"""
Module core: Architecture du modèle GPT et composants.

Contient l'implémentation du modèle Transformer:
- model.py: Classe GPTModel
- layers.py: Couches (LayerNorm, GELU, FeedForward, TransformerBlock)
- attention.py: Attention multi-têtes avec masque causal
- utils.py: Utilitaires (text_to_token_ids, token_ids_to_text, calc_loss_*, etc.)
"""

from .model import GPTModel
from .layers import LayerNorm, GELU, FeedForward, TransformerBlock
from .attention import MultiHeadAttention
from .utils import (
    text_to_token_ids,
    token_ids_to_text,
    calc_loss_batch,
    calc_loss_loader
)

__all__ = [
    "GPTModel",
    "LayerNorm",
    "GELU",
    "FeedForward",
    "TransformerBlock",
    "MultiHeadAttention",
    "text_to_token_ids",
    "token_ids_to_text",
    "calc_loss_batch",
    "calc_loss_loader"
]
