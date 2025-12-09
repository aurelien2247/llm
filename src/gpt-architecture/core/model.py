"""Modèle GPT implémentant l'architecture Transformer."""

import torch
import torch.nn as nn
from .layers import LayerNorm, TransformerBlock


class GPTModel(nn.Module):
    """Modèle GPT implémentant l'architecture Transformer pour la génération de texte."""
    
    def __init__(self, cfg):
        """Initialiser le modèle GPT avec la configuration donnée.
        
        Args:
            cfg: Dictionnaire contenant les hyperparamètres du modèle.
        """
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        """Forward pass du modèle.
        
        Args:
            in_idx: Tenseur (batch_size, seq_len) contenant les indices des jetons.
        
        Returns:
            Logits de forme (batch_size, seq_len, vocab_size).
        """
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
