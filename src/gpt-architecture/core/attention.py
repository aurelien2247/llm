"""Implémentation de l'attention multi-têtes avec masque causal."""

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """Attention multi-têtes avec masque causal pour prévenir l'attention sur les jetons futurs."""
    
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        """Initialiser la couche d'attention multi-têtes.
        
        Args:
            d_in: Dimension d'entrée.
            d_out: Dimension de sortie.
            context_length: Longueur maximale du contexte.
            dropout: Taux de dropout.
            num_heads: Nombre de têtes d'attention.
            qkv_bias: Utiliser des biais pour les projections query/key/value.
        """
        super().__init__()
        assert d_out % num_heads == 0, "d_out doit être divisible par num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        # Masque causal triangulaire supérieur
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        """Forward pass de l'attention multi-têtes.
        
        Args:
            x: Tenseur d'entrée de forme (batch_size, seq_len, d_in).
        
        Returns:
            Tenseur de contexte de forme (batch_size, seq_len, d_out).
        """
        b, num_tokens, d_in = x.shape

        # Transformer les vecteurs d'entrée en clés, requêtes et valeurs
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Calculer les scores d'attention
        attn_scores = queries @ keys.transpose(2, 3)

        # Appliquer le masque causal pour prévenir l'attention sur les jetons futurs
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / (keys.shape[-1] ** 0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec
