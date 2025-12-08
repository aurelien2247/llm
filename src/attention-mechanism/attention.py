import torch
import torch.nn as nn


# ----------------------------
# Attention causale mono-tête
# ----------------------------
class CausalAttention(nn.Module):
    """Mécanisme d'attention causale pour une seule tête."""

    def __init__(self, d_in, d_out, block_size, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        # Projections linéaires pour Query, Key, Value
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        # Masque causal triangulaire supérieur
        self.register_buffer('mask', torch.triu(torch.ones(block_size, block_size), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape 
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Calculer les scores d'attention
        attn_scores = queries @ keys.transpose(1, 2)  
        # Appliquer le masque causal
        attn_scores.masked_fill_( 
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        # Softmax avec facteur d'échelle
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights) 

        # Vecteur de contexte pondéré
        context_vec = attn_weights @ values
        return context_vec


# ------------------
# Wrapper multi-tête 
# ------------------
class MultiHeadAttentionWrapper(nn.Module):
    """Wrapper combinant plusieurs têtes d'attention."""

    def __init__(self, d_in, d_out, block_size, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, block_size, dropout, qkv_bias)
             for _ in range(num_heads)]
        )
        self.out_proj = nn.Linear(d_out*num_heads, d_out*num_heads)

    def forward(self, x):
        context_vec = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.out_proj(context_vec)


# --------------------
# Attention multi-tête 
# --------------------
class MultiHeadAttention(nn.Module):
    """Implémentation efficace de l'attention multi-tête."""
    def __init__(self, d_in, d_out, block_size, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out doit être divisible par num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads 

        # Projections linéaires pour Query, Key, Value
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  
        self.dropout = nn.Dropout(dropout)
        # Masque causal triangulaire supérieur
        self.register_buffer('mask', torch.triu(torch.ones(block_size, block_size), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # Projeter et transformer en multi-têtes
        keys = self.W_key(x)  
        queries = self.W_query(x)
        values = self.W_value(x)

        # Remodeler pour les têtes parallèles
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transposer pour paralléliser les calculs
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Calculer les scores d'attention
        attn_scores = queries @ keys.transpose(2, 3) 

        # Appliquer le masque causal
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # Softmax avec facteur d'échelle
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Pondérer les valeurs
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Concaténer les têtes et projeter
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec


