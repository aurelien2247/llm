"""
Module decoding: Stratégies de génération de texte.

Contient les stratégies de décoding pour contrôler la génération:
- strategies.py: Fonctions de scaling (temperature, top-k, top-p)
- generator.py: Fonctions de génération (generate_text_*)
"""

from .strategies import (
    softmax_avec_temperature,
    echantillonner_multinomial,
    top_k_sampling,
    top_p_sampling
)
from .generator import (
    generate_text_simple,
    generate_text_temperature,
    generate_text_top_k,
    generate_text_top_p
)

__all__ = [
    "softmax_avec_temperature",
    "echantillonner_multinomial",
    "top_k_sampling",
    "top_p_sampling",
    "generate_text_simple",
    "generate_text_temperature",
    "generate_text_top_k",
    "generate_text_top_p"
]
