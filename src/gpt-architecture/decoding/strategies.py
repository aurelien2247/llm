"""Stratégies de décodage pour la génération de texte.

Implémente différentes stratégies de contrôle de l'aléatoire lors de la génération:
- Greedy decoding (argmax)
- Temperature scaling
- Top-k sampling
- Top-p (nucleus) sampling
"""

import torch


def softmax_avec_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Appliquer le temperature scaling aux logits avant softmax.
    
    Args:
        logits: Tensor de logits bruts du modèle (batch_size, vocab_size).
        temperature: Facteur de température. 
                    - 1.0: aucun scaling (comportement normal)
                    - < 1.0: distribution plus nette (plus de confiance)
                    - > 1.0: distribution plus uniforme (plus d'aléatoire)
    
    Returns:
        Tensor de probabilités après softmax avec temperature scaling.
    
    Notes:
        - temperature = 0.1 → approche du greedy (argmax)
        - temperature = 5.0 → distribution très plate, plus de variété
    """
    if temperature <= 0:
        raise ValueError("La température doit être strictement positive")
    
    # Appliquer le temperature scaling
    logits_ajustes = logits / temperature
    
    # Appliquer softmax
    probas = torch.softmax(logits_ajustes, dim=-1)
    
    return probas


def echantillonner_multinomial(probas: torch.Tensor, num_echantillons: int = 1) -> torch.Tensor:
    """Échantillonner les token IDs selon une distribution de probabilité.
    
    Args:
        probas: Tensor de probabilités (batch_size, vocab_size).
        num_echantillons: Nombre d'échantillons à générer.
    
    Returns:
        Tensor d'indices des tokens échantillonnés.
    """
    return torch.multinomial(probas, num_samples=num_echantillons)


def top_k_sampling(logits: torch.Tensor, k: int = 50, temperature: float = 1.0) -> torch.Tensor:
    """Effectuer le top-k sampling pour filtrer les tokens peu probables.
    
    Args:
        logits: Tensor de logits bruts (batch_size, vocab_size).
        k: Nombre de tokens les plus probables à garder.
        temperature: Facteur de température.
    
    Returns:
        Tensor de probabilités filtrées (les autres probabilités mises à 0).
    
    Notes:
        Top-k sampling évite de générer des tokens très peu probables qui
        produisent souvent du texte sans sens. Par exemple:
        - k=50: garder les 50 tokens les plus probables
        - Les autres tokens obtiennent une probabilité de 0
    """
    # Obtenir les k plus grands logits
    top_k_logits, top_k_indices = torch.topk(logits, k=k, dim=-1)
    
    # Créer un tensor rempli de -inf pour masquer les tokens peu probables
    filtered_logits = torch.full_like(logits, float('-inf'))
    
    # Remplacer les logits des top-k tokens
    filtered_logits.scatter_(-1, top_k_indices, top_k_logits)
    
    # Appliquer temperature scaling et softmax
    probas = softmax_avec_temperature(filtered_logits, temperature)
    
    return probas


def top_p_sampling(logits: torch.Tensor, p: float = 0.9, temperature: float = 1.0) -> torch.Tensor:
    """Effectuer le nucleus (top-p) sampling pour un contrôle plus fin.
    
    Args:
        logits: Tensor de logits bruts (batch_size, vocab_size).
        p: Cumulative probability threshold (0 < p <= 1).
        temperature: Facteur de température.
    
    Returns:
        Tensor de probabilités filtrées.
    
    Notes:
        Le nucleus sampling (top-p) garde un ensemble de tokens dont la 
        probabilité cumulée atteint p. Cela évite de garder trop de tokens
        peu probables contrairement au top-k.
        
        Exemple avec p=0.9:
        - Les tokens dont la probabilité cumulée = 90% sont gardés
        - Les autres tokens obtiennent une probabilité de 0
    """
    # Appliquer temperature scaling
    scaled_logits = logits / temperature
    
    # Obtenir les probabilités
    probas = torch.softmax(scaled_logits, dim=-1)
    
    # Trier les probabilités en ordre décroissant
    sorted_probas, sorted_indices = torch.sort(probas, descending=True, dim=-1)
    
    # Calculer la probabilité cumulée
    cumsum_probas = torch.cumsum(sorted_probas, dim=-1)
    
    # Identifier les tokens à garder (probabilité cumulée <= p)
    # Ajouter un token supplémentaire pour s'assurer qu'on en garde au moins un
    sorted_indices_to_remove = cumsum_probas > p
    # Décaler d'un cran pour garder le premier token
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    
    # Créer un tensor de probabilités filtrées
    filtered_logits = torch.full_like(logits, float('-inf'))
    
    # Placer les logits des tokens gardés
    for i in range(logits.shape[0]):
        mask = ~sorted_indices_to_remove[i]
        filtered_logits[i, sorted_indices[i, mask]] = scaled_logits[i, sorted_indices[i, mask]]
    
    # Normaliser les probabilités
    probas_filtrees = torch.softmax(filtered_logits, dim=-1)
    
    return probas_filtrees


__all__ = [
    "softmax_avec_temperature",
    "echantillonner_multinomial",
    "top_k_sampling",
    "top_p_sampling"
]
