"""Fonctions de génération de texte avec différentes stratégies de décoding."""

import torch
from .strategies import softmax_avec_temperature, echantillonner_multinomial, top_k_sampling, top_p_sampling


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    """Générer du texte avec support complet: temperature, top-k, early stopping.
    
    Cette fonction combine toutes les techniques de génération en une seule interface unifiée.
    
    Args:
        model: Le modèle GPT à utiliser pour la génération.
        idx: Tenseur (B, T) contenant les indices des jetons du contexte initial.
        max_new_tokens: Nombre maximal de jetons à générer.
        context_size: Taille du contexte maximal.
        temperature: Facteur de température (0.0 = greedy, >0 = probabiliste).
                    - 0.0: Greedy decoding (argmax) - déterministe
                    - 0.1-0.5: Distribution nette, texte cohérent
                    - 0.7-1.0: Équilibre cohérence/variété
                    - 1.5-2.0: Plus créatif, moins prévisible
        top_k: Nombre de tokens les plus probables à conserver (None = désactivé).
              Élimine les tokens absurdes en ne gardant que les k meilleurs.
              - None: Pas de filtrage
              - 10-30: Très conservateur
              - 40-60: Équilibre (recommandé)
              - 100+: Plus permissif
        eos_id: Token ID de fin de séquence (None = désactivé).
               Si détecté, arrête la génération immédiatement.
    
    Returns:
        Tenseur (B, T+N) avec les jetons générés (N <= max_new_tokens si early stop).
    
    Notes:
        Cette fonction implémente le pipeline complet de génération:
        1. Forward pass du modèle → logits
        2. Top-k filtering (si activé) → masque tokens peu probables
        3. Temperature scaling (si activé) → contrôle variété
        4. Sampling ou greedy → sélection du prochain token
        5. Early stopping (si eos_id détecté) → arrêt anticipé
        
        Exemples d'utilisation:
        - Greedy pur: temperature=0.0, top_k=None
        - Qualité: temperature=0.7, top_k=50
        - Créatif: temperature=1.5, top_k=100
    """
    # Boucle de génération
    for _ in range(max_new_tokens):
        # Étape 1: Forward pass (même logique que generate_simple)
        # Extraire les derniers context_size jetons
        idx_cond = idx[:, -context_size:]
        
        with torch.no_grad():
            logits = model(idx_cond)
        
        # Se concentrer sur le dernier time step
        logits = logits[:, -1, :]  # (batch_size, vocab_size)

        # Étape 2: NOUVEAU - Filtrer les logits avec top-k sampling
        if top_k is not None:
            # Garder seulement les top_k valeurs
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]  # Plus petit des top-k
            # Masquer tous les logits < min_val avec -inf
            logits = torch.where(
                logits < min_val,
                torch.tensor(float("-inf")).to(logits.device),
                logits
            )

        # Étape 3: NOUVEAU - Appliquer temperature scaling
        if temperature > 0.0:
            # Diviser les logits par la température
            logits = logits / temperature

            # Appliquer softmax pour obtenir les probabilités
            probs = torch.softmax(logits, dim=-1)  # (batch_size, vocab_size)

            # Échantillonner depuis la distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Étape 4: Sinon, utiliser greedy decoding comme avant
        else:
            # Sélectionner l'index du token avec le plus haut logit
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        # Étape 5: NOUVEAU - Arrêt anticipé si token de fin détecté
        if eos_id is not None and idx_next == eos_id:
            break

        # Comme avant: ajouter l'index échantillonné à la séquence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """Générer du texte de manière simple en utilisant greedy decoding (argmax).
    
    Args:
        model: Le modèle GPT à utiliser pour la génération.
        idx: Tenseur (B, T) contenant les indices des jetons du contexte initial.
        max_new_tokens: Nombre de jetons à générer.
        context_size: Taille du contexte maximal.
    
    Returns:
        Tenseur (B, T+max_new_tokens) avec les jetons générés.
    
    Notes:
        Utilise le greedy decoding (argmax) pour sélectionner le token le plus probable.
        Voir generate_text_temperature pour une génération plus variée.
    """
    for _ in range(max_new_tokens):
        # Extraire les derniers context_size jetons
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        # Obtenir les logits du dernier jeton et prédire le suivant
        logits = logits[:, -1, :]
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def generate_text_temperature(model, idx, max_new_tokens, context_size, temperature=1.0):
    """Générer du texte avec temperature scaling pour contrôler la variété.
    
    Args:
        model: Le modèle GPT à utiliser pour la génération.
        idx: Tenseur (B, T) contenant les indices des jetons du contexte initial.
        max_new_tokens: Nombre de jetons à générer.
        context_size: Taille du contexte maximal.
        temperature: Facteur de température.
                    - 1.0: comportement normal (pas de scaling)
                    - < 1.0: distribution plus nette (moins d'aléatoire, plus de cohérence)
                    - > 1.0: distribution plus plate (plus d'aléatoire, plus de variété)
    
    Returns:
        Tenseur (B, T+max_new_tokens) avec les jetons générés.
    
    Notes:
        - temperature=0.1: très confiant, approche du greedy decoding
        - temperature=1.0: normal
        - temperature=5.0: très aléatoire, génère du texte parfois incohérent
    """
    for _ in range(max_new_tokens):
        # Extraire les derniers context_size jetons
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        # Obtenir les logits du dernier jeton
        logits = logits[:, -1, :]
        
        # Appliquer temperature scaling et softmax
        probas = softmax_avec_temperature(logits, temperature=temperature)
        
        # Échantillonner le prochain jeton selon la distribution
        idx_next = echantillonner_multinomial(probas, num_echantillons=1)
        
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def generate_text_top_k(model, idx, max_new_tokens, context_size, k=50, temperature=1.0):
    """Générer du texte avec top-k sampling.
    
    Args:
        model: Le modèle GPT à utiliser pour la génération.
        idx: Tenseur (B, T) contenant les indices des jetons du contexte initial.
        max_new_tokens: Nombre de jetons à générer.
        context_size: Taille du contexte maximal.
        k: Nombre de tokens les plus probables à conserver.
        temperature: Facteur de température.
    
    Returns:
        Tenseur (B, T+max_new_tokens) avec les jetons générés.
    
    Notes:
        Le top-k sampling évite de générer des tokens très peu probables
        qui produisent souvent du texte sans sens. Les tokens au-delà des
        k plus probables obtiennent une probabilité de 0.
        
        Exemple: k=50 garde les 50 tokens les plus probables.
    """
    for _ in range(max_new_tokens):
        # Extraire les derniers context_size jetons
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        # Obtenir les logits du dernier jeton
        logits = logits[:, -1, :]
        
        # Appliquer top-k sampling
        probas = top_k_sampling(logits, k=k, temperature=temperature)
        
        # Échantillonner le prochain jeton
        idx_next = echantillonner_multinomial(probas, num_echantillons=1)
        
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def generate_text_top_p(model, idx, max_new_tokens, context_size, p=0.9, temperature=1.0):
    """Générer du texte avec nucleus (top-p) sampling.
    
    Args:
        model: Le modèle GPT à utiliser pour la génération.
        idx: Tenseur (B, T) contenant les indices des jetons du contexte initial.
        max_new_tokens: Nombre de jetons à générer.
        context_size: Taille du contexte maximal.
        p: Cumulative probability threshold (0 < p <= 1).
        temperature: Facteur de température.
    
    Returns:
        Tenseur (B, T+max_new_tokens) avec les jetons générés.
    
    Notes:
        Le nucleus sampling (top-p) garde les tokens dont la probabilité
        cumulée atteint p. C'est plus flexible que top-k car le nombre de
        tokens conservés varie selon la distribution.
        
        Exemple: p=0.9 garde les tokens représentant 90% de la probabilité.
    """
    for _ in range(max_new_tokens):
        # Extraire les derniers context_size jetons
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        # Obtenir les logits du dernier jeton
        logits = logits[:, -1, :]
        
        # Appliquer top-p sampling
        probas = top_p_sampling(logits, p=p, temperature=temperature)
        
        # Échantillonner le prochain jeton
        idx_next = echantillonner_multinomial(probas, num_echantillons=1)
        
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


__all__ = [
    "generate",  # Fonction unifiée (recommandée)
    "generate_text_simple",
    "generate_text_temperature",
    "generate_text_top_k",
    "generate_text_top_p"
]
