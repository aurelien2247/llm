import torch


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """Générer du texte de manière simple en utilisant le modèle.
    
    Args:
        model: Le modèle GPT à utiliser pour la génération.
        idx: Tenseur (B, T) contenant les indices des jetons du contexte initial.
        max_new_tokens: Nombre de jetons à générer.
        context_size: Taille du contexte maximal.
    
    Returns:
        Tenseur (B, T+max_new_tokens) avec les jetons générés.
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
