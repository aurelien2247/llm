"""
Script de démonstration des stratégies de décodage.

Montre la différence entre:
- Greedy decoding (argmax)
- Temperature scaling
- Top-k sampling
- Top-p (nucleus) sampling

Utilisation:
    python3 demo.py
"""

import sys
import torch
import tiktoken

sys.path.insert(0, '..')
from config import GPT_CONFIG_124M, PATHS
from core import GPTModel, text_to_token_ids, token_ids_to_text
from decoding import (
    generate_text_simple,
    generate_text_temperature,
    generate_text_top_k,
    generate_text_top_p
)


def main():
    """Fonction principale de démonstration."""
    # Charger le modèle entraîné
    print("Chargement du modèle entraîné...")
    device = torch.device("cpu")
    model = GPTModel(GPT_CONFIG_124M)
    
    try:
        model.load_state_dict(torch.load(PATHS['model_checkpoint'], map_location=device))
        model.to(device)
        model.eval()
        print("✓ Modèle chargé avec succès\n")
    except FileNotFoundError:
        print(f"✗ ERREUR: {PATHS['model_checkpoint']} non trouvé")
        print("  Veuillez d'abord entraîner le modèle avec: python3 train.py")
        return

    # Configuration
    tokenizer = tiktoken.get_encoding("gpt2")
    start_context = "Every effort moves you"
    max_tokens = 50

    print(f"{'='*80}")
    print(f"DÉMONSTRATION DES STRATÉGIES DE DÉCODAGE")
    print(f"{'='*80}\n")
    print(f"Contexte initial: \"{start_context}\"")
    print(f"Tokens à générer: {max_tokens}\n")

    # 1. Greedy decoding (argmax)
    print(f"{'-'*80}")
    print("1. GREEDY DECODING (argmax)")
    print(f"{'-'*80}")
    print("Description: Sélectionne le token le plus probable à chaque étape.")
    print("Résultat: Texte déterministe et cohérent, mais peu varié.\n")
    
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model,
            idx=encoded,
            max_new_tokens=max_tokens,
            context_size=GPT_CONFIG_124M["context_length"]
        )
    text_greedy = token_ids_to_text(token_ids, tokenizer)
    print("Texte généré:")
    print(f"  {text_greedy}\n")

    # 2. Temperature scaling (faible)
    print(f"{'-'*80}")
    print("2. TEMPERATURE SCALING (température=0.3)")
    print(f"{'-'*80}")
    print("Description: Distribution très nette, approche du greedy mais avec légère variété.")
    print("Résultat: Texte cohérent avec un peu plus de variété.\n")
    
    with torch.no_grad():
        token_ids = generate_text_temperature(
            model=model,
            idx=encoded,
            max_new_tokens=max_tokens,
            context_size=GPT_CONFIG_124M["context_length"],
            temperature=0.3
        )
    text_low_temp = token_ids_to_text(token_ids, tokenizer)
    print("Texte généré:")
    print(f"  {text_low_temp}\n")

    # 3. Temperature scaling (normal)
    print(f"{'-'*80}")
    print("3. TEMPERATURE SCALING (température=1.0)")
    print(f"{'-'*80}")
    print("Description: Pas de scaling. Probabilités normales.")
    print("Résultat: Équilibre entre cohérence et variété.\n")
    
    with torch.no_grad():
        token_ids = generate_text_temperature(
            model=model,
            idx=encoded,
            max_new_tokens=max_tokens,
            context_size=GPT_CONFIG_124M["context_length"],
            temperature=1.0
        )
    text_normal_temp = token_ids_to_text(token_ids, tokenizer)
    print("Texte généré:")
    print(f"  {text_normal_temp}\n")

    # 4. Temperature scaling (élevée)
    print(f"{'-'*80}")
    print("4. TEMPERATURE SCALING (température=2.0)")
    print(f"{'-'*80}")
    print("Description: Distribution très uniforme, plus d'aléatoire.")
    print("Résultat: Texte plus varié mais parfois incohérent.\n")
    
    with torch.no_grad():
        token_ids = generate_text_temperature(
            model=model,
            idx=encoded,
            max_new_tokens=max_tokens,
            context_size=GPT_CONFIG_124M["context_length"],
            temperature=2.0
        )
    text_high_temp = token_ids_to_text(token_ids, tokenizer)
    print("Texte généré:")
    print(f"  {text_high_temp}\n")

    # 5. Top-k sampling
    print(f"{'-'*80}")
    print("5. TOP-K SAMPLING (k=50, température=1.0)")
    print(f"{'-'*80}")
    print("Description: Garde seulement les 50 tokens les plus probables.")
    print("Résultat: Texte varié mais sans tokens complètement absurdes.\n")
    
    with torch.no_grad():
        token_ids = generate_text_top_k(
            model=model,
            idx=encoded,
            max_new_tokens=max_tokens,
            context_size=GPT_CONFIG_124M["context_length"],
            k=50,
            temperature=1.0
        )
    text_top_k = token_ids_to_text(token_ids, tokenizer)
    print("Texte généré:")
    print(f"  {text_top_k}\n")

    # 6. Top-p (nucleus) sampling
    print(f"{'-'*80}")
    print("6. TOP-P SAMPLING (p=0.9, température=1.0)")
    print(f"{'-'*80}")
    print("Description: Garde les tokens représentant 90% de la probabilité cumulée.")
    print("Résultat: Équilibre flexible entre qualité et variété.\n")
    
    with torch.no_grad():
        token_ids = generate_text_top_p(
            model=model,
            idx=encoded,
            max_new_tokens=max_tokens,
            context_size=GPT_CONFIG_124M["context_length"],
            p=0.9,
            temperature=1.0
        )
    text_top_p = token_ids_to_text(token_ids, tokenizer)
    print("Texte généré:")
    print(f"  {text_top_p}\n")

    # Résumé
    print(f"{'='*80}")
    print("RÉSUMÉ DES STRATÉGIES")
    print(f"{'='*80}\n")
    
    print("┌─────────────────────────┬──────────────────┬────────────────────┐")
    print("│ Stratégie               │ Cohérence        │ Variété            │")
    print("├─────────────────────────┼──────────────────┼────────────────────┤")
    print("│ Greedy (argmax)         │ ████████████░░░░ │ ██░░░░░░░░░░░░░░░░ │")
    print("│ Temperature (T=0.3)     │ ████████████░░░░ │ ████░░░░░░░░░░░░░░ │")
    print("│ Temperature (T=1.0)     │ ██████████░░░░░░ │ ████████░░░░░░░░░░ │")
    print("│ Temperature (T=2.0)     │ ████████░░░░░░░░ │ ████████████░░░░░░ │")
    print("│ Top-k (k=50)            │ ██████████░░░░░░ │ ████████░░░░░░░░░░ │")
    print("│ Top-p (p=0.9)           │ ██████████░░░░░░ │ ████████░░░░░░░░░░ │")
    print("└─────────────────────────┴──────────────────┴────────────────────┘")
    print()
    print("Recommandations:")
    print("  • Qualité élevée: temperature=0.3 ou top-k (k=30-50)")
    print("  • Équilibre: temperature=1.0 ou top-p (p=0.9)")
    print("  • Créativité: temperature=1.5-2.0 ou top-p (p=0.95)")
    print()


if __name__ == "__main__":
    main()
