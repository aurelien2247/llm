"""
Test de la fonction generate() unifiée (suivant le tutoriel).

Ce script teste la nouvelle fonction generate() qui combine:
- Temperature scaling
- Top-k sampling
- Early stopping

Reproduit l'exemple du tutoriel:
    generate(model, idx, max_new_tokens=15, context_size=..., top_k=25, temperature=1.4)

Utilisation:
    python3 test_generate_unified.py
"""

import sys
import os
import torch
import tiktoken

sys.path.insert(0, '..')

from config import GPT_CONFIG_124M, PATHS
from core import GPTModel, text_to_token_ids, token_ids_to_text
from messages import model_loaded_from, error_not_found, MISSING_CHECKPOINT_HELP
from decoding import generate


def main():
    """Test de la fonction generate() unifiée."""
    
    print("=" * 80)
    print("TEST DE LA FONCTION generate() UNIFIÉE")
    print("=" * 80)
    print()
    
    # Charger le modèle
    print("📦 Chargement du modèle...")
    device = torch.device("cpu")
    model = GPTModel(GPT_CONFIG_124M)

    # Plusieurs chemins candidats selon d'où on exécute le script
    candidates = [
        os.path.join("..", PATHS["model_checkpoint"]),        # ../gpt-model.pt (attendu depuis src/gpt-architecture/scripts)
        PATHS["model_checkpoint"],                              # gpt-model.pt (cwd)
        os.path.join(".", PATHS["model_checkpoint"]),         # ./gpt-model.pt
        os.path.join("..", "scripts", PATHS["model_checkpoint"]),
        os.path.join("scripts", PATHS["model_checkpoint"]),
    ]

    loaded = False
    for p in candidates:
        try:
            if os.path.exists(p):
                model.load_state_dict(torch.load(p, map_location=device))
                model.to(device)
                model.eval()
                print(model_loaded_from(p))
                loaded = True
                break
        except Exception:
            # Si torch.load échoue pour une autre raison, continuer et afficher l'erreur plus bas
            continue

    if not loaded:
        print(error_not_found(PATHS['model_checkpoint']) + f" (recherché: {', '.join(candidates)})")
        print(MISSING_CHECKPOINT_HELP + " ou placer le checkpoint dans l'un des chemins ci-dessus.")
        return
    
    # Configuration
    tokenizer = tiktoken.get_encoding("gpt2")
    start_context = "Every effort moves you"
    
    print(f"Contexte initial: \"{start_context}\"")
    print(f"Max tokens: 15")
    print()
    
    # Encoder le contexte
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    
    # Test 1: Greedy (baseline)
    print("-" * 80)
    print("TEST 1: GREEDY DECODING (baseline)")
    print("-" * 80)
    print("Paramètres: temperature=0.0 (désactivé), top_k=None")
    print()
    
    torch.manual_seed(123)
    token_ids = generate(
        model=model,
        idx=encoded,
        max_new_tokens=15,
        context_size=GPT_CONFIG_124M["context_length"],
        temperature=0.0,  # Greedy
        top_k=None
    )
    text = token_ids_to_text(token_ids, tokenizer)
    print(f"Résultat: {text}")
    print()
    
    # Test 2: Temperature seule
    print("-" * 80)
    print("TEST 2: TEMPERATURE SCALING SEUL")
    print("-" * 80)
    print("Paramètres: temperature=1.4, top_k=None")
    print("Effet: Plus de variété, mais risque de tokens absurdes")
    print()
    
    torch.manual_seed(123)
    token_ids = generate(
        model=model,
        idx=encoded,
        max_new_tokens=15,
        context_size=GPT_CONFIG_124M["context_length"],
        temperature=1.4,
        top_k=None
    )
    text = token_ids_to_text(token_ids, tokenizer)
    print(f"Résultat: {text}")
    print()
    
    # Test 3: Top-k seul
    print("-" * 80)
    print("TEST 3: TOP-K SAMPLING SEUL")
    print("-" * 80)
    print("Paramètres: temperature=0.0 (greedy sur top-k), top_k=25")
    print("Effet: Élimine tokens absurdes, mais reste déterministe")
    print()
    
    torch.manual_seed(123)
    token_ids = generate(
        model=model,
        idx=encoded,
        max_new_tokens=15,
        context_size=GPT_CONFIG_124M["context_length"],
        temperature=0.0,  # Greedy dans le top-k
        top_k=25
    )
    text = token_ids_to_text(token_ids, tokenizer)
    print(f"Résultat: {text}")
    print()
    
    # Test 4: Temperature + Top-k (COMME LE TUTORIEL)
    print("-" * 80)
    print("TEST 4: TEMPERATURE + TOP-K (TUTORIEL)")
    print("-" * 80)
    print("Paramètres: temperature=1.4, top_k=25")
    print("Effet: Variété contrôlée + élimination tokens absurdes")
    print()
    
    torch.manual_seed(123)
    token_ids = generate(
        model=model,
        idx=encoded,
        max_new_tokens=15,
        context_size=GPT_CONFIG_124M["context_length"],
        temperature=1.4,
        top_k=25
    )
    text = token_ids_to_text(token_ids, tokenizer)
    print(f"Résultat: {text}")
    print()
    print("💡 Ce texte est différent du texte mémorisé généré par greedy!")
    print("   Il montre que le modèle peut générer du contenu créatif.")
    print()
    
    # Test 5: Différentes graines
    print("-" * 80)
    print("TEST 5: VARIÉTÉ AVEC DIFFÉRENTES GRAINES")
    print("-" * 80)
    print("Paramètres: temperature=1.4, top_k=25, 3 graines différentes")
    print()
    
    for seed in [42, 123, 456]:
        torch.manual_seed(seed)
        token_ids = generate(
            model=model,
            idx=encoded,
            max_new_tokens=15,
            context_size=GPT_CONFIG_124M["context_length"],
            temperature=1.4,
            top_k=25
        )
        text = token_ids_to_text(token_ids, tokenizer)
        print(f"Graine {seed}: {text}")
    
    print()
    
    # Analyse comparative
    print("=" * 80)
    print("ANALYSE COMPARATIVE")
    print("=" * 80)
    print()
    
    print("┌──────────────────────────┬─────────────┬──────────────────────────┐")
    print("│ Configuration            │ Déterministe│ Qualité                  │")
    print("├──────────────────────────┼─────────────┼──────────────────────────┤")
    print("│ Greedy (T=0, k=None)     │     ✓       │ Mémorisé, répétitif      │")
    print("│ Temperature (T=1.4)      │     ✗       │ Varié mais risqué        │")
    print("│ Top-k (k=25)             │     ✓       │ Sûr mais déterministe    │")
    print("│ T=1.4 + k=25 ⭐          │     ✗       │ Varié ET contrôlé        │")
    print("└──────────────────────────┴─────────────┴──────────────────────────┘")
    print()
    
    print("🎯 CONCLUSION:")
    print("  La combinaison temperature + top-k offre le meilleur compromis:")
    print("  • Temperature → variété et créativité")
    print("  • Top-k → élimination des tokens absurdes")
    print("  • Résultat → texte intéressant ET cohérent")
    print()
    
    print("💡 RECOMMANDATIONS:")
    print("  • QA/Précision:  temperature=0.0, top_k=None (greedy)")
    print("  • Qualité:       temperature=0.7, top_k=50")
    print("  • Équilibre:     temperature=1.0, top_k=40")
    print("  • Créatif:       temperature=1.4, top_k=25  ⭐ (tutoriel)")
    print()


if __name__ == "__main__":
    main()
