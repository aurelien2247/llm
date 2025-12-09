"""
Script d'inférence avec support des différentes stratégies de décoding.

Ce script:
1. Charge un modèle GPT entraîné
2. Permet de générer du texte avec différentes stratégies
3. Supporte: greedy, temperature, top-k, top-p

Utilisation:
    python3 infer.py --help
    python3 infer.py --prompt "Every effort" --strategy greedy --max_tokens 50
    python3 infer.py --prompt "Every effort" --strategy temperature --temperature 0.7
    python3 infer.py --prompt "Every effort" --strategy top_k --k 50
    python3 infer.py --prompt "Every effort" --strategy top_p --p 0.9
"""

import sys
import argparse
import torch
import tiktoken

sys.path.insert(0, '..')
from config import GPT_CONFIG_124M, GENERATION_CONFIG, PATHS
from core import GPTModel, text_to_token_ids, token_ids_to_text
from decoding import (
    generate_text_simple,
    generate_text_temperature,
    generate_text_top_k,
    generate_text_top_p
)
from messages import MODEL_LOADED, error_not_found, MISSING_CHECKPOINT_HELP


def main():
    """Fonction principale d'inférence."""
    parser = argparse.ArgumentParser(
        description="Charger un modèle GPT entraîné et générer du texte."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=PATHS['model_checkpoint'],
        help=f"Chemin vers le checkpoint du modèle (défaut: {PATHS['model_checkpoint']})"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Every effort moves you",
        help="Texte initial pour la génération (défaut: 'Every effort moves you')"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=GENERATION_CONFIG['max_tokens_default'],
        help=f"Nombre de tokens à générer (défaut: {GENERATION_CONFIG['max_tokens_default']})"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["greedy", "temperature", "top_k", "top_p"],
        default=GENERATION_CONFIG['default_strategy'],
        help=f"Stratégie de décodage (défaut: {GENERATION_CONFIG['default_strategy']})"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=GENERATION_CONFIG['default_temperature'],
        help=f"Facteur de température (défaut: {GENERATION_CONFIG['default_temperature']}). Utilisé avec --strategy temperature"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=GENERATION_CONFIG['default_k'],
        help=f"Nombre de tokens pour top-k (défaut: {GENERATION_CONFIG['default_k']}). Utilisé avec --strategy top_k"
    )
    parser.add_argument(
        "--p",
        type=float,
        default=GENERATION_CONFIG['default_p'],
        help=f"Seuil de probabilité pour nucleus sampling (défaut: {GENERATION_CONFIG['default_p']}). Utilisé avec --strategy top_p"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Graine pour la reproductibilité (défaut: 42)"
    )

    args = parser.parse_args()

    # Fixer la graine pour la reproductibilité
    torch.manual_seed(args.seed)

    # Charger le modèle
    print(f"Chargement du modèle depuis {args.model_path}...")
    device = torch.device("cpu")
    model = GPTModel(GPT_CONFIG_124M)
    
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.to(device)
        model.eval()
        print(MODEL_LOADED)
    except FileNotFoundError:
        print(error_not_found(args.model_path))
        print(MISSING_CHECKPOINT_HELP)
        return

    # Tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Encoder le prompt
    encoded = text_to_token_ids(args.prompt, tokenizer).to(device)

    # Afficher les paramètres
    print(f"{'='*70}")
    print("PARAMÈTRES DE GÉNÉRATION")
    print(f"{'='*70}")
    print(f"Stratégie:       {args.strategy}")
    print(f"Prompt:          {args.prompt}")
    print(f"Max tokens:      {args.max_tokens}")
    
    if args.strategy == "temperature":
        print(f"Température:     {args.temperature}")
    elif args.strategy == "top_k":
        print(f"k:               {args.k}")
    elif args.strategy == "top_p":
        print(f"p:               {args.p}")
    print(f"{'='*70}\n")

    # Générer le texte
    print("Génération en cours...\n")
    with torch.no_grad():
        if args.strategy == "greedy":
            token_ids = generate_text_simple(
                model=model,
                idx=encoded,
                max_new_tokens=args.max_tokens,
                context_size=GPT_CONFIG_124M["context_length"]
            )
        elif args.strategy == "temperature":
            token_ids = generate_text_temperature(
                model=model,
                idx=encoded,
                max_new_tokens=args.max_tokens,
                context_size=GPT_CONFIG_124M["context_length"],
                temperature=args.temperature
            )
        elif args.strategy == "top_k":
            token_ids = generate_text_top_k(
                model=model,
                idx=encoded,
                max_new_tokens=args.max_tokens,
                context_size=GPT_CONFIG_124M["context_length"],
                k=args.k,
                temperature=1.0
            )
        elif args.strategy == "top_p":
            token_ids = generate_text_top_p(
                model=model,
                idx=encoded,
                max_new_tokens=args.max_tokens,
                context_size=GPT_CONFIG_124M["context_length"],
                p=args.p,
                temperature=1.0
            )

    # Décoder et afficher
    generated_text = token_ids_to_text(token_ids, tokenizer)
    
    print(f"{'='*70}")
    print("TEXTE GÉNÉRÉ")
    print(f"{'='*70}")
    print(generated_text)
    print(f"{'='*70}\n")

    # Afficher des conseils
    print("Conseils d'utilisation:")
    print("  • temperature=0.3: Texte précis et cohérent")
    print("  • temperature=0.7: Équilibre cohérence/variété")
    print("  • temperature=1.5: Texte créatif et varié")
    print("  • top_k=50: Élimine les tokens peu probables")
    print("  • top_p=0.9: Flexible, maintient 90% de la masse de probabilité")
    print()


if __name__ == "__main__":
    main()
