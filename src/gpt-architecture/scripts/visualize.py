"""
Visualisation de l'effet du temperature scaling sur les distributions.

Crée un graphique montrant comment la température affecte la distribution
de probabilité des tokens générés.

Utilisation:
    python3 visualize.py
"""

import sys
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, '..')
from decoding import softmax_avec_temperature


def main():
    """Fonction principale de visualisation."""
    # Exemple avec un petit vocabulaire pour illustration
    vocab = {
        "closer": 0,
        "every": 1,
        "effort": 2,
        "forward": 3,
        "inches": 4,
        "moves": 5,
        "pizza": 6,
        "toward": 7,
        "you": 8,
    }

    inverse_vocab = {v: k for k, v in vocab.items()}

    # Logits générés par le modèle (exemple du tutoriel)
    next_token_logits = torch.tensor([4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79])

    print("Logits bruts du modèle:")
    for word, logit_val in zip(vocab.keys(), next_token_logits.tolist()):
        print(f"  {word:12s}: {logit_val:7.2f}")
    print()

    # Températures à tester
    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]

    # Calculer les probabilités pour chaque température
    probas_list = []
    for T in temperatures:
        probas = softmax_avec_temperature(next_token_logits, temperature=T)
        probas_list.append(probas)
        
        print(f"Temperature = {T}:")
        top_5_indices = torch.topk(probas, k=5).indices
        for idx in top_5_indices:
            word = inverse_vocab[idx.item()]
            prob = probas[idx.item()].item()
            print(f"  {word:12s}: {prob:.4f} {'█' * int(prob * 50)}")
        print()

    # Créer le graphique
    print("Création du graphique...")
    
    fig, axes = plt.subplots(1, len(temperatures), figsize=(16, 4))
    fig.suptitle('Effet du Temperature Scaling sur les Distributions de Probabilité', 
                 fontsize=14, fontweight='bold')

    x = torch.arange(len(vocab))
    colors = plt.cm.viridis(torch.linspace(0, 1, len(temperatures)))

    for idx, (T, probas, ax, color) in enumerate(zip(temperatures, probas_list, axes, colors)):
        bars = ax.bar(x, probas.numpy(), color=color, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Probabilité', fontweight='bold')
        ax.set_xlabel('Tokens', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(vocab.keys(), rotation=45, ha='right')
        ax.set_title(f'Temperature = {T}', fontweight='bold')
        ax.set_ylim([0, max(probas_list[2].max(), 0.6)])  # Normaliser par rapport à T=1.0
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Ajouter les valeurs sur les barres
        for bar, prob in zip(bars, probas):
            height = bar.get_height()
            if height > 0.02:  # Afficher seulement si > 2%
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{prob:.3f}',
                       ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig('temperature-visualization.pdf', dpi=300, bbox_inches='tight')
    print("✓ Graphique sauvegardé dans 'temperature-visualization.pdf'\n")
    # plt.show()  # Désactivé pour éviter de bloquer le script en mode headless

    # Analyse de l'effet de la température
    print(f"{'='*70}")
    print("ANALYSE DE L'EFFET DE LA TEMPÉRATURE")
    print(f"{'='*70}\n")

    print("Temperature = 0.1 (très confiant):")
    print("  → Distribution TRÈS nette")
    print("  → Le token 'forward' est sélectionné ~99% du temps")
    print("  → Comportement proche du greedy decoding (argmax)")
    print("  → ✓ Texte cohérent et prévisible")
    print("  → ✗ Peu de variété\n")

    print("Temperature = 1.0 (normal):")
    print("  → Distribution normale (pas de scaling)")
    print("  → Probabilités originales du modèle")
    print("  → ✓ Équilibre cohérence/variété")
    print("  → ✓ Bon pour la plupart des cas\n")

    print("Temperature = 5.0 (très aléatoire):")
    print("  → Distribution très uniforme")
    print("  → Tous les tokens ont des chances similaires")
    print("  → ✓ Texte très varié")
    print("  → ✗ Souvent incohérent (tokens aléatoires)")
    print("  → Exemple: 'every effort moves you pizza' ~4% du temps\n")

    print(f"{'='*70}")
    print("RECOMMANDATIONS D'UTILISATION")
    print(f"{'='*70}\n")

    print("Pour une question-réponse précise:")
    print("  → Utiliser temperature = 0.1 - 0.3 ou greedy decoding\n")

    print("Pour une conversation naturelle:")
    print("  → Utiliser temperature = 0.7 - 1.0 ou top-k/top-p\n")

    print("Pour de la génération créative:")
    print("  → Utiliser temperature = 1.2 - 1.8 ou top-p (p=0.95)\n")


if __name__ == "__main__":
    main()
