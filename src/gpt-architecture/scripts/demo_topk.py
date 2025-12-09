"""
Démonstration du Top-K Sampling (suivant le tutoriel).

Ce script illustre comment le top-k sampling améliore la génération de texte
en éliminant les tokens peu probables qui causent du texte absurde.

Concepts démontrés:
1. Problème: temperature seule peut générer des tokens absurdes
2. Solution: top-k garde seulement les k tokens les plus probables
3. Résultat: diversité conservée, mais sans tokens complètement incohérents

Utilisation:
    python3 demo_topk.py
"""

import sys
import torch

sys.path.insert(0, '..')


def main():
    """Démonstration du top-k sampling étape par étape."""
    
    print("=" * 80)
    print("DÉMONSTRATION : TOP-K SAMPLING")
    print("=" * 80)
    print()
    
    # Vocabulaire d'exemple (du tutoriel)
    vocab = {
        "closer": 0,
        "every": 1,
        "effort": 2,
        "forward": 3,
        "inches": 4,
        "moves": 5,
        "pizza": 6,  # Token absurde dans ce contexte !
        "toward": 7,
        "you": 8,
    }
    
    inverse_vocab = {v: k for k, v in vocab.items()}
    
    # Logits du modèle (du tutoriel)
    next_token_logits = torch.tensor(
        [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
    )
    
    print("📊 LOGITS BRUTS DU MODÈLE")
    print("-" * 80)
    for word, logit_val in zip(vocab.keys(), next_token_logits.tolist()):
        print(f"  {word:12s}: {logit_val:7.2f}")
    print()
    
    # Étape 1: Probabilités sans filtrage (avec temperature=1.0)
    print("📊 ÉTAPE 1: PROBABILITÉS SANS FILTRAGE (température=1.0)")
    print("-" * 80)
    probas_no_filter = torch.softmax(next_token_logits, dim=0)
    
    for idx, (word, prob) in enumerate(zip(vocab.keys(), probas_no_filter.tolist())):
        bar = "█" * int(prob * 100)
        print(f"  {word:12s}: {prob:6.4f} {bar}")
    print()
    print("⚠️  PROBLÈME: 'pizza' a 0.0018 de probabilité (0.18%)")
    print("    Avec temperature élevée, peut être sélectionné → texte absurde!")
    print()
    
    # Étape 2: Sélectionner les top-k logits
    print("📊 ÉTAPE 2: SÉLECTIONNER LES TOP-K LOGITS (k=3)")
    print("-" * 80)
    top_k = 3
    top_logits, top_pos = torch.topk(next_token_logits, top_k)
    
    print(f"Top {top_k} logits: {top_logits.tolist()}")
    print(f"Top {top_k} positions: {top_pos.tolist()}")
    print()
    
    print("Tokens correspondants:")
    for pos, logit in zip(top_pos.tolist(), top_logits.tolist()):
        word = inverse_vocab[pos]
        print(f"  {word:12s}: {logit:7.2f}")
    print()
    
    # Étape 3: Masquer les logits hors du top-k
    print("📊 ÉTAPE 3: MASQUER LES TOKENS HORS DU TOP-K")
    print("-" * 80)
    print("Méthode: mettre à -inf tous les logits < plus petit top-k logit")
    print()
    
    # Utiliser torch.where pour masquer (comme dans le tutoriel)
    new_logits = torch.where(
        condition=next_token_logits < top_logits[-1],  # Condition: logit < min(top-k)
        input=torch.tensor(float("-inf")),              # Si vrai: -inf (masqué)
        other=next_token_logits                         # Sinon: garder logit original
    )
    
    print("Logits après masquage:")
    for word, logit in zip(vocab.keys(), new_logits.tolist()):
        if logit == float("-inf"):
            print(f"  {word:12s}: -inf (MASQUÉ)")
        else:
            print(f"  {word:12s}: {logit:7.2f}")
    print()
    
    # Étape 4: Appliquer softmax pour obtenir les probabilités
    print("📊 ÉTAPE 4: APPLIQUER SOFTMAX (renormalisation)")
    print("-" * 80)
    topk_probas = torch.softmax(new_logits, dim=0)
    
    print("Probabilités après top-k sampling:")
    total_prob = 0.0
    for word, prob in zip(vocab.keys(), topk_probas.tolist()):
        if prob > 0:
            bar = "█" * int(prob * 100)
            print(f"  {word:12s}: {prob:6.4f} {bar}")
            total_prob += prob
        else:
            print(f"  {word:12s}: 0.0000 (éliminé)")
    
    print()
    print(f"✓ Somme des probabilités: {total_prob:.6f}")
    print()
    
    # Comparaison avant/après
    print("=" * 80)
    print("COMPARAISON AVANT/APRÈS")
    print("=" * 80)
    print()
    
    print("┌─────────────────────────────────────────────────────────────────────┐")
    print("│                      SANS TOP-K         │        AVEC TOP-K (k=3)   │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    
    for word in vocab.keys():
        idx = vocab[word]
        prob_before = probas_no_filter[idx].item()
        prob_after = topk_probas[idx].item()
        
        bar_before = "█" * int(prob_before * 50)
        bar_after = "█" * int(prob_after * 50)
        
        status = ""
        if prob_after == 0:
            status = "✗ ÉLIMINÉ"
        elif prob_after > prob_before * 1.5:
            status = "↑ BOOSTÉ"
        
        print(f"│ {word:10s} {prob_before:6.4f} {bar_before:20s}│ {prob_after:6.4f} {bar_after:20s} {status:10s} │")
    
    print("└─────────────────────────────────────────────────────────────────────┘")
    print()
    
    # Analyse des résultats
    print("=" * 80)
    print("ANALYSE DES RÉSULTATS")
    print("=" * 80)
    print()
    
    print("✅ AVANTAGES DU TOP-K SAMPLING:")
    print("  • Élimine les tokens absurdes ('pizza', 'effort', etc.)")
    print("  • Conserve les tokens cohérents ('forward', 'toward', 'inches')")
    print("  • Réduit le risque de génération incohérente")
    print("  • Peut être combiné avec temperature scaling")
    print()
    
    print("📊 EFFETS OBSERVÉS:")
    pizza_before = probas_no_filter[vocab["pizza"]].item()
    pizza_after = topk_probas[vocab["pizza"]].item()
    print(f"  • 'pizza' : {pizza_before:.4f} → {pizza_after:.4f} (éliminé!)")
    
    forward_before = probas_no_filter[vocab["forward"]].item()
    forward_after = topk_probas[vocab["forward"]].item()
    print(f"  • 'forward': {forward_before:.4f} → {forward_after:.4f} (boosté!)")
    print()
    
    print("💡 RECOMMANDATIONS:")
    print("  • k=3-10   : Très conservateur, texte cohérent mais répétitif")
    print("  • k=30-50  : Équilibre qualité/variété (RECOMMANDÉ)")
    print("  • k=100+   : Plus de variété, risque de tokens peu pertinents")
    print()
    
    print("🔧 COMBINAISON AVEC TEMPERATURE:")
    print("  • top_k=50 + temperature=0.7 : Génération de qualité")
    print("  • top_k=30 + temperature=1.0 : Équilibre")
    print("  • top_k=100 + temperature=1.5 : Créatif mais contrôlé")
    print()


if __name__ == "__main__":
    main()
