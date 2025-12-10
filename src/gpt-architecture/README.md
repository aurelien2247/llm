<h1 align="center">
    LLM from scratch
</h1>

<h4 align="center"> Une implémentation complète et bien structurée d'un modèle avec capacités d'entraînement, d'évaluation et de génération de texte. Conçu comme POC pour démontrer la faisabilité d'une implémentation LLM from scratch en production. </h4>

<p align="center">
  <a href="#📊-Vue d'ensemble">Vue d'ensemble</a>
  <a href="#🏛️-Architecture du Projet">Architecture du Projet</a>
  <a href="#✨-Caractéristiques principales">Caractéristiques principales</a>
  <a href="#📝-Notes d'implémentation">Notes d'implémentation</a>
  <a href="#🤠-crédits">Crédit</a>
</p>

## 📊 Vue d'ensemble

- **Modèle**: GPT-124M (Transformer decoder-only)
- **Tokenization**: tiktoken (BPE) + support tokenizers personnalisés
- **Framework**: PyTorch avec architecture modulaire
- **Stratégies de génération**: greedy, temperature, top-k, top-p (nucleus)
- **Documentation**: Complète en français avec docstrings

## 🏛️ Architecture du Projet

```
gpt-architecture/
├── core/                          # ⚙️ Architecture du modèle
│   ├── __init__.py
│   ├── model.py                   # GPTModel (classe principale)
│   ├── layers.py                  # LayerNorm, GELU, FeedForward, TransformerBlock
│   ├── attention.py               # MultiHeadAttention avec masque causal
│   └── utils.py                   # Utilitaires (text_to_token_ids, calc_loss_*)
│
├── data/                          # 📦 Pipeline de données
│   ├── __init__.py
│   ├── loader.py                  # GPTDatasetV1, create_dataloader_v1
│   └── tokenizer.py               # SimpleTokenizerV1, SimpleTokenizerV2
│
├── decoding/                      # 🎲 Stratégies de génération
│   ├── __init__.py
│   ├── strategies.py              # softmax_avec_temperature, top_k, top_p, etc.
│   └── generator.py               # Fonctions generate_text_*
│
├── scripts/                       # 🚀 Exécutables
│   ├── __init__.py
│   ├── train.py                   # Entraînement complet du modèle
│   ├── infer.py                   # Inférence avec différentes stratégies
│   ├── demo.py                    # Démonstration des stratégies côte à côte
│   ├── demo_topk.py               # Démonstration pédagogique du top-k
│   ├── test_generate_unified.py   # Tests/validation de la fonction `generate()` unifiée
│   ├── messages.py                # Messages partagés entre scripts
│   └── visualize.py               # Visualisation de température (PDF)
│
├── training.py                    # 📚 Utilitaires d'entraînement (train_model_simple, etc.)
├── config.py                      # ⚙️ Configuration centralisée
├── README.md                      # 📖 Ce fichier
└── *.pdf                          # Sorties graphiques (ignorées par git)
```
### Flux forward
```
Entrée (token IDs)
    ↓
Embedding tokens + positions
    ↓
Dropout
    ↓
N couches Transformer (attention + feed-forward)
    ↓
LayerNorm
    ↓
Projection de sortie
    ↓
Logits (vocab_size)
```

### Attention multi-têtes
- Division en 12 têtes (768 / 12 = 64 dim par tête)
- Masque causal pour prévenir l'attention sur les jetons futurs
- Softmax avec facteur d'échelle

### Bloc Transformer
```
Attention multi-têtes
    ↓ + connexion résidu
LayerNorm
    ↓
Feed-Forward (MLP)
    ↓ + connexion résidu
```

## ✨ Caractéristiques principales

### 🔐 Modularité
- **Séparation des responsabilités**: core (modèle), data (données), decoding (génération), scripts (exécution)
- **Imports explicites**: Facilement repérable où vient chaque fonction
- **Configuration centralisée**: `config.py` pour tous les hyperparamètres

### 🧠 Architecture Transformer
- Embedding + positional encoding
- Multi-head attention avec masque causal
- Couches Feed-Forward (expansion 4x)
- Layer normalization + résidus
- 12 couches × 12 têtes (768 dim)

## 📝 Notes d'implémentation

### Tokenization flexible
Le pipeline supporte deux approches :
1. **tiktoken** (par défaut) : Tokenization BPE compatible OpenAI
2. **SimpleTokenizerV2** : Tokenizer personnalisé pour expérimentation pédagogique

## Références
- "Build a Large Language Model from Scratch" - Sebastian Raschka
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani et al.

## 🤠 Crédits

<table>
    <tr>
        <td align="center">
            <a href="mailto:aurelien.moignet@imt-atlantique.net">
                <img src="https://avatars.githubusercontent.com/u/76565476?v=4" width="100px;" alt="Image de profil" style="border-radius: 100%"/>
                <br />
                <sub><b>Aurélien</b></sub>
            </a>
            <br />
        </td>
        <td align="center">
                <img src="https://avatars.githubusercontent.com/u/5618407?v=4" width="100px;" alt="Image de profil" style="border-radius: 100%"/>
                <br />
                <sub><b>Sebastian Raschka</b></sub>
                <sub><b>J'ai appris la création des llms from scratch grace aux livres <a href="https://www.amazon.fr/Build-Large-Language-Model-Scratch/dp/1633437167">Build a Large Language Model from Scratch</a></b></sub>
            <br />
        </td>
    </tr>
</table>