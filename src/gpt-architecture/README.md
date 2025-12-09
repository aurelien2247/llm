# 🏗️ Architecture GPT - Implémentation Professionnelle

Une implémentation complète et bien structurée d'un modèle GPT-124M avec capacités d'entraînement, d'évaluation et de génération de texte. Conçu comme POC pour démontrer la faisabilité d'une implémentation LLM from scratch en production.

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
│   └── visualize.py               # Visualisation de température (PDF)
│
├── training.py                    # 📚 Utilitaires d'entraînement (train_model_simple, etc.)
├── config.py                      # ⚙️ Configuration centralisée
├── README.md                      # 📖 Ce fichier
└── *.pdf                          # Sorties graphiques (ignorées par git)
```

## 🔄 Hiérarchie des dépendances

```
scripts/  ← Point d'entrée utilisateur
    ↓
training.py
    ↓
core + data + decoding
    ↓
PyTorch + tiktoken
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

### 📊 Pipeline de données
- Fenêtres glissantes (sliding windows)
- Support tiktoken (BPE) et tokenizers personnalisés
- Train/val split configurable (défaut: 90/10)
- DataLoader PyTorch standard

### 🎛️ Stratégies de génération
| Stratégie | Cohérence | Variété | Cas d'usage |
|-----------|-----------|---------|------------|
| **Greedy** | ████████ | ██░░░░░░ | QA précis |
| **Temperature 0.3** | ████████ | ████░░░░ | Texte contrôlé |
| **Temperature 1.0** | ██████░░ | ████████ | Usage général |
| **Top-k (k=50)** | ██████░░ | ████████ | Équilibre |
| **Top-p (p=0.9)** | ██████░░ | ████████ | ⭐ Recommandé |

### 📈 Monitoring & Visualisation
- Suivi des losses train/val
- Génération d'exemples tous les epochs
- Graphique PDF des courbes de perte
- Visualisation temperature effects (PDF)

## 🚀 Guide rapide

### Installation

```bash
cd /Users/moignet/Projects/llm
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 1️⃣ Entraîner le modèle

```bash
cd src/gpt-architecture/scripts
python3 train.py
```

**Résultat**: Crée `gpt-model.pt` et `loss-plot.pdf`

### 2️⃣ Générer du texte (inférence)

```bash
# Greedy (déterministe)
python3 infer.py --strategy greedy --prompt "Every effort" --max_tokens 50

# Température (variété contrôlée)
python3 infer.py --strategy temperature --temperature 0.7 --max_tokens 50

# Top-k (évite les tokens absurdes)
python3 infer.py --strategy top_k --k 50 --max_tokens 50

# Top-p (nucleus sampling)
python3 infer.py --strategy top_p --p 0.9 --max_tokens 50
```

### 3️⃣ Comparer les stratégies

```bash
python3 demo.py
```

Affiche 6 variantes côte à côte avec analyses.

### 4️⃣ Visualiser l'effet température

```bash
python3 visualize.py
```

Génère `temperature-visualization.pdf`.
```bash
python3 src/gpt-architecture/main.py
```

## 🧠 Configuration du modèle

La configuration par défaut (GPT_CONFIG_124M) :
```python
{
    "vocab_size": 50257,        # Taille du vocabulaire GPT-2
    "context_length": 1024,     # Longueur maximale du contexte
    "emb_dim": 768,             # Dimension des embeddings
    "n_heads": 12,              # Nombre de têtes d'attention
    "n_layers": 12,             # Nombre de blocs Transformer
    "drop_rate": 0.1,           # Taux de dropout
    "qkv_bias": False           # Biais pour QKV
}
```

## 📊 Résultats d'entraînement

Le modèle entraîné sur le texte "The Verdict" montre :
- **Perte d'entraînement initial** : 9.787
- **Perte d'entraînement final** : 1.314
- **Convergence** : Progressive sur 10 epochs
- **Qualité de génération** : Du texte gibberish au texte presque naturel

## 🎛️ Stratégies de décodage

Le projet implémente 4 stratégies pour contrôler l'aléatoire lors de la génération:

### 1. Greedy Decoding (argmax)
Sélectionne le token avec la plus haute probabilité à chaque étape.
- ✅ Déterministe et reproductible
- ✅ Texte cohérent
- ❌ Peu de variété

### 2. Temperature Scaling
Applique un scaling aux logits avant softmax pour contrôler la "confiance" du modèle.
- **Temperature < 1** (ex: 0.3): Distribution plus nette → texte plus cohérent
- **Temperature = 1**: Pas de scaling → comportement normal
- **Temperature > 1** (ex: 2.0): Distribution plus plate → plus d'aléatoire

Formule: `scaled_logits = logits / temperature`

### 3. Top-k Sampling
Garde seulement les k tokens les plus probables et élimine le reste.
- ✅ Évite les tokens absurdes
- ✅ Meilleure qualité que temperature seul
- ✓ Nombre de tokens constant

Exemple: k=50 garde les 50 tokens les plus probables

### 4. Top-p (Nucleus) Sampling
Garde les tokens dont la probabilité cumulée atteint p (ex: 90%).
- ✅ Flexible: ajuste le nombre de tokens selon la distribution
- ✅ Bonne qualité et variété
- ✓ Adapte le niveau de contrôle dynamiquement

Exemple: p=0.9 garde les tokens représentant 90% de la masse de probabilité

### Comparaison et recommandations

| Stratégie | Cohérence | Variété | Cas d'usage |
|-----------|-----------|---------|------------|
| Greedy | ████████░░ | ██░░░░░░░░ | QA précis |
| T=0.3 | ████████░░ | ████░░░░░░ | Texte précis |
| T=1.0 | ██████░░░░ | ████████░░ | Équilibre |
| T=2.0 | ████░░░░░░ | ██████████ | Créativité |
| Top-k | ██████░░░░ | ████████░░ | Équilibre |
| Top-p | ██████░░░░ | ████████░░ | Recommandé |

**Recommandations:**
- **QA/Précision**: temperature=0.1-0.3 ou greedy
- **Usage général**: temperature=0.7-1.0 ou top-k/top-p
- **Créativité**: temperature=1.5-2.0 ou top-p (p=0.95)

## 🔧 Architecture détaillée

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

## 📚 Références

- "Build a Large Language Model from Scratch" - Sebastian Raschka
- [OpenAI GPT-2](https://openai.com/blog/better-language-models/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani et al.

## 📝 Notes d'implémentation

### Tokenization flexible
Le pipeline supporte deux approches :
1. **tiktoken** (par défaut) : Tokenization BPE compatible OpenAI
2. **SimpleTokenizerV2** : Tokenizer personnalisé pour expérimentation pédagogique

### Gestion des erreurs
Le code inclut une gestion robuste :
- Try/except pour la compatibility des tokenizers
- Fallback gracieux pour matplotlib (si non installé)
- Gestion des appareils (CPU/GPU)

## ⚡ Performance

- **Durée d'entraînement** : ~5 minutes (10 epochs) sur CPU
- **Taille du checkpoint** : ~621 MB (modèle seul)
- **Mémoire requise** : ~2 GB pour entraînement

## 🤝 Crédits

Implémentation créée pour apprentissage pratique de l'architecture GPT et du deep learning.
Basée sur les principes pédagogiques du livre de Sebastian Raschka.
