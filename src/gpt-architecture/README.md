# Architecture GPT - Implémentation complète

Ce dossier contient une implémentation complète d'un modèle GPT-124M avec capacités d'entraînement, d'évaluation et de génération de texte.

## 📋 Vue d'ensemble

Le projet implémente l'architecture Transformer suivant les principes du livre "Build a Large Language Model from Scratch" (Sebastian Raschka).

### Caractéristiques principales
- **Tokenization flexible** : Support de tiktoken (BPE) et tokenizers personnalisés
- **Pipeline de données optimisé** : Fenêtres glissantes avec support de tokenizers multiples
- **Boucle d'entraînement complète** : Entraînement, validation et génération de texte
- **Gestion des checkpoints** : Sauvegarde et chargement du modèle entraîné
- **Génération de texte** : Génération autonome avec control de contexte

## 📁 Structure des fichiers

### Modules core
- **`model.py`** - Classe GPTModel : architecture complète du modèle
- **`layers.py`** - Composants des couches (LayerNorm, GELU, FeedForward, TransformerBlock)
- **`attention.py`** - Implémentation de l'attention multi-têtes avec masque causal
- **`generate.py`** - Fonction de génération simple de texte

### Pipeline de données et prétraitement
- **`tokenizer.py`** - SimpleTokenizerV1/V2 pour la tokenization personnalisée
- **`data.py`** - GPTDatasetV1 et create_dataloader_v1 avec support multi-tokenizer
- **`utils.py`** - Utilitaires : conversion texte/tokens, calcul de perte

### Scripts d'entraînement et inférence
- **`training.py`** - Fonctions utilitaires d'entraînement (train_model_simple, evaluate_model, etc.)
- **`train.py`** - Script complet d'entraînement avec gestion des checkpoints
- **`main.py`** - Script de démonstration simple avec génération de texte
- **`load_model.py`** - Script pour charger un modèle entraîné et générer du texte

## 🚀 Utilisation

### Installation des dépendances
```bash
cd /Users/moignet/Projects/llm
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Entraînement du modèle
```bash
python3 src/gpt-architecture/train.py
```

Cela va :
- Charger le texte depuis `data/the-verdict.txt`
- Créer les dataloaders train/val (90/10 split)
- Initialiser le modèle GPT-124M
- Entraîner pendant 10 epochs avec évaluation tous les 5 epochs
- Sauvegarder le checkpoint dans `gpt-model.pt`
- Afficher les courbes de perte (si matplotlib est disponible)

### Génération de texte avec un modèle entraîné
```bash
python3 src/gpt-architecture/load_model.py --model_path gpt-model.pt --prompt "Hello, I" --max_tokens 100
```

### Démonstration simple (modèle non entraîné)
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
