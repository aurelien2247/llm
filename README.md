<h1 align="center">
    LLM from scratch
</h1>

<h4 align="center"> Une implémentation complète et bien structurée d'un modèle  avec capacités d'entraînement, d'évaluation et de génération de texte. Ainsi que des dossiers afin d'apprendre les fonctionnement basique d'un tokenizer, du mechanisme d'attention </h4>

<p align="center">
  <a href="#🚀-Guide rapide">Guide rapide</a>
  <a href="#🤠-crédits">Crédit</a>
</p>

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

### 4️⃣ Démonstration Top-K Sampling (pédagogique)

```bash
python3 demo_topk.py
```

### 5️⃣ Visualiser l'effet température

```bash
python3 visualize.py
```
### 🔬 Test unifié de génération

```bash
python3 test_generate_unified.py
```
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