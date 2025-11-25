<h1 align="center">
    LLM from scratch
</h1>

<h4 align="center"> Ceci est un petit projet qui m'a permis de découvrir les étapes basiques de tokenization et lacréation d'embeddings pour l'entraînement d'un modèle de langage. </h4>

<p align="center">
  <a href="#🎯-Objectifs">Objectifs</a>
  <a href="#💻-Utilisation">Utilisation</a>
  <a href="#🤠-crédits">Crédit</a>
</p>

## Objectifs

Le but de ce dépôt est d'apprendre à :
- tokeniser un texte (BPE via `tiktoken`),
- construire des paires (input, target) pour la tâche de next-token prediction,
- convertir token IDs en embeddings PyTorch (`nn.Embedding`), et
- Voir l'utilisation d'un `DataLoader` simple.

## Utilisation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
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