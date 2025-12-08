import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


# ----------------------------
#  Tokenizer
# ----------------------------
# Utiliser l'encodage GPT2 de tiktoken
tokenizer = tiktoken.get_encoding("gpt2")


# ----------------------------
#  Dataset GPT
# ----------------------------
class GPTDataset(Dataset):
    """
    Découpe un texte tokenisé en séquences de longueur fixe (max_length)
    avec une fenêtre glissante (stride).
    Retourne les paires (input_ids, target_ids) pour l'entraînement.
    """
    def __init__(self, txt, tokenizer, max_length, stride):
        # Tokeniser le texte entier
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        
        self.input_ids = []
        self.target_ids = []

        # Créer des paires avec la fenêtre glissante
        for i in range(0, len(token_ids) - max_length, stride):
            inp = token_ids[i : i + max_length]
            tgt = token_ids[i + 1 : i + max_length + 1]

            self.input_ids.append(torch.tensor(inp, dtype=torch.long))
            self.target_ids.append(torch.tensor(tgt, dtype=torch.long))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


# ----------------------------
#  Créer le dataloader
# ----------------------------
def create_dataloader(text, batch_size=8, max_length=128, stride=128, shuffle=True):
    """Créer un DataLoader à partir du texte brut."""
    dataset = GPTDataset(text, tokenizer, max_length, stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ----------------------------
#  Charger le texte brut
# ----------------------------
with open("data/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()


# ----------------------------
#  Exemple de dataloader
# ----------------------------
max_length = 4
dataloader = create_dataloader(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)

inputs, targets = next(iter(dataloader))
print("Identifiants de jetons:\n", inputs)
print("Forme des entrées:", inputs.shape)


# ----------------------------
#  Embeddings
# ----------------------------
vocab_size = 50257
embed_dim = 256

token_emb = torch.nn.Embedding(vocab_size, embed_dim)
pos_emb   = torch.nn.Embedding(max_length, embed_dim)

token_embeddings = token_emb(inputs)
position_ids = torch.arange(max_length)
position_embeddings = pos_emb(position_ids)

# Diffusion de position_embeddings: (max_length, dim) → (batch, max_length, dim)
input_embeddings = token_embeddings + position_embeddings

print("Forme des embeddings de jetons:", token_embeddings.shape)
print("Forme des embeddings de position:", position_embeddings.shape)
print("Forme des embeddings d'entrée finaux:", input_embeddings.shape)
