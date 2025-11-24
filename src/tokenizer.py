import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


# ----------------------------
#  Tokenizer
# ----------------------------
tokenizer = tiktoken.get_encoding("gpt2")


# ----------------------------
#  Dataset GPT
# ----------------------------
class GPTDataset(Dataset):
    """
    Découpe un texte tokenizé en séquences de longueur fixe (max_length)
    avec un sliding window (stride).
    Retourne (input_ids, target_ids).
    """
    def __init__(self, txt, tokenizer, max_length, stride):
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        
        self.input_ids = []
        self.target_ids = []

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
#  Create dataloader
# ----------------------------
def create_dataloader(text, batch_size=8, max_length=128, stride=128, shuffle=True):
    dataset = GPTDataset(text, tokenizer, max_length, stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ----------------------------
#  Load raw text
# ----------------------------
with open("data/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()


# ----------------------------
#  Dataloader example
# ----------------------------
max_length = 4
dataloader = create_dataloader(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)

inputs, targets = next(iter(dataloader))
print("Token IDs:\n", inputs)
print("Inputs shape:", inputs.shape)


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

# broadcasting position_embeddings: (max_length, dim) → (batch, max_length, dim)
input_embeddings = token_embeddings + position_embeddings

print("Token embeddings:", token_embeddings.shape)
print("Position embeddings:", position_embeddings.shape)
print("Final input embeddings:", input_embeddings.shape)
